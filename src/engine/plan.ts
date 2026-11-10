import { kvBytesPerToken, paramCounts, type ParamCounts } from './model.ts'
import { DTYPES, type Assumptions, type Device, type Layout, type ModelSpec, type Targets, type Workload } from './types.ts'

export const GB = 1e9
/** Activations travel between devices in BF16. */
const ACT_BYTES = 2

export const DEFAULT_ASSUMPTIONS: Assumptions = { bwUtil: 0.75, mfu: 0.45, linkUtil: 0.7, reserveGB: 1.5, reserveFrac: 0.05, amortYears: 3, kwhUsd: 0.15, costMode: 'own' }

export type Bound = 'memory' | 'compute' | 'interconnect'

export interface MemoryFit {
  weightsGB: number
  kvGB: number
  reserveGB: number
  usedGB: number
  capGB: number
  weightsFit: boolean
  fits: boolean
  /** KV bytes one token adds on the most loaded device. */
  kvPerTokenDev: number
  /** Tokens of KV cache that fit beside the weights, across all sequences. */
  maxTokens: number
  maxBatch: number
  maxContext: number
}

export interface Phase { ms: number; tokS: number; computeMs: number; memMs: number; commMs: number; bound: Bound }

export interface PlanResult {
  device: Device
  count: number
  tp: number
  pp: number
  params: ParamCounts
  mem: MemoryFit
  decode: Phase & { perUserTokS: number; batch: number }
  prefill: Phase
  ttftMs: number
  requestMs: number
  cost: { perHour: number; capex: number; usdPerMTok: number; tokPerJ: number; rented: boolean }
  warnings: string[]
}

/** Expected number of distinct experts hit when n tokens each pick k of E experts uniformly. */
export function expectedExperts(E: number, k: number, n: number): number {
  if (n <= 0) return 0
  return E * (1 - Math.pow(1 - k / E, n))
}

export function validTp(m: ModelSpec, tp: number): boolean {
  if (tp < 1 || tp > m.heads || m.heads % tp !== 0) return false
  if (m.mla) return true
  return m.kvHeads % tp === 0 || tp % m.kvHeads === 0
}

/** How many ways the KV cache is split across a TP group. KV heads replicate once TP exceeds them; MLA latents always replicate. */
export const kvShard = (m: ModelSpec, tp: number) => m.mla ? 1 : Math.min(tp, m.kvHeads)

/** Ring all-reduce time in seconds for `bytes` across `n` devices. */
export function allReduceS(bytes: number, n: number, d: Device, a: Assumptions): number {
  if (n <= 1) return 0
  return (2 * (n - 1) / n) * bytes / (d.linkGBs * GB * a.linkUtil) + 2 * (n - 1) * d.linkLatencyUs * 1e-6
}

interface Split { fixed: number; expertsAll: number; perLayerAvg: number; head: number; emb: number }
function weightSplit(m: ModelSpec, p: ParamCounts, wbytes: number): Split {
  const expertsAll = m.moe ? p.moeLayers * m.moe.experts * p.perExpert * wbytes : 0
  const layers = p.layerTotal * wbytes
  return { fixed: layers - expertsAll, expertsAll, perLayerAvg: layers / m.layers, head: m.vocab * m.hidden * wbytes, emb: p.embedding * wbytes }
}

export function memoryFit(m: ModelSpec, wl: Workload, l: Layout, a: Assumptions): MemoryFit {
  const p = paramCounts(m)
  const wb = DTYPES[wl.weightDtype].bytes
  const s = weightSplit(m, p, wb)
  const layersOnStage = Math.ceil(m.layers / l.pp)
  // The heaviest stage: its share of layers plus the embedding (first stage) or head (last stage).
  const edge = l.pp === 1 ? s.emb + (m.tied ? 0 : s.head) : Math.max(s.emb, m.tied ? s.emb : s.head)
  const weights = (layersOnStage * s.perLayerAvg + edge) / l.tp
  const kvPerTokenDev = kvBytesPerToken(m, wl.kvDtype) * (layersOnStage / m.layers) / kvShard(m, l.tp)
  const cap = l.device.memGB * GB
  const reserve = a.reserveGB * GB + a.reserveFrac * cap
  const free = cap - reserve - weights
  const maxTokens = free > 0 ? Math.floor(free / kvPerTokenDev) : 0
  const kv = kvPerTokenDev * wl.context * wl.batch
  return {
    weightsGB: weights / GB, kvGB: kv / GB, reserveGB: reserve / GB, usedGB: (weights + kv + reserve) / GB, capGB: l.device.memGB,
    weightsFit: free > 0, fits: free >= kv && free > 0, kvPerTokenDev, maxTokens,
    maxBatch: Math.floor(maxTokens / Math.max(1, wl.context)),
    maxContext: Math.min(m.maxContext, Math.floor(maxTokens / Math.max(1, wl.batch))),
  }
}

const rateOf = (d: Device, wl: Workload) => (DTYPES[wl.weightDtype].rate === 'fp8' && d.tflops.fp8 ? d.tflops.fp8 : d.tflops.fp16) * 1e12

function phase(tokens: number, flops: number, bytes: number, commS: number, l: Layout, wl: Workload, a: Assumptions): Phase {
  const computeS = flops / (rateOf(l.device, wl) * a.mfu)
  const memS = bytes / (l.device.bwGBs * GB * a.bwUtil)
  const s = Math.max(computeS, memS) + commS
  const bound: Bound = commS > Math.max(computeS, memS) ? 'interconnect' : computeS >= memS ? 'compute' : 'memory'
  return { ms: s * 1e3, tokS: tokens / s, computeMs: computeS * 1e3, memMs: memS * 1e3, commMs: commS * 1e3, bound }
}

/** Per-layer attention FLOPs per (query token x key token): QK plus AV, 2 FLOPs per MAC. */
function attnFlopsPerPair(m: ModelSpec, absorbed: boolean): number {
  if (m.mla) return absorbed
    ? 2 * m.heads * (m.mla.kvLoraRank + m.mla.qkRope) + 2 * m.heads * m.mla.kvLoraRank
    : 2 * m.heads * (m.mla.qkNope + m.mla.qkRope) + 2 * m.heads * m.mla.vHead
  return 4 * m.heads * m.headDim
}

function linearActive(m: ModelSpec, p: ParamCounts) {
  // Layer parameters used per token plus the output projection (tied or not).
  return p.active - p.embedding - p.head - m.hidden + m.vocab * m.hidden
}

function commS(tokens: number, m: ModelSpec, l: Layout, a: Assumptions) {
  const msg = tokens * m.hidden * ACT_BYTES
  const tp = 2 * m.layers * allReduceS(msg, l.tp, l.device, a)
  const pp = (l.pp - 1) * (msg / (l.device.linkGBs * GB * a.linkUtil) + l.device.linkLatencyUs * 1e-6)
  return tp + pp
}

/** One decode step for `batch` sequences whose KV caches hold `kvLen` tokens. Stages run in sequence (no microbatch interleaving). */
export function decodeStep(m: ModelSpec, wl: Workload, l: Layout, a: Assumptions, batch: number, kvLen: number): Phase {
  const p = paramCounts(m)
  const s = weightSplit(m, p, DTYPES[wl.weightDtype].bytes)
  const touched = m.moe ? expectedExperts(m.moe.experts, m.moe.active, batch) / m.moe.experts : 1
  const weightRead = (s.fixed + s.expertsAll * touched + s.head) / l.tp
  const kvRead = batch * kvLen * kvBytesPerToken(m, wl.kvDtype) / kvShard(m, l.tp)
  const flops = (2 * linearActive(m, p) * batch + m.layers * attnFlopsPerPair(m, true) * kvLen * batch) / l.tp
  return phase(batch, flops, weightRead + kvRead, commS(batch, m, l, a), l, wl, a)
}

/** Prefill of one prompt of `n` tokens with causal attention. */
export function prefillPhase(m: ModelSpec, wl: Workload, l: Layout, a: Assumptions, n: number): Phase {
  const p = paramCounts(m)
  const s = weightSplit(m, p, DTYPES[wl.weightDtype].bytes)
  const touched = m.moe ? expectedExperts(m.moe.experts, m.moe.active, n) / m.moe.experts : 1
  const bytes = (s.fixed + s.expertsAll * touched + s.head) / l.tp + n * kvBytesPerToken(m, wl.kvDtype) / kvShard(m, l.tp)
  const linear = 2 * (p.active - p.embedding - p.head - m.hidden) * n + 2 * m.vocab * m.hidden
  const attn = m.layers * attnFlopsPerPair(m, false) * n * n / 2
  return phase(n, (linear + attn) / l.tp, bytes, commS(n, m, l, a), l, wl, a)
}

export function hourly(d: Device, a: Assumptions): { usd: number; rented: boolean } {
  if (a.costMode === 'rent' && d.rentPerHour) return { usd: d.rentPerHour, rented: true }
  return { usd: d.priceUsd / (a.amortYears * 8760) + (d.watts / 1000) * a.kwhUsd, rented: false }
}

export function evaluate(m: ModelSpec, wl: Workload, l: Layout, a: Assumptions): PlanResult {
  const count = l.tp * l.pp
  const warnings: string[] = []
  if (!validTp(m, l.tp)) warnings.push(`Tensor parallel ${l.tp} does not divide ${m.heads} attention heads${m.mla ? '' : ` / ${m.kvHeads} KV heads`} evenly.`)
  if (l.pp > m.layers) warnings.push(`Pipeline depth ${l.pp} exceeds ${m.layers} layers.`)
  if (wl.context > m.maxContext) warnings.push(`Context ${wl.context.toLocaleString()} exceeds the model's ${m.maxContext.toLocaleString()}-token limit.`)
  if (wl.promptLen + wl.outputLen > wl.context) warnings.push('Prompt plus output is longer than the reserved context.')
  if (!m.mla && l.tp > m.kvHeads) warnings.push(`TP ${l.tp} exceeds ${m.kvHeads} KV heads, so KV heads are replicated and the cache does not shrink further.`)
  if (count > l.device.maxCount) warnings.push(`${count} cards is more than the ${l.device.maxCount} this device is usually meshed in.`)
  if (DTYPES[wl.weightDtype].rate === 'fp8' && !l.device.tflops.fp8) warnings.push(`${l.device.name} has no FP8 rate listed; matmuls are costed at FP16.`)

  const params = paramCounts(m)
  const mem = memoryFit(m, wl, l, a)
  const batch = Math.max(1, wl.batch)
  const kvLen = wl.promptLen + wl.outputLen / 2
  const dec = decodeStep(m, wl, l, a, batch, kvLen)
  const pre = prefillPhase(m, wl, l, a, Math.max(1, wl.promptLen))
  const h = hourly(l.device, a)
  const perHour = h.usd * count
  return {
    device: l.device, count, tp: l.tp, pp: l.pp, params, mem,
    decode: { ...dec, perUserTokS: 1000 / dec.ms, batch },
    prefill: pre,
    ttftMs: pre.ms + dec.ms,
    requestMs: pre.ms + wl.outputLen * dec.ms,
    cost: { perHour, capex: l.device.priceUsd * count, usdPerMTok: perHour / (dec.tokS * 3600) * 1e6, tokPerJ: dec.tokS / (count * l.device.watts), rented: h.rented },
    warnings,
  }
}

export interface CurvePoint { batch: number; tokS: number; perUserTokS: number; tpotMs: number; fits: boolean }

/** Aggregate and per-user decode throughput as concurrency grows (powers of two). */
export function throughputCurve(m: ModelSpec, wl: Workload, l: Layout, a: Assumptions, maxBatch = 256): CurvePoint[] {
  const mem = memoryFit(m, { ...wl, batch: 1 }, l, a)
  const kvLen = wl.promptLen + wl.outputLen / 2
  const out: CurvePoint[] = []
  for (let b = 1; b <= maxBatch; b *= 2) {
    const d = decodeStep(m, wl, l, a, b, kvLen)
    out.push({ batch: b, tokS: d.tokS, perUserTokS: 1000 / d.ms, tpotMs: d.ms, fits: b * wl.context <= mem.maxTokens })
  }
  return out
}

/** All tp x pp factorisations of `count` that are legal for the model. */
export function layoutsFor(m: ModelSpec, d: Device, count: number): Layout[] {
  const out: Layout[] = []
  for (let tp = 1; tp <= count; tp++) {
    if (count % tp) continue
    const pp = count / tp
    if (validTp(m, tp) && pp <= m.layers) out.push({ device: d, tp, pp })
  }
  return out
}

/** Fastest layout for a device count that fits, else the one closest to fitting. */
export function bestLayout(m: ModelSpec, wl: Workload, d: Device, count: number, a: Assumptions): PlanResult | null {
  const rs = layoutsFor(m, d, count).map(l => evaluate(m, wl, l, a))
  if (!rs.length) return null
  const fit = rs.filter(r => r.mem.fits)
  if (fit.length) return fit.sort((x, y) => x.decode.ms - y.decode.ms)[0]
  return rs.sort((x, y) => (x.mem.usedGB - x.mem.capGB) - (y.mem.usedGB - y.mem.capGB))[0]
}

export const meshSizes = (max: number) => { const s: number[] = []; for (let n = 1; n <= max; n *= 2) s.push(n); if (s[s.length - 1] !== max) s.push(max); return s }

/** Smallest mesh of this device that holds the workload, or null. */
export function minDevices(m: ModelSpec, wl: Workload, d: Device, a: Assumptions, max = d.maxCount): PlanResult | null {
  for (const n of meshSizes(max)) {
    const r = bestLayout(m, wl, d, n, a)
    if (r?.mem.fits) return r
  }
  return null
}

export interface SearchOption {
  plan: PlanResult
  /** Independent copies of the mesh sharing the concurrent sequences (data parallel). */
  replicas: number
  totalDevices: number
  aggTokS: number
  perHour: number
  usdPerMTok: number
  ok: boolean
  reason: string
}

/**
 * For each device and mesh size, pick the best tp x pp layout, then the fewest replicas that meet the targets.
 * Ranked by hourly cost of the whole deployment, then $/1M tokens.
 */
export function search(m: ModelSpec, wl: Workload, devices: Device[], a: Assumptions, t: Targets, maxReplicas = 8): SearchOption[] {
  const out: SearchOption[] = []
  for (const d of devices) {
    for (const n of meshSizes(d.maxCount)) {
      let best: SearchOption | null = null
      for (let r = 1; r <= maxReplicas; r++) {
        const per = Math.ceil(wl.batch / r)
        const plan = bestLayout(m, { ...wl, batch: per }, d, n, a)
        if (!plan) break
        const agg = plan.decode.tokS * r
        const perHour = plan.cost.perHour * r
        const why = !plan.mem.weightsFit ? 'weights do not fit'
          : !plan.mem.fits ? `KV for ${per} x ${wl.context.toLocaleString()} tokens does not fit`
          : plan.decode.ms > t.maxTpotMs ? `${plan.decode.ms.toFixed(0)} ms/token is over ${t.maxTpotMs} ms`
          : plan.ttftMs > t.maxTtftMs ? `first token in ${fmtMs(plan.ttftMs)} is over ${fmtMs(t.maxTtftMs)}`
          : agg < t.minTokS ? `${agg.toFixed(0)} tok/s is under ${t.minTokS}`
          : ''
        const opt: SearchOption = { plan, replicas: r, totalDevices: n * r, aggTokS: agg, perHour, usdPerMTok: perHour / (agg * 3600) * 1e6, ok: !why, reason: why || 'meets targets' }
        if (!best || opt.ok) best = opt
        if (opt.ok || !plan.mem.weightsFit) break
      }
      if (best) out.push(best)
    }
  }
  return out.sort((x, y) => Number(y.ok) - Number(x.ok) || x.perHour - y.perHour || x.usdPerMTok - y.usdPerMTok)
}

export const fmtMs = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s` : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`
