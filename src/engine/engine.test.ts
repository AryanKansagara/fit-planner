import { test } from 'node:test'
import assert from 'node:assert/strict'
import { kvBytesPerToken, lineOf, manualSpec, paramCounts, parseConfig, specToConfig } from './model.ts'
import { DEVICES, devicesToCsv, mergeDevices, validateDevice } from './hardware.ts'
import { allReduceS, bestLayout, DEFAULT_ASSUMPTIONS as A, decodeStep, evaluate, expectedExperts, hourly, layoutsFor, memoryFit, meshSizes, minDevices, prefillPhase, search, throughputCurve, validTp, GB } from './plan.ts'
import { MODEL_SAMPLES } from './samples.ts'
import type { Assumptions, Device, ModelSpec, Workload } from './types.ts'

const spec = (id: string): ModelSpec => {
  const s = MODEL_SAMPLES.find(x => x.id === id)!
  const r = parseConfig(s.config, s.name)
  assert.ok(r.spec, `${id} parses`)
  return r.spec
}
const dev = (id: string) => DEVICES.find(d => d.id === id)!
const near = (actual: number, expected: number, relTol: number, msg?: string) =>
  assert.ok(Math.abs(actual - expected) <= Math.abs(expected) * relTol, `${msg ?? ''} expected ~${expected}, got ${actual}`)
const wl = (p: Partial<Workload> = {}): Workload => ({ weightDtype: 'bf16', kvDtype: 'bf16', context: 8192, batch: 1, promptLen: 1000, outputLen: 200, ...p })
// A device with a link so fast and close that communication is free.
const ideal = (d: Device): Device => ({ ...d, linkGBs: 1e9, linkLatencyUs: 0 })

// ---------- Parameter counts against published sizes ----------

test('dense parameter counts match published model sizes', () => {
  near(paramCounts(spec('llama-3.1-8b')).total, 8.03e9, 0.002, 'Llama 3.1 8B')
  near(paramCounts(spec('llama-3.3-70b')).total, 70.55e9, 0.002, 'Llama 3.3 70B')
  near(paramCounts(spec('llama-3.2-3b')).total, 3.21e9, 0.003, 'Llama 3.2 3B (tied)')
  near(paramCounts(spec('qwen2.5-32b')).total, 32.76e9, 0.003, 'Qwen2.5 32B')
})

test('tied embeddings are counted once', () => {
  const m = spec('llama-3.2-3b')
  const untied = paramCounts({ ...m, tied: false }).total
  assert.equal(untied - paramCounts(m).total, m.vocab * m.hidden)
})

test('MoE totals and active parameters match published figures', () => {
  const mix = paramCounts(spec('mixtral-8x7b'))
  near(mix.total, 46.7e9, 0.003, 'Mixtral total')
  near(mix.active, 12.9e9, 0.005, 'Mixtral active')
  const q = paramCounts(spec('qwen3-30b-a3b'))
  near(q.total, 30.5e9, 0.005, 'Qwen3-30B-A3B total')
  near(q.active, 3.3e9, 0.03, 'Qwen3-30B-A3B active')
})

test('DeepSeek V3 with MLA and shared experts lands on 671B total, 37B active', () => {
  const p = paramCounts(spec('deepseek-v3'))
  near(p.total, 671e9, 0.002)
  near(p.active, 37e9, 0.02)
})

// ---------- KV cache ----------

test('KV bytes per token = 2 x layers x kv heads x head dim x dtype', () => {
  assert.equal(kvBytesPerToken(spec('llama-3.1-8b'), 'bf16'), 2 * 32 * 8 * 128 * 2)
  assert.equal(kvBytesPerToken(spec('llama-3.3-70b'), 'fp8'), 2 * 80 * 8 * 128 * 1)
  assert.equal(kvBytesPerToken(spec('llama-3.1-8b'), 'bfp8'), 2 * 32 * 8 * 128 * 8.5 / 8)
})

test('MLA KV cache stores one latent plus RoPE key per layer', () => {
  assert.equal(kvBytesPerToken(spec('deepseek-v3'), 'bf16'), 61 * (512 + 64) * 2)
})

// ---------- Memory fit ----------

test('single-device weight memory is params x bytes per weight', () => {
  const m = spec('llama-3.1-8b')
  const f = memoryFit(m, wl(), { device: dev('nv-4090'), tp: 1, pp: 1 }, A)
  near(f.weightsGB, paramCounts(m).total * 2 / GB, 1e-6)
  const f4 = memoryFit(m, wl({ weightDtype: 'q4' }), { device: dev('nv-4090'), tp: 1, pp: 1 }, A)
  near(f4.weightsGB, paramCounts(m).total * 0.5625 / GB, 1e-6)
})

test('tensor parallel splits weights evenly; pipeline stages carry ceil(layers/pp)', () => {
  const m = spec('llama-3.3-70b')
  const d = dev('tt-p150a')
  const one = memoryFit(m, wl({ weightDtype: 'bfp8' }), { device: d, tp: 1, pp: 1 }, A)
  const tp4 = memoryFit(m, wl({ weightDtype: 'bfp8' }), { device: d, tp: 4, pp: 1 }, A)
  near(tp4.weightsGB, one.weightsGB / 4, 1e-9)
  const pp3 = memoryFit(m, wl({ weightDtype: 'bfp8' }), { device: d, tp: 1, pp: 3 }, A)
  const p = paramCounts(m)
  const perLayer = p.layerTotal * (8.5 / 8) / 80
  near(pp3.weightsGB * GB, 27 * perLayer + m.vocab * m.hidden * (8.5 / 8), 1e-9, 'heaviest stage has 27 of 80 layers plus the head')
})

test('KV stops shrinking once TP exceeds the number of KV heads', () => {
  const m = spec('llama-3.3-70b')
  const at = (tp: number) => memoryFit(m, wl(), { device: dev('nv-h100'), tp, pp: 1 }, A).kvPerTokenDev
  assert.equal(at(2), at(1) / 2)
  assert.equal(at(8), at(1) / 8)
  assert.equal(at(16), at(8))
})

test('max batch and fit follow from free memory after weights and reserve', () => {
  const m = spec('llama-3.1-8b')
  const d = dev('nv-4090')
  const f = memoryFit(m, wl({ context: 8192, batch: 2 }), { device: d, tp: 1, pp: 1 }, A)
  const free = d.memGB * GB - A.reserveGB * GB - A.reserveFrac * d.memGB * GB - f.weightsGB * GB
  assert.equal(f.maxTokens, Math.floor(free / kvBytesPerToken(m, 'bf16')))
  assert.equal(f.maxBatch, Math.floor(f.maxTokens / 8192))
  assert.equal(f.fits, 2 <= f.maxBatch)
  const tooMany = memoryFit(m, wl({ context: 8192, batch: f.maxBatch + 1 }), { device: d, tp: 1, pp: 1 }, A)
  assert.equal(tooMany.fits, false)
  assert.equal(tooMany.weightsFit, true)
})

test('a model whose weights exceed memory has no KV room', () => {
  const f = memoryFit(spec('llama-3.3-70b'), wl(), { device: dev('nv-4090'), tp: 1, pp: 1 }, A)
  assert.equal(f.weightsFit, false)
  assert.equal(f.fits, false)
  assert.equal(f.maxTokens, 0)
  assert.equal(f.maxBatch, 0)
})

test('smallest Blackhole p150a mesh for Llama 3.3 70B at BFP8 is four cards', () => {
  const m = spec('llama-3.3-70b')
  const w = wl({ weightDtype: 'bfp8', batch: 4 })
  assert.equal(bestLayout(m, w, dev('tt-p150a'), 2, A)!.mem.fits, false)
  assert.equal(minDevices(m, w, dev('tt-p150a'), A)!.count, 4)
})

// ---------- Roofline ----------

test('batch-1 decode on one device is bandwidth-bound: weights plus KV over effective bandwidth', () => {
  const m = spec('llama-3.1-8b')
  const d = dev('nv-4090')
  const ph = decodeStep(m, wl(), { device: d, tp: 1, pp: 1 }, A, 1, 1000)
  const p = paramCounts(m)
  const bytes = (p.layerTotal + m.vocab * m.hidden) * 2 + 1000 * kvBytesPerToken(m, 'bf16')
  assert.equal(ph.bound, 'memory')
  near(ph.ms, bytes / (d.bwGBs * GB * A.bwUtil) * 1000, 1e-9)
  assert.equal(ph.commMs, 0)
})

test('large batches turn decode compute-bound', () => {
  const ph = decodeStep(spec('llama-3.1-8b'), wl(), { device: dev('nv-3090'), tp: 1, pp: 1 }, A, 2048, 100)
  assert.equal(ph.bound, 'compute')
})

test('throughput curve: aggregate tok/s rises with batch while per-user speed falls', () => {
  const pts = throughputCurve(spec('llama-3.1-8b'), wl(), { device: dev('tt-p150a'), tp: 1, pp: 1 }, A, 64)
  assert.deepEqual(pts.map(p => p.batch), [1, 2, 4, 8, 16, 32, 64])
  for (let i = 1; i < pts.length; i++) {
    assert.ok(pts[i].tokS > pts[i - 1].tokS)
    assert.ok(pts[i].perUserTokS <= pts[i - 1].perUserTokS)
  }
  assert.ok(pts.some(p => !p.fits), 'large batches eventually run out of KV room')
})

test('with a free interconnect, TP=2 halves memory-bound decode time', () => {
  const m = spec('llama-3.1-8b')
  const d = ideal(dev('tt-p150a'))
  const t1 = decodeStep(m, wl(), { device: d, tp: 1, pp: 1 }, A, 1, 1000).ms
  const t2 = decodeStep(m, wl(), { device: d, tp: 2, pp: 1 }, A, 1, 1000).ms
  near(t2, t1 / 2, 1e-3)
})

test('pipeline parallel does not speed up single-stream decode', () => {
  const m = spec('llama-3.1-8b')
  const d = ideal(dev('tt-p150a'))
  const t1 = decodeStep(m, wl(), { device: d, tp: 1, pp: 1 }, A, 1, 1000).ms
  const t4 = decodeStep(m, wl(), { device: d, tp: 1, pp: 4 }, A, 1, 1000).ms
  near(t4, t1, 1e-6)
})

test('ring all-reduce: zero on one device, 2(n-1)/n x bytes over the link plus hops', () => {
  const d = dev('tt-p150a')
  assert.equal(allReduceS(1e6, 1, d, A), 0)
  const expect = (2 * 3 / 4) * 1e6 / (d.linkGBs * GB * A.linkUtil) + 2 * 3 * d.linkLatencyUs * 1e-6
  near(allReduceS(1e6, 4, d, A), expect, 1e-12)
})

test('slow PCIe links make TP decode communication-heavy compared with Ethernet mesh', () => {
  const m = spec('llama-3.3-70b')
  const w = wl({ weightDtype: 'bfp8', batch: 8 })
  const pcie = evaluate(m, w, { device: dev('tt-p100a'), tp: 4, pp: 1 }, A).decode.commMs
  const eth = evaluate(m, w, { device: dev('tt-p150a'), tp: 4, pp: 1 }, A).decode.commMs
  assert.ok(pcie > eth * 2)
})

test('prefill is compute-bound and grows faster than linearly with prompt length', () => {
  const m = spec('llama-3.1-8b')
  const l = { device: dev('nv-4090'), tp: 1, pp: 1 }
  const a = prefillPhase(m, wl(), l, A, 8000)
  const b = prefillPhase(m, wl(), l, A, 16000)
  assert.equal(a.bound, 'compute')
  assert.ok(b.ms > 2 * a.ms)
})

test('FP8-class weights use the FP8 tensor rate where the device has one', () => {
  const m = spec('llama-3.1-8b')
  const l = { device: dev('tt-p150a'), tp: 1, pp: 1 }
  const bf = prefillPhase(m, wl(), l, A, 4000).computeMs
  const f8 = prefillPhase(m, wl({ weightDtype: 'fp8' }), l, A, 4000).computeMs
  near(f8, bf * 194 / 774, 1e-9)
  const r = evaluate(m, wl({ weightDtype: 'fp8' }), { device: dev('nv-3090'), tp: 1, pp: 1 }, A)
  assert.ok(r.warnings.some(w => /no FP8/.test(w)))
})

test('MoE decode reads only routed experts at batch 1 and nearly all of them at large batch', () => {
  assert.equal(expectedExperts(8, 2, 1), 2)
  near(expectedExperts(128, 8, 512), 128, 0.001)
  assert.equal(expectedExperts(8, 2, 0), 0)
  const m = spec('mixtral-8x7b')
  const l = { device: ideal(dev('nv-h100')), tp: 1, pp: 1 }
  const moe = decodeStep(m, wl(), l, A, 1, 0).memMs
  const allExperts = decodeStep({ ...m, moe: { ...m.moe!, active: m.moe!.experts } }, wl(), l, A, 1, 0).memMs
  near(moe / allExperts, paramCounts(m).active / paramCounts(m).total, 0.02)
})

test('TTFT is prefill plus one decode step; request time adds every output token', () => {
  const r = evaluate(spec('qwen3-14b'), wl({ batch: 4 }), { device: dev('tt-p150a'), tp: 1, pp: 1 }, A)
  near(r.ttftMs, r.prefill.ms + r.decode.ms, 1e-12)
  near(r.requestMs, r.prefill.ms + 200 * r.decode.ms, 1e-12)
  near(r.decode.tokS, 4 / (r.decode.ms / 1000), 1e-9)
})

// ---------- Cost and energy ----------

test('owned hardware: amortised price plus electricity; $/1M tokens from hourly cost and throughput', () => {
  const d = dev('tt-p150a')
  const a: Assumptions = { ...A, amortYears: 3, kwhUsd: 0.2, costMode: 'own' }
  const h = d.priceUsd / (3 * 8760) + 0.3 * 0.2
  near(hourly(d, a).usd, h, 1e-12)
  const r = evaluate(spec('llama-3.3-70b'), wl({ weightDtype: 'bfp8', batch: 8 }), { device: d, tp: 4, pp: 1 }, a)
  near(r.cost.perHour, 4 * h, 1e-12)
  near(r.cost.usdPerMTok, 4 * h / (r.decode.tokS * 3600) * 1e6, 1e-12)
  near(r.cost.tokPerJ, r.decode.tokS / (4 * 300), 1e-12)
  assert.equal(r.cost.capex, 4 * d.priceUsd)
})

test('rent mode uses hourly rental where listed and falls back to owning otherwise', () => {
  const a: Assumptions = { ...A, costMode: 'rent' }
  assert.deepEqual(hourly(dev('nv-h100'), a), { usd: 2.2, rented: true })
  assert.equal(hourly(dev('tt-p150a'), a).rented, false)
})

// ---------- Layouts and search ----------

test('valid TP degrees divide attention heads and align with KV heads', () => {
  const q = spec('qwen2.5-32b') // 40 heads, 8 KV heads
  assert.ok(validTp(q, 8))
  assert.ok(!validTp(q, 16))
  assert.ok(!validTp(q, 5))
  assert.deepEqual(layoutsFor(q, dev('tt-p150a'), 8).map(l => `${l.tp}x${l.pp}`), ['1x8', '2x4', '4x2', '8x1'])
  assert.deepEqual(meshSizes(8), [1, 2, 4, 8])
  assert.deepEqual(meshSizes(6), [1, 2, 4, 6])
})

test('search ranks options meeting targets by hourly cost and explains misses', () => {
  const m = spec('llama-3.3-70b')
  const opts = search(m, wl({ weightDtype: 'bfp8', batch: 8, promptLen: 2000, outputLen: 300 }), DEVICES, A, { minTokS: 100, maxTpotMs: 80, maxTtftMs: 3000 })
  const ok = opts.filter(o => o.ok)
  assert.ok(ok.length > 3)
  for (let i = 1; i < ok.length; i++) assert.ok(ok[i].perHour >= ok[i - 1].perHour)
  assert.ok(opts.findIndex(o => !o.ok) > opts.findLastIndex(o => o.ok), 'all passing options come first')
  const single4090 = opts.find(o => o.plan.device.id === 'nv-4090' && o.plan.count === 1)!
  assert.equal(single4090.reason, 'weights do not fit')
  for (const o of ok) {
    assert.ok(o.aggTokS >= 100 && o.plan.decode.ms <= 80 && o.plan.ttftMs <= 3000 && o.plan.mem.fits)
  }
})

test('search adds data-parallel replicas when one mesh cannot reach the throughput target', () => {
  const m = spec('llama-3.1-8b')
  const opts = search(m, wl({ batch: 32 }), [dev('tt-n150s')], A, { minTokS: 1500, maxTpotMs: 1000, maxTtftMs: 60000 })
  const withReplicas = opts.find(o => o.ok && o.replicas > 1)
  assert.ok(withReplicas, 'at least one option uses replicas')
  assert.equal(withReplicas.totalDevices, withReplicas.plan.count * withReplicas.replicas)
  near(withReplicas.aggTokS, withReplicas.plan.decode.tokS * withReplicas.replicas, 1e-9)
})

// ---------- Import ----------

test('invalid JSON reports the line of the error', () => {
  const r = parseConfig('{\n  "hidden_size": 4096,\n  "num_hidden_layers": 32\n  "vocab_size": 1\n}')
  assert.equal(r.spec, null)
  assert.equal(r.issues[0].level, 'error')
  assert.equal(r.issues[0].line, 4)
})

test('missing GQA field assumes multi-head attention with a warning; bad head ratio is an error with its line', () => {
  const base = { model_type: 'llama', hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, intermediate_size: 11008, vocab_size: 32000, max_position_embeddings: 4096 }
  const r = parseConfig(JSON.stringify(base, null, 2))
  assert.equal(r.spec!.kvHeads, 32)
  assert.ok(r.issues.some(i => i.level === 'warn' && i.field === 'kvHeads'))
  const text = JSON.stringify({ ...base, num_key_value_heads: 5 }, null, 2)
  const bad = parseConfig(text)
  assert.equal(bad.spec, null)
  const e = bad.issues.find(i => i.level === 'error')!
  assert.equal(e.line, lineOf(text, 'num_key_value_heads'))
})

test('nested text_config, GPT-2 style plain MLP, and quantisation hints are understood', () => {
  const mm = parseConfig(JSON.stringify({ model_type: 'llava', text_config: { model_type: 'llama', hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, intermediate_size: 14336, vocab_size: 128256, max_position_embeddings: 8192 } }))
  assert.equal(mm.spec!.hidden, 4096)
  assert.ok(mm.issues.some(i => /text_config/.test(i.message)))
  const g = parseConfig(JSON.stringify({ model_type: 'gpt2', n_embd: 768, n_layer: 12, n_head: 12, vocab_size: 50257, n_positions: 1024 }))
  assert.equal(g.spec!.gated, false)
  assert.equal(g.spec!.tied, true)
  near(paramCounts(g.spec!).total, 124e6, 0.01, 'GPT-2 small (no biases counted)')
  assert.equal(parseConfig(MODEL_SAMPLES.find(s => s.id === 'deepseek-v3')!.config).suggestedDtype, 'fp8')
})

test('manual specs round-trip through config.json text', () => {
  const m = manualSpec({ name: 'Mine', hidden: 4096, layers: 32, heads: 32, kvHeads: 8, intermediate: 14336, vocab: 128256, maxContext: 131072 })
  const back = parseConfig(specToConfig(m)).spec!
  assert.equal(paramCounts(back).total, paramCounts(m).total)
  assert.equal(kvBytesPerToken(back, 'bf16'), kvBytesPerToken(m, 'bf16'))
})

// ---------- Hardware table ----------

test('device edits override the curated table by id; validation and CSV quoting', () => {
  const edited = { ...dev('tt-p150a'), priceUsd: 1000 }
  const merged = mergeDevices([edited, { ...edited, id: 'custom-x', name: 'Box, "v2"' }])
  assert.equal(merged.length, DEVICES.length + 1)
  assert.equal(merged.find(d => d.id === 'tt-p150a')!.priceUsd, 1000)
  assert.deepEqual(validateDevice(dev('tt-p150a')), [])
  assert.ok(validateDevice({ ...edited, memGB: 0, name: ' ' }).length >= 2)
  const csv = devicesToCsv(merged)
  assert.ok(csv.includes('"Box, ""v2"""'))
  assert.equal(csv.split('\n').length, merged.length + 1)
})
