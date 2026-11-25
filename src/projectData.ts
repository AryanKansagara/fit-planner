import { mergeDevices, validateDevice, DEVICES } from './engine/hardware.ts'
import { KV_BYTES, parseConfig, paramCounts, type ParseResult } from './engine/model.ts'
import { evaluate, minDevices, type PlanResult } from './engine/plan.ts'
import { MODEL_SAMPLES, WORKLOAD_SAMPLES } from './engine/samples.ts'
import { DTYPES, type Device, type ModelSpec, type Targets, type Workload } from './engine/types.ts'
import { getDefaults, saveProject, uuid, type Project, type Snapshot } from './store.ts'

const parsed = new Map<string, ParseResult>()
/** Parse results are cached by config text; the spec name comes from the repo or sample. */
export function parseProjectModel(p: Pick<Project, 'model'>): ParseResult {
  const key = p.model.config + '\u0000' + (p.model.repo ?? '')
  let r = parsed.get(key)
  if (!r) { r = parseConfig(p.model.config, p.model.repo ? sampleName(p.model.repo) : undefined); parsed.set(key, r) }
  return r
}
const sampleName = (repo: string) => MODEL_SAMPLES.find(s => s.repo === repo)?.name ?? repo

export const projectSpec = (p: Pick<Project, 'model'>): ModelSpec | null => parseProjectModel(p).spec
export const projectDevices = (p: Pick<Project, 'devices'>): Device[] => mergeDevices(p.devices)
export const projectDevice = (p: Pick<Project, 'devices' | 'deviceId'>): Device => projectDevices(p).find(d => d.id === p.deviceId) ?? DEVICES[0]

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const finite = (o: unknown, keys: string[]) => isObj(o) && keys.every(k => typeof o[k] === 'number' && Number.isFinite(o[k]))
const text = (o: unknown, keys: string[]) => isObj(o) && keys.every(k => typeof o[k] === 'string')

function deviceOk(d: unknown): boolean {
  if (!text(d, ['id', 'name', 'vendor', 'linkKind']) || !finite(d, ['memGB', 'bwGBs', 'linkGBs', 'linkLatencyUs', 'priceUsd', 'watts', 'maxCount'])) return false
  const { tflops, rentPerHour } = d as Device
  if (!finite(tflops, ['fp16']) || (tflops.fp8 !== undefined && !finite(tflops, ['fp8']))) return false
  return (rentPerHour === undefined || (Number.isFinite(rentPerHour) && rentPerHour >= 0)) && validateDevice(d as Device).length === 0
}

const snapshotOk = (s: unknown) => isObj(s) && text(s, ['label', 'modelName', 'deviceName']) && finite(s, ['createdAt', 'count', 'tp', 'pp'])
  && finite(s.workload, ['context', 'batch']) && text(s.workload, ['weightDtype', 'kvDtype'])
  && finite(s.result, ['usedGB', 'tokS', 'perUserTokS', 'ttftMs', 'maxBatch', 'perHour', 'usdPerMTok', 'capex', 'tokPerJ'])

/** Why a stored or imported plan cannot be sized. The planner never writes such data, but a hand-edited file can carry it. */
export function planDataErrors(p: Project): string[] {
  const errs: string[] = []
  // Object.hasOwn: names such as "toString" must not pass as dtypes.
  const dtype = (field: string, v: unknown, known: object) => {
    if (typeof v !== 'string' || !Object.hasOwn(known, v)) errs.push(`plan.workload.${field} must be one of ${Object.keys(known).join(', ')} (got ${JSON.stringify(v) ?? 'nothing'}).`)
  }
  dtype('weightDtype', p.workload?.weightDtype, DTYPES)
  dtype('kvDtype', p.workload?.kvDtype, KV_BYTES)
  if (!finite(p.workload, ['context', 'batch', 'promptLen', 'outputLen'])) errs.push('plan.workload needs numbers for context, batch, promptLen and outputLen.')
  if (!finite(p.targets, ['minTokS', 'maxTpotMs', 'maxTtftMs'])) errs.push('plan.targets needs numbers for minTokS, maxTpotMs and maxTtftMs.')
  if (!finite(p.assumptions, ['bwUtil', 'mfu', 'linkUtil', 'reserveGB', 'reserveFrac', 'amortYears', 'kwhUsd'])) errs.push('plan.assumptions has a value that is not a number.')
  if (p.model.repo !== undefined && typeof p.model.repo !== 'string') errs.push('plan.model.repo must be text.')
  if (!Array.isArray(p.devices) || !p.devices.every(deviceOk)) errs.push('plan.devices has an entry that is not a valid device.')
  if (!Array.isArray(p.snapshots) || !p.snapshots.every(snapshotOk)) errs.push('A saved snapshot is incomplete.')
  return errs
}

export function projectPlan(p: Project): PlanResult | null {
  const m = projectSpec(p)
  if (planDataErrors(p).length) return null
  return m ? evaluate(m, p.workload, { device: projectDevice(p), tp: p.tp, pp: p.pp }, p.assumptions) : null
}

export const DEFAULT_WORKLOAD: Workload = { weightDtype: 'bfp8', kvDtype: 'bf16', context: 8192, batch: 4, promptLen: 1500, outputLen: 300 }
export const DEFAULT_TARGETS: Targets = { minTokS: 60, maxTpotMs: 80, maxTtftMs: 3000 }

function workloadFrom(id: string | undefined): { workload: Workload; targets: Targets } {
  const w = WORKLOAD_SAMPLES.find(x => x.id === id)
  if (!w) return { workload: DEFAULT_WORKLOAD, targets: DEFAULT_TARGETS }
  return {
    workload: { weightDtype: w.weightDtype, kvDtype: 'bf16', context: w.context, batch: w.batch, promptLen: w.promptLen, outputLen: w.outputLen },
    targets: { minTokS: w.minTokS, maxTpotMs: w.maxTpotMs, maxTtftMs: w.maxTtftMs },
  }
}

/** Picks the smallest mesh of the default device that holds the model, so a new plan starts out fitting. */
export function autoLayout(p: Project): Project {
  const m = projectSpec(p)
  if (!m) return p
  const best = minDevices(m, p.workload, projectDevice(p), p.assumptions)
  return best ? { ...p, tp: best.tp, pp: best.pp } : p
}

export function newProject(name: string, modelSampleId = 'llama-3.1-8b', workloadId = 'team-chat'): Project {
  const d = getDefaults()
  const s = MODEL_SAMPLES.find(x => x.id === modelSampleId) ?? MODEL_SAMPLES[1]
  const { workload, targets } = workloadFrom(workloadId)
  const now = Date.now()
  const p: Project = {
    id: uuid(), name, createdAt: now, updatedAt: now,
    model: { kind: 'sample', repo: s.repo, sampleId: s.id, config: s.config, at: now },
    workload, targets, assumptions: d.assumptions, deviceId: d.deviceId, tp: 1, pp: 1, devices: [], snapshots: [], workloadSampleId: workloadId,
  }
  const r = parseProjectModel(p)
  if (r.suggestedDtype === 'fp8' && workload.weightDtype === 'bf16') p.workload = { ...workload, weightDtype: 'fp8' }
  return autoLayout(p)
}

export async function createSampleProject(modelId: string, workloadId = 'team-chat') {
  const s = MODEL_SAMPLES.find(x => x.id === modelId) ?? MODEL_SAMPLES[1]
  const w = WORKLOAD_SAMPLES.find(x => x.id === workloadId)
  return saveProject({ ...newProject(`${s.name} · ${w?.name.replace(' (synthetic)', '') ?? 'plan'}`, s.id, workloadId), sample: true })
}

export const createBlankProject = (name: string) => saveProject(newProject(name))

export function snapshotOf(p: Project, label: string): Snapshot | null {
  const m = projectSpec(p)
  const r = projectPlan(p)
  if (!m || !r) return null
  const pc = paramCounts(m)
  return {
    id: uuid(), label, createdAt: Date.now(), modelName: m.name, paramsB: pc.total / 1e9, activeB: pc.active / 1e9,
    deviceName: r.device.name, count: r.count, tp: r.tp, pp: r.pp, workload: p.workload, costMode: p.assumptions.costMode,
    result: {
      fits: r.mem.fits, weightsGB: r.mem.weightsGB, kvGB: r.mem.kvGB, usedGB: r.mem.usedGB, capGB: r.mem.capGB, maxBatch: r.mem.maxBatch,
      tokS: r.decode.tokS, perUserTokS: r.decode.perUserTokS, tpotMs: r.decode.ms, ttftMs: r.ttftMs, prefillTokS: r.prefill.tokS,
      usdPerMTok: r.cost.usdPerMTok, perHour: r.cost.perHour, capex: r.cost.capex, tokPerJ: r.cost.tokPerJ, bound: r.decode.bound,
    },
  }
}

// ---------- Plan JSON (share / import) ----------

export const PLAN_FORMAT = 'fit-planner/plan@1'

export function planJson(p: Project): string {
  const { id: _i, createdAt: _c, updatedAt: _u, snapshots, ...rest } = p
  return JSON.stringify({ format: PLAN_FORMAT, exportedAt: new Date().toISOString(), plan: rest, snapshots }, null, 2)
}

export interface PlanImport { project: Project | null; errors: string[] }

/** Validates a shared plan file and returns a fresh project (new id) ready to save. */
export function importPlanJson(text: string): PlanImport {
  let o: Record<string, unknown>
  try { o = JSON.parse(text) } catch (e) { return { project: null, errors: [`Not valid JSON: ${(e as Error).message}`] } }
  if (o?.format !== PLAN_FORMAT || typeof o.plan !== 'object' || !o.plan) return { project: null, errors: [`Expected a file with "format": "${PLAN_FORMAT}" (use Share plan > Download JSON).`] }
  const plan = o.plan as Partial<Project>
  const errors: string[] = []
  if (!plan.model || typeof plan.model.config !== 'string') errors.push('plan.model.config is missing.')
  if (!plan.workload || typeof plan.workload.context !== 'number') errors.push('plan.workload is missing or incomplete.')
  if (typeof plan.tp !== 'number' || typeof plan.pp !== 'number' || plan.tp < 1 || plan.pp < 1) errors.push('plan.tp and plan.pp must be positive numbers.')
  if (errors.length) return { project: null, errors }
  const base = newProject(String(plan.name ?? 'Imported plan'))
  const now = Date.now()
  const project: Project = {
    ...base, ...plan, id: base.id, name: base.name, createdAt: now, updatedAt: now,
    assumptions: { ...base.assumptions, ...plan.assumptions }, targets: { ...base.targets, ...plan.targets },
    devices: Array.isArray(plan.devices) ? plan.devices : [],
    snapshots: Array.isArray(o.snapshots) ? (o.snapshots as Snapshot[]).map(s => ({ ...s, id: uuid() })) : [],
    sample: false,
  }
  errors.push(...planDataErrors(project))
  const r = parseProjectModel(project)
  if (!r.spec) errors.push(`The plan's model config does not parse: ${r.issues.find(i => i.level === 'error')?.message ?? 'unknown error'}`)
  return { project: errors.length ? null : project, errors }
}
