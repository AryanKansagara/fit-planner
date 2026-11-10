import { mergeDevices, DEVICES } from './engine/hardware.ts'
import { parseConfig, paramCounts, type ParseResult } from './engine/model.ts'
import { evaluate, minDevices, type PlanResult } from './engine/plan.ts'
import { MODEL_SAMPLES, WORKLOAD_SAMPLES } from './engine/samples.ts'
import type { Device, ModelSpec, Targets, Workload } from './engine/types.ts'
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

export function projectPlan(p: Project): PlanResult | null {
  const m = projectSpec(p)
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
    ...base, ...plan, id: base.id, createdAt: now, updatedAt: now,
    assumptions: { ...base.assumptions, ...plan.assumptions }, targets: { ...base.targets, ...plan.targets },
    devices: Array.isArray(plan.devices) ? plan.devices : [],
    snapshots: Array.isArray(o.snapshots) ? (o.snapshots as Snapshot[]).map(s => ({ ...s, id: uuid() })) : [],
    sample: false,
  }
  const r = parseProjectModel(project)
  if (!r.spec) errors.push(`The plan's model config does not parse: ${r.issues.find(i => i.level === 'error')?.message ?? 'unknown error'}`)
  return { project: errors.length ? null : project, errors }
}
