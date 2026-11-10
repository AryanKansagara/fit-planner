import { useSyncExternalStore } from 'react'
import { DEFAULT_ASSUMPTIONS } from './engine/plan.ts'
import type { Assumptions, Device, Targets, Workload } from './engine/types.ts'

export interface ModelSource {
  kind: 'sample' | 'hf' | 'paste' | 'manual'
  /** Hugging Face repo id when fetched from the Hub or taken from a sample. */
  repo?: string
  sampleId?: string
  /** The config.json text the spec is parsed from. */
  config: string
  at: number
}

export interface SnapResult {
  fits: boolean
  weightsGB: number
  kvGB: number
  usedGB: number
  capGB: number
  maxBatch: number
  tokS: number
  perUserTokS: number
  tpotMs: number
  ttftMs: number
  prefillTokS: number
  usdPerMTok: number
  perHour: number
  capex: number
  tokPerJ: number
  bound: string
}

export interface Snapshot {
  id: string
  label: string
  createdAt: number
  modelName: string
  paramsB: number
  activeB: number
  deviceName: string
  count: number
  tp: number
  pp: number
  workload: Workload
  costMode: Assumptions['costMode']
  result: SnapResult
}

export interface Project {
  id: string
  name: string
  createdAt: number
  updatedAt: number
  model: ModelSource
  workload: Workload
  targets: Targets
  assumptions: Assumptions
  deviceId: string
  tp: number
  pp: number
  /** Edited copies of curated devices plus custom devices. */
  devices: Device[]
  snapshots: Snapshot[]
  workloadSampleId?: string
  /** Created from a built-in sample. */
  sample?: boolean
}

// ---------- Tiny IndexedDB wrapper (falls back to memory in private modes) ----------

const DB = 'fit-planner'
let dbp: Promise<IDBDatabase | null> | null = null
function db(): Promise<IDBDatabase | null> {
  return dbp ??= new Promise(res => {
    try {
      const req = indexedDB.open(DB, 1)
      req.onupgradeneeded = () => {
        req.result.createObjectStore('projects', { keyPath: 'id' })
      }
      req.onsuccess = () => res(req.result)
      req.onerror = () => res(null)
    } catch { res(null) }
  })
}
const mem = { projects: new Map<string, unknown>() }

async function tx<T>(store: 'projects', mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const d = await db()
  if (!d) throw new Error('no-idb')
  return new Promise((res, rej) => {
    const r = fn(d.transaction(store, mode).objectStore(store))
    r.onsuccess = () => res(r.result)
    r.onerror = () => rej(r.error)
  })
}

async function idbGetAll<T>(store: 'projects'): Promise<T[]> {
  try { return await tx(store, 'readonly', s => s.getAll()) as T[] } catch { return [...mem[store].values()] as T[] }
}
async function idbGet<T>(store: 'projects', key: string): Promise<T | undefined> {
  try { return await tx(store, 'readonly', s => s.get(key)) as T | undefined } catch { return mem[store].get(key) as T | undefined }
}
async function idbPut(store: 'projects', value: unknown, key?: string) {
  try { await tx(store, 'readwrite', s => key === undefined ? s.put(value) : s.put(value, key)) } catch { mem[store].set(key ?? (value as { id: string }).id, value) }
}
async function idbDel(store: 'projects', key: string) {
  try { await tx(store, 'readwrite', s => s.delete(key)) } catch { mem[store].delete(key) }
}

// ---------- In-memory project list with subscribers ----------

let projects: Project[] = []
let loaded = false
const subs = new Set<() => void>()
const emit = () => { projects = [...projects]; subs.forEach(f => f()) }
const changeHooks = new Set<(p: Project) => void>()

export const onProjectChange = (f: (p: Project) => void) => { changeHooks.add(f); return () => changeHooks.delete(f) }

export async function loadProjects() {
  projects = (await idbGetAll<Project>('projects')).sort((a, b) => b.updatedAt - a.updatedAt)
  loaded = true
  emit()
}

export function useProjects() {
  return useSyncExternalStore(f => { subs.add(f); return () => subs.delete(f) }, () => loaded ? projects : null)
}

export const getProjects = () => projects
export const getProject = (id: string) => projects.find(p => p.id === id)

export async function saveProject(p: Project, opts: { touch?: boolean; silent?: boolean } = {}) {
  const next = opts.touch === false ? p : { ...p, updatedAt: Date.now() }
  await idbPut('projects', next)
  const i = projects.findIndex(x => x.id === p.id)
  if (i >= 0) projects[i] = next
  else projects.unshift(next)
  projects.sort((a, b) => b.updatedAt - a.updatedAt)
  emit()
  if (!opts.silent) changeHooks.forEach(f => f(next))
  return next
}

/** Deletes locally; returns an undo function that restores the project. */
export async function deleteProject(id: string): Promise<() => Promise<void>> {
  const p = getProject(id)
  projects = projects.filter(x => x.id !== id)
  emit()
  await idbDel('projects', id)
  addTombstone(id)
  return async () => {
    if (!p) return
    removeTombstone(id)
    await saveProject(p)
  }
}

export const uuid = () => crypto.randomUUID()

// Tombstones keep a project deleted offline from coming back on the next sync.
const TOMB = 'fit-planner.tombstones'
export const tombstones = (): string[] => { try { return JSON.parse(localStorage.getItem(TOMB) ?? '[]') } catch { return [] } }
const addTombstone = (id: string) => localStorage.setItem(TOMB, JSON.stringify([...new Set([...tombstones(), id])]))
const removeTombstone = (id: string) => localStorage.setItem(TOMB, JSON.stringify(tombstones().filter(x => x !== id)))
export const clearTombstones = (ids: string[]) => localStorage.setItem(TOMB, JSON.stringify(tombstones().filter(x => !ids.includes(x))))

// ---------- App defaults (settings page) ----------

export interface Defaults { assumptions: Assumptions; deviceId: string }
const DEF_KEY = 'fit-planner.defaults'
const BASE_DEFAULTS: Defaults = { assumptions: DEFAULT_ASSUMPTIONS, deviceId: 'tt-p150a' }
export function getDefaults(): Defaults {
  try {
    const d = JSON.parse(localStorage.getItem(DEF_KEY) ?? '{}')
    return { ...BASE_DEFAULTS, ...d, assumptions: { ...DEFAULT_ASSUMPTIONS, ...d.assumptions } }
  } catch { return BASE_DEFAULTS }
}
export const setDefaults = (d: Defaults) => localStorage.setItem(DEF_KEY, JSON.stringify(d))
export const resetDefaults = () => localStorage.removeItem(DEF_KEY)
