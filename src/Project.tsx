import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode } from 'react'
import { TABS, type Tab } from './App.tsx'
import { kvBytesPerToken, paramCounts } from './engine/model.ts'
import { layoutsFor, meshSizes, minDevices, throughputCurve, fmtMs, DEFAULT_ASSUMPTIONS } from './engine/plan.ts'
import { WORKLOAD_SAMPLES } from './engine/samples.ts'
import { DTYPES, type Assumptions, type KvDtype, type WeightDtype, type Workload } from './engine/types.ts'
import { autoLayout, parseProjectModel, planJson, projectDevice, projectDevices, projectPlan, snapshotOf } from './projectData.ts'
import { getProject, saveProject, useProjects, type Project, type Snapshot } from './store.ts'
import { deleteSnapshotCloud } from './sync.ts'
import { Code, CopyButton, download, Empty, fmtB, fmtDate, fmtGB, fmtNum, fmtUsd, go, isTyping, Link, Modal, NotFound, SyntheticBadge, toast } from './ui.tsx'
import { CurveChart, Dial, Mesh, MeshLegend, PhaseBars } from './viz.tsx'

const ModelTab = lazy(() => import('./ModelTab.tsx'))
const HardwareTab = lazy(() => import('./HardwareTab.tsx'))
const SearchTab = lazy(() => import('./SearchTab.tsx'))

const TAB_LABEL: Record<Tab, string> = { plan: 'Plan', model: 'Model', hardware: 'Hardware', search: 'Search', history: 'History' }

export type Update = (patch: Partial<Project>) => void

export default function ProjectView({ id, tab }: { id: string; tab: Tab }) {
  const list = useProjects()
  const p = list?.find(x => x.id === id)
  const [sharing, setSharing] = useState(false)
  useEffect(() => { if (p) document.title = `${p.name} · ${TAB_LABEL[tab]} · Fit Planner` }, [p?.name, tab])

  const update: Update = patch => { const cur = getProject(id); if (cur) void saveProject({ ...cur, ...patch }) }
  const snapshot = () => {
    const cur = getProject(id)
    if (!cur) return
    const r = projectPlan(cur)
    const s = snapshotOf(cur, r ? `${r.count}x ${r.device.name} · ${cur.workload.weightDtype.toUpperCase()} · batch ${cur.workload.batch}` : 'Snapshot')
    if (!s) { toast('Fix the model config before saving a snapshot.', { tone: 'error' }); return }
    void saveProject({ ...cur, snapshots: [...cur.snapshots, s] })
    toast('Configuration saved to history.', { tone: 'ok', action: { label: 'View', run: () => go(`/app/p/${id}/history`) } })
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey || document.querySelector('dialog[open]')) return
      const i = Number(e.key) - 1
      if (i >= 0 && i < TABS.length) { e.preventDefault(); go(`/app/p/${id}/${TABS[i]}`, { keepScroll: true }) }
      if (e.key === 's') { e.preventDefault(); snapshot() }
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [id])

  if (list === null) return <div className="boot" role="status"><span className="spinner" /> Loading</div>
  if (!p) return <NotFound />
  return (
    <section className="page">
      <div className="proj-head">
        <div className="grow">
          <p className="eyebrow"><Link href="/app">Plans</Link> / {p.sample ? <SyntheticBadge /> : 'Your plan'}</p>
          <input className="title-input" aria-label="Plan name" value={p.name} onChange={e => update({ name: e.target.value })} />
        </div>
        <div className="row gap-sm wrap">
          <button type="button" className="btn ghost sm" onClick={snapshot} title="Save to history (S)">Save snapshot</button>
          <button type="button" className="btn ghost sm" onClick={() => setSharing(true)}>Share plan</button>
          <Link href={`/app/p/${p.id}/report`} className="btn primary sm">Report</Link>
        </div>
      </div>
      <nav className="tabs" aria-label="Plan sections">
        {TABS.map((t, i) => <Link key={t} href={`/app/p/${p.id}/${t}`} aria-current={t === tab ? 'page' : undefined} title={`${TAB_LABEL[t]} (${i + 1})`}>{TAB_LABEL[t]}{t === 'history' && p.snapshots.length > 0 && <span className="count">{p.snapshots.length}</span>}</Link>)}
        <span className="tabs-spacer" />
        <span className="kbd-hint muted small"><kbd>1</kbd>-<kbd>5</kbd> tabs · <kbd>S</kbd> save</span>
      </nav>
      <Suspense fallback={<div className="boot" role="status"><span className="spinner" /> Loading</div>}>
        {tab === 'plan' && <PlanTab p={p} update={update} />}
        {tab === 'model' && <ModelTab p={p} update={update} />}
        {tab === 'hardware' && <HardwareTab p={p} update={update} />}
        {tab === 'search' && <SearchTab p={p} update={update} />}
        {tab === 'history' && <HistoryTab p={p} update={update} />}
      </Suspense>
      <Modal open={sharing} onClose={() => setSharing(false)} title="Share this plan" wide>
        <ShareBody p={p} />
      </Modal>
    </section>
  )
}

function ShareBody({ p }: { p: Project }) {
  const json = planJson(p)
  const slug = p.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'plan'
  return (
    <div className="stack">
      <p className="muted small">The plan file carries the model config, workload, targets, assumptions, hardware edits and history. Anyone can open it with Plans &gt; Import plan JSON. Nothing is uploaded.</p>
      <div className="row gap-sm wrap">
        <button type="button" className="btn primary sm" onClick={() => download(`${slug}.plan.json`, json, 'application/json')}>Download JSON</button>
        <CopyButton text={json} label="Copy JSON" />
      </div>
      <Code code={json.length > 6000 ? json.slice(0, 6000) + '\n…' : json} label="plan.json" />
    </div>
  )
}

// ---------- Small inputs ----------

export function NumField({ label, value, onChange, min = 0, max, step = 1, suffix, hint }: { label: string; value: number; onChange(v: number): void; min?: number; max?: number; step?: number; suffix?: string; hint?: string }) {
  const [text, setText] = useState(String(value))
  useEffect(() => { if (Number(text) !== value) setText(String(value)) }, [value])
  const bad = text.trim() === '' || !Number.isFinite(Number(text)) || Number(text) < min || (max !== undefined && Number(text) > max)
  return (
    <label className={`field${bad ? ' invalid' : ''}`}>
      <span>{label}{suffix && <em> {suffix}</em>}</span>
      <input type="number" inputMode="decimal" value={text} min={min} max={max} step={step} aria-invalid={bad} title={hint}
        onChange={e => { setText(e.target.value); const v = Number(e.target.value); if (e.target.value.trim() !== '' && Number.isFinite(v) && v >= min && (max === undefined || v <= max)) onChange(v) }} />
    </label>
  )
}

function Kpi({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'ok' | 'bad' | 'warn' }) {
  return <div className={`kpi${tone ? ' ' + tone : ''}`}><div className="kpi-label">{label}</div><div className="kpi-value">{value}</div>{sub && <div className="kpi-sub">{sub}</div>}</div>
}

// ---------- Plan tab ----------

function PlanTab({ p, update }: { p: Project; update: Update }) {
  const parsed = parseProjectModel(p)
  const m = parsed.spec
  const r = useMemo(() => projectPlan(p), [p])
  const device = projectDevice(p)
  const devices = projectDevices(p)
  const curve = useMemo(() => m ? throughputCurve(m, p.workload, { device, tp: p.tp, pp: p.pp }, p.assumptions, 256) : [], [m, p.workload, device, p.tp, p.pp, p.assumptions])
  if (!m || !r) return (
    <Empty title="The model config has errors">
      <p className="muted">{parsed.issues.find(i => i.level === 'error')?.message}</p>
      <Link href={`/app/p/${p.id}/model`} className="btn primary">Fix the model</Link>
    </Empty>
  )
  const pc = paramCounts(m)
  const count = p.tp * p.pp
  const wl = (patch: Partial<Workload>) => update({ workload: { ...p.workload, ...patch }, workloadSampleId: undefined })
  const as = (patch: Partial<Assumptions>) => update({ assumptions: { ...p.assumptions, ...patch } })
  const setCount = (n: number) => {
    const ls = layoutsFor(m, device, n)
    if (!ls.length) { toast(`No legal split of ${n} devices for ${m.heads} heads.`, { tone: 'error' }); return }
    const best = ls.reduce((a, b) => (b.tp > a.tp ? b : a))
    update({ tp: best.tp, pp: best.pp })
  }
  const auto = () => {
    const best = minDevices(m, p.workload, device, p.assumptions)
    if (!best) { toast(`Even ${device.maxCount}x ${device.name} cannot hold this workload. Try a smaller dtype, context or batch, or another device.`, { tone: 'error', ttl: 7000 }); return }
    update({ tp: best.tp, pp: best.pp })
    toast(`${best.count}x ${device.name} (TP ${best.tp} x PP ${best.pp}) is the smallest mesh that fits.`, { tone: 'ok' })
  }
  const counts = [...new Set([...meshSizes(Math.max(device.maxCount, count)), count])].sort((a, b) => a - b)
  const fill = { weightsGB: r.mem.weightsGB, kvGB: r.mem.kvGB, reserveGB: r.mem.reserveGB, capGB: r.mem.capGB }

  return (
    <div className="plan-grid">
      <aside className="plan-inputs">
        <div className="card pad stack-sm">
          <div className="row between"><h2 className="h5">Model</h2><Link href={`/app/p/${p.id}/model`} className="small">Change</Link></div>
          <strong>{m.name}</strong>
          <dl className="facts">
            <dt>Parameters</dt><dd>{fmtB(pc.total)}{m.moe && <> ({fmtB(pc.active)} active)</>}</dd>
            <dt>Layers / hidden</dt><dd>{m.layers} / {fmtNum(m.hidden)}</dd>
            <dt>Heads (KV)</dt><dd>{m.heads} ({m.mla ? 'MLA latent' : m.kvHeads})</dd>
            <dt>KV per token</dt><dd>{fmtNum(kvBytesPerToken(m, p.workload.kvDtype) / 1024, 0)} KB</dd>
            <dt>Max context</dt><dd>{fmtNum(m.maxContext)}</dd>
          </dl>
        </div>

        <div className="card pad stack-sm">
          <h2 className="h5">Hardware</h2>
          <label className="field"><span>Device</span>
            <select value={p.deviceId} onChange={e => { const d = devices.find(x => x.id === e.target.value)!; const next = autoLayout({ ...p, deviceId: d.id }); update({ deviceId: d.id, tp: next.tp, pp: next.pp }) }}>
              {['Tenstorrent', ...new Set(devices.map(d => d.vendor).filter(v => v !== 'Tenstorrent'))].map(v => (
                <optgroup key={v} label={v}>{devices.filter(d => d.vendor === v).map(d => <option key={d.id} value={d.id}>{d.name} · {d.memGB} GB</option>)}</optgroup>
              ))}
            </select>
          </label>
          <div className="row gap-sm">
            <label className="field grow"><span>Devices</span>
              <select value={count} onChange={e => setCount(+e.target.value)}>{counts.map(n => <option key={n} value={n}>{n}</option>)}</select>
            </label>
            <label className="field grow"><span>Split (TP x PP)</span>
              <select value={`${p.tp}x${p.pp}`} onChange={e => { const [tp, pp] = e.target.value.split('x').map(Number); update({ tp, pp }) }}>
                {layoutsFor(m, device, count).map(l => <option key={`${l.tp}x${l.pp}`} value={`${l.tp}x${l.pp}`}>{l.tp} x {l.pp}</option>)}
                {!layoutsFor(m, device, count).some(l => l.tp === p.tp && l.pp === p.pp) && <option value={`${p.tp}x${p.pp}`}>{p.tp} x {p.pp} (invalid)</option>}
              </select>
            </label>
          </div>
          <button type="button" className="btn ghost sm block" onClick={auto}>Smallest mesh that fits</button>
          <p className="muted small">{device.linkKind} · {device.linkGBs} GB/s per device · {device.bwGBs} GB/s memory</p>
        </div>

        <div className="card pad stack-sm">
          <div className="row between"><h2 className="h5">Workload</h2>
            <select aria-label="Load a workload profile" className="mini" value={p.workloadSampleId ?? ''} onChange={e => {
              const w = WORKLOAD_SAMPLES.find(x => x.id === e.target.value)
              if (w) update({ workload: { ...p.workload, weightDtype: w.weightDtype, context: w.context, batch: w.batch, promptLen: w.promptLen, outputLen: w.outputLen }, targets: { minTokS: w.minTokS, maxTpotMs: w.maxTpotMs, maxTtftMs: w.maxTtftMs }, workloadSampleId: w.id })
            }}>
              <option value="">Custom</option>
              {WORKLOAD_SAMPLES.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
            </select>
          </div>
          <div className="row gap-sm">
            <label className="field grow"><span>Weights</span>
              <select value={p.workload.weightDtype} onChange={e => wl({ weightDtype: e.target.value as WeightDtype })}>{Object.entries(DTYPES).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}</select>
            </label>
            <label className="field grow"><span>KV cache</span>
              <select value={p.workload.kvDtype} onChange={e => wl({ kvDtype: e.target.value as KvDtype })}><option value="bf16">BF16</option><option value="fp8">FP8</option><option value="bfp8">BFP8</option></select>
            </label>
          </div>
          <div className="two">
            <NumField label="Context / seq" value={p.workload.context} min={128} max={10_000_000} step={1024} onChange={v => wl({ context: v })} />
            <NumField label="Concurrent seqs" value={p.workload.batch} min={1} max={4096} onChange={v => wl({ batch: Math.round(v) })} />
            <NumField label="Prompt tokens" value={p.workload.promptLen} min={1} max={10_000_000} step={100} onChange={v => wl({ promptLen: Math.round(v) })} />
            <NumField label="Output tokens" value={p.workload.outputLen} min={1} max={1_000_000} step={50} onChange={v => wl({ outputLen: Math.round(v) })} />
          </div>
        </div>

        <details className="card pad">
          <summary className="h5">Assumptions</summary>
          <div className="two mt-sm">
            <NumField label="Bandwidth util." value={p.assumptions.bwUtil} min={0.05} max={1} step={0.05} onChange={v => as({ bwUtil: v })} />
            <NumField label="Compute util. (MFU)" value={p.assumptions.mfu} min={0.05} max={1} step={0.05} onChange={v => as({ mfu: v })} />
            <NumField label="Link util." value={p.assumptions.linkUtil} min={0.05} max={1} step={0.05} onChange={v => as({ linkUtil: v })} />
            <NumField label="Reserve / device" suffix="GB" value={p.assumptions.reserveGB} min={0} max={64} step={0.5} onChange={v => as({ reserveGB: v })} />
            <NumField label="Amortise over" suffix="years" value={p.assumptions.amortYears} min={0.25} max={10} step={0.5} onChange={v => as({ amortYears: v })} />
            <NumField label="Electricity" suffix="$/kWh" value={p.assumptions.kwhUsd} min={0} max={2} step={0.01} onChange={v => as({ kwhUsd: v })} />
          </div>
          <div className="seg mt-sm" role="group" aria-label="Cost basis">
            <button type="button" aria-pressed={p.assumptions.costMode === 'own'} onClick={() => as({ costMode: 'own' })}>Own (amortised)</button>
            <button type="button" aria-pressed={p.assumptions.costMode === 'rent'} onClick={() => as({ costMode: 'rent' })}>Rent hourly</button>
          </div>
          <button type="button" className="linkish small mt-sm" onClick={() => update({ assumptions: { ...DEFAULT_ASSUMPTIONS } })}>Reset to defaults</button>
        </details>
      </aside>

      <div className="plan-out stack">
        <div className={`fit-banner ${r.mem.fits ? 'ok' : 'bad'}`} role="status">
          <strong>{r.mem.fits ? 'Fits' : r.mem.weightsFit ? 'Weights fit, KV cache does not' : 'Does not fit'}</strong>
          <span>{fmtGB(r.mem.usedGB)} of {fmtGB(r.mem.capGB)} on the busiest device · {r.count}x {device.name} as TP {r.tp} x PP {r.pp}</span>
        </div>

        <div className="card pad mesh-card">
          <div className="mesh-wrap">
            <div className="grow">
              <div className="row between wrap gap-sm"><h2 className="h5">Per-device memory</h2><span className="muted small">rows = pipeline stages, columns = tensor-parallel group</span></div>
              <Mesh tp={r.tp} pp={r.pp} fill={fill} label={device.name} />
              <MeshLegend fill={fill} />
            </div>
            <Dial value={r.decode.tokS} max={Math.max(50, niceMax(Math.max(r.decode.tokS * 1.25, ...curve.filter(c => c.fits).map(c => c.tokS))))} label="tok/s decode" sub={`${fmtNum(r.decode.perUserTokS, 1)} tok/s per sequence`} />
          </div>
        </div>

        <div className="kpis">
          <Kpi label="Decode throughput" value={`${fmtNum(r.decode.tokS)} tok/s`} sub={`${fmtNum(r.decode.ms, 1)} ms per step at batch ${r.decode.batch}`} tone={r.decode.tokS >= p.targets.minTokS ? 'ok' : 'warn'} />
          <Kpi label="Time to first token" value={fmtMs(r.ttftMs)} sub={`${fmtNum(p.workload.promptLen)}-token prompt, ${fmtNum(r.prefill.tokS)} tok/s prefill`} tone={r.ttftMs <= p.targets.maxTtftMs ? 'ok' : 'warn'} />
          <Kpi label="Cost per 1M output tokens" value={fmtUsd(r.cost.usdPerMTok, r.cost.usdPerMTok < 10 ? 2 : 0)} sub={`${fmtUsd(r.cost.perHour, 2)}/h ${r.cost.rented ? 'rented' : 'owned'} · ${fmtUsd(r.cost.capex)} hardware`} />
          <Kpi label="Energy" value={`${fmtNum(r.cost.tokPerJ, 2)} tok/J`} sub={`${fmtNum(r.count * device.watts)} W at board power`} />
          <Kpi label={`Max sequences at ${fmtNum(p.workload.context)} ctx`} value={fmtNum(r.mem.maxBatch)} sub={`${fmtNum(r.mem.maxTokens)} KV tokens total`} tone={r.mem.maxBatch >= p.workload.batch ? 'ok' : 'bad'} />
          <Kpi label={`Max context at ${p.workload.batch} seq`} value={fmtNum(Math.max(0, r.mem.maxContext))} sub={`model limit ${fmtNum(m.maxContext)}`} />
        </div>

        {r.warnings.length > 0 && <ul className="warnings">{r.warnings.map(w => <li key={w}>{w}</li>)}</ul>}

        <div className="card pad">
          <div className="row between wrap gap-sm"><h2 className="h5">Throughput vs concurrency</h2><span className="muted small">{device.name} x{r.count}, {DTYPES[p.workload.weightDtype].label}</span></div>
          <CurveChart points={curve} current={p.workload.batch} />
        </div>

        <div className="two-col">
          <div className="card pad"><PhaseBars phase={r.decode} title={`Decode step · batch ${r.decode.batch}`} /></div>
          <div className="card pad"><PhaseBars phase={r.prefill} title={`Prefill · ${fmtNum(p.workload.promptLen)} tokens`} /></div>
        </div>
        <p className="muted small">Roofline estimate: each phase takes the slower of compute (at {Math.round(p.assumptions.mfu * 100)}% of peak) and memory traffic (at {Math.round(p.assumptions.bwUtil * 100)}% of bandwidth), plus all-reduce and pipeline hops over the interconnect. Real kernels add overheads this does not see. <Link href="/docs#method">How it is calculated</Link></p>
      </div>
    </div>
  )
}

const niceMax = (v: number) => { const e = Math.pow(10, Math.floor(Math.log10(Math.max(1, v)))); return Math.ceil(v / e) * e }

// ---------- History tab ----------

function HistoryTab({ p, update }: { p: Project; update: Update }) {
  const [sel, setSel] = useState<string[]>(() => p.snapshots.slice(-2).map(s => s.id))
  const current = snapshotOf(p, 'Current (unsaved)')
  const pick = (id: string) => setSel(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id].slice(-2))
  if (!p.snapshots.length) return (
    <Empty title="No saved configurations yet">
      <p className="muted">Save a snapshot (press <kbd>S</kbd>) each time you settle on a configuration, then compare them side by side.</p>
      <button type="button" className="btn primary" onClick={() => { if (current) update({ snapshots: [{ ...current, label: current.deviceName ? `${current.count}x ${current.deviceName}` : 'Snapshot' }] }) }}>Save the current plan</button>
    </Empty>
  )
  const chosen = sel.map(id => p.snapshots.find(s => s.id === id)).filter(Boolean) as Snapshot[]
  const [a, b] = chosen.length === 2 ? chosen : [chosen[0] ?? p.snapshots[p.snapshots.length - 1], current!]
  return (
    <div className="stack">
      <div className="card table-card">
        <table className="vt">
          <caption className="sr-only">Saved configurations</caption>
          <thead><tr><th scope="col">Compare</th><th scope="col">Saved</th><th scope="col">Configuration</th><th scope="col" className="num">tok/s</th><th scope="col" className="num">TTFT</th><th scope="col" className="num">$/1M</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
          <tbody>
            {[...p.snapshots].reverse().map(s => (
              <tr key={s.id}>
                <td><input type="checkbox" aria-label={`Compare ${s.label}`} checked={sel.includes(s.id)} onChange={() => pick(s.id)} /></td>
                <td className="nowrap small">{fmtDate(s.createdAt)}</td>
                <th scope="row"><input className="inline-edit" aria-label="Snapshot label" value={s.label} onChange={e => update({ snapshots: p.snapshots.map(x => x.id === s.id ? { ...x, label: e.target.value } : x) })} /><div className="muted small">{s.modelName} · {s.count}x {s.deviceName} (TP {s.tp} x PP {s.pp}){!s.result.fits && <span className="bad-t"> · does not fit</span>}</div></th>
                <td className="num">{fmtNum(s.result.tokS)}</td>
                <td className="num">{fmtMs(s.result.ttftMs)}</td>
                <td className="num">{fmtUsd(s.result.usdPerMTok, 2)}</td>
                <td><button type="button" className="btn danger-ghost sm" onClick={() => {
                  const before = p.snapshots
                  update({ snapshots: before.filter(x => x.id !== s.id) })
                  setSel(x => x.filter(i => i !== s.id))
                  toast('Snapshot deleted.', { undo: () => { const cur = getProject(p.id); if (cur) void saveProject({ ...cur, snapshots: before }) } })
                  setTimeout(() => { if (!getProject(p.id)?.snapshots.some(x => x.id === s.id)) void deleteSnapshotCloud(s.id) }, 9000)
                }}>Delete</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {a && b && <Compare a={a} b={b} />}
    </div>
  )
}

export function Compare({ a, b }: { a: Snapshot; b: Snapshot }) {
  type Row = [string, (s: Snapshot) => number | string, ((n: number) => string)?, ('higher' | 'lower')?]
  const rows: Row[] = [
    ['Model', s => s.modelName],
    ['Hardware', s => `${s.count}x ${s.deviceName}`],
    ['Split', s => `TP ${s.tp} x PP ${s.pp}`],
    ['Weights / KV dtype', s => `${s.workload.weightDtype} / ${s.workload.kvDtype}`],
    ['Context x sequences', s => `${fmtNum(s.workload.context)} x ${s.workload.batch}`],
    ['Fits', s => s.result.fits ? 'yes' : 'no'],
    ['Memory used per device', s => s.result.usedGB, fmtGB, 'lower'],
    ['Decode tok/s', s => s.result.tokS, n => fmtNum(n), 'higher'],
    ['Per-sequence tok/s', s => s.result.perUserTokS, n => fmtNum(n, 1), 'higher'],
    ['Time to first token', s => s.result.ttftMs, fmtMs, 'lower'],
    ['Max sequences at context', s => s.result.maxBatch, n => fmtNum(n), 'higher'],
    ['Hourly cost', s => s.result.perHour, n => fmtUsd(n, 2), 'lower'],
    ['$ per 1M tokens', s => s.result.usdPerMTok, n => fmtUsd(n, 2), 'lower'],
    ['Hardware cost', s => s.result.capex, n => fmtUsd(n), 'lower'],
    ['Tokens per joule', s => s.result.tokPerJ, n => fmtNum(n, 2), 'higher'],
  ]
  return (
    <div className="card table-card">
      <table className="vt compare">
        <caption className="h5">What changed</caption>
        <thead><tr><th scope="col">Metric</th><th scope="col">{a.label}</th><th scope="col">{b.label}</th><th scope="col" className="num">Change</th></tr></thead>
        <tbody>
          {rows.map(([label, get, fmt, better]) => {
            const va = get(a), vb = get(b)
            const num = typeof va === 'number' && typeof vb === 'number'
            const d = num ? (vb as number) - (va as number) : 0
            const pct = num && va ? d / Math.abs(va as number) : 0
            const good = better && d !== 0 ? (better === 'higher') === d > 0 : null
            return (
              <tr key={label} className={!num && va !== vb ? 'changed' : ''}>
                <th scope="row">{label}</th>
                <td>{num && fmt ? fmt(va as number) : String(va)}</td>
                <td>{num && fmt ? fmt(vb as number) : String(vb)}</td>
                <td className={`num ${good === null ? '' : good ? 'ok-t' : 'bad-t'}`}>{num ? (Math.abs(pct) < 0.0005 ? 'same' : `${pct > 0 ? '+' : ''}${(pct * 100).toFixed(0)}%`) : va === vb ? '' : 'changed'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
