import { useEffect, useRef, useState } from 'react'
import { MODEL_SAMPLES, WORKLOAD_SAMPLES } from './engine/samples.ts'
import { createBlankProject, createSampleProject, importPlanJson, projectDevice, projectPlan, projectSpec } from './projectData.ts'
import { deleteProject, saveProject, useProjects } from './store.ts'
import { Empty, fmtNum, go, Link, Modal, SyntheticBadge, toast } from './ui.tsx'
import { Mesh } from './viz.tsx'

export default function Workspace() {
  const list = useProjects()
  const [naming, setNaming] = useState(false)
  const [importing, setImporting] = useState(false)
  useEffect(() => {
    const q = new URLSearchParams(location.search)
    if (q.get('new') === '1') { history.replaceState({}, '', '/app'); setNaming(true) }
    if (q.get('import') === '1') { history.replaceState({}, '', '/app'); setImporting(true) }
    const s = q.get('sample'), w = q.get('workload') ?? undefined, tab = q.get('tab')
    if (s && MODEL_SAMPLES.some(x => x.id === s)) void createSampleProject(s, w).then(p => go(`/app/p/${p.id}${tab ? '/' + tab : ''}`, { replace: true }))
  }, [])

  const create = async (name: string) => {
    const p = await createBlankProject(name || 'New deployment plan')
    go(`/app/p/${p.id}/model?import=1`)
  }
  const sample = async (id: string, w?: string) => { const p = await createSampleProject(id, w); go(`/app/p/${p.id}`) }

  if (list === null) return <div className="boot" role="status"><span className="spinner" /> Loading</div>
  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Workspace</p>
          <h1>Deployment plans</h1>
          <p className="muted">One plan per model you want to serve. Each keeps its config, hardware edits, targets and a history of saved configurations.</p>
        </div>
        <div className="row gap-sm wrap">
          <button type="button" className="btn ghost" onClick={() => setImporting(true)}>Import plan JSON</button>
          <button type="button" className="btn primary" onClick={() => setNaming(true)}>New plan</button>
        </div>
      </div>

      {list.length === 0 ? (
        <Empty title="No plans yet" icon={<div className="empty-mesh" aria-hidden="true">{Array.from({ length: 8 }, (_, i) => <i key={i} />)}</div>}>
          <p className="muted">Start from Llama 3.3 70B on a four-card Blackhole mesh, or bring your own model's config.json.</p>
          <div className="row gap center-x wrap">
            <button type="button" className="btn primary" onClick={() => void sample('llama-3.3-70b', 'team-chat')}>Try the sample</button>
            <button type="button" className="btn ghost" onClick={() => setNaming(true)}>Import your model</button>
          </div>
        </Empty>
      ) : (
        <div className="proj-grid">
          {list.map(p => {
            const m = projectSpec(p)
            const r = projectPlan(p)
            return (
              <article key={p.id} className="card proj">
                <Link href={`/app/p/${p.id}`} className="proj-link">
                  <h3>{p.name} {p.sample && <SyntheticBadge />}</h3>
                  <p className="muted small">{m?.name ?? 'Model config needs fixing'} · {r ? `${r.count}x ${projectDevice(p).name}` : ''}</p>
                  {r && <Mesh tp={r.tp} pp={r.pp} fill={{ weightsGB: r.mem.weightsGB, kvGB: r.mem.kvGB, reserveGB: r.mem.reserveGB, capGB: r.mem.capGB }} compact maxTiles={16} />}
                  <div className="proj-meta">
                    {r ? <><span className={r.mem.fits ? 'ok-t' : 'bad-t'}>{r.mem.fits ? 'Fits' : 'Does not fit'}</span><span>{fmtNum(r.decode.tokS)} tok/s</span><span>{p.snapshots.length} saved</span></> : <span className="bad-t">Fix the model config</span>}
                  </div>
                </Link>
                <div className="row gap-sm proj-actions">
                  <Link href={`/app/p/${p.id}/search`} className="btn ghost sm">Search</Link>
                  <Link href={`/app/p/${p.id}/report`} className="btn ghost sm">Report</Link>
                  <button type="button" className="btn danger-ghost sm" onClick={async () => {
                    const undo = await deleteProject(p.id)
                    toast(`Deleted ${p.name}.`, { undo: () => void undo() })
                  }}>Delete</button>
                </div>
              </article>
            )
          })}
        </div>
      )}

      <div className="samples">
        <h2 className="h4">Start from a public architecture</h2>
        <p className="muted small">Configs copied from each model's Hugging Face config.json. Paired with the synthetic "team chat" workload; change it on the plan.</p>
        <div className="sample-grid">
          {MODEL_SAMPLES.map(s => (
            <button key={s.id} type="button" className="card sample-card" onClick={() => void sample(s.id)}>
              <strong>{s.name}</strong>
              <span className="mono small muted">{s.repo}</span>
              <span className="muted small">{s.blurb}</span>
            </button>
          ))}
        </div>
        <h2 className="h4 mt">Synthetic workload profiles</h2>
        <div className="sample-grid">
          {WORKLOAD_SAMPLES.map(w => (
            <button key={w.id} type="button" className="card sample-card" onClick={() => void sample('llama-3.1-8b', w.id)}>
              <strong>{w.name.replace(' (synthetic)', '')}</strong> <SyntheticBadge />
              <span className="muted small">{w.blurb}</span>
              <span className="mono small muted">{w.batch} seq · {fmtNum(w.promptLen)} in / {w.outputLen} out · {w.minTokS}+ tok/s</span>
            </button>
          ))}
        </div>
      </div>

      <Modal open={naming} onClose={() => setNaming(false)} title="New deployment plan">
        <form onSubmit={e => { e.preventDefault(); void create(String(new FormData(e.currentTarget).get('name') ?? '').trim()) }} className="stack">
          <label className="field"><span>Plan name</span><input name="name" autoFocus placeholder="Homelab Llama 70B" /></label>
          <p className="muted small">Next: paste a config.json, fetch one from the Hugging Face Hub by repo id, or type the dimensions by hand.</p>
          <div className="row end gap"><button type="button" className="btn ghost" onClick={() => setNaming(false)}>Cancel</button><button className="btn primary">Create and import</button></div>
        </form>
      </Modal>
      <ImportPlan open={importing} onClose={() => setImporting(false)} />
    </section>
  )
}

function ImportPlan({ open, onClose }: { open: boolean; onClose(): void }) {
  const [text, setText] = useState('')
  const [errors, setErrors] = useState<string[]>([])
  const file = useRef<HTMLInputElement>(null)
  const run = async (t: string) => {
    const r = importPlanJson(t)
    setErrors(r.errors)
    if (r.project) { const p = await saveProject(r.project); toast(`Imported ${p.name}.`, { tone: 'ok' }); onClose(); go(`/app/p/${p.id}`) }
  }
  return (
    <Modal open={open} onClose={onClose} title="Import a shared plan">
      <div className="stack">
        <p className="muted small">A plan file from Share plan &gt; Download JSON. It brings the model config, workload, targets, hardware edits and saved snapshots.</p>
        <div className="row gap-sm"><button type="button" className="btn ghost sm" onClick={() => file.current?.click()}>Choose file</button><span className="muted small">or paste below</span></div>
        <input ref={file} type="file" accept=".json,application/json" hidden onChange={async e => { const f = e.target.files?.[0]; if (f) { const t = await f.text(); setText(t); void run(t) } }} />
        <label className="field"><span>Plan JSON</span><textarea rows={8} className="mono" value={text} onChange={e => setText(e.target.value)} placeholder='{ "format": "fit-planner/plan@1", ... }' /></label>
        {errors.length > 0 && <ul className="issues" role="alert">{errors.map(e => <li key={e} className="error">{e}</li>)}</ul>}
        <div className="row end gap"><button type="button" className="btn ghost" onClick={onClose}>Cancel</button><button type="button" className="btn primary" disabled={!text.trim()} onClick={() => void run(text)}>Import</button></div>
      </div>
    </Modal>
  )
}
