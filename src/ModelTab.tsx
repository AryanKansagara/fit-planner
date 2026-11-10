import { useMemo, useRef, useState } from 'react'
import { kvBytesPerToken, manualSpec, paramCounts, parseConfig, specToConfig } from './engine/model.ts'
import { MODEL_SAMPLES } from './engine/samples.ts'
import { autoLayout, parseProjectModel } from './projectData.ts'
import type { Project, ModelSource } from './store.ts'
import { fmtB, fmtDate, fmtNum, toast } from './ui.tsx'
import { NumField, type Update } from './Project.tsx'

type Src = 'sample' | 'hf' | 'paste' | 'manual'

/** Accepts "org/name", a huggingface.co URL, or a URL to config.json. */
export function repoFromInput(s: string): string | null {
  const t = s.trim().replace(/^https?:\/\/(www\.)?huggingface\.co\//, '').replace(/\/(resolve|blob|tree)\/.*$/, '').replace(/\/+$/, '')
  return /^[\w.-]+\/[\w.-]+$/.test(t) ? t : null
}

export default function ModelTab({ p, update }: { p: Project; update: Update }) {
  const importing = new URLSearchParams(location.search).get('import') === '1'
  const [src, setSrc] = useState<Src>(importing ? 'hf' : p.model.kind === 'sample' ? 'sample' : p.model.kind)
  const [draft, setDraft] = useState<{ text: string; repo?: string; kind: ModelSource['kind']; sampleId?: string } | null>(null)
  const current = parseProjectModel(p)
  const shown = draft ? parseConfig(draft.text, draft.repo ? MODEL_SAMPLES.find(s => s.repo === draft.repo)?.name ?? draft.repo : undefined) : current
  const text = draft?.text ?? p.model.config

  const apply = () => {
    if (!draft || !shown.spec) return
    const model: ModelSource = { kind: draft.kind, repo: draft.repo, sampleId: draft.sampleId, config: draft.text, at: Date.now() }
    let workload = p.workload
    if (shown.suggestedDtype && shown.suggestedDtype !== 'bf16' && workload.weightDtype === 'bf16') workload = { ...workload, weightDtype: shown.suggestedDtype }
    const next = autoLayout({ ...p, model, workload })
    update({ model, workload, tp: next.tp, pp: next.pp, sample: draft.kind === 'sample' ? p.sample : false })
    setDraft(null)
    toast(`Now planning ${shown.spec.name}. Mesh resized to the smallest that fits.`, { tone: 'ok' })
    if (importing) history.replaceState({}, '', location.pathname)
  }

  return (
    <div className="model-grid">
      <div className="stack">
        <div className="card pad stack">
          <h2 className="h5">Where the model comes from</h2>
          <div className="seg" role="group" aria-label="Model source">
            {([['sample', 'Public samples'], ['hf', 'Hugging Face'], ['paste', 'Paste config.json'], ['manual', 'Manual']] as const).map(([k, l]) => (
              <button key={k} type="button" aria-pressed={src === k} onClick={() => setSrc(k)}>{l}</button>
            ))}
          </div>
          {src === 'sample' && <SamplePicker onPick={s => setDraft({ text: s.config, repo: s.repo, kind: 'sample', sampleId: s.id })} selected={draft?.sampleId ?? p.model.sampleId} />}
          {src === 'hf' && <HfFetch onFetched={(t, repo) => setDraft({ text: t, repo, kind: 'hf' })} />}
          {src === 'paste' && <Paste onText={t => setDraft({ text: t, kind: 'paste' })} />}
          {src === 'manual' && <Manual onSpec={t => setDraft({ text: t, kind: 'manual' })} />}
        </div>

        <Validation result={shown} text={text} draft={!!draft} />

        {draft && (
          <div className="sticky-apply card pad row between wrap gap-sm">
            <span>{shown.spec ? <>Ready: <strong>{shown.spec.name}</strong>, {fmtB(paramCounts(shown.spec).total)} parameters.</> : <span className="error">Fix the errors above before using this config.</span>}</span>
            <div className="row gap-sm"><button type="button" className="btn ghost sm" onClick={() => setDraft(null)}>Discard</button><button type="button" className="btn primary sm" disabled={!shown.spec} onClick={apply}>Use this model</button></div>
          </div>
        )}
      </div>

      <div className="stack">
        {shown.spec && <Breakdown result={shown} />}
        <div className="card pad">
          <div className="row between"><h2 className="h5">{draft ? 'Draft config.json' : 'Current config.json'}</h2>{!draft && <span className="muted small">{p.model.kind === 'hf' ? `Fetched from ${p.model.repo}` : p.model.kind === 'sample' ? `Sample: ${p.model.repo}` : p.model.kind} · {fmtDate(p.model.at)}</span>}</div>
          <Numbered text={text} marks={new Set(shown.issues.filter(i => i.line && i.level !== 'info').map(i => i.line!))} />
        </div>
      </div>
    </div>
  )
}

function SamplePicker({ onPick, selected }: { onPick(s: typeof MODEL_SAMPLES[number]): void; selected?: string }) {
  return (
    <div className="pick-list" role="list">
      {MODEL_SAMPLES.map(s => (
        <button key={s.id} type="button" role="listitem" className={`pick${selected === s.id ? ' on' : ''}`} onClick={() => onPick(s)}>
          <strong>{s.name}</strong><span className="mono small muted">{s.repo}</span><span className="small muted">{s.blurb}</span>
        </button>
      ))}
    </div>
  )
}

function HfFetch({ onFetched }: { onFetched(text: string, repo: string): void }) {
  const [input, setInput] = useState('')
  const [state, setState] = useState<{ busy: boolean; err: string | null }>({ busy: false, err: null })
  const go = async () => {
    const repo = repoFromInput(input)
    if (!repo) { setState({ busy: false, err: 'Enter a repo id like Qwen/Qwen3-14B or a huggingface.co model URL.' }); return }
    setState({ busy: true, err: null })
    try {
      const res = await fetch(`https://huggingface.co/${repo}/resolve/main/config.json`)
      if (res.status === 401 || res.status === 403) throw new Error(`${repo} is gated or private. Accept its license on huggingface.co, download config.json and paste it, or try an ungated mirror (for Llama, unsloth/<model>).`)
      if (res.status === 404) throw new Error(`No config.json found for ${repo}. Check the spelling; GGUF-only repos have no config.json, so use the original model repo.`)
      if (!res.ok) throw new Error(`Hugging Face returned ${res.status}.`)
      const t = await res.text()
      setState({ busy: false, err: null })
      onFetched(t, repo)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setState({ busy: false, err: /fetch|network|load failed/i.test(msg) ? 'Could not reach huggingface.co. Check your connection, or paste the config instead.' : msg })
    }
  }
  return (
    <form className="stack-sm" onSubmit={e => { e.preventDefault(); void go() }}>
      <label className="field"><span>Repo id or URL</span><input value={input} onChange={e => setInput(e.target.value)} placeholder="Qwen/Qwen3-32B" autoFocus spellCheck={false} /></label>
      <div className="row gap-sm wrap">
        <button className="btn primary sm" disabled={state.busy || !input.trim()}>{state.busy ? 'Fetching' : 'Fetch config.json'}</button>
        {['Qwen/Qwen3-32B', 'mistralai/Mistral-Small-24B-Instruct-2501', 'google/gemma-3-27b-it'].map(r => <button key={r} type="button" className="chip" onClick={() => setInput(r)}>{r}</button>)}
      </div>
      {state.err && <p className="error small" role="alert">{state.err}</p>}
      <p className="muted small">Fetched directly from your browser via the public Hub (no token, so gated repos need pasting).</p>
    </form>
  )
}

function Paste({ onText }: { onText(t: string): void }) {
  const [t, setT] = useState('')
  const file = useRef<HTMLInputElement>(null)
  return (
    <div className="stack-sm">
      <label className="field"><span>config.json contents</span><textarea className="mono" rows={10} value={t} onChange={e => { setT(e.target.value); if (e.target.value.trim()) onText(e.target.value) }} placeholder='{ "hidden_size": 4096, "num_hidden_layers": 32, ... }' spellCheck={false} /></label>
      <div className="row gap-sm"><button type="button" className="btn ghost sm" onClick={() => file.current?.click()}>Open a file</button><span className="muted small">Validated as you type.</span></div>
      <input ref={file} type="file" accept=".json,application/json" hidden onChange={async e => { const f = e.target.files?.[0]; if (f) { const x = await f.text(); setT(x); onText(x) } }} />
    </div>
  )
}

function Manual({ onSpec }: { onSpec(text: string): void }) {
  const [v, setV] = useState({ name: 'My model', hidden: 4096, layers: 32, heads: 32, kvHeads: 8, intermediate: 14336, vocab: 128256, maxContext: 131072, experts: 0, activeExperts: 2, expertInter: 0 })
  const set = (k: keyof typeof v, x: number | string) => { const n = { ...v, [k]: x }; setV(n); onSpec(specToConfig(manualSpec(n))) }
  return (
    <div className="stack-sm">
      <label className="field"><span>Name</span><input value={v.name} onChange={e => set('name', e.target.value)} /></label>
      <div className="three">
        <NumField label="Hidden size" value={v.hidden} min={1} onChange={x => set('hidden', x)} />
        <NumField label="Layers" value={v.layers} min={1} onChange={x => set('layers', x)} />
        <NumField label="Attention heads" value={v.heads} min={1} onChange={x => set('heads', x)} />
        <NumField label="KV heads" value={v.kvHeads} min={1} onChange={x => set('kvHeads', x)} />
        <NumField label="MLP intermediate" value={v.intermediate} min={1} onChange={x => set('intermediate', x)} />
        <NumField label="Vocabulary" value={v.vocab} min={1} onChange={x => set('vocab', x)} />
        <NumField label="Max context" value={v.maxContext} min={1} onChange={x => set('maxContext', x)} />
        <NumField label="Experts (0 = dense)" value={v.experts} min={0} onChange={x => set('experts', x)} />
        {v.experts > 1 && <NumField label="Active experts" value={v.activeExperts} min={1} onChange={x => set('activeExperts', x)} />}
      </div>
      <p className="muted small">Assumes a SwiGLU decoder with untied embeddings. The form writes a config.json, so the same validation applies.</p>
    </div>
  )
}

function Validation({ result, text, draft }: { result: ReturnType<typeof parseConfig>; text: string; draft: boolean }) {
  const errs = result.issues.filter(i => i.level === 'error').length
  const warns = result.issues.filter(i => i.level === 'warn').length
  return (
    <div className="card pad stack-sm validation">
      <div className="row between wrap gap-sm">
        <h2 className="h5">Validation {draft ? '(draft)' : ''}</h2>
        <span className="v-sum"><span className="badge">{result.format}</span> <span className={errs ? 'bad-t' : 'ok-t'}>{errs} errors</span> · <span className={warns ? 'warn-t' : ''}>{warns} warnings</span> · {text.split('\n').length} lines</span>
      </div>
      {result.issues.length === 0 ? <p className="ok-t small">Every sizing field was found. Nothing assumed.</p> : (
        <ul className="issues">
          {result.issues.map((i, k) => <li key={k} className={`iss ${i.level}`}><span className="iss-lvl">{i.level}</span>{i.line && <span className="mono">line {i.line}</span>}<span>{i.message}</span></li>)}
        </ul>
      )}
      {result.mapping.length > 0 && (
        <details>
          <summary className="small">Field mapping ({result.mapping.length} fields)</summary>
          <table className="vt compact">
            <thead><tr><th scope="col">Planner field</th><th scope="col">Value</th><th scope="col">From</th></tr></thead>
            <tbody>{result.mapping.map(f => <tr key={String(f.field)}><th scope="row">{f.label}</th><td className="mono">{String(f.value)}</td><td className="mono small">{f.source}{f.line ? ` (line ${f.line})` : ''}</td></tr>)}</tbody>
          </table>
        </details>
      )}
    </div>
  )
}

function Breakdown({ result }: { result: ReturnType<typeof parseConfig> }) {
  const m = result.spec!
  const pc = useMemo(() => paramCounts(m), [m])
  const mlp = pc.layerTotal - m.layers * (pc.attnPerLayer + 2 * m.hidden)
  const rows: [string, number][] = [
    ['Embedding', pc.embedding],
    [m.tied ? 'Output head (tied, shared)' : 'Output head', pc.head],
    ['Attention (all layers)', m.layers * pc.attnPerLayer],
    [m.moe ? 'MLP and experts (all layers)' : 'MLP (all layers)', mlp],
  ]
  return (
    <div className="card pad stack-sm">
      <h2 className="h5">{m.name}</h2>
      <div className="kpis three-up">
        <div className="kpi"><div className="kpi-label">Total parameters</div><div className="kpi-value">{fmtB(pc.total)}</div></div>
        <div className="kpi"><div className="kpi-label">Active per token</div><div className="kpi-value">{fmtB(pc.active)}</div></div>
        <div className="kpi"><div className="kpi-label">KV per token (BF16)</div><div className="kpi-value">{fmtNum(kvBytesPerToken(m, 'bf16') / 1024)} KB</div></div>
      </div>
      <div className="stackbar" role="img" aria-label="Parameter breakdown">
        {rows.filter(r => r[1] > 0).map(([l, v], i) => <i key={l} className={`sb${i}`} style={{ flexGrow: v }} title={`${l}: ${fmtB(v)}`} />)}
      </div>
      <ul className="legend">{rows.map(([l, v], i) => <li key={l}><i className={`sb${i}`} />{l} {fmtB(v)}</li>)}</ul>
      {m.moe && <p className="muted small">{m.moe.experts} experts, {m.moe.active} routed per token{m.moe.shared ? ` plus ${m.moe.shared} shared` : ''}{m.moe.denseLayers ? `; first ${m.moe.denseLayers} layers dense` : ''}. Every expert must be resident, but decode only reads the experts the batch actually hits.</p>}
      {m.mla && <p className="muted small">Latent attention: {m.mla.kvLoraRank}-dim KV latent plus {m.mla.qkRope}-dim RoPE key per layer, replicated across a tensor-parallel group.</p>}
    </div>
  )
}

function Numbered({ text, marks }: { text: string; marks: Set<number> }) {
  const lines = text.split('\n')
  return (
    <pre className="numbered" tabIndex={0} aria-label="config.json with line numbers">
      {lines.slice(0, 400).map((l, i) => <div key={i} className={marks.has(i + 1) ? 'mark' : ''}><span className="ln">{i + 1}</span>{l || ' '}</div>)}
      {lines.length > 400 && <div className="muted">… {lines.length - 400} more lines</div>}
    </pre>
  )
}
