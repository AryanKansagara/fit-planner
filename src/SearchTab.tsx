import { useMemo, useState } from 'react'
import { fmtMs, search, type SearchOption } from './engine/plan.ts'
import type { Targets } from './engine/types.ts'
import { projectDevices, projectSpec } from './projectData.ts'
import type { Project } from './store.ts'
import { download, Empty, fmtNum, fmtUsd, Link, toast } from './ui.tsx'
import { NumField, type Update } from './Project.tsx'

export function searchCsv(opts: SearchOption[]): string {
  const head = 'rank,meets_targets,device,cards_per_mesh,tp,pp,replicas,total_cards,agg_tok_s,ms_per_token,ttft_ms,usd_per_hour,usd_per_1m_tokens,hardware_usd,tok_per_joule,memory_used_gb,memory_gb,reason'
  const esc = (s: string) => /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  return [head, ...opts.map((o, i) => [i + 1, o.ok, esc(o.plan.device.name), o.plan.count, o.plan.tp, o.plan.pp, o.replicas, o.totalDevices, o.aggTokS.toFixed(1), o.plan.decode.ms.toFixed(2), o.plan.ttftMs.toFixed(0), o.perHour.toFixed(3), o.usdPerMTok.toFixed(3), o.plan.cost.capex * o.replicas, o.plan.cost.tokPerJ.toFixed(3), o.plan.mem.usedGB.toFixed(2), o.plan.mem.capGB, esc(o.reason)].join(','))].join('\n')
}

export function useSearch(p: Project) {
  const m = projectSpec(p)
  const devices = projectDevices(p)
  return useMemo(() => m ? search(m, p.workload, devices, p.assumptions, p.targets) : [], [m, p.workload, p.devices, p.assumptions, p.targets])
}

export default function SearchTab({ p, update }: { p: Project; update: Update }) {
  const opts = useSearch(p)
  const [showMisses, setShowMisses] = useState(false)
  const [vendor, setVendor] = useState('all')
  const t = (patch: Partial<Targets>) => update({ targets: { ...p.targets, ...patch } })
  if (!projectSpec(p)) return <Empty title="Fix the model config first"><Link href={`/app/p/${p.id}/model`} className="btn primary">Model</Link></Empty>
  const vendors = [...new Set(opts.map(o => o.plan.device.vendor))]
  const filtered = opts.filter(o => vendor === 'all' || o.plan.device.vendor === vendor)
  const ok = filtered.filter(o => o.ok)
  const misses = filtered.filter(o => !o.ok && o.plan.mem.weightsFit)
  const useOpt = (o: SearchOption) => {
    update({ deviceId: o.plan.device.id, tp: o.plan.tp, pp: o.plan.pp, workload: { ...p.workload, batch: Math.ceil(p.workload.batch / o.replicas) } })
    toast(o.replicas > 1 ? `Plan set to one of ${o.replicas} replicas: ${o.plan.count}x ${o.plan.device.name}, ${Math.ceil(p.workload.batch / o.replicas)} sequences each.` : `Plan set to ${o.plan.count}x ${o.plan.device.name}.`, { tone: 'ok' })
  }
  return (
    <div className="stack">
      <div className="card pad stack-sm">
        <div className="row between wrap gap-sm"><h2 className="h5">Targets</h2><span className="muted small">{p.workload.batch} concurrent sequences, {fmtNum(p.workload.promptLen)}-token prompts, {p.workload.weightDtype.toUpperCase()} weights. Change these on the Plan tab.</span></div>
        <div className="row gap wrap">
          <NumField label="Aggregate throughput at least" suffix="tok/s" value={p.targets.minTokS} min={0} step={10} onChange={v => t({ minTokS: v })} />
          <NumField label="Time per output token at most" suffix="ms" value={p.targets.maxTpotMs} min={1} step={5} onChange={v => t({ maxTpotMs: v })} />
          <NumField label="Time to first token at most" suffix="ms" value={p.targets.maxTtftMs} min={1} step={100} onChange={v => t({ maxTtftMs: v })} />
          <label className="field"><span>Vendor</span><select value={vendor} onChange={e => setVendor(e.target.value)}><option value="all">All</option>{vendors.map(v => <option key={v}>{v}</option>)}</select></label>
        </div>
        <p className="muted small">{fmtNum(p.targets.maxTpotMs ? 1000 / p.targets.maxTpotMs : 0, 0)} tok/s per sequence is the floor implied by the latency target. Options add data-parallel replicas (up to 8) when one mesh is too slow.</p>
      </div>

      <div className="row between wrap gap-sm">
        <h2 className="h5">{ok.length ? `${ok.length} configurations meet the targets, cheapest first` : 'Nothing meets every target'}</h2>
        <div className="row gap-sm">
          <button type="button" className="btn ghost sm" onClick={() => download(`${p.name.replace(/\W+/g, '-')}-search.csv`, searchCsv(filtered))}>Export CSV</button>
          <Link href={`/app/p/${p.id}/report`} className="btn ghost sm">Printable report</Link>
        </div>
      </div>
      {ok.length > 0 ? <OptionTable opts={ok} onUse={useOpt} current={p} /> : <p className="muted">Loosen a target, shrink the context or batch, use a smaller weight dtype, or add a custom device. The closest misses are below.</p>}

      <button type="button" className="linkish small" onClick={() => setShowMisses(s => !s)} aria-expanded={showMisses || !ok.length}>{showMisses || !ok.length ? 'Hide' : 'Show'} {misses.length} near misses (weights fit, a target does not)</button>
      {(showMisses || !ok.length) && misses.length > 0 && <OptionTable opts={misses.slice(0, 30)} onUse={useOpt} current={p} />}
    </div>
  )
}

export function OptionTable({ opts, onUse, current, print }: { opts: SearchOption[]; onUse?(o: SearchOption): void; current?: Project; print?: boolean }) {
  return (
    <div className="card table-card scroll-x">
      <table className="vt options">
        <thead><tr>
          <th scope="col">#</th><th scope="col">Hardware</th><th scope="col" className="num">tok/s</th><th scope="col" className="num">ms/token</th><th scope="col" className="num">TTFT</th>
          <th scope="col" className="num">$/hour</th><th scope="col" className="num">$/1M tok</th><th scope="col" className="num">Capex</th><th scope="col" className="num">tok/J</th>{!print && <th scope="col"><span className="sr-only">Use</span></th>}
        </tr></thead>
        <tbody>
          {opts.map((o, i) => {
            const isCur = current && current.deviceId === o.plan.device.id && current.tp === o.plan.tp && current.pp === o.plan.pp && o.replicas === 1
            return (
              <tr key={`${o.plan.device.id}-${o.plan.count}`} className={isCur ? 'current' : ''}>
                <td className="muted">{i + 1}</td>
                <th scope="row">
                  <strong>{o.replicas > 1 ? `${o.replicas} x ` : ''}{o.plan.count}x {o.plan.device.name}</strong>
                  <div className="muted small">TP {o.plan.tp} x PP {o.plan.pp} · {fmtNum(o.plan.mem.usedGB, 1)}/{o.plan.mem.capGB} GB · {o.plan.decode.bound}-bound{o.ok ? '' : <span className="bad-t"> · {o.reason}</span>}</div>
                </th>
                <td className="num">{fmtNum(o.aggTokS)}</td>
                <td className="num">{fmtNum(o.plan.decode.ms, 1)}</td>
                <td className="num">{fmtMs(o.plan.ttftMs)}</td>
                <td className="num">{fmtUsd(o.perHour, 2)}</td>
                <td className="num">{fmtUsd(o.usdPerMTok, 2)}</td>
                <td className="num">{fmtUsd(o.plan.cost.capex * o.replicas)}</td>
                <td className="num">{fmtNum(o.plan.cost.tokPerJ, 2)}</td>
                {!print && <td>{isCur ? <span className="badge on">current</span> : onUse && <button type="button" className="btn ghost sm" onClick={() => onUse(o)}>Use</button>}</td>}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
