import { useEffect } from 'react'
import { HARDWARE_AS_OF } from './engine/hardware.ts'
import { kvBytesPerToken, paramCounts } from './engine/model.ts'
import { fmtMs } from './engine/plan.ts'
import { DTYPES } from './engine/types.ts'
import { projectPlan, projectSpec } from './projectData.ts'
import { useProjects } from './store.ts'
import { OptionTable, searchCsv, useSearch } from './SearchTab.tsx'
import { download, fmtB, fmtGB, fmtNum, fmtUsd, Link, Monogram, NotFound, ThemeToggle } from './ui.tsx'
import { Mesh, MeshLegend } from './viz.tsx'
import type { Project } from './store.ts'

export default function Report({ id }: { id: string }) {
  const list = useProjects()
  const p = list?.find(x => x.id === id)
  useEffect(() => { if (p) document.title = `${p.name} report · Fit Planner` }, [p?.name])
  if (list === null) return <div className="boot" role="status"><span className="spinner" /> Loading</div>
  if (!p) return <NotFound />
  return <ReportBody p={p} />
}

function ReportBody({ p }: { p: Project }) {
  const m = projectSpec(p)
  const r = projectPlan(p)
  const opts = useSearch(p)
  const ok = opts.filter(o => o.ok)
  if (!m || !r) return <section className="page narrow"><h1>Report unavailable</h1><p>The model config has errors. <Link href={`/app/p/${p.id}/model`}>Fix it</Link>.</p></section>
  const pc = paramCounts(m)
  const today = new Date().toISOString().slice(0, 10)
  const fill = { weightsGB: r.mem.weightsGB, kvGB: r.mem.kvGB, reserveGB: r.mem.reserveGB, capGB: r.mem.capGB }
  const shown = (ok.length ? ok : opts.filter(o => o.plan.mem.weightsFit)).slice(0, 12)
  return (
    <div className="report-page">
      <div className="report-bar no-print">
        <Link href={`/app/p/${p.id}`} className="btn ghost sm">Back to plan</Link>
        <span className="grow" />
        <ThemeToggle />
        <button type="button" className="btn ghost sm" onClick={() => download(`${p.name.replace(/\W+/g, '-')}-options.csv`, searchCsv(opts))}>CSV</button>
        <button type="button" className="btn primary sm" onClick={() => print()}>Print or save PDF</button>
      </div>
      <article className="report">
        <header className="report-cover">
          <div className="row gap-sm"><Monogram /><span className="utility">Fit Planner · deployment comparison</span></div>
          <h1>{p.name}</h1>
          <p className="report-sub">{m.name} · {fmtB(pc.total)} parameters{m.moe ? ` (${fmtB(pc.active)} active)` : ''} · {DTYPES[p.workload.weightDtype].label} weights · prepared {today}</p>
        </header>

        <section className="report-sec">
          <h2>Workload and targets</h2>
          <div className="report-kpis">
            <div><span>Concurrent sequences</span><b>{p.workload.batch}</b></div>
            <div><span>Context reserved</span><b>{fmtNum(p.workload.context)}</b></div>
            <div><span>Prompt / output</span><b>{fmtNum(p.workload.promptLen)} / {fmtNum(p.workload.outputLen)}</b></div>
            <div><span>Throughput target</span><b>{fmtNum(p.targets.minTokS)} tok/s</b></div>
            <div><span>Per-token latency</span><b>{p.targets.maxTpotMs} ms</b></div>
            <div><span>First token within</span><b>{fmtMs(p.targets.maxTtftMs)}</b></div>
          </div>
        </section>

        <section className="report-sec">
          <h2>Current plan: {r.count}x {r.device.name}</h2>
          <div className="report-plan">
            <div>
              <Mesh tp={r.tp} pp={r.pp} fill={fill} compact maxTiles={32} />
              <MeshLegend fill={fill} />
            </div>
            <dl className="facts">
              <dt>Fits</dt><dd className={r.mem.fits ? 'ok-t' : 'bad-t'}>{r.mem.fits ? 'Yes' : 'No'} ({fmtGB(r.mem.usedGB)} of {fmtGB(r.mem.capGB)} per device)</dd>
              <dt>Split</dt><dd>TP {r.tp} x PP {r.pp}</dd>
              <dt>Decode</dt><dd>{fmtNum(r.decode.tokS)} tok/s total, {fmtNum(r.decode.perUserTokS, 1)} per sequence ({r.decode.bound}-bound)</dd>
              <dt>First token</dt><dd>{fmtMs(r.ttftMs)}</dd>
              <dt>Max sequences</dt><dd>{fmtNum(r.mem.maxBatch)} at {fmtNum(p.workload.context)} tokens</dd>
              <dt>KV per token</dt><dd>{fmtNum(kvBytesPerToken(m, p.workload.kvDtype) / 1024)} KB ({p.workload.kvDtype})</dd>
              <dt>Cost</dt><dd>{fmtUsd(r.cost.perHour, 2)}/h · {fmtUsd(r.cost.usdPerMTok, 2)} per 1M tokens · {fmtUsd(r.cost.capex)} hardware</dd>
              <dt>Energy</dt><dd>{fmtNum(r.cost.tokPerJ, 2)} tok/J</dd>
            </dl>
          </div>
          {r.warnings.length > 0 && <ul className="warnings">{r.warnings.map(w => <li key={w}>{w}</li>)}</ul>}
        </section>

        <section className="report-sec">
          <h2>{ok.length ? 'Options that meet the targets, cheapest first' : 'Closest options (none meets every target)'}</h2>
          <OptionTable opts={shown} print />
          <p className="small muted">{ok.length} of {opts.length} configurations meet every target. Hourly cost is {p.assumptions.costMode === 'rent' ? 'rental where listed, otherwise ' : ''}hardware amortised over {p.assumptions.amortYears} years plus electricity at ${p.assumptions.kwhUsd}/kWh at board power.</p>
        </section>

        <section className="report-sec">
          <h2>Assumptions</h2>
          <p className="small">Roofline model: decode reads active weights and the batch's KV cache at {Math.round(p.assumptions.bwUtil * 100)}% of peak bandwidth; prefill runs at {Math.round(p.assumptions.mfu * 100)}% of dense peak; tensor-parallel layers add two ring all-reduces each at {Math.round(p.assumptions.linkUtil * 100)}% of link bandwidth; pipeline stages run in sequence. Each device keeps {p.assumptions.reserveGB} GB plus {Math.round(p.assumptions.reserveFrac * 100)}% free for runtime and activations. Hardware figures as of {HARDWARE_AS_OF}{p.devices.length ? `, with ${p.devices.length} edited or custom device${p.devices.length > 1 ? 's' : ''}` : ''}. These are planning estimates, not benchmarks: measure before you buy.</p>
        </section>
        <footer className="report-foot small muted">Fit Planner · independent project, not affiliated with Tenstorrent · {location.host}</footer>
      </article>
    </div>
  )
}
