import type { CurvePoint, Phase } from './engine/plan.ts'
import { fmtGB, fmtNum, fmtTok } from './ui.tsx'

export interface TileFill { weightsGB: number; kvGB: number; reserveGB: number; capGB: number }

/** Grid of devices: columns are the tensor-parallel group, rows are pipeline stages. Each tile fills with weights, KV cache and reserve. */
export function Mesh({ tp, pp, fill, label, maxTiles = 64, compact }: { tp: number; pp: number; fill: TileFill; label?: string; maxTiles?: number; compact?: boolean }) {
  const total = tp * pp
  const shownPp = Math.max(1, Math.min(pp, Math.floor(maxTiles / tp)))
  const used = fill.weightsGB + fill.kvGB + fill.reserveGB
  const over = used > fill.capGB
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / fill.capGB) * 100))}%`
  // Stack from the bottom: reserve, weights, KV.
  const kvShown = Math.max(0, Math.min(fill.kvGB, fill.capGB - fill.weightsGB - fill.reserveGB))
  return (
    <div className={`mesh${compact ? ' compact' : ''}`} role="img" aria-label={`${total} devices, ${tp} tensor-parallel by ${pp} pipeline stages. Each holds ${fmtGB(fill.weightsGB)} of weights and ${fmtGB(fill.kvGB)} of KV cache out of ${fmtGB(fill.capGB)}${over ? ', over capacity' : ''}.`}>
      <div className="mesh-grid" style={{ gridTemplateColumns: `repeat(${Math.min(tp, 16)}, minmax(0, 1fr))` }}>
        {Array.from({ length: shownPp * Math.min(tp, 16) }, (_, i) => (
          <div key={i} className={`tile${over ? ' over' : ''}`} style={{ animationDelay: `${(i % 16) * 25 + Math.floor(i / 16) * 40}ms` }}>
            <i className="t-res" style={{ height: pct(fill.reserveGB) }} />
            <i className="t-w" style={{ height: pct(fill.weightsGB), bottom: pct(fill.reserveGB) }} />
            <i className="t-kv" style={{ height: pct(kvShown), bottom: pct(fill.reserveGB + fill.weightsGB) }} />
            {!compact && i === 0 && <span className="tile-lbl">{label ?? ''}</span>}
          </div>
        ))}
      </div>
      {(shownPp < pp || tp > 16) && <p className="muted small">Showing {shownPp * Math.min(tp, 16)} of {total} devices.</p>}
    </div>
  )
}

export function MeshLegend({ fill }: { fill: TileFill }) {
  const free = Math.max(0, fill.capGB - fill.weightsGB - fill.kvGB - fill.reserveGB)
  return (
    <ul className="legend">
      <li><i className="lg-w" />Weights {fmtGB(fill.weightsGB)}</li>
      <li><i className="lg-kv" />KV cache {fmtGB(fill.kvGB)}</li>
      <li><i className="lg-res" />Reserve {fmtGB(fill.reserveGB)}</li>
      <li><i className="lg-free" />{free > 0 ? `Free ${fmtGB(free)}` : `Short ${fmtGB(fill.weightsGB + fill.kvGB + fill.reserveGB - fill.capGB)}`}</li>
    </ul>
  )
}

/** Semicircular gauge for tokens per second, scaled to `max`. */
export function Dial({ value, max, label, sub }: { value: number; max: number; label: string; sub?: string }) {
  const f = Math.max(0, Math.min(1, value / max))
  const r = 80, cx = 100, cy = 96
  const arc = (t: number) => { const a = Math.PI * (1 - t); return [cx + r * Math.cos(a), cy - r * Math.sin(a)] }
  const [ex, ey] = arc(f)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <figure className="dial">
      <svg viewBox="0 0 200 118" aria-hidden="true">
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} className="dial-track" />
        {f > 0.001 && <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${ex} ${ey}`} className="dial-val" />}
        {ticks.map(t => { const [x1, y1] = arc(t); const a = Math.PI * (1 - t); return <line key={t} x1={x1} y1={y1} x2={cx + (r - 14) * Math.cos(a)} y2={cy - (r - 14) * Math.sin(a)} className="dial-tick" /> })}
        <line x1={cx} y1={cy} x2={cx + (r - 22) * Math.cos(Math.PI * (1 - f))} y2={cy - (r - 22) * Math.sin(Math.PI * (1 - f))} className="dial-needle" />
        <circle cx={cx} cy={cy} r="5" className="dial-hub" />
      </svg>
      <figcaption>
        <strong aria-live="polite">{value >= 100 ? fmtNum(value) : fmtNum(value, 1)}</strong>
        <span>{label}</span>
        {sub && <small>{sub}</small>}
      </figcaption>
    </figure>
  )
}

/** Aggregate and per-user decode throughput against concurrency, log2 x axis. */
export function CurveChart({ points, current }: { points: CurvePoint[]; current: number }) {
  const W = 640, H = 240, L = 52, R = 52, T = 14, B = 34
  if (!points.length) return null
  const maxAgg = Math.max(...points.map(p => p.tokS)) * 1.1
  const maxUser = Math.max(...points.map(p => p.perUserTokS)) * 1.1
  const x = (i: number) => L + (i / Math.max(1, points.length - 1)) * (W - L - R)
  const yA = (v: number) => T + (1 - v / maxAgg) * (H - T - B)
  const yU = (v: number) => T + (1 - v / maxUser) * (H - T - B)
  const firstMiss = points.findIndex(p => !p.fits)
  const curIdx = points.reduce((best, p, i) => Math.abs(Math.log2(p.batch) - Math.log2(current)) < Math.abs(Math.log2(points[best].batch) - Math.log2(current)) ? i : best, 0)
  const path = (f: (p: CurvePoint) => number) => points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${f(p).toFixed(1)}`).join(' ')
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Decode throughput from batch 1 to ${points[points.length - 1].batch}. Aggregate peaks at ${fmtNum(Math.max(...points.map(p => p.tokS)))} tokens per second.`}>
        {firstMiss >= 0 && <rect x={x(firstMiss) - (firstMiss ? (x(1) - x(0)) / 2 : 0)} y={T} width={W - R - x(firstMiss) + (firstMiss ? (x(1) - x(0)) / 2 : 0)} height={H - T - B} className="ch-nofit" />}
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={T + t * (H - T - B)} y2={T + t * (H - T - B)} className="ch-grid" />
            <text x={L - 8} y={T + t * (H - T - B) + 4} textAnchor="end" className="ch-ax a">{fmtTok(maxAgg * (1 - t))}</text>
            <text x={W - R + 8} y={T + t * (H - T - B) + 4} className="ch-ax u">{fmtNum(maxUser * (1 - t))}</text>
          </g>
        ))}
        {points.map((p, i) => <text key={p.batch} x={x(i)} y={H - B + 18} textAnchor="middle" className="ch-ax">{p.batch}</text>)}
        <text x={(L + W - R) / 2} y={H - 2} textAnchor="middle" className="ch-ax">concurrent sequences</text>
        <line x1={x(curIdx)} x2={x(curIdx)} y1={T} y2={H - B} className="ch-cur" />
        <path d={path(p => yA(p.tokS))} className="ch-agg" />
        <path d={path(p => yU(p.perUserTokS))} className="ch-user" />
        {points.map((p, i) => <circle key={p.batch} cx={x(i)} cy={yA(p.tokS)} r={i === curIdx ? 5 : 3} className="ch-dot"><title>{`Batch ${p.batch}: ${fmtNum(p.tokS)} tok/s total, ${fmtNum(p.perUserTokS, 1)} tok/s each${p.fits ? '' : ' (KV does not fit)'}`}</title></circle>)}
      </svg>
      <figcaption><ul className="legend">
        <li><i className="lg-agg" />Aggregate tok/s (left)</li>
        <li><i className="lg-user" />Per-sequence tok/s (right)</li>
        {firstMiss >= 0 && <li><i className="lg-nofit" />KV cache does not fit</li>}
      </ul></figcaption>
    </figure>
  )
}

/** Where the time goes in one step: compute, memory traffic and interconnect. */
export function PhaseBars({ phase, title }: { phase: Phase; title: string }) {
  const max = Math.max(phase.computeMs, phase.memMs, phase.commMs, 1e-9)
  const rows: [string, number, string][] = [['Compute', phase.computeMs, 'pb-c'], ['Memory traffic', phase.memMs, 'pb-m'], ['Interconnect', phase.commMs, 'pb-i']]
  return (
    <div className="phase">
      <div className="phase-head"><span className="utility">{title}</span><span className={`bound b-${phase.bound}`}>{phase.bound}-bound</span></div>
      {rows.map(([l, v, c]) => (
        <div key={l} className="pb-row">
          <span>{l}</span>
          <div className="pb-track"><i className={c} style={{ width: `${(v / max) * 100}%` }} /></div>
          <span className="mono">{v >= 100 ? fmtNum(v) : fmtNum(v, 2)} ms</span>
        </div>
      ))}
    </div>
  )
}
