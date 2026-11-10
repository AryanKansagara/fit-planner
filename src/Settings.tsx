import { useState } from 'react'
import { DEVICES } from './engine/hardware.ts'
import type { Assumptions } from './engine/types.ts'
import { getDefaults, resetDefaults, setDefaults, type Defaults } from './store.ts'
import { setTheme, toast, useDark } from './ui.tsx'
import { useSync } from './sync.ts'
import { NumField } from './Project.tsx'

export default function Settings() {
  const [d, setD] = useState<Defaults>(getDefaults)
  const dark = useDark()
  const s = useSync()
  const save = (patch: Partial<Defaults>) => { const n = { ...d, ...patch }; setD(n); setDefaults(n) }
  const as = (patch: Partial<Assumptions>) => save({ assumptions: { ...d.assumptions, ...patch } })
  return (
    <section className="page narrow">
      <p className="eyebrow">Settings</p>
      <h1>Settings</h1>
      <div className="card pad stack">
        <h2 className="h5">Defaults for new plans</h2>
        <label className="field"><span>Starting device</span>
          <select value={d.deviceId} onChange={e => save({ deviceId: e.target.value })}>{DEVICES.map(x => <option key={x.id} value={x.id}>{x.vendor} {x.name}</option>)}</select>
        </label>
        <div className="three">
          <NumField label="Bandwidth utilisation" value={d.assumptions.bwUtil} min={0.05} max={1} step={0.05} onChange={v => as({ bwUtil: v })} />
          <NumField label="Compute utilisation (MFU)" value={d.assumptions.mfu} min={0.05} max={1} step={0.05} onChange={v => as({ mfu: v })} />
          <NumField label="Link utilisation" value={d.assumptions.linkUtil} min={0.05} max={1} step={0.05} onChange={v => as({ linkUtil: v })} />
          <NumField label="Reserve per device" suffix="GB" value={d.assumptions.reserveGB} min={0} max={64} step={0.5} onChange={v => as({ reserveGB: v })} />
          <NumField label="Amortise hardware over" suffix="years" value={d.assumptions.amortYears} min={0.25} max={10} step={0.5} onChange={v => as({ amortYears: v })} />
          <NumField label="Electricity" suffix="$/kWh" value={d.assumptions.kwhUsd} min={0} max={2} step={0.01} onChange={v => as({ kwhUsd: v })} />
        </div>
        <div className="seg" role="group" aria-label="Default cost basis">
          <button type="button" aria-pressed={d.assumptions.costMode === 'own'} onClick={() => as({ costMode: 'own' })}>Own (amortised)</button>
          <button type="button" aria-pressed={d.assumptions.costMode === 'rent'} onClick={() => as({ costMode: 'rent' })}>Rent hourly</button>
        </div>
        <p className="muted small">Saved as you type. Existing plans keep their own assumptions (Plan tab, Assumptions).</p>
      </div>
      <div className="card pad stack">
        <h2 className="h5">Appearance</h2>
        <div className="seg" role="group" aria-label="Theme">
          <button type="button" aria-pressed={!dark} onClick={() => setTheme('light')}>Light</button>
          <button type="button" aria-pressed={dark} onClick={() => setTheme('dark')}>Dark</button>
        </div>
      </div>
      <div className="card pad stack">
        <h2 className="h5">Sync and data</h2>
        <p className="muted small">{s.user ? `Signed in as ${s.user.email}. Use the account menu in the header to sync, export, or delete cloud data.` : 'Signed out. Everything is stored in this browser (IndexedDB). Sign in from the account menu to sync plans across devices.'}</p>
        <button type="button" className="btn ghost sm" onClick={() => { resetDefaults(); setD(getDefaults()); toast('Defaults restored.') }}>Restore default settings</button>
      </div>
      <div className="card pad stack">
        <h2 className="h5">Keyboard</h2>
        <ul className="kbd-list small">
          <li><kbd>1</kbd> to <kbd>5</kbd> switch plan tabs</li>
          <li><kbd>S</kbd> save the current configuration to history</li>
          <li><kbd>Ctrl</kbd>/<kbd>Cmd</kbd> + <kbd>Z</kbd> undo the last delete</li>
        </ul>
      </div>
    </section>
  )
}
