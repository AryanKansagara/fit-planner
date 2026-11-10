import { useState } from 'react'
import { DEVICES, devicesToCsv, HARDWARE_AS_OF, validateDevice } from './engine/hardware.ts'
import type { Device } from './engine/types.ts'
import { autoLayout, projectDevices } from './projectData.ts'
import { getProject, saveProject, uuid, type Project } from './store.ts'
import { download, fmtNum, fmtUsd, Modal, toast } from './ui.tsx'
import { NumField, type Update } from './Project.tsx'

const BLANK: Device = { id: '', name: 'My accelerator', vendor: 'Custom', kind: 'custom', memGB: 32, bwGBs: 500, tflops: { fp16: 100, fp8: 200 }, linkGBs: 25, linkLatencyUs: 10, linkKind: 'PCIe', priceUsd: 1000, watts: 300, maxCount: 8 }

export default function HardwareTab({ p, update }: { p: Project; update: Update }) {
  const devices = projectDevices(p)
  const [editing, setEditing] = useState<Device | null>(null)
  const edited = new Set(p.devices.map(d => d.id))
  const curated = new Set(DEVICES.map(d => d.id))

  const save = (d: Device) => {
    const id = d.id || `custom-${uuid().slice(0, 8)}`
    const next = [...p.devices.filter(x => x.id !== id), { ...d, id }]
    update({ devices: next })
    setEditing(null)
    toast(curated.has(id) ? `Saved your edits to ${d.name} for this plan.` : `Added ${d.name}.`, { tone: 'ok' })
  }
  const remove = (d: Device) => {
    const before = p.devices
    const patch: Partial<Project> = { devices: before.filter(x => x.id !== d.id) }
    if (!curated.has(d.id) && p.deviceId === d.id) Object.assign(patch, { deviceId: DEVICES[3].id })
    update(patch)
    toast(curated.has(d.id) ? `${d.name} reset to the published figures.` : `Removed ${d.name}.`, { undo: () => { const cur = getProject(p.id); if (cur) void saveProject({ ...cur, devices: before, deviceId: p.deviceId }) } })
  }
  const use = (d: Device) => {
    const next = autoLayout({ ...p, deviceId: d.id })
    update({ deviceId: d.id, tp: next.tp, pp: next.pp })
    toast(`Planning on ${next.tp * next.pp}x ${d.name}.`, { tone: 'ok' })
  }

  return (
    <div className="stack">
      <div className="row between wrap gap-sm">
        <p className="muted small grow">Published dense (no sparsity) figures as of {HARDWARE_AS_OF}. Edits and custom devices belong to this plan. Interconnect is per-device bandwidth usable for collectives.</p>
        <div className="row gap-sm">
          <button type="button" className="btn ghost sm" onClick={() => download(`hardware-${HARDWARE_AS_OF}.csv`, devicesToCsv(devices))}>Export CSV</button>
          <button type="button" className="btn primary sm" onClick={() => setEditing({ ...BLANK })}>Add custom device</button>
        </div>
      </div>
      <div className="card table-card scroll-x">
        <table className="vt hw">
          <caption className="sr-only">Hardware profiles</caption>
          <thead><tr>
            <th scope="col">Device</th><th scope="col" className="num">Memory</th><th scope="col" className="num">Bandwidth</th><th scope="col" className="num">FP16 / FP8 TFLOPS</th>
            <th scope="col" className="num">Link</th><th scope="col" className="num">Price</th><th scope="col" className="num">Power</th><th scope="col"><span className="sr-only">Actions</span></th>
          </tr></thead>
          <tbody>
            {devices.map(d => (
              <tr key={d.id} className={d.id === p.deviceId ? 'current' : ''}>
                <th scope="row">
                  <div className="row gap-sm wrap"><span className={`vendor v-${d.vendor.toLowerCase()}`}>{d.vendor}</span><strong>{d.name}</strong>{edited.has(d.id) && <span className="badge">{curated.has(d.id) ? 'edited' : 'custom'}</span>}</div>
                  {d.notes && <div className="muted small">{d.notes}</div>}
                </th>
                <td className="num">{d.memGB} GB</td>
                <td className="num">{fmtNum(d.bwGBs)} GB/s</td>
                <td className="num">{d.tflops.fp16} / {d.tflops.fp8 ?? '-'}</td>
                <td className="num" title={d.linkKind}>{d.linkGBs} GB/s<div className="muted small">{d.linkLatencyUs} us</div></td>
                <td className="num">{fmtUsd(d.priceUsd)}{d.rentPerHour ? <div className="muted small">{fmtUsd(d.rentPerHour, 2)}/h</div> : null}</td>
                <td className="num">{d.watts} W</td>
                <td className="nowrap">
                  {d.id === p.deviceId ? <span className="badge on">in use</span> : <button type="button" className="btn ghost sm" onClick={() => use(d)}>Use</button>}
                  <button type="button" className="btn ghost sm" onClick={() => setEditing(d)}>Edit</button>
                  {edited.has(d.id) && <button type="button" className="btn danger-ghost sm" onClick={() => remove(d)}>{curated.has(d.id) ? 'Reset' : 'Remove'}</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Modal open={!!editing} onClose={() => setEditing(null)} title={editing?.id ? `Edit ${editing.name}` : 'Add a custom device'} wide>
        {editing && <DeviceForm initial={editing} onSave={save} onCancel={() => setEditing(null)} />}
      </Modal>
    </div>
  )
}

function DeviceForm({ initial, onSave, onCancel }: { initial: Device; onSave(d: Device): void; onCancel(): void }) {
  const [d, setD] = useState<Device>(initial)
  const errs = validateDevice(d)
  const set = (patch: Partial<Device>) => setD(x => ({ ...x, ...patch }))
  return (
    <form className="stack" onSubmit={e => { e.preventDefault(); if (!errs.length) onSave(d) }}>
      <div className="row gap-sm wrap">
        <label className="field grow"><span>Name</span><input value={d.name} onChange={e => set({ name: e.target.value })} required /></label>
        <label className="field"><span>Vendor</span><input value={d.vendor} onChange={e => set({ vendor: e.target.value })} /></label>
      </div>
      <div className="three">
        <NumField label="Memory" suffix="GB" value={d.memGB} step={1} onChange={v => set({ memGB: v })} />
        <NumField label="Memory bandwidth" suffix="GB/s" value={d.bwGBs} step={8} onChange={v => set({ bwGBs: v })} />
        <NumField label="Dense FP16" suffix="TFLOPS" value={d.tflops.fp16} step={1} onChange={v => set({ tflops: { ...d.tflops, fp16: v } })} />
        <NumField label="Dense FP8 (0 = none)" suffix="TFLOPS" value={d.tflops.fp8 ?? 0} step={1} onChange={v => set({ tflops: { ...d.tflops, fp8: v || undefined } })} />
        <NumField label="Interconnect" suffix="GB/s" value={d.linkGBs} step={1} onChange={v => set({ linkGBs: v })} />
        <NumField label="Link latency" suffix="us" value={d.linkLatencyUs} step={0.5} onChange={v => set({ linkLatencyUs: v })} />
        <NumField label="Price" suffix="USD" value={d.priceUsd} step={50} onChange={v => set({ priceUsd: v })} />
        <NumField label="Rental (0 = none)" suffix="$/h" value={d.rentPerHour ?? 0} step={0.05} onChange={v => set({ rentPerHour: v || undefined })} />
        <NumField label="Board power" suffix="W" value={d.watts} step={10} onChange={v => set({ watts: v })} />
        <NumField label="Max cards in a mesh" value={d.maxCount} min={1} max={256} onChange={v => set({ maxCount: Math.round(v) })} />
      </div>
      <label className="field"><span>Interconnect description</span><input value={d.linkKind} onChange={e => set({ linkKind: e.target.value })} placeholder="PCIe 4.0 x16, 400G Ethernet, NVLink" /></label>
      {errs.length > 0 && <ul className="issues" role="alert">{errs.map(e => <li key={e} className="error small">{e}</li>)}</ul>}
      <div className="row end gap"><button type="button" className="btn ghost" onClick={onCancel}>Cancel</button><button className="btn primary" disabled={errs.length > 0}>Save device</button></div>
    </form>
  )
}
