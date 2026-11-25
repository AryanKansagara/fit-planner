import { test } from 'node:test'
import assert from 'node:assert/strict'
import { MODEL_SAMPLES, WORKLOAD_SAMPLES } from './engine/samples.ts'
import { importPlanJson, newProject, planDataErrors, planJson, projectPlan, snapshotOf } from './projectData.ts'

const custom = { id: 'c1', name: 'Card', vendor: '', kind: 'custom' as const, memGB: 32, bwGBs: 450, tflops: { fp16: 100 }, linkGBs: 25, linkLatencyUs: 0, linkKind: '', priceUsd: 0, watts: 300, maxCount: 8 }
const full = () => {
  const p = { ...newProject('Mine'), devices: [custom] }
  return { ...p, snapshots: [snapshotOf(p, 'A')!] }
}
/** The exported file of a full plan with `edit` applied to its parsed JSON. */
const file = (edit: (o: any) => void) => { const o = JSON.parse(planJson(full())); edit(o); return JSON.stringify(o) }

test('every sample model and workload survives export and import', () => {
  for (const m of MODEL_SAMPLES) for (const w of WORKLOAD_SAMPLES) {
    const p = newProject(m.name, m.id, w.id)
    const r = importPlanJson(planJson(p))
    assert.deepEqual(r.errors, [], `${m.id} ${w.id}`)
    assert.equal(projectPlan(r.project!)!.mem.usedGB, projectPlan(p)!.mem.usedGB)
  }
  const r = importPlanJson(planJson(full()))
  assert.deepEqual(r.errors, [])
  assert.deepEqual(r.project!.devices, [custom])
  assert.equal(r.project!.snapshots.length, 1)
})

test('an unknown weight or KV dtype is refused, naming the field', () => {
  for (const v of ['zz', 'BF16', '', 'toString', '__proto__', 'constructor', null, 4, ['bf16'], {}, undefined]) {
    for (const field of ['weightDtype', 'kvDtype']) {
      const r = importPlanJson(file(o => { o.plan.workload[field] = v }))
      assert.equal(r.project, null, `${field} ${JSON.stringify(v)}`)
      assert.match(r.errors.join(), new RegExp(`plan\\.workload\\.${field} must be one of`))
    }
  }
  // Weight dtypes that are not KV dtypes.
  assert.match(importPlanJson(file(o => { o.plan.workload.kvDtype = 'q4' })).errors.join(), /kvDtype must be one of bf16, fp8, bfp8/)
})

test('other data the pages cannot render is refused instead of stored', () => {
  const bad: [string, (o: any) => void][] = [
    ['null device', o => { o.plan.devices = [null] }],
    ['device without tflops', o => { delete o.plan.devices[0].tflops }],
    ['device with text memory', o => { o.plan.devices[0].memGB = 'lots' }],
    ['null snapshot', o => { o.snapshots = [null] }],
    ['snapshot without a result', o => { delete o.snapshots[0].result }],
    ['text batch', o => { o.plan.workload.batch = 'x' }],
    ['missing prompt length', o => { delete o.plan.workload.promptLen }],
    ['null target', o => { o.plan.targets.minTokS = null }],
    ['text assumption', o => { o.plan.assumptions.reserveGB = '1 GB' }],
    ['repo object', o => { o.plan.model.repo = {} }],
  ]
  for (const [name, edit] of bad) {
    const r = importPlanJson(file(edit))
    assert.equal(r.project, null, name)
    assert.ok(r.errors.length > 0, name)
  }
  assert.equal(importPlanJson(file(o => { o.plan.name = 7 })).project!.name, '7')
})

test('a stored plan with an unknown dtype has no result and a reason', () => {
  const p = full()
  const bad = { ...p, workload: { ...p.workload, weightDtype: 'zz' as never } }
  assert.equal(projectPlan(bad), null)
  assert.equal(snapshotOf(bad, 'x'), null)
  assert.match(planDataErrors(bad).join(), /weightDtype must be one of/)
  assert.deepEqual(planDataErrors(p), [])
})
