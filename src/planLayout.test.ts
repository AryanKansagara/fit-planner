import { test } from 'node:test'
import assert from 'node:assert/strict'
import { importPlanJson, newProject, planJson } from './projectData.ts'

const CFG = JSON.stringify({
  model_type: 'llama',
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  intermediate_size: 14336,
  vocab_size: 128256,
  max_position_embeddings: 131072,
})

const makePlan = (patch: Record<string, unknown>, wlPatch: Record<string, unknown> = {}) => JSON.stringify({
  format: 'fit-planner/plan@1',
  plan: {
    name: 'Test',
    model: { kind: 'paste', config: CFG, at: 0 },
    workload: { weightDtype: 'bf16', kvDtype: 'bf16', context: 8192, batch: 1, promptLen: 1000, outputLen: 200, ...wlPatch },
    tp: 1,
    pp: 1,
    ...patch,
  },
})

test('importPlanJson refuses illegal tp, pp, mesh size, device and workload', () => {
  for (const [label, json, re] of [
    ['non-divisible tp', makePlan({ tp: 3 }), /tp \(3\).*32 attention heads/],
    ['tp over the heads', makePlan({ tp: 64 }), /tp \(64\)/],
    ['fractional tp', makePlan({ tp: 1.5 }), /whole numbers/],
    ['fractional pp', makePlan({ pp: 1.5 }), /whole numbers/],
    ['infinite tp', makePlan({ tp: 1 }).replace('"tp":1', '"tp":1e999'), /whole numbers/],
    ['pp over layers', makePlan({ pp: 1000 }), /exceeds the model's 32 layers/],
    ['mesh over device max', makePlan({ deviceId: 'nv-4090', tp: 8, pp: 2 }), /16 devices .* is more than the 8 that GeForce RTX 4090/],
    ['unknown device', makePlan({ deviceId: 'toString' }), /"toString" is not a known or custom device/],
    ['context 0', makePlan({}, { context: 0 }), /context must be between 128 and 10,000,000 \(got 0\)/],
    ['context 127', makePlan({}, { context: 127 }), /context must be between 128/],
    ['context too large', makePlan({}, { context: 10_000_001 }), /context must be between 128/],
    ['batch 0', makePlan({}, { batch: 0 }), /batch must be a whole number between 1 and 4,096/],
    ['batch 1.5', makePlan({}, { batch: 1.5 }), /batch must be a whole number/],
    ['negative prompt', makePlan({}, { promptLen: -1 }), /promptLen/],
    ['output 0', makePlan({}, { outputLen: 0 }), /outputLen/],
  ] as const) {
    const r = importPlanJson(json)
    assert.equal(r.project, null, label)
    assert.ok(r.errors.some(e => re.test(e)), `${label}: got ${JSON.stringify(r.errors)}`)
  }
})

test('importPlanJson accepts what the Plan tab allows, including the limits', () => {
  for (const [label, json] of [
    ['replicated KV heads', makePlan({ tp: 16 })],
    ['full mesh', makePlan({ tp: 8, pp: 4 })],
    ['device max', makePlan({ deviceId: 'nv-4090', tp: 8 })],
    ['fractional context (the field accepts it)', makePlan({}, { context: 8192.5 })],
    ['lower limits', makePlan({}, { context: 128, batch: 1, promptLen: 1, outputLen: 1 })],
    ['upper limits', makePlan({}, { context: 10_000_000, batch: 4096, promptLen: 10_000_000, outputLen: 1_000_000 })],
  ] as const) {
    assert.deepEqual(importPlanJson(json).errors, [], label)
  }
  const sample = newProject('Valid 8B', 'llama-3.1-8b', 'team-chat')
  assert.deepEqual(importPlanJson(planJson(sample)).errors, [])
})
