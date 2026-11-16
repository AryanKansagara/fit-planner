// Real config.json files from the Hugging Face Hub (sizing keys kept verbatim, bookkeeping keys trimmed),
// checked against the parameter totals the Hub reports from each repo's safetensors headers (retrieved 2026-11-13).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { attnPairs, kvBytesForSeq, kvBytesPerToken, paramCounts, parseConfig, seqLenForBytes, specToConfig } from './model.ts'
import { DEFAULT_ASSUMPTIONS as A, GB, memoryFit } from './plan.ts'
import { DEVICES } from './hardware.ts'
import type { ModelSpec, Workload } from './types.ts'

interface Fixture { repo: string; safetensorsTotal: number | null; config: Record<string, unknown> }
const FIX: Fixture[] = JSON.parse(readFileSync(new URL('./fixtures/hf-configs.json', import.meta.url), 'utf8'))
const load = (repo: string) => {
  const f = FIX.find(x => x.repo === repo)!
  const r = parseConfig(JSON.stringify(f.config, null, 2))
  assert.ok(r.spec, `${repo}: ${r.issues.map(i => i.message).join('; ')}`)
  return { spec: r.spec, issues: r.issues, total: f.safetensorsTotal }
}
const near = (actual: number, expected: number, relTol: number, msg?: string) =>
  assert.ok(Math.abs(actual - expected) <= Math.abs(expected) * relTol, `${msg ?? ''} expected ~${expected}, got ${actual}`)
const wl = (p: Partial<Workload> = {}): Workload => ({ weightDtype: 'bf16', kvDtype: 'bf16', context: 8192, batch: 1, promptLen: 1000, outputLen: 200, ...p })

test('every real config parses and matches the Hub safetensors parameter total within 0.1%', () => {
  for (const f of FIX) {
    const { spec, total } = load(f.repo)
    if (total) near(paramCounts(spec).total, total, 0.001, f.repo)
  }
})

test('Falcon multi_query means one shared KV head; BLOOM reads n_embed', () => {
  const falcon = load('tiiuae/falcon-7b').spec
  assert.equal(falcon.kvHeads, 1)
  assert.equal(kvBytesPerToken(falcon, 'bf16'), 2 * 32 * 1 * 64 * 2)
  assert.equal(load('bigscience/bloom-560m').spec.hidden, 1024)
})

test('Gemma, BLOOM and OPT tie the output head when tie_word_embeddings is absent', () => {
  for (const r of ['google/gemma-2-9b', 'bigscience/bloom-560m', 'facebook/opt-1.3b']) assert.equal(load(r).spec.tied, true, r)
  near(paramCounts(load('facebook/opt-1.3b').spec).total, 1.31e9, 0.01, 'OPT 1.3B (learned positions not counted)')
})

test('MoE totals and active counts match model cards: Kimi K2, gpt-oss-20b, Qwen1.5-MoE', () => {
  const kimi = paramCounts(load('moonshotai/Kimi-K2-Instruct').spec)
  near(kimi.active, 32.9e9, 0.01, 'Kimi K2 active (32B per the card, plus embeddings)')
  const oss = load('openai/gpt-oss-20b')
  const p = paramCounts(oss.spec)
  near(p.total, 20.91e9, 0.002, 'gpt-oss-20b total')
  // The model card's 3.61B active excludes the embedding table.
  near(p.active - p.embedding, 3.61e9, 0.01, 'gpt-oss-20b active')
  assert.equal(oss.spec.moe!.active, 4)
  near(paramCounts(load('Qwen/Qwen1.5-MoE-A2.7B').spec).active, 2.69e9, 0.01, 'Qwen1.5-MoE-A2.7B')
})

test('Llama 4 Scout: shared expert per layer and dense width from intermediate_size_mlp', () => {
  const { spec } = load('meta-llama/Llama-4-Scout-17B-16E-Instruct')
  assert.equal(spec.moe!.shared, 1)
  assert.equal(spec.moe!.expertInter, 8192)
  assert.equal(spec.intermediate, 16384)
  const p = paramCounts(spec)
  near(p.active, 17.2e9, 0.01, 'Scout active (17B per the card)')
  // 109B on the card includes the vision encoder, which is not counted.
  assert.ok(p.total > 107e9 && p.total < 109e9, `Scout total ${p.total}`)
})

test('interleaved MoE: decoder_sparse_step 2 makes every other layer dense', () => {
  const base = { model_type: 'qwen2_moe', hidden_size: 2048, num_hidden_layers: 24, num_attention_heads: 16, num_key_value_heads: 16, intermediate_size: 5632, moe_intermediate_size: 1408, num_experts: 60, num_experts_per_tok: 4, vocab_size: 151936, max_position_embeddings: 8192 }
  const every = load('Qwen/Qwen1.5-MoE-A2.7B').spec
  const half = parseConfig(JSON.stringify({ ...base, decoder_sparse_step: 2 })).spec!
  assert.equal(half.moe!.denseLayers, 12)
  assert.equal(paramCounts(half).moeLayers, 12)
  assert.equal(every.moe!.denseLayers, 0)
  const only = parseConfig(JSON.stringify({ ...base, mlp_only_layers: [0, 23] })).spec!
  assert.equal(only.moe!.denseLayers, 2)
})

test('sliding-window layers are read from layer_types, gemma2 alternation, or all layers', () => {
  const oss = load('openai/gpt-oss-20b').spec
  assert.deepEqual(oss.window, { size: 128, layers: 12 })
  assert.deepEqual(load('google/gemma-2-9b').spec.window, { size: 4096, layers: 21 })
  assert.deepEqual(load('mistralai/Mistral-7B-v0.1').spec.window, { size: 4096, layers: 32 })
  // Qwen configs declare a window but switch it off.
  const q = parseConfig(JSON.stringify({ model_type: 'qwen2', hidden_size: 1024, num_hidden_layers: 4, num_attention_heads: 8, num_key_value_heads: 8, intermediate_size: 2048, vocab_size: 1000, max_position_embeddings: 32768, sliding_window: 4096, use_sliding_window: false })).spec!
  assert.equal(q.window, undefined)
  assert.deepEqual(parseConfig(specToConfig(oss)).spec!.window, oss.window)
})

test('sliding-window KV: full layers grow with context, windowed layers stop at the window', () => {
  const oss = load('openai/gpt-oss-20b').spec
  const per = kvBytesPerToken(oss, 'bf16') / oss.layers
  assert.equal(kvBytesForSeq(oss, 'bf16', 100), per * 24 * 100)
  assert.equal(kvBytesForSeq(oss, 'bf16', 131072), per * (12 * 131072 + 12 * 128))
  for (const n of [50, 128, 129, 5000, 131072]) assert.equal(seqLenForBytes(oss, 'bf16', kvBytesForSeq(oss, 'bf16', n)), n)
  const mistral = load('mistralai/Mistral-7B-v0.1').spec
  assert.equal(kvBytesForSeq(mistral, 'bf16', 32768), kvBytesForSeq(mistral, 'bf16', 4096))
  assert.equal(seqLenForBytes(mistral, 'bf16', kvBytesForSeq(mistral, 'bf16', 4096) + 1), Infinity)
})

test('memory fit uses windowed KV: gpt-oss-20b at 128k context needs about half the full-attention cache', () => {
  const oss = load('openai/gpt-oss-20b').spec
  const l = { device: DEVICES.find(d => d.id === 'nv-h100')!, tp: 1, pp: 1 }
  const f = memoryFit(oss, wl({ context: 131072, batch: 4 }), l, A)
  near(f.kvGB, 4 * kvBytesForSeq(oss, 'bf16', 131072) / GB, 1e-9)
  near(f.kvGB, 4 * 131072 * kvBytesPerToken(oss, 'bf16') / 2 / GB, 0.01)
  const free = (f.capGB - f.reserveGB - f.weightsGB) * GB
  assert.equal(f.maxBatch, Math.floor(free / kvBytesForSeq(oss, 'bf16', 131072)))
})

test('attention work without a window is layers x n^2 / 2; a window caps it per layer', () => {
  const dense: ModelSpec = { ...load('mistralai/Mistral-7B-v0.1').spec, window: undefined }
  assert.equal(attnPairs(dense, 1000), 32 * 1000 * 1000 / 2)
  const w = load('mistralai/Mistral-7B-v0.1').spec
  assert.equal(attnPairs(w, 1000), attnPairs(dense, 1000))
  assert.equal(attnPairs(w, 8192), 32 * (8192 * 4096 - 4096 * 4096 / 2))
})
