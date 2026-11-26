import { test } from 'node:test'
import assert from 'node:assert/strict'
import { manualSpec, parseConfig, specToConfig } from './model.ts'

const BASE = { model_type: 'llama', hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, head_dim: 128, intermediate_size: 14336, vocab_size: 128256, max_position_embeddings: 131072 }
const MOE = { ...BASE, num_local_experts: 8, num_experts_per_tok: 2, decoder_sparse_step: 2 }
const without = (o: object, ...keys: string[]) => Object.fromEntries(Object.entries(o).filter(([k]) => !keys.includes(k)))

test('no message or mapping row reports NaN or Infinity, whichever fields are missing or bad', () => {
  const keys = [...Object.keys(BASE), 'num_experts_per_tok']
  for (const base of [BASE, MOE, without(BASE, 'num_key_value_heads', 'head_dim', 'intermediate_size')]) {
    for (const k of keys) for (const k2 of keys) {
      for (const bad of [undefined, null, 0, -4, 2.5, '32', [], {}]) {
        const text = JSON.stringify({ ...base, [k]: bad, [k2]: undefined })
        const r = parseConfig(text)
        const shown = JSON.stringify([r.issues.map(i => i.message), r.mapping.map(m => String(m.value))])
        assert.doesNotMatch(shown, /NaN|Infinity/, text)
        if (r.issues.some(i => /cannot be derived/.test(i.message))) assert.equal(r.spec, null, text)
      }
    }
  }
})

test('an underivable field names what it needs; a derivable one is still assumed', () => {
  const r = parseConfig('{}')
  assert.equal(r.spec, null)
  assert.deepEqual(r.issues.filter(i => /cannot be derived/.test(i.message)).map(i => i.message), [
    'KV heads: num_key_value_heads missing and cannot be derived (no GQA field, so full multi-head attention) because num_attention_heads is missing or invalid.',
    'Head dim: head_dim missing and cannot be derived (hidden size / heads) because hidden_size and num_attention_heads are missing or invalid.',
    'MLP intermediate: intermediate_size missing and cannot be derived (4 x hidden size) because hidden_size is missing or invalid.',
  ])
  const ok = parseConfig(JSON.stringify(without(BASE, 'num_key_value_heads', 'head_dim', 'intermediate_size')))
  assert.equal(ok.spec!.kvHeads, 32)
  assert.equal(ok.spec!.headDim, 128)
  assert.equal(ok.spec!.intermediate, 16384)
})

// Issue #13.
test('a hidden size smaller than the head count leaves no head dim and is an error', () => {
  const noDim = without(BASE, 'head_dim')
  for (const head_dim of [undefined, null]) for (const [hidden, heads] of [[10, 32], [15, 32], [16, 32], [31, 32], [1, 8]]) {
    const text = JSON.stringify({ ...noDim, hidden_size: hidden, num_attention_heads: heads, head_dim }, null, 2)
    const r = parseConfig(text)
    assert.equal(r.spec, null, text)
    assert.deepEqual(r.issues, [{ level: 'error', field: 'headDim', line: 3, message: `Head dim: head_dim missing and cannot be derived (hidden size / heads) because hidden size ${hidden} is smaller than ${heads} heads.` }], text)
    assert.equal(r.mapping.some(m => m.field === 'headDim'), false)
  }
  // The head dim is only derived when the config has none: an explicit one, or latent attention, does not need it.
  const tiny = { ...noDim, hidden_size: 10 }
  assert.equal(parseConfig(JSON.stringify({ ...tiny, head_dim: 64 })).spec!.headDim, 64)
  assert.equal(parseConfig(JSON.stringify({ ...tiny, kv_channels: 64 })).spec!.headDim, 64)
  assert.ok(parseConfig(JSON.stringify({ ...tiny, kv_lora_rank: 512, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 })).spec)
  // One unit per head is the smallest that works.
  assert.equal(parseConfig(JSON.stringify({ ...noDim, hidden_size: 32 })).spec!.headDim, 1)
  // An unusable input is reported as such, not compared with the other.
  assert.equal(parseConfig(JSON.stringify({ ...noDim, hidden_size: -4 })).issues.some(i => /smaller than/.test(i.message)), false)
})

test('the manual form cannot round a head dim up from nothing either', () => {
  for (const hidden of [10, 16, 31]) {
    const r = parseConfig(specToConfig(manualSpec({ name: 'Tiny', hidden, layers: 2, heads: 32, kvHeads: 8, intermediate: 64, vocab: 1000, maxContext: 1024 })))
    assert.equal(r.spec, null)
    assert.ok(r.issues.some(i => i.level === 'error' && /head_dim must be a positive integer \(got 0\)/.test(i.message)))
  }
})
