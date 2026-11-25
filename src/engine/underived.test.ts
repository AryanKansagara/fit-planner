import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseConfig } from './model.ts'

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
