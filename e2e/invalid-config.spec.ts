import { test, expect } from './fixtures.ts'
import { LLAMA_8B, json, newBlankPlan, openSample } from './helpers.ts'

const without = (...keys: string[]) => Object.fromEntries(Object.entries(LLAMA_8B).filter(([k]) => !keys.includes(k)))

/** [name, text typed or pasted, message the validation must show]. All must be rejected. */
const BAD: [string, string, RegExp][] = [
  ['an empty object', '{}', /Hidden size: none of hidden_size, n_embd, n_embed, d_model, dim found\./],
  ['an array', '[]', /Expected a JSON object/],
  ['a bare number', '42', /Expected a JSON object/],
  ['JSON null', 'null', /Expected a JSON object/],
  ['a string', '"llama"', /Expected a JSON object/],
  ['an HTML error page', '<html><body>404 Not Found</body></html>', /Not valid JSON/],
  ['plain prose', 'please size my model', /Not valid JSON/],
  ['a truncated file (cut mid-string)', json(LLAMA_8B).slice(0, 120), /Not valid JSON/],
  ['a truncated file (cut after a comma)', json(LLAMA_8B).slice(0, json(LLAMA_8B).indexOf('"head_dim"')), /Not valid JSON/],
  ['a trailing comma', json(LLAMA_8B).replace(/\n}$/, ',\n}'), /Not valid JSON/],
  ['single quotes', json(LLAMA_8B).replaceAll('"', "'"), /Not valid JSON/],
  ['no hidden size', json(without('hidden_size')), /Hidden size: none of hidden_size/],
  ['no layer count', json(without('num_hidden_layers')), /Layers: none of num_hidden_layers/],
  ['no head count', json(without('num_attention_heads')), /Attention heads: none of num_attention_heads/],
  ['no vocabulary', json(without('vocab_size')), /Vocabulary: none of vocab_size, padded_vocab_size found\./],
  ['zero layers', json({ ...LLAMA_8B, num_hidden_layers: 0 }), /num_hidden_layers must be a positive integer \(got 0\)\./],
  ['negative hidden size', json({ ...LLAMA_8B, hidden_size: -4096 }), /hidden_size must be a positive integer \(got -4096\)\./],
  ['fractional layers', json({ ...LLAMA_8B, num_hidden_layers: 1.5 }), /num_hidden_layers must be a positive integer \(got 1\.5\)\./],
  ['layers given as a string', json({ ...LLAMA_8B, num_hidden_layers: '32' }), /num_hidden_layers should be a number \(got "32"\)\./],
  ['layers null', json({ ...LLAMA_8B, num_hidden_layers: null }), /Layers: none of num_hidden_layers/],
  ['heads not a multiple of KV heads', json({ ...LLAMA_8B, num_attention_heads: 30 }), /Attention heads \(30\) must be a multiple of KV heads \(8\)\./],
  ['more KV heads than heads', json({ ...LLAMA_8B, num_key_value_heads: 64 }), /must be a multiple of KV heads \(64\)/],
  // Issue #13.
  ['a hidden size smaller than the head count', json({ ...without('head_dim'), hidden_size: 10 }), /Head dim: head_dim missing and cannot be derived \(hidden size \/ heads\) because hidden size 10 is smaller than 32 heads\./],
  ['more active experts than experts', json({ ...LLAMA_8B, num_local_experts: 4, num_experts_per_tok: 8, moe_intermediate_size: 1024 }), /Active experts \(8\) exceeds experts \(4\)\./],
]

test.describe('a wrong, empty or truncated config.json is rejected with a reason and no result', () => {
  for (const [name, text, message] of BAD) {
    test(`pasted: ${name}`, async ({ page }) => {
      await openSample(page, 'Llama 3.1 8B')
      const before = await page.locator('.fit-banner').innerText()
      await page.getByRole('link', { name: 'Change' }).click()
      await page.getByRole('button', { name: 'Paste config.json' }).click()
      await page.getByLabel('config.json contents').fill(text)

      const validation = page.locator('.validation')
      await expect(validation).toContainText(message)
      await expect(validation.locator('.v-sum')).not.toContainText('0 errors')
      await expect(page.locator('.sticky-apply')).toContainText('Fix the errors above before using this config.')
      await expect(page.getByRole('button', { name: 'Use this model' })).toBeDisabled()
      // No model summary or parameter count is offered for it.
      await expect(page.locator('.kpis.three-up')).toHaveCount(0)
      await expect(page.locator('.sticky-apply')).not.toContainText('parameters')

      // Discarding leaves the plan exactly as it was.
      await page.getByRole('button', { name: 'Discard' }).click()
      await page.getByRole('link', { name: 'Plan', exact: true }).click()
      await expect(page.locator('.plan-inputs')).toContainText('Llama 3.1 8B')
      expect(await page.locator('.fit-banner').innerText()).toBe(before)
    })
  }

  test('a blank or whitespace-only box offers nothing to apply', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    const box = page.getByLabel('config.json contents')
    await box.fill('   \n\t  ')
    await expect(page.locator('.sticky-apply')).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Use this model' })).toHaveCount(0)
  })

  test('errors point at the line of the offending key', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    const text = json({ ...LLAMA_8B, num_hidden_layers: 0 })
    await page.getByLabel('config.json contents').fill(text)
    const line = text.split('\n').findIndex(l => l.includes('"num_hidden_layers"')) + 1
    await expect(page.locator('.iss.error')).toContainText(`line ${line}`)
    await expect(page.locator('pre.numbered .mark')).toContainText('"num_hidden_layers": 0')
  })

  test('a truncated file picked from disk is rejected the same way', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    const half = json(LLAMA_8B).slice(0, 200)
    await page.locator('input[type=file]').setInputFiles({ name: 'config.json', mimeType: 'application/json', buffer: Buffer.from(half) })
    await expect(page.locator('.validation')).toContainText('Not valid JSON')
    await expect(page.getByRole('button', { name: 'Use this model' })).toBeDisabled()
  })

  test('a plan whose saved config no longer parses says so instead of showing numbers', async ({ page }) => {
    await page.goto('/app')
    await page.getByRole('button', { name: 'Import plan JSON' }).click()
    const bad = { format: 'fit-planner/plan@1', plan: { name: 'Broken config', model: { kind: 'paste', config: '{"hidden_size": 4096', at: 0 }, workload: { weightDtype: 'bf16', kvDtype: 'bf16', context: 8192, batch: 1, promptLen: 1000, outputLen: 200 }, tp: 1, pp: 1 } }
    await page.locator('dialog textarea').fill(JSON.stringify(bad))
    await page.getByRole('button', { name: 'Import', exact: true }).click()
    await expect(page.locator('dialog .issues')).toContainText("The plan's model config does not parse: Not valid JSON")
    await expect(page).toHaveURL(/\/app$/)
    await page.getByRole('button', { name: 'Cancel' }).click()
    await expect(page.locator('.proj')).toHaveCount(0)
  })
})

test.describe('unknown fields are flagged, never silently treated as zero', () => {
  test('missing optional fields are assumed visibly, with the assumption in the numbers', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    // No KV heads, MLP width or context limit. Assumed: KV heads = heads (32), MLP = 4 x hidden (16384), context 4096.
    const sparse = { model_type: 'llama', hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, vocab_size: 128256 }
    await page.getByLabel('config.json contents').fill(json(sparse))
    const validation = page.locator('.validation')
    await expect(validation).toContainText('0 errors · 3 warnings')
    await expect(validation).toContainText('KV heads: num_key_value_heads missing, assumed 32 (no GQA field, so full multi-head attention).')
    await expect(validation).toContainText('MLP intermediate: intermediate_size missing, assumed 16384 (4 x hidden size).')
    await expect(validation).toContainText('Max context: max_position_embeddings missing, assumed 4096 (no position limit in the config).')
    await expect(validation).not.toContainText('Every sizing field was found')
    await validation.getByText('Field mapping').click()
    await expect(validation.locator('tr', { hasText: 'KV heads' })).toContainText('assumed: no GQA field, so full multi-head attention')
    await expect(validation.locator('tr', { hasText: 'MLP intermediate' })).toContainText('16384')

    // Hand-worked with those assumptions: layer = 4*4096^2 + 3*4096*16384 + 2*4096 = 268,443,648;
    // total = 32*268,443,648 + 2*525,336,576 + 4096 = 9,640,873,... -> 9.6B. KV/token = 2*32*32*128*2 B = 512 KB.
    await expect(page.locator('.sticky-apply')).toContainText('Ready: llama, 9.6B parameters.')
    await expect(page.locator('.kpi').filter({ hasText: 'Total parameters' })).toContainText('9.6B')
    await expect(page.locator('.kpi').filter({ hasText: 'KV per token' })).toContainText('512 KB')
  })

  test('a complete config reports that nothing was assumed', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.getByLabel('config.json contents').fill(json(LLAMA_8B))
    await expect(page.locator('.validation')).toContainText('Every sizing field was found. Nothing assumed.')
    await expect(page.locator('.validation')).toContainText('0 errors · 0 warnings')
  })

  test('without a hidden size or head count nothing is derived from them', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.getByLabel('config.json contents').fill(json(without('hidden_size', 'num_attention_heads', 'head_dim')))
    await expect(page.locator('.sticky-apply')).toContainText('Fix the errors above')
    await expect(page.getByRole('button', { name: 'Use this model' })).toBeDisabled()
    await expect(page.locator('.sticky-apply')).not.toContainText('parameters')
  })

  // Issue #6.
  test('derived assumptions are never reported as NaN', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.getByLabel('config.json contents').fill('{}')
    const validation = page.locator('.validation')
    await expect(validation).toContainText('Head dim: head_dim missing and cannot be derived (hidden size / heads) because hidden_size and num_attention_heads are missing or invalid.')
    await expect(validation).toContainText('KV heads: num_key_value_heads missing and cannot be derived')
    await expect(validation).not.toContainText('NaN')
    await validation.getByText('Field mapping').click()
    await expect(validation.locator('table')).not.toContainText('NaN')
    await expect(page.getByRole('button', { name: 'Use this model' })).toBeDisabled()
  })
})
