import { expect, type Locator, type Page } from '@playwright/test'

/** A labelled control on the plan page: <label class="field"><span>Label</span> ...</label>. */
export const field = (page: Page, label: string): Locator =>
  page.locator('label.field').filter({ has: page.locator('span', { hasText: new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`) }) })

export const selectOf = (page: Page, label: string) => field(page, label).locator('select')
export const inputOf = (page: Page, label: string) => field(page, label).locator('input')

/** Opens the workspace and starts a plan from one of the public architectures. */
export async function openSample(page: Page, modelName: string) {
  await page.goto('/app')
  await page.locator('.sample-card', { hasText: modelName }).first().click()
  await expect(page).toHaveURL(/\/app\/p\/[\w-]+$/)
  await expect(page.locator('.fit-banner')).toBeVisible()
}

export const banner = (page: Page) => page.locator('.fit-banner')
/** The big number of a KPI tile, found by its label. */
export const kpi = (page: Page, label: RegExp | string) => page.locator('.kpi').filter({ hasText: label }).locator('.kpi-value')
export const projectId = (page: Page) => new URL(page.url()).pathname.split('/')[3]

/** Starts a blank plan named `name` and lands on its Model tab (the Hugging Face source is preselected). */
export async function newBlankPlan(page: Page, name = 'E2E plan') {
  await page.goto('/app?new=1')
  await page.getByLabel('Plan name').fill(name)
  await page.getByRole('button', { name: 'Create and import' }).click()
  await expect(page).toHaveURL(/\/app\/p\/[\w-]+\/model/)
}

/** A complete Llama-style config.json (8.03B parameters). Individual tests copy and break it. */
export const LLAMA_8B = {
  model_type: 'llama',
  architectures: ['LlamaForCausalLM'],
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  head_dim: 128,
  intermediate_size: 14336,
  vocab_size: 128256,
  max_position_embeddings: 131072,
  tie_word_embeddings: false,
  torch_dtype: 'bfloat16',
}

export const json = (o: unknown) => JSON.stringify(o, null, 2)
