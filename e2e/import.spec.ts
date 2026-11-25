import type { Page, Route } from '@playwright/test'
import { test, expect } from './fixtures.ts'
import { LLAMA_8B, json, newBlankPlan, openSample } from './helpers.ts'

/*
 * Llama-style config in LLAMA_8B: hidden 4096, 32 layers, 32 heads, 8 KV heads, MLP 14336, vocab 128256, untied.
 * Parameters (hand-worked): 32 * 218,112,000 + 2 * 525,336,576 + 4096 = 8,030,261,248 -> "8.0B".
 * KV per token (BF16) = 2 * 32 * 8 * 128 * 2 = 131,072 B -> "128 KB".
 */
const NAMED = { ...LLAMA_8B, _name_or_path: 'acme/Test-8B' }

const HF_URL = 'https://huggingface.co/acme/Test-8B/resolve/main/config.json'

/** Stubs every Hugging Face request and records the URLs the page asked for. */
async function stubHub(page: Page, respond: (route: Route) => Promise<void> | void) {
  const requested: string[] = []
  await page.route('https://huggingface.co/**', route => { requested.push(route.request().url()); return respond(route) })
  return requested
}

async function useModel(page: Page) {
  await page.getByRole('button', { name: 'Use this model' }).click()
  await expect(page.locator('.toasts')).toContainText('Now planning')
}

async function expectPlanning(page: Page, name: string) {
  await page.getByRole('link', { name: 'Plan', exact: true }).click()
  await expect(page.locator('.plan-inputs')).toContainText(name)
  const facts = page.locator('.facts')
  await expect(facts).toContainText('8.0B')
  await expect(facts).toContainText('32 / 4,096')
  await expect(facts).toContainText('128 KB')
}

test.describe('import a config.json', () => {
  test('paste: validated as you type, applied on request', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.getByLabel('config.json contents').fill(json(NAMED))

    const validation = page.locator('.validation')
    await expect(validation).toContainText('Validation (draft)')
    await expect(validation).toContainText('0 errors')
    await expect(validation).toContainText('Field mapping (')
    await expect(page.locator('.sticky-apply')).toContainText('Ready: acme/Test-8B, 8.0B parameters.')
    // The draft is not the plan yet.
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    await expect(page.locator('.plan-inputs')).toContainText('Llama 3.1 8B')
    await page.getByRole('link', { name: 'Model', exact: true }).click()

    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.getByLabel('config.json contents').fill(json(NAMED))
    await useModel(page)
    await expectPlanning(page, 'acme/Test-8B')
  })

  test('file: a .json file is read and validated like pasted text', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.locator('input[type=file]').setInputFiles({ name: 'config.json', mimeType: 'application/json', buffer: Buffer.from(json(NAMED)) })
    await expect(page.getByLabel('config.json contents')).toHaveValue(json(NAMED))
    await expect(page.locator('.sticky-apply')).toContainText('Ready: acme/Test-8B')
    await useModel(page)
    await expectPlanning(page, 'acme/Test-8B')
  })

  test('file: a file that is not JSON gives an error and nothing to apply', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.locator('input[type=file]').setInputFiles({ name: 'config.json', mimeType: 'application/json', buffer: Buffer.from('<!doctype html><title>Sign in</title>') })
    await expect(page.locator('.validation')).toContainText('Not valid JSON')
    await expect(page.getByRole('button', { name: 'Use this model' })).toBeDisabled()
  })

  test('Hugging Face by repo id: stubbed fetch, nothing leaves the machine', async ({ page }) => {
    const requested = await stubHub(page, route => route.fulfill({ status: 200, contentType: 'application/json', body: json(LLAMA_8B) }))
    await newBlankPlan(page)
    await page.getByLabel('Repo id or URL').fill('acme/Test-8B')
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await expect(page.locator('.sticky-apply')).toContainText('Ready: acme/Test-8B, 8.0B parameters.')
    expect(requested).toEqual([HF_URL])
    await useModel(page)
    await page.getByRole('link', { name: 'Model', exact: true }).click()
    await expect(page.getByText('Fetched from acme/Test-8B')).toBeVisible()
    await expectPlanning(page, 'acme/Test-8B')
  })

  test('Hugging Face by URL: the repo id is taken from the page URL', async ({ page }) => {
    const requested = await stubHub(page, route => route.fulfill({ status: 200, body: json(LLAMA_8B) }))
    await newBlankPlan(page)
    for (const input of [
      'https://huggingface.co/acme/Test-8B',
      'https://huggingface.co/acme/Test-8B/blob/main/config.json',
      'https://huggingface.co/acme/Test-8B/resolve/main/config.json',
      '  acme/Test-8B  ',
    ]) {
      await page.getByLabel('Repo id or URL').fill(input)
      await page.getByRole('button', { name: 'Fetch config.json' }).click()
      await expect.poll(() => requested.length).toBeGreaterThan(0)
      expect(requested.at(-1)).toBe(HF_URL)
      requested.length = 0
    }
  })

  test('Hugging Face: input that is not a repo is refused without a request', async ({ page }) => {
    const requested = await stubHub(page, route => route.fulfill({ status: 200, body: json(LLAMA_8B) }))
    await newBlankPlan(page)
    for (const input of ['llama', 'a/b/c', 'https://example.com/acme/Test-8B', 'acme/Test 8B', 'acme/']) {
      await page.getByLabel('Repo id or URL').fill(input)
      await page.getByRole('button', { name: 'Fetch config.json' }).click()
      await expect(page.getByRole('alert')).toContainText('Enter a repo id like Qwen/Qwen3-14B or a huggingface.co model URL.')
    }
    expect(requested).toEqual([])
    await expect(page.locator('.sticky-apply')).toHaveCount(0)
  })

  const FAILURES: [string, (route: Route) => Promise<void>, RegExp][] = [
    ['404', route => route.fulfill({ status: 404, body: 'Entry not found' }), /No config\.json found for acme\/Test-8B\. Check the spelling; GGUF-only repos have no config\.json/],
    ['401', route => route.fulfill({ status: 401, body: 'Unauthorized' }), /acme\/Test-8B is gated or private\. Accept its license/],
    ['403', route => route.fulfill({ status: 403, body: 'Forbidden' }), /acme\/Test-8B is gated or private/],
    ['429', route => route.fulfill({ status: 429, body: 'Too Many Requests' }), /^Hugging Face returned 429\.$/],
    ['500', route => route.fulfill({ status: 500, body: 'Internal Server Error' }), /^Hugging Face returned 500\.$/],
    ['network failure', route => route.abort('connectionrefused'), /Could not reach huggingface\.co\. Check your connection, or paste the config instead\./],
  ]
  for (const [name, respond, message] of FAILURES) {
    test(`Hugging Face failure (${name}) says what went wrong and offers no model`, async ({ page }) => {
      await stubHub(page, respond)
      await newBlankPlan(page)
      await page.getByLabel('Repo id or URL').fill('acme/Test-8B')
      await page.getByRole('button', { name: 'Fetch config.json' }).click()
      await expect(page.getByRole('alert')).toHaveText(message)
      await expect(page.locator('.sticky-apply')).toHaveCount(0)
      await expect(page.getByRole('button', { name: 'Use this model' })).toHaveCount(0)
      await expect(page.getByRole('button', { name: 'Fetch config.json' })).toBeEnabled()
    })
  }

  test('Hugging Face answering 200 with a login page or an empty body is an error, not a model', async ({ page }) => {
    let body = '<!doctype html><html><title>Log in</title></html>'
    await stubHub(page, route => route.fulfill({ status: 200, contentType: 'text/html', body }))
    await newBlankPlan(page)
    await page.getByLabel('Repo id or URL').fill('acme/Test-8B')
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await expect(page.locator('.validation')).toContainText('Not valid JSON')
    await expect(page.getByRole('button', { name: 'Use this model' })).toBeDisabled()
    body = ''
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await expect(page.locator('.validation')).toContainText('1 errors')
    await expect(page.getByRole('button', { name: 'Use this model' })).toBeDisabled()
  })

  // Issue #5: the draft from an earlier fetch survives a failed one.
  test('a failed fetch discards the earlier draft', async ({ page }) => {
    let status = 200
    await stubHub(page, route => status === 200 ? route.fulfill({ status, body: json(LLAMA_8B) }) : route.fulfill({ status, body: 'nope' }))
    await newBlankPlan(page)
    await page.getByLabel('Repo id or URL').fill('acme/Test-8B')
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await expect(page.locator('.sticky-apply')).toContainText('Ready: acme/Test-8B')
    status = 404
    await page.getByLabel('Repo id or URL').fill('acme/Other-70B')
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await expect(page.getByRole('alert')).toContainText('No config.json found for acme/Other-70B')
    await expect(page.getByRole('button', { name: 'Use this model' })).toHaveCount(0)
  })

  // Issue #5, second half.
  test('emptying the paste box discards the draft', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    const box = page.getByLabel('config.json contents')
    await box.fill(json(NAMED))
    await expect(page.locator('.sticky-apply')).toContainText('Ready')
    await box.fill('')
    await expect(page.getByRole('button', { name: 'Use this model' })).toHaveCount(0)
    // Whitespace counts as empty, and typing again brings a draft back.
    await box.fill(json(NAMED))
    await expect(page.locator('.sticky-apply')).toContainText('Ready')
    await box.fill('  \n ')
    await expect(page.getByRole('button', { name: 'Use this model' })).toHaveCount(0)
  })

  test('an input that is not a repo id also discards the earlier draft', async ({ page }) => {
    await stubHub(page, route => route.fulfill({ status: 200, body: json(LLAMA_8B) }))
    await newBlankPlan(page)
    await page.getByLabel('Repo id or URL').fill('acme/Test-8B')
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await expect(page.locator('.sticky-apply')).toContainText('Ready: acme/Test-8B')
    await page.getByLabel('Repo id or URL').fill('not a repo')
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await expect(page.getByRole('alert')).toContainText('Enter a repo id')
    await expect(page.getByRole('button', { name: 'Use this model' })).toHaveCount(0)
  })

  test('a slow fetch does not replace a config pasted in the meantime', async ({ page }) => {
    let release = () => {}
    const held = new Promise<void>(r => { release = r })
    await stubHub(page, async route => { await held; await route.fulfill({ status: 200, body: json(LLAMA_8B) }) })
    await newBlankPlan(page)
    await page.getByLabel('Repo id or URL').fill('acme/Slow-8B')
    await page.getByRole('button', { name: 'Fetch config.json' }).click()
    await page.getByRole('button', { name: 'Paste config.json' }).click()
    await page.getByLabel('config.json contents').fill(json(NAMED))
    const ready = await page.locator('.sticky-apply').innerText()
    const done = page.waitForResponse(/huggingface\.co/)
    release()
    await done
    await page.waitForTimeout(300)
    expect(await page.locator('.sticky-apply').innerText()).toBe(ready)
  })

  test('manual entry writes a config.json that goes through the same validation', async ({ page }) => {
    await newBlankPlan(page)
    await page.getByRole('button', { name: 'Manual' }).click()
    // The form defaults are the Llama 3.1 8B shape: 8.0B parameters.
    await page.getByLabel('Name', { exact: true }).fill('Hand typed 8B')
    await expect(page.locator('.sticky-apply')).toContainText('Ready: Hand typed 8B, 8.0B parameters.')
    await expect(page.locator('.validation')).toContainText('0 errors')
    await useModel(page)
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    await expect(page.locator('.plan-inputs')).toContainText('Hand typed 8B')
  })

  test('public samples can be swapped in and the mesh is resized to fit', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    await page.getByRole('link', { name: 'Change' }).click()
    await page.getByRole('button', { name: 'Public samples' }).click()
    await page.locator('.pick', { hasText: 'Llama 3.3 70B' }).click()
    await expect(page.locator('.sticky-apply')).toContainText('Ready: Llama 3.3 70B, 70.6B parameters.')
    await page.getByRole('button', { name: 'Use this model' }).click()
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    // BFP8 weights of the 70B (74.96 GB) cannot sit on one 32 GB card: the mesh grew, and it fits.
    await expect(page.locator('.fit-banner')).toContainText('Fits')
    await expect(page.locator('.fit-banner')).not.toContainText(' 1x Blackhole')
  })
})
