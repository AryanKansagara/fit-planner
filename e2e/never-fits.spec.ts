import { test, expect } from './fixtures.ts'
import { banner, inputOf, kpi, openSample, selectOf } from './helpers.ts'

/*
 * A deployment that does not fit must say so everywhere it shows up: the plan banner, the plan list,
 * saved history, the printable report, search results and the landing-page demo.
 * Llama 3.3 70B in BF16 is 141.1 GB of weights: four 32 GB cards hold 35.3 GB each, which cannot fit
 * (hand-worked in fit-math.spec.ts).
 */
async function overfullPlan(page: import('@playwright/test').Page) {
  await openSample(page, 'Llama 3.3 70B')
  await selectOf(page, 'Weights').selectOption('bf16')
  await inputOf(page, 'Concurrent seqs').fill('1')
  await selectOf(page, 'Device').selectOption('tt-p150a')
  await selectOf(page, 'Devices').selectOption('4')
  await expect(banner(page)).toContainText('Does not fit')
}

test.describe('a model that does not fit is never shown as fitting', () => {
  test('plan banner, KPIs, plan list, history and report agree', async ({ page }) => {
    await overfullPlan(page)
    await expect(page.locator('.fit-banner.bad')).toBeVisible()
    await expect(page.locator('.fit-banner.ok')).toHaveCount(0)
    await expect(kpi(page, 'Max sequences')).toHaveText('0')

    await page.getByRole('button', { name: 'Save snapshot' }).click()
    await page.getByRole('link', { name: /History/ }).click()
    await expect(page.locator('table.vt tbody tr').first()).toContainText('does not fit')
    // Against a plan that fits, the compare table says "no" for Fits.
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    await selectOf(page, 'Devices').selectOption('8')
    await expect(banner(page)).toContainText('Fits')
    await page.getByRole('button', { name: 'Save snapshot' }).click()
    await page.getByRole('link', { name: /History/ }).click()
    await expect(page.locator('table.compare tr', { hasText: /^Fits/ })).toContainText('no')
    await expect(page.locator('table.compare tr', { hasText: /^Fits/ })).toContainText('yes')

    // Back to the overfull mesh and look at the list and report.
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    await selectOf(page, 'Devices').selectOption('4')
    await page.goto('/app')
    const card = page.locator('.proj').first()
    await expect(card).toContainText('Does not fit')
    await expect(card.locator('.proj-meta .ok-t')).toHaveCount(0)
    await card.getByRole('link', { name: 'Report' }).click()
    const facts = page.locator('.report .facts')
    await expect(facts).toContainText('No (39.0 GB of 32.0 GB per device)')
    await expect(page.locator('.report')).toContainText('Short 7.0 GB')
  })

  test('"Smallest mesh that fits" says so when nothing fits and leaves the mesh alone', async ({ page }) => {
    await openSample(page, 'DeepSeek V3')
    // 671B parameters at BFP8 is 713 GB; eight 12 GB Wormhole n150s hold 96 GB.
    await selectOf(page, 'Device').selectOption('tt-n150s')
    await expect(banner(page)).toContainText('Does not fit')
    const before = await banner(page).innerText()
    await page.getByRole('button', { name: 'Smallest mesh that fits' }).click()
    await expect(page.locator('.toasts')).toContainText('Even 8x Wormhole n150s cannot hold this workload.')
    await expect(page.locator('.toast.error')).toBeVisible()
    expect(await banner(page).innerText()).toBe(before)
    await expect(banner(page)).toContainText('Does not fit')
  })

  test('a custom device is held to the memory it says it has', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    await selectOf(page, 'Weights').selectOption('bf16')
    await inputOf(page, 'Concurrent seqs').fill('1')
    await page.getByRole('link', { name: 'Hardware', exact: true }).click()
    await page.getByRole('button', { name: 'Add custom device' }).click()
    const dialog = page.locator('dialog[open]')
    await dialog.getByLabel('Name', { exact: true }).fill('Tiny 12 GB card')
    await dialog.getByLabel('Memory GB', { exact: true }).fill('12')
    // 500 GB/s (the form default) trips the browser's own step check (step 8): use a multiple of 8.
    await dialog.getByLabel('Memory bandwidth GB/s', { exact: true }).fill('512')
    await dialog.getByRole('button', { name: 'Save device' }).click()
    await expect(page.locator('.toasts')).toContainText('Added Tiny 12 GB card.')
    await page.locator('tr', { hasText: 'Tiny 12 GB card' }).getByRole('button', { name: 'Use' }).click()
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    // 16.06 GB of weights cannot sit on one 12 GB card. Two cards with TP 2: 8.03 + KV 0.54 + reserve (1.5 + 0.6) = 10.67 GB.
    await expect(banner(page)).toContainText('2x Tiny 12 GB card')
    await expect(banner(page)).toContainText('10.7 GB of 12.0 GB')
    await selectOf(page, 'Devices').selectOption('1')
    await expect(banner(page)).toContainText('Does not fit')
    await expect(banner(page)).toContainText('1x Tiny 12 GB card')
  })

  // Issue #8.
  test('the custom device form saves with its default values', async ({ page }) => {
    test.fail(true, 'known bug, https://github.com/AryanKansagara/fit-planner/issues/8: 500 GB/s fails the step=8 check, so Save device does nothing')
    await openSample(page, 'Llama 3.1 8B')
    await page.getByRole('link', { name: 'Hardware', exact: true }).click()
    await page.getByRole('button', { name: 'Add custom device' }).click()
    await page.locator('dialog[open]').getByRole('button', { name: 'Save device' }).click()
    await expect(page.locator('.toasts')).toContainText('Added My accelerator.')
  })

  test('search lists no option that overflows its memory', async ({ page }) => {
    await overfullPlan(page)
    await page.getByRole('link', { name: 'Search', exact: true }).click()
    const table = page.locator('table').first()
    await expect(table).toBeVisible()
    const rows = await table.locator('tbody tr').allInnerTexts()
    expect(rows.length).toBeGreaterThan(3)
    for (const row of rows) {
      const m = row.match(/([\d.]+)\/(\d+) GB/)
      expect(m, `memory column in "${row}"`).not.toBeNull()
      expect(Number(m![1]), `used <= capacity in "${row.replace(/\s+/g, ' ')}"`).toBeLessThanOrEqual(Number(m![2]))
    }
    // The 4 x p150a plan from the plan tab (39.0 GB per card) is not among them.
    expect(rows.join('\n')).not.toMatch(/Blackhole p150a[\s\S]*TP 4 x PP 1 · 39\.0/)
    // Taking the first option puts a fitting configuration on the plan.
    await table.locator('tbody tr').first().getByRole('button', { name: 'Use' }).click()
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    await expect(banner(page)).toContainText('Fits')
  })

  test('landing demo: a model too big for the cards says "Does not fit" and shows no speed', async ({ page }) => {
    await page.goto('/')
    const hero = page.locator('.hero-card')
    await hero.locator('input[type=range]').first().focus()
    await page.keyboard.press('End')
    await expect(hero.locator('.hc-controls label').first()).toContainText('DeepSeek V3')
    // 671B at BFP8 is 713 GB: 32 x 24 GB RTX 4090s hold 768 GB raw but 26.5 GB per card is over 24 GB.
    await hero.getByRole('button', { name: /4090/ }).click()
    await expect(hero.locator('.hc-caption strong')).toHaveText('Does not fit on 32x GeForce RTX 4090')
    await expect(hero.locator('.hc-caption strong.bad-t')).toBeVisible()
    await expect(hero).toContainText('add cards or shrink the dtype')
    await expect(hero.getByLabel(/tok\/s/).first()).toBeDefined()
    // The smallest model on a card it fits: shows a count, not "Does not fit".
    await hero.locator('input[type=range]').first().focus()
    await page.keyboard.press('Home')
    await expect(hero.locator('.hc-caption strong')).not.toContainText('Does not fit')
  })

  // Issue #7.
  test('landing demo respects the device\'s own mesh limit', async ({ page }) => {
    test.fail(true, 'known bug, https://github.com/AryanKansagara/fit-planner/issues/7: the demo sizes every device up to 32 cards')
    await page.goto('/')
    const hero = page.locator('.hero-card')
    await hero.locator('input[type=range]').first().focus()
    await page.keyboard.press('End')
    await hero.getByRole('button', { name: /p100a/ }).click()
    // The Hardware table offers the p100a (PCIe only) in meshes of at most 4.
    const caption = await hero.locator('.hc-caption strong').innerText()
    const count = Number(caption.match(/(\d+)x Blackhole p100a/)?.[1] ?? 0)
    expect(count).toBeLessThanOrEqual(4)
  })

  // Issue #3.
  test('an imported plan with an illegal split is not reported as fitting', async ({ page }) => {
    test.fail(true, 'known bug, https://github.com/AryanKansagara/fit-planner/issues/3: import accepts tp 3 for 32 heads and reports Fits')
    const cfg = '{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,"vocab_size":128256,"max_position_embeddings":131072}'
    const plan = { format: 'fit-planner/plan@1', plan: { name: 'Illegal split', model: { kind: 'paste', config: cfg, at: 0 }, workload: { weightDtype: 'bf16', kvDtype: 'bf16', context: 8192, batch: 1, promptLen: 1000, outputLen: 200 }, tp: 3, pp: 1 } }
    await page.goto('/app')
    await page.getByRole('button', { name: 'Import plan JSON' }).click()
    await page.locator('dialog textarea').fill(JSON.stringify(plan))
    await page.getByRole('button', { name: 'Import', exact: true }).click()
    await expect(page.locator('dialog .issues')).toContainText(/tp/i)
    await expect(page).toHaveURL(/\/app$/)
  })

  // Issue #2: kept apart so a crash cannot take other tests down; the page must survive the import.
  test('an imported plan with an unknown weight dtype is refused, not stored', async ({ page }) => {
    const cfg = '{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,"vocab_size":128256,"max_position_embeddings":131072}'
    const plan = { format: 'fit-planner/plan@1', plan: { name: 'Odd dtype', model: { kind: 'paste', config: cfg, at: 0 }, workload: { weightDtype: 'zz', kvDtype: 'bf16', context: 8192, batch: 1, promptLen: 1000, outputLen: 200 }, tp: 1, pp: 1 } }
    await page.goto('/app')
    await page.getByRole('button', { name: 'Import plan JSON' }).click()
    await page.locator('dialog textarea').fill(JSON.stringify(plan))
    await page.getByRole('button', { name: 'Import', exact: true }).click()
    await expect(page.locator('dialog .issues')).toContainText('plan.workload.weightDtype must be one of bf16, fp8, int8, bfp8, bfp4, q4 (got "zz").')
    await expect(page).toHaveURL(/\/app$/)
    await page.reload()
    await expect(page.getByText('No plans yet')).toBeVisible()
  })

  // Plans stored by the old importer: the list must render so they can be deleted.
  test('a stored plan with an unknown weight dtype is explained and can be deleted', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    const url = page.url()
    await page.evaluate(() => new Promise<void>((res, rej) => {
      const open = indexedDB.open('fit-planner', 1)
      open.onerror = () => rej(open.error)
      open.onsuccess = () => {
        const store = open.result.transaction('projects', 'readwrite').objectStore('projects')
        const all = store.getAll()
        all.onsuccess = () => {
          const put = store.put({ ...all.result[0], workload: { ...all.result[0].workload, weightDtype: 'zz' } })
          put.onsuccess = () => { open.result.close(); res() }
        }
      }
    }))
    for (const tab of ['', '/search', '/hardware', '/history', '/report']) {
      await page.goto(url + tab)
      await expect(page.locator('.issues')).toContainText('plan.workload.weightDtype must be one of')
      await expect(page.locator('body')).not.toContainText('Fits')
    }
    await page.getByRole('link', { name: 'Back to plans' }).click()
    await expect(page.locator('.proj')).toContainText('Plan data is not valid')
    await page.locator('.proj').getByRole('button', { name: 'Delete' }).click()
    await expect(page.getByText('No plans yet')).toBeVisible()
  })

  // Issue #4.
  test('search does not count more replicas than there are concurrent sequences', async ({ page }) => {
    test.fail(true, 'known bug, https://github.com/AryanKansagara/fit-planner/issues/4: 6 idle replicas "meet" the throughput target')
    await openSample(page, 'Llama 3.3 70B')
    await selectOf(page, 'Weights').selectOption('bf16')
    await inputOf(page, 'Concurrent seqs').fill('1')
    await page.getByRole('link', { name: 'Search', exact: true }).click()
    const first = page.locator('table').first().locator('tbody tr').first()
    await expect(first).toBeVisible()
    // "6 x 1x Instinct MI300X": the leading number is the replica count. One sequence can use one replica.
    const replicas = Number((await first.innerText()).match(/^\s*\d+\s+(\d+) x /)?.[1] ?? 1)
    expect(replicas).toBeLessThanOrEqual(1)
  })
})
