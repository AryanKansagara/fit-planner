import { test, expect } from './fixtures.ts'
import { banner, inputOf, openSample, projectId, selectOf } from './helpers.ts'

/* The app promises local-first storage: "Plans live in your browser" (IndexedDB), plus a few settings in localStorage. */

test.describe('state survives a reload where the app promises persistence', () => {
  test('a plan keeps its name, workload, hardware, mesh and history', async ({ page }) => {
    await openSample(page, 'Llama 3.3 70B')
    const id = projectId(page)
    await page.getByLabel('Plan name').fill('Office rack')
    await selectOf(page, 'Weights').selectOption('bf16')
    await inputOf(page, 'Concurrent seqs').fill('1')
    await inputOf(page, 'Context / seq').fill('8192')
    await selectOf(page, 'Device').selectOption('tt-p150a')
    await selectOf(page, 'Devices').selectOption('8')
    await selectOf(page, 'Split (TP x PP)').selectOption('4x2')
    await expect(banner(page)).toContainText('21.1 GB of 32.0 GB on the busiest device · 8x Blackhole p150a as TP 4 x PP 2')
    await page.getByRole('button', { name: 'Save snapshot' }).click()
    await expect(page.locator('.toasts')).toContainText('Configuration saved to history.')
    // Let the save settle (the store writes to IndexedDB asynchronously), then reload on the same URL.
    await expect(page.getByRole('link', { name: /History/ }).locator('.count')).toHaveText('1')
    await page.reload()

    await expect(page).toHaveURL(new RegExp(`/app/p/${id}$`))
    await expect(page.getByLabel('Plan name')).toHaveValue('Office rack')
    await expect(selectOf(page, 'Weights')).toHaveValue('bf16')
    await expect(inputOf(page, 'Concurrent seqs')).toHaveValue('1')
    await expect(inputOf(page, 'Context / seq')).toHaveValue('8192')
    await expect(selectOf(page, 'Device')).toHaveValue('tt-p150a')
    await expect(selectOf(page, 'Devices')).toHaveValue('8')
    await expect(selectOf(page, 'Split (TP x PP)')).toHaveValue('4x2')
    await expect(banner(page)).toContainText('21.1 GB of 32.0 GB on the busiest device · 8x Blackhole p150a as TP 4 x PP 2')
    await page.getByRole('link', { name: /History/ }).click()
    await expect(page.locator('table.vt:not(.compare) tbody tr')).toHaveCount(1)
    await expect(page.locator('table.vt:not(.compare) tbody tr')).toContainText('8x Blackhole p150a (TP 4 x PP 2)')

    // The plan list shows it too, and a deep link to another tab works after a cold load.
    await page.goto('/app')
    await expect(page.locator('.proj')).toContainText('Office rack')
    await page.goto(`/app/p/${id}/search`)
    await expect(page.getByRole('heading', { name: /configurations meet the targets|Nothing meets every target/ })).toBeVisible()
  })

  test('edited hardware persists with the plan', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    await selectOf(page, 'Device').selectOption('nv-4090')
    await page.getByRole('link', { name: 'Hardware', exact: true }).click()
    // The RTX 4090's published figures all pass the form's step checks (the p150a's $1,299 does not: issue #8).
    await page.locator('tr', { hasText: 'GeForce RTX 4090' }).getByRole('button', { name: 'Edit' }).click()
    await page.locator('dialog[open]').getByLabel('Memory GB', { exact: true }).fill('16')
    await page.locator('dialog[open]').getByRole('button', { name: 'Save device' }).click()
    await expect(page.locator('.toasts')).toContainText('Saved your edits to GeForce RTX 4090 for this plan.')
    await page.getByRole('link', { name: 'Plan', exact: true }).click()
    // BFP8 weights 8.53 GB + 4 sequences x 1.07 GB of KV + reserve (1.5 + 5% of 16 = 2.3) = 15.1 GB of 16.
    await inputOf(page, 'Concurrent seqs').fill('4')
    await expect(banner(page)).toContainText('15.1 GB of 16.0 GB')
    await page.reload()
    await expect(banner(page)).toContainText('15.1 GB of 16.0 GB')
    await page.getByRole('link', { name: 'Hardware', exact: true }).click()
    await expect(page.locator('tr', { hasText: 'GeForce RTX 4090' })).toContainText('16 GB')
    await expect(page.locator('tr', { hasText: 'GeForce RTX 4090' }).locator('.badge', { hasText: 'edited' })).toBeVisible()
  })

  test('a deleted plan stays deleted after a reload, and an undone delete stays', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    await page.goto('/app')
    await page.locator('.proj').getByRole('button', { name: 'Delete' }).click()
    await expect(page.locator('.proj')).toHaveCount(0)
    await page.reload()
    await expect(page.getByText('No plans yet')).toBeVisible()
    await expect(page.locator('.proj')).toHaveCount(0)

    await page.locator('.sample-card', { hasText: 'Qwen3 14B' }).click()
    await expect(page).toHaveURL(/\/app\/p\/[\w-]+$/)
    await page.goto('/app')
    await page.locator('.proj').getByRole('button', { name: 'Delete' }).click()
    await page.locator('.toasts').getByRole('button', { name: 'Undo' }).click()
    await expect(page.locator('.proj')).toHaveCount(1)
    await page.reload()
    await expect(page.locator('.proj')).toHaveCount(1)
    await expect(page.locator('.proj')).toContainText('Qwen3 14B')
  })

  test('plans are shared by every tab of the same browser profile', async ({ page, context }) => {
    await openSample(page, 'Qwen3 14B')
    const id = projectId(page)
    await page.getByLabel('Plan name').fill('Shared across tabs')
    await expect(page.getByLabel('Plan name')).toHaveValue('Shared across tabs')
    const other = await context.newPage()
    await other.goto(`/app/p/${id}`)
    await expect(other.getByLabel('Plan name')).toHaveValue('Shared across tabs')
    await other.close()
  })

  test('theme choice and default settings persist, and the defaults seed new plans', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: 'Dark', exact: true }).click()
    await expect(page.locator('html')).toHaveAttribute('data-theme-mode', 'dark')
    await page.getByLabel('Starting device').selectOption('nv-4090')
    await page.getByLabel('Reserve per device GB').fill('4')
    await page.getByRole('button', { name: 'Rent hourly' }).click()
    await page.reload()
    await expect(page.locator('html')).toHaveAttribute('data-theme-mode', 'dark')
    await expect(page.getByLabel('Starting device')).toHaveValue('nv-4090')
    await expect(page.getByLabel('Reserve per device GB')).toHaveValue('4')
    await expect(page.getByRole('button', { name: 'Rent hourly' })).toHaveAttribute('aria-pressed', 'true')

    // A new plan starts on the chosen device with the chosen reserve: Llama 3.1 8B BFP8 on a 24 GB card.
    // weights 8.53 + KV 8 x 1.07 = 8.59 + reserve (4 + 0.05 x 24 = 5.2) = 22.3 GB.
    await page.goto('/app')
    await page.locator('.sample-card', { hasText: 'Llama 3.1 8B' }).click()
    await expect(selectOf(page, 'Device')).toHaveValue('nv-4090')
    await expect(banner(page)).toContainText('22.3 GB of 24.0 GB on the busiest device · 1x GeForce RTX 4090')

    await page.goto('/settings')
    await page.getByRole('button', { name: 'Restore default settings' }).click()
    await expect(page.getByLabel('Starting device')).toHaveValue('tt-p150a')
    await page.reload()
    await expect(page.getByLabel('Starting device')).toHaveValue('tt-p150a')
    await expect(page.getByRole('button', { name: 'Own (amortised)' })).toHaveAttribute('aria-pressed', 'true')
  })

  test('an unknown plan id is a clear "not found", not a blank page', async ({ page }) => {
    await page.goto('/app/p/00000000-0000-4000-8000-000000000000')
    await expect(page.getByRole('heading', { name: 'This page did not fit' })).toBeVisible()
    await page.goto('/app/p/00000000-0000-4000-8000-000000000000/report')
    await expect(page.getByRole('heading', { name: 'This page did not fit' })).toBeVisible()
  })
})
