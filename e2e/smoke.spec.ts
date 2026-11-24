import { test, expect } from './fixtures.ts'
import { banner, openSample } from './helpers.ts'

test.describe('pages, navigation and keyboard', () => {
  test('every page renders signed out, with no uncaught errors and no third-party requests', async ({ page, net }) => {
    const sameOrigin = new URL(test.info().project.use.baseURL ?? 'http://127.0.0.1').host
    const seen = new Set<string>()
    page.on('request', r => seen.add(new URL(r.url()).host))

    await page.goto('/')
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Size the mesh before you buy the cards.')
    await expect(page).toHaveTitle('Fit Planner: size LLM deployments across accelerator meshes')
    await page.getByRole('link', { name: 'Docs', exact: true }).first().click()
    await expect(page).toHaveURL(/\/docs$/)
    await expect(page).toHaveTitle('Docs · Fit Planner')
    await page.getByRole('link', { name: 'Settings', exact: true }).first().click()
    await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible()
    await expect(page.getByText('Signed out. Everything is stored in this browser (IndexedDB).')).toBeVisible()
    await page.getByRole('link', { name: 'Plans', exact: true }).first().click()
    await expect(page.getByRole('heading', { name: 'Deployment plans' })).toBeVisible()
    await expect(page.getByText('No plans yet')).toBeVisible()
    await page.goto('/no/such/page')
    await expect(page.getByRole('heading', { name: 'This page did not fit' })).toBeVisible()
    await expect(page).toHaveTitle('Not found · Fit Planner')

    // Nothing was sent anywhere but the preview server; the only attempts elsewhere are the webfonts, which were blocked.
    const external = [...seen].filter(h => h !== sameOrigin)
    expect(external.filter(h => !['fonts.googleapis.com', 'fonts.gstatic.com'].includes(h))).toEqual([])
    expect(net.blocked.filter(h => h.includes('supabase') || h.includes('huggingface'))).toEqual([])
  })

  test('the Supabase client is not loaded and sync stays local', async ({ page }) => {
    const scripts: string[] = []
    page.on('response', r => { if (r.request().resourceType() === 'script') scripts.push(r.url()) })
    await openSample(page, 'Llama 3.1 8B')
    await expect(page.locator('.sync-pill')).toHaveCount(0)
    expect(scripts.filter(u => /supabase/i.test(u))).toEqual([])
  })

  test('landing: the live demo and the "Try the sample" path', async ({ page }) => {
    await page.goto('/')
    const hero = page.locator('.hero-card')
    await expect(hero).toContainText('Live · real engine')
    // Default demo: Llama 3.3 70B, 8k context, BFP8, Blackhole p150a. It shows the smallest mesh that holds it.
    await expect(hero.locator('.hc-caption strong')).toContainText('Blackhole p150a')
    await expect(hero.locator('.hc-caption strong')).not.toContainText('Does not fit')
    const before = await hero.locator('.hc-caption').innerText()
    await hero.getByRole('button', { name: 'BFP4' }).click()
    await expect(hero.getByRole('button', { name: 'BFP4' })).toHaveAttribute('aria-pressed', 'true')
    expect(await hero.locator('.hc-caption').innerText()).not.toBe(before)

    await page.getByRole('link', { name: 'Try the sample' }).click()
    await expect(page).toHaveURL(/\/app\/p\/[\w-]+$/)
    await expect(page.locator('.plan-inputs')).toContainText('Llama 3.3 70B')
    await expect(banner(page)).toContainText('Fits')
  })

  test('keyboard: digits switch tabs, S saves, shortcuts are ignored while typing', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    const tabs = ['plan', 'model', 'hardware', 'search', 'history']
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    for (const [i, tab] of tabs.entries()) {
      await page.keyboard.press(String(i + 1))
      await expect(page).toHaveURL(new RegExp(`/app/p/[\\w-]+(/${tab})?$`))
      await expect(page.getByRole('link', { name: new RegExp(`^${tab}`, 'i') }).first()).toHaveAttribute('aria-current', 'page')
    }
    await page.keyboard.press('1')
    await page.keyboard.press('s')
    await expect(page.getByRole('link', { name: /History/ }).locator('.count')).toHaveText('1')
    // Typing "1", "s" and "5" into a field must not switch tabs or save another snapshot.
    await page.keyboard.press('2')
    await page.getByRole('button', { name: 'Manual' }).click()
    await page.getByLabel('Name', { exact: true }).fill('')
    await page.getByLabel('Name', { exact: true }).pressSequentially('1s5', { delay: 20 })
    await expect(page.getByLabel('Name', { exact: true })).toHaveValue('1s5')
    await expect(page).toHaveURL(/\/app\/p\/[\w-]+\/model$/)
    await expect(page.getByRole('link', { name: /History/ }).locator('.count')).toHaveText('1')
  })

  // Issue #9.
  test('the plan name keeps every character typed in quick succession', async ({ page }) => {
    test.fail(true, 'known bug, https://github.com/AryanKansagara/fit-planner/issues/9: keystrokes are dropped until the IndexedDB write finishes')
    await openSample(page, 'Llama 3.1 8B')
    const name = page.getByLabel('Plan name')
    await name.click()
    await page.keyboard.press('Control+a')
    await page.keyboard.type('Homelab rack number two')
    await page.waitForTimeout(500)
    await expect(name).toHaveValue('Homelab rack number two')
  })

  test('docs and settings pages mention what the planner computes', async ({ page }) => {
    await page.goto('/docs')
    await expect(page.locator('main')).toContainText('KV')
    await page.goto('/settings')
    await expect(page.getByRole('heading', { name: 'Defaults for new plans' })).toBeVisible()
  })
})
