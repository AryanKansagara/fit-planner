import { readFile } from 'node:fs/promises'
import { test, expect } from './fixtures.ts'
import { banner, inputOf, openSample, projectId, selectOf } from './helpers.ts'

/** A plan with two saved snapshots whose compare table can be worked by hand (see fit-math.spec.ts):
 *   A: Llama 3.1 8B, BF16, 1 sequence x 8,192, one Blackhole p150a: 20.2 GB used, 11 sequences max, $1,299 hardware.
 *   B: the same on an RTX 4090: 19.8 GB used, 4 sequences max, $1,800 hardware. */
async function planWithTwoSnapshots(page: import('@playwright/test').Page) {
  await openSample(page, 'Llama 3.1 8B')
  await selectOf(page, 'Weights').selectOption('bf16')
  await inputOf(page, 'Concurrent seqs').fill('1')
  await inputOf(page, 'Context / seq').fill('8192')
  await expect(banner(page)).toContainText('20.2 GB of 32.0 GB')
  await page.getByRole('button', { name: 'Save snapshot' }).click()
  await expect(page.locator('.toasts')).toContainText('Configuration saved to history.')
  await selectOf(page, 'Device').selectOption('nv-4090')
  await expect(banner(page)).toContainText('19.8 GB of 24.0 GB')
  // The keyboard shortcut does the same as the button.
  await page.locator('body').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('s')
  await expect(page.getByRole('link', { name: /History/ }).locator('.count')).toHaveText('2')
}

test.describe('save, compare, export and re-import', () => {
  test('two snapshots compare against each other with the hand-worked differences', async ({ page }) => {
    await planWithTwoSnapshots(page)
    await page.getByRole('link', { name: /History/ }).click()

    const rows = page.locator('table.vt:not(.compare) tbody tr')
    await expect(rows).toHaveCount(2)
    // Newest first.
    await expect(rows.nth(0)).toContainText('1x GeForce RTX 4090')
    await expect(rows.nth(1)).toContainText('1x Blackhole p150a')

    const compare = page.locator('table.compare')
    const row = (label: string) => compare.locator('tr', { has: page.locator('th', { hasText: new RegExp(`^${label}$`) }) })
    await expect(compare.locator('thead')).toContainText('1x Blackhole p150a')
    await expect(compare.locator('thead')).toContainText('1x GeForce RTX 4090')
    await expect(row('Hardware')).toContainText('changed')
    await expect(row('Fits')).toContainText('yes')
    await expect(row('Memory used per device').locator('td').nth(0)).toHaveText('20.2 GB')
    await expect(row('Memory used per device').locator('td').nth(1)).toHaveText('19.8 GB')
    // (19.834 - 20.234) / 20.234 = -1.98%
    await expect(row('Memory used per device').locator('td').nth(2)).toHaveText('-2%')
    await expect(row('Max sequences at context').locator('td').nth(0)).toHaveText('11')
    await expect(row('Max sequences at context').locator('td').nth(1)).toHaveText('4')
    // 1800 / 1299 = +38.6%
    await expect(row('Hardware cost').locator('td').nth(0)).toHaveText('$1,299')
    await expect(row('Hardware cost').locator('td').nth(1)).toHaveText('$1,800')
    await expect(row('Hardware cost').locator('td').nth(2)).toHaveText('+39%')
    // Same workload on both sides: nothing to report.
    await expect(row('Context x sequences').locator('td').nth(2)).toHaveText('')
  })

  test('snapshots can be relabelled and deleted with undo', async ({ page }) => {
    await planWithTwoSnapshots(page)
    await page.getByRole('link', { name: /History/ }).click()
    const labels = page.getByLabel('Snapshot label')
    await labels.nth(1).fill('Cheap card')
    await expect(page.locator('table.compare thead')).toContainText('Cheap card')

    // Deleting one snapshot leaves one; Undo brings it back with its label.
    await page.locator('table.vt:not(.compare) tbody tr').nth(1).getByRole('button', { name: 'Delete' }).click()
    await expect(page.locator('table.vt:not(.compare) tbody tr')).toHaveCount(1)
    await expect(page.locator('.toasts')).toContainText('Snapshot deleted.')
    await page.locator('.toasts').getByRole('button', { name: 'Undo' }).click()
    await expect(page.locator('table.vt:not(.compare) tbody tr')).toHaveCount(2)
    await expect(labels.nth(1)).toHaveValue('Cheap card')
  })

  test('share: Download JSON, then Import plan JSON restores the plan and its history', async ({ page }) => {
    await planWithTwoSnapshots(page)
    const originalId = projectId(page)
    const originalBanner = (await banner(page).textContent())!

    await page.getByRole('button', { name: 'Share plan' }).click()
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download JSON' }).click()])
    expect(download.suggestedFilename()).toBe('llama-3-1-8b-team-chat.plan.json')
    const path = await download.path()
    const exported = JSON.parse(await readFile(path, 'utf8'))
    expect(exported.format).toBe('fit-planner/plan@1')
    expect(exported.plan).toMatchObject({ deviceId: 'nv-4090', tp: 1, pp: 1, workload: { weightDtype: 'bf16', context: 8192, batch: 1 } })
    expect(exported.plan.model.config).toContain('"hidden_size": 4096')
    expect(exported.snapshots).toHaveLength(2)
    expect(exported.snapshots.map((s: { result: { usedGB: number } }) => Math.round(s.result.usedGB * 10) / 10)).toEqual([20.2, 19.8])
    expect(exported).not.toHaveProperty('plan.id')

    // Re-import it as a file from the workspace.
    await page.keyboard.press('Escape')
    await page.goto('/app')
    await page.getByRole('button', { name: 'Import plan JSON' }).click()
    await page.locator('dialog input[type=file]').setInputFiles(path)
    await expect(page).toHaveURL(/\/app\/p\/[\w-]+$/)
    expect(projectId(page)).not.toBe(originalId)
    await expect(page.locator('.toasts')).toContainText('Imported Llama 3.1 8B · Team chat.')
    await expect(banner(page)).toHaveText(originalBanner)
    await expect(selectOf(page, 'Device')).toHaveValue('nv-4090')
    await expect(selectOf(page, 'Weights')).toHaveValue('bf16')
    await page.getByRole('link', { name: /History/ }).click()
    await expect(page.locator('table.vt:not(.compare) tbody tr')).toHaveCount(2)
    await expect(page.locator('table.compare')).toContainText('-2%')

    // Both plans now exist.
    await page.goto('/app')
    await expect(page.locator('.proj')).toHaveCount(2)
  })

  test('share: the plan JSON can be pasted back in', async ({ page }) => {
    await planWithTwoSnapshots(page)
    await page.getByRole('button', { name: 'Share plan' }).click()
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download JSON' }).click()])
    const text = await readFile(await download.path(), 'utf8')
    await page.keyboard.press('Escape')
    await page.goto('/app')
    await page.getByRole('button', { name: 'Import plan JSON' }).click()
    await expect(page.getByRole('button', { name: 'Import', exact: true })).toBeDisabled()
    await page.locator('dialog textarea').fill(text)
    await page.getByRole('button', { name: 'Import', exact: true }).click()
    await expect(page).toHaveURL(/\/app\/p\/[\w-]+$/)
    await expect(banner(page)).toContainText('19.8 GB of 24.0 GB')
  })

  const BAD_PLANS: [string, string, RegExp][] = [
    ['text that is not JSON', 'this is not json', /Not valid JSON/],
    ['a truncated export', '{"format":"fit-planner/plan@1","plan":{"name":"x"', /Not valid JSON/],
    ['JSON without the format tag', '{"plan":{}}', /Expected a file with "format": "fit-planner\/plan@1"/],
    ['another tool\'s format', '{"format":"other/plan@2","plan":{}}', /Expected a file with "format"/],
    ['an array', '[]', /Expected a file with "format"/],
    ['a plan without a model', '{"format":"fit-planner/plan@1","plan":{"workload":{"context":8192},"tp":1,"pp":1}}', /plan\.model\.config is missing\./],
    ['a plan without a workload', '{"format":"fit-planner/plan@1","plan":{"model":{"config":"{}"},"tp":1,"pp":1}}', /plan\.workload is missing or incomplete\./],
    ['a plan with tp 0', '{"format":"fit-planner/plan@1","plan":{"model":{"config":"{}"},"workload":{"context":8192},"tp":0,"pp":1}}', /plan\.tp and plan\.pp must be positive numbers\./],
    ['a plan with a negative pp', '{"format":"fit-planner/plan@1","plan":{"model":{"config":"{}"},"workload":{"context":8192},"tp":1,"pp":-2}}', /plan\.tp and plan\.pp must be positive numbers\./],
    ['a plan whose model config is empty', '{"format":"fit-planner/plan@1","plan":{"model":{"config":"{}"},"workload":{"context":8192},"tp":1,"pp":1}}', /The plan's model config does not parse: Hidden size: none of/],
  ]
  for (const [name, text, message] of BAD_PLANS) {
    test(`import refuses ${name}`, async ({ page }) => {
      await page.goto('/app')
      await page.getByRole('button', { name: 'Import plan JSON' }).click()
      await page.locator('dialog textarea').fill(text)
      await page.getByRole('button', { name: 'Import', exact: true }).click()
      await expect(page.locator('dialog .issues')).toContainText(message)
      await expect(page).toHaveURL(/\/app$/)
      await page.getByRole('button', { name: 'Cancel' }).click()
      await expect(page.locator('.proj')).toHaveCount(0)
      await expect(page.getByText('No plans yet')).toBeVisible()
    })
  }

  test('CSV exports: search options and the hardware table', async ({ page }) => {
    await planWithTwoSnapshots(page)
    await page.getByRole('link', { name: 'Search', exact: true }).click()
    const [search] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Export CSV' }).click()])
    expect(search.suggestedFilename()).toMatch(/-search\.csv$/)
    const lines = (await readFile(await search.path(), 'utf8')).split('\n')
    expect(lines[0]).toBe('rank,meets_targets,device,cards_per_mesh,tp,pp,replicas,total_cards,agg_tok_s,ms_per_token,ttft_ms,usd_per_hour,usd_per_1m_tokens,hardware_usd,tok_per_joule,memory_used_gb,memory_gb,reason')
    expect(lines.length).toBeGreaterThan(10)
    for (const line of lines.slice(1)) expect(line.split(',').length).toBeGreaterThanOrEqual(18)
    // Rows are ranked 1..n and options that meet the targets come first.
    expect(lines.slice(1).map(l => Number(l.split(',')[0]))).toEqual(lines.slice(1).map((_, i) => i + 1))
    const flags = lines.slice(1).map(l => l.split(',')[1])
    expect(flags.join('')).toMatch(/^(true)*(false)*$/)

    await page.getByRole('link', { name: 'Hardware', exact: true }).click()
    const [hw] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Export CSV' }).click()])
    expect(hw.suggestedFilename()).toBe('hardware-2026-11-01.csv')
    const hwLines = (await readFile(await hw.path(), 'utf8')).trim().split('\n')
    expect(hwLines[0]).toBe('id,name,vendor,memGB,bwGBs,fp16TFLOPS,fp8TFLOPS,linkGBs,linkLatencyUs,linkKind,priceUsd,rentPerHour,watts,maxCount')
    expect(hwLines).toContain('tt-p150a,Blackhole p150a,Tenstorrent,32,512,194,774,400,3,4x QSFP-DD 800G Ethernet,1299,,300,32')
    expect(hwLines).toContain('nv-4090,GeForce RTX 4090,NVIDIA,24,1008,165,330,25,10,PCIe 4.0 x16,1800,0.4,450,8')
  })

  test('the printable report shows the plan and what it was sized from', async ({ page }) => {
    await planWithTwoSnapshots(page)
    await page.getByRole('link', { name: 'Report', exact: true }).click()
    const report = page.locator('article.report')
    await expect(report.locator('h1')).toHaveText('Llama 3.1 8B · Team chat')
    await expect(report).toContainText('Current plan: 1x GeForce RTX 4090')
    await expect(report.locator('.facts')).toContainText('Yes (19.8 GB of 24.0 GB per device)')
    await expect(report.locator('.facts')).toContainText('Max sequences4 at 8,192 tokens')
    await expect(report).toContainText('8.0B parameters')
    const [csv] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'CSV' }).click()])
    expect(csv.suggestedFilename()).toBe('Llama-3-1-8B-Team-chat-options.csv')
  })

  test('deleting a plan can be undone with the button or Ctrl+Z', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    await page.goto('/app')
    const cards = page.locator('.proj')
    await expect(cards).toHaveCount(1)
    await cards.first().getByRole('button', { name: 'Delete' }).click()
    await expect(cards).toHaveCount(0)
    await page.locator('.toasts').getByRole('button', { name: 'Undo' }).click()
    await expect(cards).toHaveCount(1)
    await cards.first().getByRole('button', { name: 'Delete' }).click()
    await expect(cards).toHaveCount(0)
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('Control+z')
    await expect(cards).toHaveCount(1)
  })
})
