import { test, expect } from './fixtures.ts'
import { LLAMA_8B } from './helpers.ts'

const basePlan = (patch: Record<string, unknown>, wlPatch: Record<string, unknown> = {}) => JSON.stringify({
  format: 'fit-planner/plan@1',
  plan: {
    name: 'Repro',
    model: { kind: 'paste', config: JSON.stringify(LLAMA_8B), at: 0 },
    workload: { weightDtype: 'bf16', kvDtype: 'bf16', context: 8192, batch: 1, promptLen: 1000, outputLen: 200, ...wlPatch },
    tp: 1,
    pp: 1,
    ...patch,
  },
})

test.describe('plan import rejects illegal meshes and workloads (#3)', () => {
  for (const [name, payload, msg] of [
    ['fractional tp', basePlan({ tp: 1.5 }), /whole numbers/i],
    ['pp exceeding layers and device max', basePlan({ pp: 1000 }), /exceeds.*layers/i],
    ['mesh exceeding device max', basePlan({ deviceId: 'nv-4090', tp: 8, pp: 2 }), /16 devices .* is more than the 8/],
    ['zero context', basePlan({}, { context: 0 }), /plan\.workload\.context must be between 128/],
    ['an unknown device', basePlan({ deviceId: 'tt-p999' }), /"tt-p999" is not a known or custom device/],
  ] as const) {
    test(`refuses ${name}`, async ({ page }) => {
      await page.goto('/app')
      await page.getByRole('button', { name: 'Import plan JSON' }).click()
      await page.locator('dialog textarea').fill(payload)
      await page.getByRole('button', { name: 'Import', exact: true }).click()
      await expect(page.locator('dialog .issues')).toContainText(msg)
      await expect(page).toHaveURL(/\/app$/)
      await page.getByRole('button', { name: 'Cancel' }).click()
      await expect(page.locator('.proj')).toHaveCount(0)
    })
  }
})
