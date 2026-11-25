import { test, expect } from './fixtures.ts'
import { openSample } from './helpers.ts'

type Box = { x: number; y: number; width: number; height: number }
const overlaps = (a: Box, b: Box) => a.x < b.x + b.width - 0.5 && b.x < a.x + a.width - 0.5 && a.y < b.y + b.height - 0.5 && b.y < a.y + a.height - 0.5

test.describe('plan page layout', () => {
  for (const width of [390, 1024, 1081, 1280, 1440]) {
    test(`the inputs never cover the results at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 })
      for (const model of ['Llama 3.1 8B', 'DeepSeek V3']) {
        await openSample(page, model)
        await page.getByText('Assumptions', { exact: true }).click()
        const [left, right] = await page.evaluate(() => ['.plan-inputs > .card', '.plan-out, .plan-out > *, .kpi'].map(sel =>
          [...document.querySelectorAll(sel)].map(el => { const { x, y, width, height } = el.getBoundingClientRect(); return { x, y, width, height } })))
        expect(left.length).toBe(4)
        expect(right.length).toBeGreaterThan(6)
        for (const a of left) for (const b of right) expect(overlaps(a, b), `${model}: ${JSON.stringify(a)} covers ${JSON.stringify(b)}`).toBe(false)
        // Neither the page nor a control inside a card sticks out to the right.
        expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(0)
        const out = await page.locator('.plan-inputs .card').evaluateAll(cards => cards.flatMap(c => {
          const right = c.getBoundingClientRect().right
          return [...c.querySelectorAll('select, input, button')].filter(el => el.getBoundingClientRect().right > right + 0.5).map(el => el.outerHTML.slice(0, 60))
        }))
        expect(out).toEqual([])
      }
    })
  }
})
