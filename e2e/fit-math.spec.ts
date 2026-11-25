import { test, expect } from './fixtures.ts'
import { banner, inputOf, kpi, openSample, selectOf } from './helpers.ts'

/*
 * Hand-worked memory figures. These do not call the engine: they come from the model's dimensions
 * (the public config.json values in the sample) and the app's stated assumptions (1.5 GB + 5% of
 * the card reserved per device; 1 GB = 1e9 bytes).
 *
 * Llama 3.1 8B: hidden 4096, 32 layers, 32 heads, 8 KV heads, head_dim 128, MLP 14336, vocab 128256, untied.
 *   per layer  = attention 4096*4096*2 + 2*4096*1024 = 41,943,040
 *              + MLP 3*4096*14336                    = 176,160,768
 *              + two norms 2*4096                    =       8,192      = 218,112,000
 *   embedding = head = 128256*4096                                     = 525,336,576
 *   BF16 weights = 2 bytes * (32*218,112,000 + 2*525,336,576)          = 16,060,522,496 B = 16.06 GB
 *   KV per token = 2 (K,V) * 32 layers * 8 KV heads * 128 * 2 bytes    = 131,072 B = 128 KB
 *   KV for 1 sequence of 8,192 tokens = 1.074 GB; of 131,072 tokens = 17.18 GB
 *
 * Llama 3.3 70B: hidden 8192, 80 layers, 64 heads, 8 KV heads, head_dim 128, MLP 28672, untied.
 *   per layer 855,654,400; embedding = head = 1,050,673,152
 *   BF16 weights = 141,107,396,608 B = 141.1 GB, KV per token 327,680 B = 320 KB
 */

test.describe('memory fit matches hand-worked numbers', () => {
  test('Llama 3.1 8B: change dtype, hardware, context and mesh', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    await selectOf(page, 'Weights').selectOption('bf16')
    await inputOf(page, 'Concurrent seqs').fill('1')
    await inputOf(page, 'Context / seq').fill('8192')

    // One Blackhole p150a (32 GB, reserve 1.5 + 0.05*32 = 3.1 GB): 16.06 + 1.07 + 3.1 = 20.23 GB.
    await expect(selectOf(page, 'Device')).toHaveValue('tt-p150a')
    await expect(banner(page)).toContainText('Fits')
    await expect(banner(page)).toContainText('20.2 GB of 32.0 GB on the busiest device · 1x Blackhole p150a as TP 1 x PP 1')
    // Free after weights and reserve = 32 - 3.1 - 16.06 = 12.84 GB, so floor(12.84 / 1.074) = 11 sequences.
    await expect(kpi(page, 'Max sequences')).toHaveText('11')
    await expect(page.locator('.facts')).toContainText('128 KB')

    // 131,072 tokens of context: KV alone is 17.18 GB, more than the 12.84 GB left. Used 36.34 GB of 32.
    await inputOf(page, 'Context / seq').fill('131072')
    await expect(banner(page)).toContainText('Weights fit, KV cache does not')
    await expect(banner(page)).toContainText('36.3 GB of 32.0 GB')
    await expect(kpi(page, 'Max sequences')).toHaveText('0')
    await expect(page.locator('.fit-banner.bad')).toBeVisible()

    // "Smallest mesh that fits": two cards each hold half the weights and half the KV (8 KV heads split):
    // 8.03 + 8.59 + 3.1 = 19.72 GB.
    await page.getByRole('button', { name: 'Smallest mesh that fits' }).click()
    await expect(banner(page)).toContainText('Fits')
    await expect(banner(page)).toContainText('19.7 GB of 32.0 GB')
    await expect(banner(page)).toContainText('2x Blackhole p150a')

    // Back to 8,192 on an RTX 4090 (24 GB, reserve 1.5 + 1.2 = 2.7 GB): 16.06 + 1.07 + 2.7 = 19.83 GB, 4 sequences fit.
    await inputOf(page, 'Context / seq').fill('8192')
    await selectOf(page, 'Device').selectOption('nv-4090')
    await expect(banner(page)).toContainText('Fits')
    await expect(banner(page)).toContainText('19.8 GB of 24.0 GB on the busiest device · 1x GeForce RTX 4090')
    await expect(kpi(page, 'Max sequences')).toHaveText('4')

    // Five sequences of 8,192 tokens need 5.37 GB of KV but only 24 - 2.7 - 16.06 = 5.24 GB is free.
    await inputOf(page, 'Concurrent seqs').fill('5')
    await expect(banner(page)).toContainText('Weights fit, KV cache does not')
    await expect(banner(page)).toContainText('24.1 GB of 24.0 GB')
    // Four are exactly what fits: 16.06 + 4.29 + 2.7 = 23.06 GB.
    await inputOf(page, 'Concurrent seqs').fill('4')
    await expect(banner(page)).toContainText('Fits')
    await expect(banner(page)).toContainText('23.1 GB of 24.0 GB')
  })

  test('Llama 3.3 70B: mesh size and split decide the fit', async ({ page }) => {
    await openSample(page, 'Llama 3.3 70B')
    await selectOf(page, 'Weights').selectOption('bf16')
    await inputOf(page, 'Concurrent seqs').fill('1')
    await inputOf(page, 'Context / seq').fill('8192')
    await selectOf(page, 'Device').selectOption('tt-p150a')

    // Four 32 GB cards: weights 141.1/4 = 35.28 GB per card cannot fit whatever the split.
    await selectOf(page, 'Devices').selectOption('4')
    await expect(banner(page)).toContainText('Does not fit')
    await expect(banner(page)).toContainText('39.0 GB of 32.0 GB')
    await expect(banner(page)).not.toContainText('Fits')
    await expect(kpi(page, 'Max sequences')).toHaveText('0')
    // 2 x 2 and 1 x 4 are no better: weights are split four ways in every layout.
    await selectOf(page, 'Split (TP x PP)').selectOption('2x2')
    await expect(banner(page)).toContainText('Does not fit')
    await selectOf(page, 'Split (TP x PP)').selectOption('1x4')
    await expect(banner(page)).toContainText('Does not fit')

    // Eight cards, TP 8: weights 17.64 GB, KV (327,680 B * 8192 / 8 KV shards) 0.34 GB, reserve 3.1 GB = 21.07 GB.
    await selectOf(page, 'Devices').selectOption('8')
    await expect(banner(page)).toContainText('Fits')
    await expect(banner(page)).toContainText('21.1 GB of 32.0 GB on the busiest device · 8x Blackhole p150a as TP 8 x PP 1')
    // TP 4 x PP 2 holds the same weights per card (half the layers, half the shards).
    await selectOf(page, 'Split (TP x PP)').selectOption('4x2')
    await expect(banner(page)).toContainText('21.1 GB of 32.0 GB')
    await expect(banner(page)).toContainText('as TP 4 x PP 2')

    // Sixteen cards TP 8 x PP 2: weights 8.82 GB, KV 0.17 GB, reserve 3.1 GB = 12.09 GB.
    await selectOf(page, 'Devices').selectOption('16')
    await selectOf(page, 'Split (TP x PP)').selectOption('8x2')
    await expect(banner(page)).toContainText('12.1 GB of 32.0 GB')

    // The boundary: eight cards, 32,768 tokens x 8 sequences is 10.74 GB of KV against 32 - 3.1 - 17.64 = 11.26 GB free.
    await selectOf(page, 'Devices').selectOption('8')
    await selectOf(page, 'Split (TP x PP)').selectOption('8x1')
    await inputOf(page, 'Context / seq').fill('32768')
    await inputOf(page, 'Concurrent seqs').fill('8')
    await expect(banner(page)).toContainText('Fits')
    await expect(banner(page)).toContainText('31.5 GB of 32.0 GB')
    await expect(kpi(page, 'Max sequences')).toHaveText('8')
    // A ninth sequence needs 12.08 GB of KV: 32.8 GB used, does not fit.
    await inputOf(page, 'Concurrent seqs').fill('9')
    await expect(banner(page)).toContainText('Weights fit, KV cache does not')
    await expect(banner(page)).toContainText('32.8 GB of 32.0 GB')
  })

  test('a smaller weight dtype shrinks the weights by the stated ratio', async ({ page }) => {
    await openSample(page, 'Llama 3.1 8B')
    await inputOf(page, 'Concurrent seqs').fill('1')
    await inputOf(page, 'Context / seq').fill('8192')
    await selectOf(page, 'Devices').selectOption('1')
    // BF16 2 bytes -> 16.06 GB; BFP8 8.5/8 bytes -> 8.53 GB; BFP4 4.5/8 bytes -> 4.51 GB. Plus 1.07 KV and 3.1 reserve.
    for (const [dtype, used] of [['bf16', '20.2'], ['bfp8', '12.7'], ['bfp4', '8.7']] as const) {
      await selectOf(page, 'Weights').selectOption(dtype)
      await expect(banner(page)).toContainText(`${used} GB of 32.0 GB`)
    }
  })
})
