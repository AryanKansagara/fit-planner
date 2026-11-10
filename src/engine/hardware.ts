import type { Device } from './types.ts'

/** Date the curated table was last checked against public spec sheets and list prices. */
export const HARDWARE_AS_OF = '2026-11-01'

// Public spec-sheet figures, dense (no sparsity). Prices are list or typical street prices in USD.
// Interconnect figures are per-device usable bandwidth for collectives; PCIe-only cards use a
// practical PCIe number. Values marked "approx" in notes are estimates: edit them for your setup.
export const DEVICES: Device[] = [
  { id: 'tt-n150s', name: 'Wormhole n150s', vendor: 'Tenstorrent', kind: 'accelerator', memGB: 12, bwGBs: 288, tflops: { fp16: 74, fp8: 262 }, linkGBs: 50, linkLatencyUs: 5, linkKind: '2x QSFP-DD 200G Ethernet', priceUsd: 1099, watts: 160, maxCount: 8, notes: 'Single Wormhole chip. Link latency approx.' },
  { id: 'tt-n300s', name: 'Wormhole n300s', vendor: 'Tenstorrent', kind: 'accelerator', memGB: 24, bwGBs: 576, tflops: { fp16: 131, fp8: 466 }, linkGBs: 50, linkLatencyUs: 5, linkKind: '2x QSFP-DD 200G Ethernet', priceUsd: 1399, watts: 300, maxCount: 8, notes: 'Two Wormhole chips on one card, modelled as one device. Four in a T3000-class box.' },
  { id: 'tt-p100a', name: 'Blackhole p100a', vendor: 'Tenstorrent', kind: 'accelerator', memGB: 28, bwGBs: 448, tflops: { fp16: 166, fp8: 664 }, linkGBs: 25, linkLatencyUs: 10, linkKind: 'PCIe 5.0 x16 only', priceUsd: 999, watts: 300, maxCount: 4, notes: 'No Ethernet ports, so multi-card runs over PCIe. FP16 rate approx.' },
  { id: 'tt-p150a', name: 'Blackhole p150a', vendor: 'Tenstorrent', kind: 'accelerator', memGB: 32, bwGBs: 512, tflops: { fp16: 194, fp8: 774 }, linkGBs: 400, linkLatencyUs: 3, linkKind: '4x QSFP-DD 800G Ethernet', priceUsd: 1299, watts: 300, maxCount: 32, notes: 'Four in a QuietBox-class box (about $12k). FP16 rate and link latency approx.' },
  { id: 'nv-3090', name: 'GeForce RTX 3090', vendor: 'NVIDIA', kind: 'gpu', memGB: 24, bwGBs: 936, tflops: { fp16: 71 }, linkGBs: 25, linkLatencyUs: 10, linkKind: 'PCIe 4.0 x16', priceUsd: 800, watts: 350, maxCount: 8, notes: 'Used price. No FP8. NVLink bridge not modelled.' },
  { id: 'nv-4090', name: 'GeForce RTX 4090', vendor: 'NVIDIA', kind: 'gpu', memGB: 24, bwGBs: 1008, tflops: { fp16: 165, fp8: 330 }, linkGBs: 25, linkLatencyUs: 10, linkKind: 'PCIe 4.0 x16', priceUsd: 1800, rentPerHour: 0.4, watts: 450, maxCount: 8 },
  { id: 'nv-5090', name: 'GeForce RTX 5090', vendor: 'NVIDIA', kind: 'gpu', memGB: 32, bwGBs: 1792, tflops: { fp16: 209, fp8: 419 }, linkGBs: 50, linkLatencyUs: 10, linkKind: 'PCIe 5.0 x16', priceUsd: 2400, rentPerHour: 0.7, watts: 575, maxCount: 8, notes: 'Street price above MSRP.' },
  { id: 'nv-l40s', name: 'L40S 48GB', vendor: 'NVIDIA', kind: 'gpu', memGB: 48, bwGBs: 864, tflops: { fp16: 362, fp8: 733 }, linkGBs: 25, linkLatencyUs: 10, linkKind: 'PCIe 4.0 x16', priceUsd: 8000, rentPerHour: 0.9, watts: 350, maxCount: 8 },
  { id: 'nv-a100', name: 'A100 80GB SXM', vendor: 'NVIDIA', kind: 'gpu', memGB: 80, bwGBs: 2039, tflops: { fp16: 312 }, linkGBs: 300, linkLatencyUs: 3, linkKind: 'NVLink 3 (600 GB/s bidirectional)', priceUsd: 15000, rentPerHour: 1.3, watts: 400, maxCount: 8 },
  { id: 'nv-h100', name: 'H100 80GB SXM', vendor: 'NVIDIA', kind: 'gpu', memGB: 80, bwGBs: 3350, tflops: { fp16: 989, fp8: 1979 }, linkGBs: 450, linkLatencyUs: 3, linkKind: 'NVLink 4 (900 GB/s bidirectional)', priceUsd: 25000, rentPerHour: 2.2, watts: 700, maxCount: 8 },
  { id: 'amd-mi300x', name: 'Instinct MI300X', vendor: 'AMD', kind: 'gpu', memGB: 192, bwGBs: 5300, tflops: { fp16: 1307, fp8: 2615 }, linkGBs: 448, linkLatencyUs: 4, linkKind: 'Infinity Fabric (7 links)', priceUsd: 15000, rentPerHour: 2.0, watts: 750, maxCount: 8, notes: 'Price approx.' },
]

/** Curated table with a project's edits and custom devices applied (same id replaces). */
export function mergeDevices(overrides: Device[]): Device[] {
  const byId = new Map(DEVICES.map(d => [d.id, d]))
  for (const o of overrides) byId.set(o.id, o)
  return [...byId.values()]
}

export function validateDevice(d: Device): string[] {
  const errs: string[] = []
  if (!d.name.trim()) errs.push('Name is required.')
  for (const [k, v] of [['Memory', d.memGB], ['Bandwidth', d.bwGBs], ['FP16 TFLOPS', d.tflops.fp16], ['Link bandwidth', d.linkGBs], ['Power', d.watts], ['Max cards', d.maxCount]] as const) {
    if (!(v > 0)) errs.push(`${k} must be greater than zero.`)
  }
  if (d.tflops.fp8 !== undefined && !(d.tflops.fp8 >= 0)) errs.push('FP8 TFLOPS must be zero or more.')
  if (!(d.priceUsd >= 0)) errs.push('Price must be zero or more.')
  if (!(d.linkLatencyUs >= 0)) errs.push('Link latency must be zero or more.')
  return errs
}

const CSV_COLS = ['id', 'name', 'vendor', 'memGB', 'bwGBs', 'fp16TFLOPS', 'fp8TFLOPS', 'linkGBs', 'linkLatencyUs', 'linkKind', 'priceUsd', 'rentPerHour', 'watts', 'maxCount'] as const

export function devicesToCsv(list: Device[]): string {
  const esc = (v: unknown) => { const s = v === undefined ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s }
  return [CSV_COLS.join(','), ...list.map(d => [d.id, d.name, d.vendor, d.memGB, d.bwGBs, d.tflops.fp16, d.tflops.fp8, d.linkGBs, d.linkLatencyUs, d.linkKind, d.priceUsd, d.rentPerHour, d.watts, d.maxCount].map(esc).join(','))].join('\n')
}
