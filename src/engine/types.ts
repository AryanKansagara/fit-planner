export type WeightDtype = 'bf16' | 'fp8' | 'int8' | 'bfp8' | 'bfp4' | 'q4'
export type KvDtype = 'bf16' | 'fp8' | 'bfp8'

/** Bytes per element, and which tensor-core rate the matmuls run at. */
export const DTYPES: Record<WeightDtype, { label: string; bytes: number; rate: 'fp16' | 'fp8' }> = {
  bf16: { label: 'BF16 / FP16', bytes: 2, rate: 'fp16' },
  fp8: { label: 'FP8 (e4m3)', bytes: 1, rate: 'fp8' },
  int8: { label: 'INT8', bytes: 1, rate: 'fp8' },
  // Block floating point: 8- or 4-bit mantissas with one shared 8-bit exponent per 16 values.
  bfp8: { label: 'BFP8 (block FP8)', bytes: 8.5 / 8, rate: 'fp8' },
  bfp4: { label: 'BFP4 (block FP4)', bytes: 4.5 / 8, rate: 'fp8' },
  // GGUF Q4_K_M / AWQ / GPTQ style 4-bit with group scales, dequantised to FP16 for the matmul.
  q4: { label: '4-bit (Q4_K / AWQ)', bytes: 4.5 / 8, rate: 'fp16' },
}

export interface MoeSpec {
  experts: number
  active: number
  expertInter: number
  shared: number
  sharedInter: number
  /** Leading layers that use a dense MLP instead of experts. */
  denseLayers: number
}

export interface MlaSpec { qLoraRank: number; kvLoraRank: number; qkNope: number; qkRope: number; vHead: number }

export interface ModelSpec {
  name: string
  arch: string
  hidden: number
  layers: number
  heads: number
  kvHeads: number
  headDim: number
  intermediate: number
  vocab: number
  tied: boolean
  maxContext: number
  /** SwiGLU-style MLP (gate, up, down) vs a plain two-matrix MLP. */
  gated: boolean
  qkvBias: boolean
  qkNorm: boolean
  moe?: MoeSpec
  mla?: MlaSpec
}

export interface Device {
  id: string
  name: string
  vendor: string
  kind: 'accelerator' | 'gpu' | 'custom'
  memGB: number
  bwGBs: number
  /** Dense (no sparsity) peak TFLOPS. */
  tflops: { fp16: number; fp8?: number }
  /** Per-device bandwidth to peers for collectives, GB/s, and one-hop latency. */
  linkGBs: number
  linkLatencyUs: number
  linkKind: string
  priceUsd: number
  rentPerHour?: number
  watts: number
  /** Largest mesh this device is offered in (cards per host or per pod). */
  maxCount: number
  notes?: string
}

export interface Workload {
  weightDtype: WeightDtype
  kvDtype: KvDtype
  /** Context window reserved per sequence (prompt + output). */
  context: number
  /** Concurrent sequences decoding together. */
  batch: number
  promptLen: number
  outputLen: number
}

export interface Assumptions {
  /** Achievable fraction of peak memory bandwidth during decode. */
  bwUtil: number
  /** Achievable fraction of peak dense FLOPS (model FLOPS utilisation). */
  mfu: number
  /** Achievable fraction of link bandwidth for collectives. */
  linkUtil: number
  /** Fixed runtime reserve per device (firmware, activations, scratch), GB. */
  reserveGB: number
  /** Fraction of device memory kept free for fragmentation. */
  reserveFrac: number
  amortYears: number
  kwhUsd: number
  /** Own the hardware (amortised price plus power) or rent it by the hour. */
  costMode: 'own' | 'rent'
}

export interface Layout { device: Device; tp: number; pp: number }

export interface Targets { minTokS: number; maxTpotMs: number; maxTtftMs: number }
