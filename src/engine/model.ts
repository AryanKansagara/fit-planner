import type { KvDtype, ModelSpec, WeightDtype } from './types.ts'

export interface Issue { level: 'error' | 'warn' | 'info'; message: string; line?: number; field?: string }
export interface FieldMap { field: keyof ModelSpec | string; label: string; value: string | number | boolean; source: string; line?: number }
export interface ParseResult {
  spec: ModelSpec | null
  issues: Issue[]
  mapping: FieldMap[]
  /** Dtype the checkpoint ships in, if the config says. */
  suggestedDtype?: WeightDtype
  format: string
}

const PLAIN_MLP = new Set(['gpt2', 'gpt_neox', 'gptj', 'opt', 'bloom', 'falcon', 'gpt_bigcode', 'phi', 'starcoder2', 'mpt'])
const QKV_BIAS = new Set(['qwen2', 'qwen2_moe', 'qwen2_vl'])
const QK_NORM = new Set(['qwen3', 'qwen3_moe', 'olmo2', 'gemma3', 'gemma3_text'])

/** 1-based line of the first `"key":` in the JSON text. */
export function lineOf(text: string, key: string): number | undefined {
  const i = text.search(new RegExp(`"${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"\\s*:`))
  return i < 0 ? undefined : text.slice(0, i).split('\n').length
}

/** Line number for a JSON.parse error, from the position in the message. */
export function jsonErrorLine(text: string, err: unknown): number | undefined {
  const m = String(err instanceof Error ? err.message : err)
  const ln = m.match(/line (\d+)/)
  if (ln) return +ln[1]
  const pos = m.match(/position (\d+)/)
  return pos ? text.slice(0, +pos[1]).split('\n').length : undefined
}

/** Parses a Hugging Face config.json into a ModelSpec, reporting every field it mapped or assumed. */
export function parseConfig(text: string, nameHint?: string): ParseResult {
  const issues: Issue[] = []
  const mapping: FieldMap[] = []
  let root: unknown
  try { root = JSON.parse(text) } catch (e) {
    const line = jsonErrorLine(text, e)
    return { spec: null, issues: [{ level: 'error', message: `Not valid JSON: ${(e as Error).message.replace(/^JSON\.parse: /, '')}`, line }], mapping, format: 'unknown' }
  }
  if (!root || typeof root !== 'object' || Array.isArray(root)) return { spec: null, issues: [{ level: 'error', message: 'Expected a JSON object (the contents of config.json).', line: 1 }], mapping, format: 'unknown' }
  let o = root as Record<string, unknown>
  if (o.text_config && typeof o.text_config === 'object') {
    issues.push({ level: 'info', message: 'Multimodal config: using text_config for the language model. Vision tower weights are not counted.', line: lineOf(text, 'text_config') })
    o = { ...(o.text_config as Record<string, unknown>), model_type: (o.text_config as Record<string, unknown>).model_type ?? o.model_type, architectures: o.architectures }
  }
  const modelType = String(o.model_type ?? 'unknown')
  const arch = Array.isArray(o.architectures) && o.architectures.length ? String(o.architectures[0]) : modelType

  const num = (field: string, label: string, keys: string[], fallback?: { value: number; why: string; level?: Issue['level'] }): number => {
    for (const k of keys) {
      const v = o[k]
      if (typeof v === 'number' && Number.isFinite(v)) {
        if (v <= 0 || !Number.isInteger(v)) issues.push({ level: 'error', field, message: `${k} must be a positive integer (got ${v}).`, line: lineOf(text, k) })
        mapping.push({ field, label, value: v, source: k, line: lineOf(text, k) })
        return v
      }
      if (v !== undefined && v !== null) issues.push({ level: 'error', field, message: `${k} should be a number (got ${JSON.stringify(v)}).`, line: lineOf(text, k) })
    }
    if (fallback) {
      mapping.push({ field, label, value: fallback.value, source: `assumed: ${fallback.why}` })
      issues.push({ level: fallback.level ?? 'warn', field, message: `${label}: ${keys[0]} missing, assumed ${fallback.value} (${fallback.why}).` })
      return fallback.value
    }
    issues.push({ level: 'error', field, message: `${label}: none of ${keys.join(', ')} found.` })
    return NaN
  }

  const hidden = num('hidden', 'Hidden size', ['hidden_size', 'n_embd', 'd_model', 'dim'])
  const layers = num('layers', 'Layers', ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers'])
  const heads = num('heads', 'Attention heads', ['num_attention_heads', 'n_head', 'n_heads'])
  const kvHeads = num('kvHeads', 'KV heads', ['num_key_value_heads', 'n_head_kv', 'num_kv_heads', 'multi_query_group_num'], { value: heads, why: 'no GQA field, so full multi-head attention' })
  const isMla = typeof o.kv_lora_rank === 'number'
  const headDim = isMla
    ? Number(o.qk_nope_head_dim ?? 0) + Number(o.qk_rope_head_dim ?? 0)
    : num('headDim', 'Head dim', ['head_dim', 'kv_channels'], { value: Math.round(hidden / heads), why: 'hidden size / heads', level: 'info' })
  const intermediate = num('intermediate', 'MLP intermediate', ['intermediate_size', 'n_inner', 'ffn_dim', 'ffn_hidden_size'], { value: 4 * hidden, why: '4 x hidden size' })
  const vocab = num('vocab', 'Vocabulary', ['vocab_size', 'padded_vocab_size'])
  const maxContext = num('maxContext', 'Max context', ['max_position_embeddings', 'n_positions', 'seq_length', 'max_sequence_length'], { value: 4096, why: 'no position limit in the config' })

  const tiedRaw = o.tie_word_embeddings
  const tied = typeof tiedRaw === 'boolean' ? tiedRaw : modelType === 'gpt2'
  mapping.push({ field: 'tied', label: 'Tied embeddings', value: tied, source: typeof tiedRaw === 'boolean' ? 'tie_word_embeddings' : 'assumed from model_type', line: lineOf(text, 'tie_word_embeddings') })

  const gated = !PLAIN_MLP.has(modelType)
  mapping.push({ field: 'gated', label: 'Gated MLP (3 matrices)', value: gated, source: `model_type ${modelType}`, line: lineOf(text, 'model_type') })
  const qkvBias = o.attention_bias === true || (o.attention_bias === undefined && QKV_BIAS.has(modelType))
  const qkNorm = QK_NORM.has(modelType)

  // Mixture of experts
  let moe: ModelSpec['moe']
  const expertKey = ['num_local_experts', 'num_experts', 'n_routed_experts', 'moe_num_experts'].find(k => typeof o[k] === 'number' && (o[k] as number) > 1)
  if (expertKey) {
    const experts = num('moe.experts', 'Experts', [expertKey])
    const active = num('moe.active', 'Active experts per token', ['num_experts_per_tok', 'num_selected_experts', 'moe_topk', 'top_k'], { value: 2, why: 'common top-2 routing' })
    const expertInter = num('moe.expertInter', 'Expert intermediate', ['moe_intermediate_size', 'expert_intermediate_size'], { value: intermediate, why: 'same as intermediate_size', level: 'info' })
    let shared = 0, sharedInter = 0
    if (typeof o.n_shared_experts === 'number') { shared = num('moe.shared', 'Shared experts', ['n_shared_experts']); sharedInter = expertInter }
    else if (typeof o.shared_expert_intermediate_size === 'number' && o.shared_expert_intermediate_size > 0) { shared = 1; sharedInter = num('moe.sharedInter', 'Shared expert intermediate', ['shared_expert_intermediate_size']) }
    let denseLayers = 0
    if (typeof o.first_k_dense_replace === 'number') denseLayers = num('moe.denseLayers', 'Leading dense layers', ['first_k_dense_replace'])
    else if (Array.isArray(o.mlp_only_layers)) denseLayers = o.mlp_only_layers.length
    if (typeof o.decoder_sparse_step === 'number' && o.decoder_sparse_step > 1) issues.push({ level: 'warn', message: `decoder_sparse_step is ${o.decoder_sparse_step}: only every ${o.decoder_sparse_step}th layer is MoE. Treated as all-MoE, so totals run high.`, line: lineOf(text, 'decoder_sparse_step') })
    if (active > experts) issues.push({ level: 'error', field: 'moe.active', message: `Active experts (${active}) exceeds experts (${experts}).`, line: lineOf(text, 'num_experts_per_tok') })
    moe = { experts, active, expertInter, shared, sharedInter, denseLayers }
  }

  // Multi-head latent attention (DeepSeek V2/V3, Kimi K2)
  let mla: ModelSpec['mla']
  if (typeof o.kv_lora_rank === 'number') {
    mla = {
      qLoraRank: typeof o.q_lora_rank === 'number' ? o.q_lora_rank : 0,
      kvLoraRank: num('mla.kvLoraRank', 'KV latent rank', ['kv_lora_rank']),
      qkNope: num('mla.qkNope', 'QK no-RoPE head dim', ['qk_nope_head_dim']),
      qkRope: num('mla.qkRope', 'QK RoPE head dim', ['qk_rope_head_dim']),
      vHead: num('mla.vHead', 'V head dim', ['v_head_dim']),
    }
    issues.push({ level: 'info', message: 'Multi-head latent attention: the KV cache stores one compressed latent per layer, not per-head K and V.', line: lineOf(text, 'kv_lora_rank') })
  }

  if (Number.isFinite(heads) && Number.isFinite(kvHeads) && heads % kvHeads !== 0) issues.push({ level: 'error', field: 'kvHeads', message: `Attention heads (${heads}) must be a multiple of KV heads (${kvHeads}).`, line: lineOf(text, 'num_key_value_heads') })
  if (!mla && o.head_dim === undefined && Number.isFinite(hidden) && hidden % heads !== 0) issues.push({ level: 'warn', field: 'headDim', message: `hidden_size ${hidden} is not divisible by ${heads} heads; head_dim rounded.` })
  if (typeof o.sliding_window === 'number' && o.use_sliding_window !== false) issues.push({ level: 'info', message: `Sliding window of ${o.sliding_window} tokens declared. KV is sized for full attention, which is the upper bound.`, line: lineOf(text, 'sliding_window') })

  let suggestedDtype: WeightDtype | undefined
  const q = o.quantization_config as Record<string, unknown> | undefined
  if (q && typeof q === 'object') {
    const method = String(q.quant_method ?? '')
    const bits = Number(q.bits ?? q.w_bit ?? 0)
    if (method === 'fp8') suggestedDtype = 'fp8'
    else if (bits === 4 || /awq|gptq|bnb|mxfp4/.test(method)) suggestedDtype = 'q4'
    else if (bits === 8) suggestedDtype = 'int8'
    issues.push({ level: 'info', message: `Checkpoint is quantised (${method || 'unknown method'}${bits ? `, ${bits}-bit` : ''}).${suggestedDtype ? ` Weight dtype set to ${suggestedDtype}.` : ''}`, line: lineOf(text, 'quantization_config') })
  } else if (o.torch_dtype === 'bfloat16' || o.torch_dtype === 'float16') suggestedDtype = 'bf16'

  const name = nameHint || (typeof o._name_or_path === 'string' && o._name_or_path ? o._name_or_path : arch)
  if (issues.some(i => i.level === 'error')) return { spec: null, issues, mapping, suggestedDtype, format: `config.json (${modelType})` }
  return {
    spec: { name, arch: modelType, hidden, layers, heads, kvHeads, headDim, intermediate, vocab, tied, maxContext, gated, qkvBias, qkNorm, moe, mla },
    issues, mapping, suggestedDtype, format: `config.json (${modelType})`,
  }
}

export interface ParamCounts {
  total: number
  /** Parameters touched per token (all of a dense model; routed experts only for MoE). */
  active: number
  embedding: number
  head: number
  attnPerLayer: number
  /** Total parameters in transformer layers (excludes embedding, head, final norm). */
  layerTotal: number
  /** Bytes-relevant split for MoE: per-expert parameters in one MoE layer. */
  perExpert: number
  moeLayers: number
}

export function attnParams(m: ModelSpec): number {
  const h = m.hidden
  if (m.mla) {
    const { qLoraRank, kvLoraRank, qkNope, qkRope, vHead } = m.mla
    const qHead = qkNope + qkRope
    const q = qLoraRank > 0 ? h * qLoraRank + qLoraRank + qLoraRank * m.heads * qHead : h * m.heads * qHead
    const kvA = h * (kvLoraRank + qkRope) + kvLoraRank
    const kvB = kvLoraRank * m.heads * (qkNope + vHead)
    return q + kvA + kvB + m.heads * vHead * h
  }
  const qd = m.heads * m.headDim, kvd = m.kvHeads * m.headDim
  return h * qd + 2 * h * kvd + qd * h + (m.qkvBias ? qd + 2 * kvd : 0) + (m.qkNorm ? 2 * m.headDim : 0)
}

export function paramCounts(m: ModelSpec): ParamCounts {
  const h = m.hidden
  const embedding = m.vocab * h
  const head = m.tied ? 0 : m.vocab * h
  const attn = attnParams(m)
  const norms = 2 * h
  const dense = (m.gated ? 3 : 2) * h * m.intermediate
  let layerTotal: number, layerActive: number, perExpert = 0, moeLayers = 0
  if (m.moe) {
    const e = m.moe
    perExpert = 3 * h * e.expertInter
    const shared = e.shared * 3 * h * e.sharedInter
    const router = h * e.experts
    moeLayers = Math.max(0, m.layers - e.denseLayers)
    const denseL = m.layers - moeLayers
    layerTotal = m.layers * (attn + norms) + denseL * dense + moeLayers * (e.experts * perExpert + shared + router)
    layerActive = m.layers * (attn + norms) + denseL * dense + moeLayers * (e.active * perExpert + shared + router)
  } else {
    layerTotal = layerActive = m.layers * (attn + norms + dense)
  }
  return {
    total: embedding + head + layerTotal + h,
    active: embedding + head + layerActive + h,
    embedding, head, attnPerLayer: attn, layerTotal, perExpert, moeLayers,
  }
}

export const KV_BYTES: Record<KvDtype, number> = { bf16: 2, fp8: 1, bfp8: 8.5 / 8 }

/** KV-cache bytes for one token of one sequence across all layers. */
export function kvBytesPerToken(m: ModelSpec, dtype: KvDtype): number {
  const b = KV_BYTES[dtype]
  if (m.mla) return m.layers * (m.mla.kvLoraRank + m.mla.qkRope) * b
  return 2 * m.layers * m.kvHeads * m.headDim * b
}

/** Manual entry: a dense GQA decoder from a handful of numbers. */
export function manualSpec(p: { name: string; hidden: number; layers: number; heads: number; kvHeads: number; intermediate: number; vocab: number; maxContext: number; experts?: number; activeExperts?: number; expertInter?: number }): ModelSpec {
  return {
    name: p.name, arch: 'manual', hidden: p.hidden, layers: p.layers, heads: p.heads, kvHeads: p.kvHeads, headDim: Math.round(p.hidden / p.heads),
    intermediate: p.intermediate, vocab: p.vocab, tied: false, maxContext: p.maxContext, gated: true, qkvBias: false, qkNorm: false,
    moe: p.experts && p.experts > 1 ? { experts: p.experts, active: p.activeExperts || 2, expertInter: p.expertInter || p.intermediate, shared: 0, sharedInter: 0, denseLayers: 0 } : undefined,
  }
}

/** Builds a config.json text from a spec, so manual models round-trip through the same importer. */
export function specToConfig(m: ModelSpec): string {
  const o: Record<string, unknown> = {
    _name_or_path: m.name, model_type: m.arch === 'manual' ? 'llama' : m.arch, hidden_size: m.hidden, num_hidden_layers: m.layers, num_attention_heads: m.heads,
    num_key_value_heads: m.kvHeads, head_dim: m.headDim, intermediate_size: m.intermediate, vocab_size: m.vocab, max_position_embeddings: m.maxContext,
    tie_word_embeddings: m.tied,
  }
  if (m.moe) Object.assign(o, { num_local_experts: m.moe.experts, num_experts_per_tok: m.moe.active, moe_intermediate_size: m.moe.expertInter })
  return JSON.stringify(o, null, 2)
}
