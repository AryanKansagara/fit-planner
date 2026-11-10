import { DEVICES, HARDWARE_AS_OF } from './engine/hardware.ts'
import { DEFAULT_ASSUMPTIONS as A } from './engine/plan.ts'
import { DTYPES } from './engine/types.ts'
import { Code, fmtNum, fmtUsd, Link } from './ui.tsx'

const SECTIONS = [['config', 'Getting config.json'], ['fields', 'What is read from it'], ['hardware', 'Hardware profiles'], ['method', 'How it is calculated'], ['plans', 'Plan files and CSV'], ['data', 'Your data and sync'], ['limits', 'Limits']] as const

export default function Docs() {
  return (
    <section className="page docs">
      <nav className="docs-nav" aria-label="Docs sections">
        <p className="utility">Docs</p>
        {SECTIONS.map(([id, l]) => <a key={id} href={`#${id}`}>{l}</a>)}
      </nav>
      <div className="docs-body">
        <h1>Using Fit Planner</h1>
        <p className="lede">Everything the planner needs is in a model's config.json and a hardware spec sheet. This page explains where to get both, and exactly what is done with them.</p>

        <h2 id="config">Getting config.json</h2>
        <p>Every Transformers-format model on the Hugging Face Hub ships a <code>config.json</code> next to its weights. You do not need the weights.</p>
        <h3>Fetch it in the app</h3>
        <p>On a plan's Model tab choose <b>Hugging Face</b> and enter a repo id such as <code>Qwen/Qwen3-32B</code> or paste the model page URL. Your browser downloads <code>https://huggingface.co/&lt;repo&gt;/resolve/main/config.json</code> directly; nothing goes through a server of ours.</p>
        <h3>Gated models (Llama, Gemma, some Mistral)</h3>
        <p>Gated repos return 401 without a token, and the app never asks for one. Either accept the license on the model page and download the file, or use an ungated mirror with the same architecture (for Llama, the <code>unsloth/</code> copies). Then paste it on the Model tab.</p>
        <Code label="shell" code={`# with a Hugging Face token that has accepted the license
curl -s -H "Authorization: Bearer $HF_TOKEN" \\
  https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/resolve/main/config.json > config.json

# or with the CLI
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct config.json --local-dir .`} />
        <h3>GGUF, llama.cpp and Ollama</h3>
        <p>GGUF-only repos have no config.json. Use the original model's repo for the architecture, then pick the precision you will run: <b>4-bit (Q4_K / AWQ)</b> is costed at 4.5 bits per weight, close to Q4_K_M. For Ollama, <code>ollama show &lt;model&gt; --modelfile</code> names the base model.</p>
        <h3>Multimodal and local files</h3>
        <p>Configs with a <code>text_config</code> block (Llama 4, Gemma 3, LLaVA) are read from that block; the vision tower is not counted. Local fine-tunes keep the base model's config.json in the output folder: open that file with <b>Paste config.json &gt; Open a file</b>.</p>

        <h2 id="fields">What is read from it</h2>
        <div className="table-card card scroll-x">
          <table className="vt compact">
            <thead><tr><th scope="col">Planner field</th><th scope="col">config.json keys (first found wins)</th><th scope="col">If missing</th></tr></thead>
            <tbody>
              <tr><th scope="row">Hidden size</th><td className="mono">hidden_size, n_embd, d_model, dim</td><td>error</td></tr>
              <tr><th scope="row">Layers</th><td className="mono">num_hidden_layers, n_layer, num_layers</td><td>error</td></tr>
              <tr><th scope="row">Attention heads</th><td className="mono">num_attention_heads, n_head</td><td>error</td></tr>
              <tr><th scope="row">KV heads</th><td className="mono">num_key_value_heads, n_head_kv, num_kv_heads</td><td>warning, assumes full multi-head attention</td></tr>
              <tr><th scope="row">Head dim</th><td className="mono">head_dim, kv_channels</td><td>hidden / heads</td></tr>
              <tr><th scope="row">MLP width</th><td className="mono">intermediate_size, n_inner, ffn_dim</td><td>warning, 4 x hidden</td></tr>
              <tr><th scope="row">Vocabulary, context</th><td className="mono">vocab_size; max_position_embeddings, n_positions</td><td>error; 4,096</td></tr>
              <tr><th scope="row">Experts</th><td className="mono">num_local_experts, num_experts, n_routed_experts; num_experts_per_tok; moe_intermediate_size; n_shared_experts; first_k_dense_replace</td><td>dense model</td></tr>
              <tr><th scope="row">Latent attention</th><td className="mono">kv_lora_rank, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim</td><td>standard attention</td></tr>
              <tr><th scope="row">Precision hint</th><td className="mono">quantization_config, torch_dtype</td><td>BF16</td></tr>
            </tbody>
          </table>
        </div>
        <p>The validation report lists every error and assumption with its line number, and the field mapping shows which key each number came from. Configs with errors cannot be used until fixed.</p>

        <h2 id="hardware">Hardware profiles (as of {HARDWARE_AS_OF})</h2>
        <p>Figures come from public vendor spec sheets and list or typical street prices. TFLOPS are dense (no 2:4 sparsity). Interconnect is the per-device bandwidth realistically available to collectives: Ethernet port totals for Tenstorrent cards, one direction of NVLink, and a practical PCIe number for cards without a direct link. Treat entries marked approximate as starting points and edit them on the Hardware tab; edits stay with the plan.</p>
        <div className="table-card card scroll-x">
          <table className="vt compact">
            <thead><tr><th scope="col">Device</th><th scope="col" className="num">GB</th><th scope="col" className="num">GB/s</th><th scope="col" className="num">FP16 / FP8</th><th scope="col">Link</th><th scope="col" className="num">Price</th><th scope="col" className="num">W</th></tr></thead>
            <tbody>{DEVICES.map(d => <tr key={d.id}><th scope="row">{d.vendor} {d.name}</th><td className="num">{d.memGB}</td><td className="num">{fmtNum(d.bwGBs)}</td><td className="num">{d.tflops.fp16} / {d.tflops.fp8 ?? '-'}</td><td className="small">{d.linkKind}, {d.linkGBs} GB/s</td><td className="num">{fmtUsd(d.priceUsd)}{d.rentPerHour ? ` · ${fmtUsd(d.rentPerHour, 2)}/h` : ''}</td><td className="num">{d.watts}</td></tr>)}</tbody>
          </table>
        </div>
        <p>A Wormhole n300 carries two chips; it is modelled as one 24 GB device, which slightly flatters it because the on-card link is not charged. Four Blackhole p150 cards approximate a QuietBox-class desktop.</p>

        <h2 id="method">How it is calculated</h2>
        <h3>Weights and KV cache</h3>
        <ul>
          <li>Parameters: embedding (vocab x hidden), output head unless tied, and per layer the attention projections (Q, K, V, O, plus biases or QK-norm where the architecture has them), two norms, and a gated MLP of 3 x hidden x intermediate. MoE layers hold every expert plus shared experts and the router. Checked against published sizes: Llama 3.1 8B is 8.03B, Mixtral 8x7B is 46.7B (12.9B active), DeepSeek V3 is 671B (37B active).</li>
          <li>Bytes per weight: {Object.values(DTYPES).map(d => `${d.label} ${+d.bytes.toFixed(3)}`).join(', ')}. Block formats share one 8-bit exponent across 16 values.</li>
          <li>KV bytes per token = 2 x layers x KV heads x head dim x bytes. Latent attention stores (kv_lora_rank + qk_rope_head_dim) per layer instead.</li>
          <li>Per device: the heaviest pipeline stage gets ceil(layers / PP) layers plus the embedding or head, divided by TP. KV divides by min(TP, KV heads); past that, heads are replicated. Each device keeps {A.reserveGB} GB plus {A.reserveFrac * 100}% free by default.</li>
        </ul>
        <h3>Speed</h3>
        <ul>
          <li>Decode step = max(FLOPs / (peak x MFU), bytes / (bandwidth x utilisation)) + communication. Bytes are the non-expert weights, the head, the experts the batch is expected to touch (E x (1 - (1 - k/E)^batch)), and every sequence's KV cache at the average length prompt + output/2.</li>
          <li>Prefill of P tokens = 2 x active params x P plus causal attention (layers x P² x per-head work / 2), compute-bound for any real prompt. Time to first token = prefill + one decode step.</li>
          <li>Tensor parallel adds two ring all-reduces per layer: 2(n-1)/n x tokens x hidden x 2 bytes over the link, plus 2(n-1) hops of link latency. Pipeline stages run one after another (no microbatch interleaving), so PP saves memory but not decode latency.</li>
          <li>FP8-class weights (FP8, INT8, BFP8, BFP4) use the device's FP8 rate when it has one; 4-bit GGUF/AWQ weights are dequantised and run at FP16.</li>
          <li>Defaults: {A.bwUtil * 100}% of bandwidth, {A.mfu * 100}% MFU, {A.linkUtil * 100}% of link bandwidth. Change them per plan.</li>
        </ul>
        <h3>Cost, energy and search</h3>
        <ul>
          <li>Owned: price / (years x 8,760 h) + board watts x $/kWh. Rented: the listed hourly rate, falling back to owned where none is listed. $/1M tokens = hourly cost / (decode tok/s x 3,600) x 10⁶. Tokens per joule = decode tok/s / total board watts.</li>
          <li>Search tries each device at 1, 2, 4, 8... cards up to its mesh limit, picks the fastest legal TP x PP split that fits, then adds data-parallel replicas (up to 8, sharing the concurrent sequences) until the throughput target is met. Options meeting every target are ranked by hourly cost.</li>
        </ul>

        <h2 id="plans">Plan files and CSV</h2>
        <p><b>Share plan &gt; Download JSON</b> writes everything needed to reproduce a plan. <b>Plans &gt; Import plan JSON</b> opens one as a new plan. The format:</p>
        <Code label="plan.json" code={`{
  "format": "fit-planner/plan@1",
  "exportedAt": "2026-11-10T12:00:00.000Z",
  "plan": {
    "name": "Llama 3.3 70B on Blackhole",
    "model": { "kind": "hf", "repo": "unsloth/Llama-3.3-70B-Instruct", "config": "{ ...config.json text... }" },
    "workload": { "weightDtype": "bfp8", "kvDtype": "bf16", "context": 8192, "batch": 8, "promptLen": 2000, "outputLen": 300 },
    "targets": { "minTokS": 160, "maxTpotMs": 50, "maxTtftMs": 1500 },
    "assumptions": { "bwUtil": 0.75, "mfu": 0.45, "linkUtil": 0.7, "reserveGB": 1.5, "reserveFrac": 0.05, "amortYears": 3, "kwhUsd": 0.15, "costMode": "own" },
    "deviceId": "tt-p150a", "tp": 4, "pp": 1,
    "devices": []
  },
  "snapshots": []
}`} />
        <p>The Search tab and report export CSV with one row per option: device, cards, TP, PP, replicas, aggregate tok/s, ms per token, TTFT, $/hour, $/1M tokens, hardware cost, tok/J, memory used, and why it missed if it did. The Hardware tab exports the device table as CSV.</p>

        <h2 id="data">Your data and sync</h2>
        <p>Plans are stored in your browser's IndexedDB and work fully offline once loaded. Signing in (email and password, optional) syncs plans, hardware edits and snapshots to a database where row-level security limits each account to its own rows. When both copies changed, the most recent edit wins. Pasted configs over 1.5 MB stay on the device; the plan still syncs. The account menu exports everything as JSON, deletes your cloud copy, or deletes the account.</p>

        <h2 id="limits">Limits</h2>
        <ul>
          <li>This is a roofline model. Kernel efficiency, scheduler overhead, speculative decoding, paged-attention fragmentation and prefix caching move real numbers by tens of percent. Calibrate the utilisation knobs with one measured run of your own.</li>
          <li>Sliding-window and hybrid attention are sized as full attention, an upper bound on KV.</li>
          <li>Prefill and decode are costed separately; continuous batching that mixes them is not simulated.</li>
          <li>Expert parallelism is modelled as tensor-parallel experts; all-to-all routing traffic is not costed.</li>
        </ul>
        <p><Link href="/app" className="btn primary">Open the planner</Link></p>
      </div>
    </section>
  )
}
