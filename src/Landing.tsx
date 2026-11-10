import { useMemo, useState } from 'react'
import { DEVICES } from './engine/hardware.ts'
import { paramCounts, parseConfig } from './engine/model.ts'
import { bestLayout, DEFAULT_ASSUMPTIONS, minDevices } from './engine/plan.ts'
import { MODEL_SAMPLES } from './engine/samples.ts'
import type { WeightDtype, Workload } from './engine/types.ts'
import { fmtB, fmtGB, fmtNum, fmtTok, Link } from './ui.tsx'
import { Dial, Mesh, MeshLegend } from './viz.tsx'

const MODELS = MODEL_SAMPLES.map(s => ({ ...s, spec: parseConfig(s.config, s.name).spec! }))
  .map(s => ({ ...s, params: paramCounts(s.spec).total }))
  .sort((a, b) => a.params - b.params)
const CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
const HERO_DEVICES = ['tt-p150a', 'tt-n300s', 'tt-p100a', 'nv-4090'].map(id => DEVICES.find(d => d.id === id)!)
const DTYPE_CHOICES: [WeightDtype, string][] = [['bf16', 'BF16'], ['bfp8', 'BFP8'], ['bfp4', 'BFP4']]

function Hero() {
  const [mi, setMi] = useState(MODELS.findIndex(m => m.id === 'llama-3.3-70b'))
  const [ci, setCi] = useState(2)
  const [devId, setDevId] = useState('tt-p150a')
  const [dtype, setDtype] = useState<WeightDtype>('bfp8')
  const model = MODELS[mi]
  const device = HERO_DEVICES.find(d => d.id === devId)!
  const wl: Workload = { weightDtype: dtype, kvDtype: 'bf16', context: CONTEXTS[ci], batch: 4, promptLen: Math.min(2000, CONTEXTS[ci] / 2), outputLen: 300 }
  const r = useMemo(() => minDevices(model.spec, wl, device, DEFAULT_ASSUMPTIONS, 32) ?? bestLayout(model.spec, wl, device, 32, DEFAULT_ASSUMPTIONS), [mi, ci, devId, dtype])
  if (!r) return null
  const fill = { weightsGB: r.mem.weightsGB, kvGB: r.mem.kvGB, reserveGB: r.mem.reserveGB, capGB: r.mem.capGB }
  return (
    <div className="hero-card" aria-label="Live sizing demo">
      <div className="hc-head">
        <span className="utility">Live · real engine</span>
        <div className="seg sm" role="group" aria-label="Device">
          {HERO_DEVICES.map(d => <button key={d.id} type="button" aria-pressed={d.id === devId} onClick={() => setDevId(d.id)}>{d.name.replace('GeForce ', '').replace('Blackhole ', '').replace('Wormhole ', '')}</button>)}
        </div>
      </div>
      <div className="hc-body">
        <div className="hc-mesh">
          <Mesh tp={r.tp} pp={r.pp} fill={fill} compact maxTiles={32} />
          <p className="hc-caption">
            <strong className={r.mem.fits ? '' : 'bad-t'}>{r.mem.fits ? `${r.count}x ${device.name}` : `Does not fit on 32x ${device.name}`}</strong>
            <span>TP {r.tp} x PP {r.pp} · {fmtGB(r.mem.usedGB)} of {device.memGB} GB each</span>
          </p>
          <MeshLegend fill={fill} />
        </div>
        <Dial value={r.mem.fits ? r.decode.tokS : 0} max={Math.max(100, Math.ceil(r.decode.tokS / 100) * 150)} label="tok/s, 4 users" sub={r.mem.fits ? `${fmtNum(r.decode.perUserTokS, 1)} tok/s each · first token ${r.ttftMs < 1000 ? fmtNum(r.ttftMs) + ' ms' : fmtNum(r.ttftMs / 1000, 1) + ' s'}` : 'add cards or shrink the dtype'} />
      </div>
      <div className="hc-controls">
        <label className="slider">
          <span>Model <b>{model.name}</b> <em>{fmtB(model.params)}</em></span>
          <input type="range" min={0} max={MODELS.length - 1} value={mi} onChange={e => setMi(+e.target.value)} aria-valuetext={`${model.name}, ${fmtB(model.params)} parameters`} />
        </label>
        <label className="slider">
          <span>Context per user <b>{fmtTok(CONTEXTS[ci])} tokens</b></span>
          <input type="range" min={0} max={CONTEXTS.length - 1} value={ci} onChange={e => setCi(+e.target.value)} aria-valuetext={`${CONTEXTS[ci]} tokens`} />
        </label>
        <div className="seg sm" role="group" aria-label="Weight precision">
          {DTYPE_CHOICES.map(([k, l]) => <button key={k} type="button" aria-pressed={dtype === k} onClick={() => setDtype(k)}>{l}</button>)}
        </div>
      </div>
      <Link href={`/app?sample=${model.id}&workload=team-chat`} className="hc-open">Open this in the planner →</Link>
    </div>
  )
}

export function Landing() {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <p className="utility eyebrow-pill">LLM deployment planner</p>
          <h1>Size the mesh <em>before</em> you buy the cards.</h1>
          <p className="lede">Pick a model, or paste its config.json, and the hardware you are considering. Fit Planner works out whether it fits, how to split it across devices, how much KV cache is left, and what decode and prefill speed and cost per million tokens to expect.</p>
          <div className="row gap wrap">
            <Link href="/app?sample=llama-3.3-70b&workload=team-chat" className="btn primary lg">Try the sample</Link>
            <Link href="/app?new=1" className="btn ghost lg">Import your model</Link>
          </div>
          <p className="muted small">Free, runs in your browser, no account needed.</p>
        </div>
        <Hero />
      </section>

      <section className="band">
        <div className="container">
          <p className="utility">The questions people keep asking</p>
          <h2>Every hardware thread asks the same three things.</h2>
          <div className="evidence">
            <figure className="ev"><blockquote>Has anyone run local models on these cards? How does 512 GB/s on a 32 GB card compare, and is chaining cards over Ethernet better than PCIe?</blockquote><figcaption>r/LocalLLaMA, paraphrased</figcaption></figure>
            <figure className="ev"><blockquote>400 tokens per second on a $12,000 four-card box: is that per user or aggregate, and what would my model do?</blockquote><figcaption>r/RISCV, paraphrased</figcaption></figure>
            <figure className="ev"><blockquote>Is anyone crazy enough to try one of these for self-hosting? I need help deciding between them and a used GPU.</blockquote><figcaption>r/selfhosted, paraphrased</figcaption></figure>
            <figure className="ev"><blockquote>Now we know how Blackhole performs. Which models actually fit, and at what context?</blockquote><figcaption>r/LocalLLaMA, paraphrased</figcaption></figure>
          </div>
          <p className="muted small">Fit, speed and cost depend on the model's shape, your context and concurrency, and the interconnect. That is arithmetic you can do before spending money.</p>
        </div>
      </section>

      <section className="container steps-sec">
        <p className="utility">How it works</p>
        <h2>From config.json to a shopping list.</h2>
        <ol className="steps">
          <li><span className="step-n">01</span><h3>Bring the model</h3><p>Fetch config.json from the Hugging Face Hub by repo id, paste it, or type the dimensions. GQA, MoE and latent attention are read from the config, with every assumed field flagged by line.</p></li>
          <li><span className="step-n">02</span><h3>Describe the job</h3><p>Weight and KV precision, context per user, concurrent sequences, prompt and output length. Pick a curated device or add your own with its memory, bandwidth, TFLOPS, link and price.</p></li>
          <li><span className="step-n">03</span><h3>Search and decide</h3><p>Every device and mesh size, split tensor- and pipeline-parallel, ranked by hourly cost for your throughput and latency targets. Save snapshots, compare them, print the report.</p></li>
        </ol>
      </section>

      <section className="container bento">
        <div className="bento-item b-purple"><h3>Memory fit per device</h3><p>Weights by dtype including block FP8 and FP4, KV cache per token from layers x KV heads x head dim, runtime reserve, and the most loaded pipeline stage.</p></div>
        <div className="bento-item b-teal"><h3>Tensor and pipeline splits</h3><p>Only legal TP degrees for the head counts. KV stops shrinking once TP passes the KV heads, and the planner says so.</p></div>
        <div className="bento-item b-yellow"><h3>Roofline speed</h3><p>Decode limited by bandwidth, prefill by compute, plus ring all-reduce time over Ethernet, NVLink or PCIe. A throughput curve shows where batching stops paying.</p></div>
        <div className="bento-item b-sand"><h3>Cost and energy</h3><p>Dollars per million tokens from amortised price and power or hourly rental, and tokens per joule at board power.</p></div>
        <div className="bento-item b-pink"><h3>Mixture of experts, done right</h3><p>All experts resident, but decode reads only the experts a batch actually routes to. DeepSeek-style latent KV is sized as a latent.</p></div>
        <div className="bento-item b-sand"><h3>Yours, everywhere</h3><p>Plans live in your browser. Sign in to sync them across machines, export everything as JSON, or delete it all.</p></div>
      </section>

      <section className="container">
        <div className="cta-band">
          <h2>What does your model need?</h2>
          <p>Eight public architectures from 3B to 671B are ready to plan, or bring your own config.json.</p>
          <div className="row gap center-x wrap"><Link href="/app" className="btn primary lg">Open the planner</Link><Link href="/docs" className="btn ghost lg">Read the method</Link></div>
        </div>
      </section>
    </>
  )
}
