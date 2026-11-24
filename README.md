# Fit Planner

Plan an LLM deployment across accelerator meshes before you buy or rent the cards.

Pick a model (or paste / fetch its Hugging Face `config.json`) and the hardware you are considering. Fit Planner works out whether it fits, how to split it with tensor and pipeline parallelism, how much KV cache is left, and roofline estimates for decode and prefill speed, time to first token, $/1M tokens and tokens per joule. A search ranks every device and mesh size by hourly cost for your throughput and latency targets.

**Live:** https://fit-planner-xi.vercel.app

## The pain

"Will a 70B fit on four 32 GB cards? At what context? Is 400 tok/s per user or aggregate? Ethernet mesh or PCIe?" Every hardware thread asks this, and the answer is arithmetic over the model's shape, your workload and the interconnect.

## Features

- Import `config.json` by paste, file, or Hugging Face repo id; manual entry. Validation report with line numbers and a field mapping. Handles GQA, MoE (incl. shared experts and leading dense layers), multi-head latent attention, `text_config`, quantisation hints.
- Engine (plain TS, 43 tests): parameter counts checked against published sizes and against the Hub safetensors totals of 13 real config.json files (Falcon, BLOOM, Gemma 2, Mistral, Phi-3.5-MoE, Qwen MoE, OLMoE, DeepSeek V2, Kimi K2, gpt-oss, Llama 4...), sliding-window KV, weights by dtype incl. BFP8/BFP4, KV bytes per token, per-device memory for TP x PP, max batch and context, roofline decode and prefill with ring all-reduce over the interconnect, MoE expert-touch model, throughput curve, cost and energy, config search with data-parallel replicas.
- Curated, dated hardware table (Tenstorrent Wormhole / Blackhole, NVIDIA, AMD), editable per plan, plus custom devices.
- Saved snapshots with side-by-side compare, printable comparison report, plan JSON share/import, CSV export.
- Local-first (IndexedDB); optional email sign-in syncs plans via Supabase. Light and dark themes, keyboard shortcuts, responsive.

Samples: nine public architectures copied from their config.json files, plus synthetic workload profiles (labelled).

## Run locally

```
npm install
npm run dev      # http://localhost:5173
npm test         # engine tests
npm run build
npm run test:e2e # browser tests against the production build (see below)
```

Copy `.env` with `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY` to enable sync; without it the app runs local-only.

## Tests

- `npm test`: engine unit tests (`node --test`).
- `npm run test:e2e`: builds the app, serves it with `vite preview` and runs the Playwright (Chromium) suite in `e2e/`. One-time setup: `npx playwright install chromium` (on Linux add `--with-deps`). The server listens on `127.0.0.1:4173`; set `E2E_PORT` to use another port (`E2E_PORT=4510 npm run test:e2e`). The port must be free, the run never attaches to an existing server. Run a single file with `npx playwright test e2e/fit-math.spec.ts` after `npm run build`.
- The suite runs signed out with a fresh browser profile per test. Every request to a host other than the preview server is aborted (Hugging Face is stubbed per test), and a test fails on any uncaught page error. It checks the memory-fit numbers against hand-worked values, config.json import (paste, file, Hub), rejected and truncated input, snapshots, compare, plan JSON export and re-import, CSV downloads, and that state survives a reload.
- A few tests are marked `test.fail()` with a link to the open bug they document; they flip to a failure when the bug is fixed, which is the cue to delete the marker.
- CI (`.github/workflows/ci.yml`, job `test`) runs `npm ci`, build, unit tests and the E2E suite on every pull request and push to `main`; failed runs upload the Playwright report.

Preview deployments are off (`vercel.json` `git.deploymentEnabled`): only pushes to `main` deploy, and pull requests are gated by CI instead.

Built by Aryan Kansagara.
