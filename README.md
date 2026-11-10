# Fit Planner

Plan an LLM deployment across accelerator meshes before you buy or rent the cards.

Pick a model (or paste / fetch its Hugging Face `config.json`) and the hardware you are considering. Fit Planner works out whether it fits, how to split it with tensor and pipeline parallelism, how much KV cache is left, and roofline estimates for decode and prefill speed, time to first token, $/1M tokens and tokens per joule. A search ranks every device and mesh size by hourly cost for your throughput and latency targets.

**Live:** https://fit-planner-xi.vercel.app

## The pain

"Will a 70B fit on four 32 GB cards? At what context? Is 400 tok/s per user or aggregate? Ethernet mesh or PCIe?" Every hardware thread asks this, and the answer is arithmetic over the model's shape, your workload and the interconnect.

## Features

- Import `config.json` by paste, file, or Hugging Face repo id; manual entry. Validation report with line numbers and a field mapping. Handles GQA, MoE (incl. shared experts and leading dense layers), multi-head latent attention, `text_config`, quantisation hints.
- Engine (plain TS, 33 tests): parameter counts checked against published sizes (Llama 3.1 8B, Mixtral, DeepSeek V3...), weights by dtype incl. BFP8/BFP4, KV bytes per token, per-device memory for TP x PP, max batch and context, roofline decode and prefill with ring all-reduce over the interconnect, MoE expert-touch model, throughput curve, cost and energy, config search with data-parallel replicas.
- Curated, dated hardware table (Tenstorrent Wormhole / Blackhole, NVIDIA, AMD), editable per plan, plus custom devices.
- Saved snapshots with side-by-side compare, printable comparison report, plan JSON share/import, CSV export.
- Local-first (IndexedDB); optional email sign-in syncs plans via Supabase. Light and dark themes, keyboard shortcuts, responsive.

Samples: eight public architectures copied from their config.json files, plus synthetic workload profiles (labelled).

## Run locally

```
npm install
npm run dev      # http://localhost:5173
npm test         # engine tests
npm run build
```

Copy `.env` with `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY` to enable sync; without it the app runs local-only.

Built by Aryan Kansagara. Independent project, not affiliated with Tenstorrent.
