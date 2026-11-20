# inkling-metal: scratch plan

speed up local inference of thinkingmachines' inkling-2-mini (26.2B MoE,
3.4B active, middle 12 of 24 layers looped 3x, hybrid local/global attention)
on an m2 macbook air, 24GB. llama.cpp edits are fair game. just ideas, not
settled; measure before trusting any of it.

## current setup (working, stable)

- weights: `unsloth/Inkling-2-Mini-GGUF` UD-Q3_K_XL (13.6GB, top-1 96.3% vs
  Q8_K_XL), in the HF cache. IQ4_XS, dspark drafter and mmproj were deleted.
- engine: unsloth llama.cpp prebuilt `b12568-mix-8eebad4` (carries the looped
  layer + dspark pins; stock llama.cpp can't run it) in
  `~/.llama-app/unsloth-b12568/`.
- served by the Llama menu bar app (router on `127.0.0.1:9931`). the app
  checks `/opt/homebrew/bin/llama` first and overwrites `~/.llama-app/llama`
  on launch, so brew's llama.cpp is unlinked and `/opt/homebrew/bin/llama`
  is a symlink to the unsloth build. `brew link llama.cpp` undoes it.
- settings: `~/.config/llama/models.user.ini`, section
  `[unsloth/Inkling-2-Mini-GGUF:Q3_K_XL]`: ctx 32768, ubatch 128, parallel 1,
  cache-ram 512, ctx-checkpoints 2, spec-type none. ~14.8GB on device.
- hax preset `inkling` in `~/.config/hax/config.json`.

## hazards

- the first attempt (IQ4_XS + mmproj + drafter, 64k ctx, default 8GB
  cache-ram) froze the machine and needed a hard reboot. swap shows 0MB and
  `iogpu.wired_limit_mb` is 22528, so overcommit hangs instead of slowing.
- always run experiments under a watchdog: poll
  `sysctl -n kern.memorystatus_vm_pressure_level` (4 = critical) and
  `memory_pressure` free %, and `kill -9` the server before it gets there.
  a 32k/ub128 run peaks at level 2 (warn), ~11-13% free.
- use `llama fit-params -m ... -c N -ub N -fit off -lv 4` to read the
  projected memory breakdown without loading anything.

## observations so far

- decode ~12.3 tok/s, small prompts only (no real pp numbers yet).
- rough roofline: ~5.4B effective params/token at ~3.5 bits is ~2.4GB/token;
  at ~100GB/s that's ~40 tok/s. we're at ~30%, so overhead-bound.
- compute buffer is O(ubatch x ctx): 1.8GB @16k, 6.6GB @64k (ub 512);
  0.85GB @32k ub 128. `-fa on` changes nothing, so attention isn't hitting
  the flash attention path.
- dspark drafter (n-max 3): 52% acceptance but 9.5 tok/s vs 12.3 without.

## ideas, roughly in order

1. baseline. `llama-bench` pp512/tg128 at depth 0/8k/32k; verbose logs for
   graph splits and CPU fallbacks; xcode metal capture of one decode step
   for a per-kernel breakdown. sets the order of everything else.
2. attention -> flash attention. suspects: extra inkling tensors
   (`inkling_rel_idx`, `inkling_tau`, `inkling_attn_sconv`), i.e. a relative
   position bias, temperature and short conv. fold bias into the FA mask and
   tau into the scale, or add a metal FA variant. wins: constant compute
   buffer, ub 512 back (faster prefill), 128k ctx in budget, flat decode at
   depth, and quantized KV (q8_0 needs FA).
3. drafter verify cost. MoE verify at batch 4 touches ~4x experts. bench
   pp2..pp4 vs tg; tune metal `mul_mat_id` for tiny batches; try
   `--spec-draft-n-max 1/2`. 52% acceptance should pay if verify is cheap.
4. loop overhead. check the graph isn't duplicating work per loop pass
   (per-pass scale copies, rebuilt views); confirm metal fusions fire for
   this arch (rms_norm+mul, moe gate/up, shared expert add). compare against
   upstream ggml-org/llama.cpp#32694 (`inkling2_loop` layout, unsloth's
   `Shard_Rewrite/` files) which needs the loop-pass scale re-share fix.
5. quant mix, only once bandwidth-bound: attention/shared tensors Q4-Q6,
   routed experts Q3; K-quants over IQ on metal. compare KLD and tg against
   UD-Q3_K_XL.
6. housekeeping: build the fork from source here and pin it, instead of the
   brew-slot symlink; turn swap back on (needs sudo) so mistakes degrade
   rather than freeze.
