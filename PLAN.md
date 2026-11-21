# inkling-metal

local inference of thinkingmachines' inkling-2-mini (26.2B MoE, 3.4B active,
middle 12 of 24 layers looped 3x, hybrid local/global attention) on an m2
macbook air, 24GB, through a patched llama.cpp.

## current setup

- weights: `unsloth/Inkling-2-Mini-GGUF` UD-Q3_K_XL (13.6GB), in the HF cache.
  expert tensors of 20 layers are NVFP4 with per-expert scales.
- engine: built here by `./build`: ggml-org/llama.cpp `b12568` + the 30
  unsloth pins in `pins.txt` (merged in order, as their CI does) +
  `patches/`. installed in `~/.llama-app/inkling-metal/`, and
  `/opt/homebrew/bin/llama` (what the Llama menu bar app runs) links to it.
  the unsloth prebuilt is still in `~/.llama-app/unsloth-b12568/`; relink to
  go back. `brew link llama.cpp` restores brew's.
- `./build` was run here from the build step on; its clone-and-merge
  branch is the same sequence that made `src-mix` but was not re-run.
- the installed prebuilt `b12568-mix-8eebad4` was 28 of those pins: it has no
  chunked (#385) or banded (#386) attention. `git -C src-mix diff
  mix-8eebad4 mix` shows the difference.
- settings: `~/.config/llama/models.user.ini`, section
  `[unsloth/Inkling-2-Mini-GGUF:Q3_K_XL]`: ctx 102400, ubatch 256,
  flash-attn on, parallel 1, cache-ram 512, ctx-checkpoints 2, spec-type
  none. ~16.6GB on device (12.7 model, 3.4 f16 KV, 0.5 compute).
- hax preset `inkling` in `~/.config/hax/config.json`.

## what changed and what it bought

measured on this machine, one run each, nothing else loaded.

| | prebuilt, 32k | now, 100k |
| --- | --- | --- |
| prefill, 512 tokens | 85 t/s (ub 128) | 121 t/s (ub 256), 126 (ub 512) |
| decode, short context | 12.7 t/s (11.6 via app) | 14.4 t/s (15.7 via app) |
| device memory, projected | 14.8GB | 16.6GB |
| same settings at 100k | 22.2GB (does not fit) | 16.6GB |

- `patches/0001`: metal kernel for `FLASH_ATTN_EXT_BANDED`. the relative
  position bias and the per-query sink go into the half8x8 flash attention
  kernel, so prefill no longer builds an n_kv x n_ubatch x n_head bias. the
  compute buffer stops growing with context. also takes a q8_0 KV cache.
  `test-backend-ops -o FLASH_ATTN_EXT_BANDED -b MTL0`: 46/46 against CPU.
- `patches/0002`: NVFP4 expert mat-vec kernel wasted 25-44% of its lane
  iterations on this model's row widths. decode 11.8 -> 13.8 t/s in a
  back-to-back pair.
- `patches/0006-0008, 0010`: fewer graph nodes per decoded token (12.6k ->
  11.2k) in the bias path, `mix()` and `clip()`. GPU time 65.8 -> 61.5 ms
  per token. the `mix()` and `clip()` rewrites are algebraically the same
  but round differently: on a 2.3k-token prompt the temp-0 text diverges
  after 15 tokens, and perplexity on 4 chunks moves 4.580 -> 4.558. drop
  0007 and 0008 (about 3.5 ms per token) for strict identity.
- `patches/0003-0005, 0009`: perf cases and env-gated profiling, inert unless set.
  `GGML_METAL_PROF=1` prints encode/gpu/wall per graph, `=2` a per-op table,
  `GGML_METAL_PROF_SKIP=",MUL_MAT,"` skips op types for ablation.
- temp-0 output of the final build is byte-identical to the prebuilt on a
  390-token prompt + 48 tokens. a code planted 400 tokens into a 31k-token
  prompt is retrieved at 100k context.

## depth

- decode slows with cache depth. at 8.3k tokens deep: f16 KV 12.4 t/s, q8_0
  KV 11.0 t/s (prefill to that depth averaged 89 and 79 t/s). at 2k deep
  the extra 7 ms per token is half flash attention, half the dense
  bias and mask plumbing around it. the 8
  full-attention layer visits read the whole cache for every token, so this
  is partly inherent: 100k of f16 KV is 3.2GB per token.
- not measured cleanly beyond 8k. one 31k run with q8_0 KV gave 59 t/s
  average prefill and 5.9 t/s decode, but a build was running beside it.
- q8_0 KV (`cache-type-k = q8_0`, `cache-type-v = q8_0`) brings 100k down to
  15.0GB with the same text in tests, at the cost above.

## hazards

- overcommit froze the machine once (swap shows 0MB, `iogpu.wired_limit_mb`
  is 22528). two loaded models is enough: metal reports `Insufficient
  Memory` and every request fails with `Compute error`.
- run every experiment through `./wd cmd...`: it takes a machine-wide lock
  and kills the command if `kern.memorystatus_vm_pressure_level` reaches 4
  or free memory drops below `$MINFREE` % (default 6). the lock does not
  cover the app: unload its model first
  (`curl 127.0.0.1:9931/models/unload -d '{"model":"..."}'`).
- `lockf` needs `-k`; without it the lock file is unlinked on release and
  two holders get through.
- at 100k/ub256 the app peaked at pressure level 1, 17% free on a short
  request. a full 100k context has not been run.
- `llama fit-params -m ... -c N -ub N -fa on -fit off -lv 4` reads the
  projected memory breakdown without loading anything.
- `llama-bench -d N` cannot restore state with a q8_0 KV cache
  (`failed to restore kv cache`); its depth numbers are then depth 0.

## where decode time goes now

from in-situ ablation before the node trimming, 1-token graph, ~66 ms GPU
per token, roughly +-3 ms: 12.6k graph nodes, ~7k metal kernels, one command buffer, no CPU fallback.
MUL_MAT 21-29 ms (q8_0 attention projections, bandwidth-bound, and the q5_K
output head), MUL_MAT_ID ~23 ms, ~7k small kernels ~19 ms, flash attention
and conv ~2 ms. about 3.6GB is read per token, so the ceiling at ~90GB/s is
near 25 t/s, not 40.

## ideas left

1. decode at depth: a banded variant for one query would drop the dense
   per-head bias and mask chain (CONT, CPY, GET_ROWS, ADD over n_kv x 24
   per full layer); then check the vec kernel against bandwidth.
2. fused control, SITU, convolve and clip kernels: 1-2 ms each.
3. the lane fix of `patches/0002` for the q5_K output head and the 18
   expert tensors that are not NVFP4: ~1 ms and ~1.5 ms.
4. dspark drafter: 52% acceptance but slower than no drafter on the old
   build; verify cost is now lower, not re-measured.
5. quant mix: attention tensors are already bandwidth-bound, so smaller
   quants there would pay; compare KLD against UD-Q3_K_XL first.
6. upstream now has Inkling 2 Mini itself (ggml-org/llama.cpp#32694);
   rebase the patches onto it when a tag carries it.
7. turn swap back on (needs sudo) so mistakes degrade rather than freeze.
