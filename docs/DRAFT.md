# Speculative decoding

Gall uses Inkling-2-Mini’s dspark drafter to propose tokens, then verifies
them with the main model. The server enables it when
`dspark-Inkling-2-Mini-Q8_0.gguf` is beside the model. `gall up` downloads
that file automatically; `--no-draft` disables it. Drafting is disabled
for requests with `response_format`.

The benefit depends on the answer. On the M2 MacBook Air, the recorded
server tests showed little change for prose and larger gains for a proof.
Gall estimates whether drafting will pay and falls back to ordinary
decode when it will not.

## Results

Measured on 22 November 2026, on a cool 24 GB M2 MacBook Air. Each server
request generated 120 tokens; the runs without and with drafting used
the same process settings and ran sequentially under the GPU lock. Rates are
the server’s `predicted_per_second`.

| Request | Temp 0, no draft | Temp 0, draft | Default sampling, no draft | Default sampling, draft |
| --- | ---: | ---: | ---: | ---: |
| Explain how a transformer predicts | 20.5 | 20.7 | 20.2 | 20.5 |
| Induction proof | 20.4 | 23.4 | 20.0 | 21.8 |
| Rust key-value store | 20.4 | 21.4 | 20.2 | 20.4 |

All rates are tokens/s. Complete messages matched in every pair, including
sampled runs with the same seed. These are individual measurements, not
averages. Thinking text reads as prose, which helps explain why the
server gains were smaller than gains on raw code and mathematics.

Additional checks covered two turns with prefix reuse, a repeated prompt,
an 830-token system prefix saved and restored, idle unload, and an image
request. Messages matched with and without drafting. The 64-step
teacher-forced engine check retained zero greedy mismatches and mean KL
divergence of 1.05e-4.

## Drafter architecture

The 402 MB GGUF file has architecture `dflash` and 54 tensors. It contains
three transformer layers with hidden size 2304, 24 query heads, 8 KV heads
and an FFN width of 9216. It drafts a block of eight positions, with
non-causal attention within the block. Layers 0 and 2 see 1,024 context
positions; layer 1 sees the full context. Projections use Q8_0.

The drafter borrows the main model’s token embeddings, embedding norm and
output head. It consumes four target features per committed token:

| Feature | Source |
| --- | --- |
| `f0` | Residual entering visit 5, physical layer 5 |
| `f1` | Residual entering visit 23, physical layer 11 on pass 1 |
| `f2` | Residual entering visit 41, physical layer 17 on pass 2 |
| `f3` | Normalized output mix (`result_norm`) |

For token position `p`, with `w_r = softmax(dspark_fuse[r])`:

```text
c_l    = rms(sum_k w_l[k] f_k) * dflare_norm_l       l = 0, 1, 2
K_l[p] = rope(rms_head(k_ctx_l c_l) * k_ctx_norm_l, p)
V_l[p] = v_ctx_l c_l
f_rnn  = rms(sum_k w_3[k] f_k) * dspark_norm
```

Its context occupies two rings and one full cache, about 4 KB per token
in f16 (428 MB at 102,400 tokens). Context injection costs six 4.7 MB
matrix-vector operations per committed token, measured at about 0.4 ms.

The first draft position contains the anchor token; the remaining seven
start with mask embeddings. After the transformer layers, a recurrent
head produces proposed tokens in sequence. The full block is evaluated
even when only some proposals will be verified.

The reference implementations are
`llama/src-mix/src/models/dspark.cpp` (`graph_inkling2` and
`build_inkling2_head`) and `llama/src-mix/common/speculative.cpp`
(`common_speculative_impl_draft_dspark`). Gall’s implementation is in
`src/dr.rs`, `src/dr.metal` and `src/drm.metal`.

## Verification and sampling

`Dr::pass` evaluates an anchor and up to seven proposed tokens in one
batch. `Dr::commit` retains accepted rows and rolls back the rest through
`Model::truncate`.

In the server, `Dr::next` exposes the main model’s logits one position at
a time. A verified row is reused only when the server’s sampled token
matches the proposal for that row. Otherwise Gall discards the later rows
and starts a new pass. The sampler therefore continues to use the main
model’s distribution. Stop strings, tool calls, token limits and client
disconnections do not need a separate acceptance path.

Reset, truncate, prefix restoration and prefill discard pending rows.
Snapshots record how far speculative work wrote into convolution rings
so rollback remains valid. Under `response_format`, the server uses
ordinary decode.

## Batched kernels

The weight kernels dequantize a unit once and dot it with several inputs.
The input count is a compile-time constant. For experts, `uni` groups
slabs by how many rows use them, then issues indirect dispatches for each
use count. A slab is read once, with one dot product per row that uses it.

Small operations dispatch all rows together on grid z. Convolution inputs
are stored before any row reads them. Row 0 is always retained, so its
drafter context can be injected in the same command buffer.

Measured pass times in the integrated build:

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU ms | 49.5 | 70.3 | 95.7 | 117.0 | 149.9 | 181.4 | 235.1 | 243.9 |

On 23 November 2026, NVFP4 and IQ4_NL slabs with one use moved to two
weight rows per threadgroup, as in single-token decode. Best pass times
by `gall dr --tv 1,2,3,4` went from 46.9, 71.4, 97.8 and 115.3 ms to
46.9, 67.4, 92.6 and 113.8 ms. Greedy output with `--block 3` matched
ordinary decode.

Two or four rows cost less than serial decoding of the same number of
tokens, but the draft itself must also be paid for. Larger batches are
not automatically better: expert arithmetic and register pressure grow
with the number of rows.

## Choosing when to draft

`Spec` tracks observed acceptance in 12 probability bins. A proposed row
is worthwhile when its probability of being retained exceeds the token
rate multiplied by the extra verification time:

```text
p > rate * (tv[j + 2] - tv[j + 1])
```

Here `p` includes the chance that preceding proposals are also accepted.
The rate is the better observed rate of serial and drafted decoding.
An extra row costs roughly 22–27 ms in the small batches.

A proposal after the last verified row can be checked against the new
pending token without another verification pass. If it matches, Gall
keeps the rest of that draft for the next round. The controller estimates
whether each new draft will repay its cost, disables drafting when it
will not, and occasionally probes again. Draft length follows the recent
number of accepted proposals, bounded between three and seven.

This matters most for prose. In an earlier test, a fixed block of three
without the controller reduced prose throughput to 14.5 tokens/s.

## Prefill, snapshots and reload

`Dr::feed` uses batched prefill with feature taps. It collects the residuals
entering visits 5, 23 and 41 and the normalized output mix, then injects
context in groups of eight tokens.

The drafter attends only to context injected in an unbroken run. Restored
prefixes have no saved drafter context; the drafter starts with the tokens
processed since restoration. The main model still reuses the saved prefix.

The recurrent head uses a precomputed projection, `P = output * hidden_out`,
to avoid a full output-head matrix-vector operation for each proposal.
Computing it on every load originally made reloads take 0.71–0.75 seconds.
Gall now caches `P` and its bias in a 104 MB file in `~/.cache/gall`, keyed
by model sizes and the drafter’s modification time. Recorded reloads then
took 0.07–0.10 seconds. The unloaded server footprint was 229–231 MB,
compared with 224 MB without the drafter.

## Remaining work

- Sampled proposals with the corresponding acceptance rule could improve
  acceptance at higher temperatures; current proposals use argmax.
- The draft costs about 15–31 ms, much of it in the output head. A cheaper
  head used only for proposals could reduce that cost.
- Small operations still scale with row count: an earlier four-row profile
  took 13.4 ms, compared with about 5 ms serially.
- Seven- and eight-row verification needs further kernel work.
- Saving drafter context could improve acceptance just after a restore,
  at the cost of larger snapshots and more disk I/O.
- A batch crossing 4,096 tokens uses f16 throughout, while serial decode
  would switch to the int8-assisted attention path partway through.
- `KV=q8` with batched verification remains untested.
