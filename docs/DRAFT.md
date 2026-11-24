# Speculative decoding

Gall uses Inkling-2-Mini’s dspark drafter to propose tokens, then verifies
them with the main model. The server enables it when
`dspark-Inkling-2-Mini-Q8_0.gguf` is beside the model. `gall up` downloads
that file automatically; `--no-draft` disables it. Drafting also runs
under `response_format`: the schema still picks each token from the main
model’s logits.

The benefit depends on the answer. On the M2 MacBook Air, code edits,
code and JSON gain most; drafted prose is about level with serial decode
(0.97–1.05 of its rate). Gall estimates whether drafting will pay and
falls back to ordinary decode when it will not. A change to keep prose
at or above the serial rate is not done.

## Results

Measured on 23–24 November 2026, on a warm 24 GB M2 MacBook Air, as
medians of interleaved runs. Server requests used temperature 0 with
reasoning off; rates are tokens/s.

| Server request | Start of 23 November | 24 November |
| --- | ---: | ---: |
| JSON schema | 18.9 | 31.7–31.9 |
| Tool call | 19.6 | 27.5–28.9 |
| Code edit (1,518 tokens) | 22.1 | 35–36 |
| Quote rewrite | 20.6 | 29–30 |
| Prose | 19.4 | 23.1–23.4 |
| Proof | 19.3 | 25.4 |
| Rust | 19.4 | 26.7 |

`gall run --draft` over 400 tokens, as a ratio to serial decode in the
same build:

| Workload | Code | Maths | Prose | Essay | Explanation | Dialogue | Summary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drafted / serial | 1.46 | 1.14 | 0.97 | 1.10 | 0.99 | 1.02 | 1.28 |

Part of the server gain comes from serial decode, which went from 45.2 to
39.5–39.8 ms of GPU time per token in the same period
([engine notes](ENGINE.md)).

## Exactness

Drafted greedy output is a greedy decode of the model under a different
rounding: a pass of several rows sums in another order than a step. It is
as close to the llama.cpp reference as serial decode. Over a 400-token
teacher-forced history of the `ref/short` prompt, by 100 tokens, the
median difference over the 32 best logits was:

| Comparison | 1–100 | 101–200 | 201–300 | 301–400 |
| --- | ---: | ---: | ---: | ---: |
| Serial against llama.cpp | 0.24 | 0.13 | 0.30 | 0.98 |
| 8-row passes against llama.cpp | 0.24 | 0.14 | 0.25 | 0.64 |
| 8-row passes against serial | 0.0007 | 0.03 | 0.24 | 0.61 |
| KL from the reference, serial | 9.3e-4 | 6.5e-4 | 4.2e-4 | 2.5e-3 |
| KL from the reference, passes | 9.7e-4 | 7.5e-4 | 4.1e-4 | 2.0e-3 |

The argmax differed from the reference at 3 of 400 positions for serial
decode and 2 for passes. Drafted output equals serial output until a
near-tie meets the accumulated difference, typically after 100–300
tokens. A margin guard cannot prevent this, because the difference is in
the cache history, not in the row. Sampled output with the same seed can
differ in the same way. Drafted output is the same from run to run.

Check it against `ref/short.400`, a reference dump of the `ref/short`
prompt with `n_gen` 400, `n_batch` 1 and 401 logit rows (see
[reference tools](../tools/README.md)):

```sh
target/release/gall dr "$(gall model)" DRAFTER ref/short.400 --drift 1,2,8 -o 400
```

`--drift` also prints a hash of each history’s logit bits, so a kernel
change can be shown to keep them. `--cmp R` compares one pass of `R` rows
with stepping (logit difference and margin), and `--tv` times passes by
row count.

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

The head scores a shortlist of 32,768 tokens, not all 200,058: the 28,672
with the greatest bias, and a ring of the last 4,096 new tokens fed. A
7-token draft went from 23.7 to about 8 ms. Teacher-forced tokens a pass
at block 7, full head against shortlist: code 5.51 and 5.23 (identifiers
outside the list), maths 3.71 and 3.70, prose 2.52 and 2.51. A 16k list
gave 4.51 for code.

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
so rollback remains valid.

## Copy drafts

The server also proposes tokens copied from context. It finds the last
places where the last 12 and the last 3 tokens occurred in the prompt or
the answer, takes the longer match, and proposes up to 7 tokens that
followed it. A copy replaces the dspark draft when it is longer, and is
verified like any draft. Under `response_format`, a copy stops where the schema would not
accept it. On a code edit, 96–97% of copied rows were kept, about 6 rows
a pass. A single minimum match of 3 or of 8 tokens kept 66% or 72–81%.

## Batched kernels

`uni` sorts expert slabs by how many rows of a pass use them, in three
classes. Slabs of 3 or more uses, and dense matrices in passes of 3 or
more rows, go through 8 × 8 `simdgroup_matrix` tiles (`x_<fmt>` in
`drm.metal`). Slabs of 1 or 2 uses stay dot products (`x_<fmt>_1`,
`_2`), because a tile costs 8 multiplies per weight whatever its use
count: two rows on tiles took 97 ms against 75 ms as dots. IQ2_S, IQ2_XS
and IQ3_XXS dots take two units a lane, as the serial mat-vecs do. The
drafter’s 8-row and head matrix products use the same kernels.

Verify decodes weights as the serial mat-vecs do (subnormal halves, signs
by one multiply, the powers of two in the scales). The IQ2_XXS, IQ2_XS
and IQ3_XXS grids and the IQ4_NL pair table are in threadgroup memory;
IQ2_S's 8 KB grid stays in the constant cache, as a threadgroup copy cost
6.5 ms at 8 rows. Q8_0 staged through threadgroup memory was 12 ms
slower.

The small operations `low`, `ctl`, `rows` and `post` take every row of a
pass in one threadgroup, so their f16 weights are read once; together
they take about 3 ms at 8 rows. `post` stores the convolution inputs
before any row reads them. Row 0 is always retained, so its drafter
context can be injected in the same command buffer.

Pass times on 24 November 2026, medians of interleaved runs:

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before tiles, ms | 47 | 67 | 93 | 114 | 150 | 181 | 235 | 244 |
| Now, ms | 41.5 | 61 | 84 | 96 | 105 | 115 | 125 | 135 |

The convolution ring caps a pass at 13 rows. A 13-row build gave 18.6 ms
a token against 19.0 ms at 8 rows (2%), and was not kept.

## Choosing when to draft

`Spec` holds the acceptance of a proposed row in 12 probability bins, in
two contexts: after fewer than 3, or 3 or more, tokens in a row that the
drafts in hand predicted. The bins start at what drafts at every position
kept over 13 workloads of 400 tokens (0.05–0.91 and 0.05–0.96), and move
with what each pass keeps. A row is verified while its probability of
being retained exceeds the token rate multiplied by its extra cost:

```text
p > rate * (tv[j + 2] - tv[j + 1])
```

Here `p` includes the chance that preceding proposals are also accepted.
The costs are a fixed table, not timed: a step 40.2 ms, passes of 2–8
rows 63, 85, 97, 107, 117, 126 and 136 ms, drafts of 1–7 tokens 6.3–8.8
ms. The plan therefore follows from the tokens alone, and drafted output
is the same from run to run.

A proposal after the last verified row can be checked against the new
pending token without another verification pass. If it matches, Gall
keeps the rest of that draft for the next round. The controller estimates
whether each new draft will repay its cost and turns drafting off when it
will not. While off, it drafts again 8 tokens on if the last draft would
have paid, otherwise 16 and then 32 tokens on. Draft length follows the
recent number of accepted proposals, bounded between three and seven.

Not kept: drafting while the pass verifies, with the drafter’s context 4
tokens late, cut first-position acceptance from 0.89 to 0.48 for code,
0.77 to 0.34 for maths and 0.61 to 0.22 for prose. Verifying the
drafter’s second choice was right at the first break 0.33 (code), 0.38
(maths) and 0.24 (prose) of the time; a branch row returned at most 0.27
tokens against a cost of 0.23–0.55, under 0.5% net.

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

- Drafted prose is about level with serial decode. The controller should
  keep it at or above the serial rate; that work is not done.
- Tool calls have no grammar. One built from the request’s `tools` with
  the existing schema matcher would make them valid. Tokens forced by a
  grammar would also need a canonical form: of 201 constrained positions,
  none allowed exactly one token.
- Sampled proposals with the corresponding acceptance rule could improve
  acceptance at higher temperatures; current proposals use argmax.
- Saving drafter context could improve acceptance just after a restore,
  at the cost of larger snapshots and more disk I/O.
- A batch crossing 4,096 tokens uses f16 throughout, while serial decode
  would switch to the int8-assisted attention path partway through.
- `KV=q8` with batched verification remains untested.
