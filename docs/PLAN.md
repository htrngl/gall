# optimisation plan

written 24 November 2026 from LIMITS.md, ENGINE.md, DRAFT.md and the
README. nothing here is measured: each item gives the reasoning, the
first cheap measurement, and the gate that drops it. estimates are
arithmetic on figures already in the docs. order is expected gain in
real use, which is a hax session of 50-64k tokens in a tool loop, not
the short-context benches the work so far was tuned on.

classes: E exact (same bits), R another rounding (as passes are now),
L lossy (off by default, behind an env setting, with a kl gate as `VT`
has).

## 0. a long-session bench first

every figure in the docs is short context, 10.5k or 29.9k. the server
that died on 24 November held 54,612 tokens. build one replay: a saved
hax request chain (system, tools, 50k+ tokens, 10 tool turns) through
`tools/smoke.py`-style http, reporting per turn: reload, snapshot read,
int8 rebuild, prefill tokens and ms, first token, tokens/s, snapshot
write. also record the temperature and sampler fields hax sends (item 5
depends on them). all items below are judged on this and on the
existing short benches, interleaved.

## 1. keep the model through a tool loop (E, serve.rs)

the app unloads after 1 idle second. a `tool_calls` finish means the
client returns as soon as its tool ends, so each tool turn pays an
unload (snapshot write), a reload (0.1 s), a snapshot read, the int8
rebuild and cold weight pages: ttft 1.18 s at 805 tokens, unmeasured at
54k where the f16 rows are about 1.8 GB (33 KB a token) and int8 1 GB.

- after `finish_reason` `tool_calls`, idle limit = max(set, min(30 s,
  4 x the median gap of the last turns)); after `stop`, the set limit.
  memory pressure warn for 3 s still parks at once (up.rs rule holds).
- gate: bench 0 turn time; footprint held for the extra seconds is the
  cost, state it in the README table.
- estimate: the whole reload path a tool turn, 1 s at short context and
  likely several at 54k.

## 2. snapshots that cost what changed (E, snap.rs)

- append: a session's snapshot is rewritten whole at each unload. write
  only rows past the last write plus the window and conv tail (`ROWS`).
  at 54k a turn of 300 new tokens writes 10 MB plus the tail, not 1.8 GB.
- save the int8 rows too (18 KB a token): restore then reads int8 for
  the first stage and f16 only for chosen keys, and drops the rebuild.
  first measure the rebuild and the read apart at 30k and 54k.
- map, do not read: wrap the snap file's pages as the cache buffer
  (`newBufferWithBytesNoCopy`, as gguf does) so restore is O(1) and the
  first token faults what it uses. needs page-aligned row layout.
- ladder: a snapshot restores only at its end (minus 12 steps), so a
  client that edits mid-history (hax compaction, pruned tool output)
  loses everything after the system prefix, and prefill is 106 tokens/s:
  10k lost tokens are 94 s. store window rows and conv state every 4,096
  tokens; restore at the highest rung under the common prefix (full
  attention rows are per token, so they truncate free). measure rung
  size first; gate at under 5% of the snapshot.
- gate for all: restored logits equal unrestored (the existing check).

## 3. prefill rate (R as now, pf.rs, pfm.metal)

prefill is 130 tokens/s (7.7 ms a token, about 620 G weights/s), only
5 x a serial step, and a cold 54k prompt is about 9 minutes. no floor
table exists for it. make one as LIMITS.md has for decode: ms by part
(experts by format, Q8_0, k-quants, attention scores, small ops) against
bytes at 87 GB/s and against tile arithmetic.

- expert tile occupancy: a batch of B rows gives each expert B/16 rows
  on average. at 64 that is 4 rows in an 8-wide tile and one slab
  decode per 4 uses. count rows per expert per visit; try `ub` 256, 512,
  1024 (memory: the stream is about 80 N halves a row) and sort rows by
  expert as `uni` does. estimate 1.2-1.5 x on the expert part if tiles
  run half empty now.
- weights decoded once per tile column of 8 rows: with 256+ rows, decode
  a weight tile once and multiply it against every 8-row group (decode
  hoisted out of the row loop). check whether `pfm.metal` already does.
- the last visit's attention output and experts feed only the head and
  tap f3. without the drafter, skip them for all rows but the last
  (1/48 of the expert and o work, E).
- gate: `gall check` on ref/2k and ref/8k.b256, kl no worse.

## 4. verify long copies at prefill cost (R, dr.rs, pf.rs, serve.rs)

a copy draft is capped at 7 tokens and a pass at 8 rows: 135 ms, 16.9 ms
a row, with 96-97% of copied rows kept. prefill does a row in 7.7 ms.
when the copy match runs on (match of 12+, last pass all kept), verify
32-64 copied tokens in one prefill-shaped batch with the head on every
row (0.46 G weights a row, under 1 ms on tiles), keep the longest
prefix whose argmax (or sampled token, item 5) equals the copy, drop the
rest.

- rollback: the conv ring caps a pass at 13 rows. keep the batch's conv
  inputs (post already stores them before rows read them) and rebuild
  the ring at the kept length; window and full rows truncate.
- first measurement, cpu only: on the code-edit and quote-rewrite
  outputs, the share of output tokens inside copy runs of 16+, 32+, 64+.
  gate: under 25% in runs of 16+ drops the item.
- estimate: about 9 ms a token on those runs against 17-28 now; code
  edit 35 -> 50+ tokens/s if half the answer is long copies.

## 5. sampled requests keep their drafts (E for the sampler, serve.rs, dr.rs)

a row is kept only when the server's sampled token equals the proposal,
and proposals are argmax, so at temperature > 0 acceptance falls with
entropy. couple the noise, with no acceptance rule: sample by
gumbel-max, token = argmax(l_i / t + g(seed, position, i)) after top_k,
top_p, min_p, where g is a hash, not a stream. the drafter proposes
argmax(q_i / t + g) with the same g over its shortlist. both sides see
the same noise, so they agree whenever p and q are close, and the output
is exactly the main model's sample for that seed. the recurrent head
feeds the proposed token on, as now.

- cost: the sampler changes its random stream (same distribution;
  note it in API.md). g is needed only for tokens that survive the
  filters and for the shortlist.
- first measurement: bench 0's temperature. if hax sends 0, park this.
- gate: acceptance at t 0.7 within 0.1 of greedy acceptance on the 13
  workloads; a chi-square of sampled tokens against `sample()`.

## 6. the cost table knows the context length (E, dr.rs)

`Spec` plans with fixed costs: step 40.2 ms, passes 63-136 ms. at 29.9k
a step is 58.6 ms, and a pass past 4,096 tokens reads f16 only, once
for all rows. so at long context the plan is wrong in both directions,
and extra rows are cheaper relative to a step than the table says. fit
step and pass cost as a + b n from `--tv` at 0, 10k, 30k, 60k (still a
table, so output stays run-to-run identical). also do the open item:
int8-assisted attention in passes past 4,096.

- estimate: drafting should pay more at 30k+ than at short context;
  measure prose and code ratios at 30k before and after.

## 7. drafter context across a restore (R, snap.rs, dr.rs)

with instant unload every turn restores, and the drafter then knows
only the tokens fed since. layers 0 and 2 see 1,024 positions, layer 1
all. measure acceptance against drafter context length (0, 256, 1,024,
4,096, all) on code and prose at 10k. then save the shortest tail that
holds acceptance (1,024 rows are about 4 MB). item 1 removes most
restores inside a loop; this covers the first turn after a pause.

## 8. a floor table for passes (E, drm.metal)

by bytes, a pass reads the dense 1.6 GB once and distinct experts x
slab: 2 rows about 3.7 GB, 42 ms; 8 rows about 6.4 GB, 73 ms. measured:
61 and 135. so passes are arithmetic-bound by 19 and 62 ms, and serial
got 142 -> 204 G weights/s from decode work alone. per kernel and row
count, print ms against the byte floor; the largest gaps first. two
guesses to test: Q8_0 at 2 rows should cost near one read (13-14 ms,
not 2 x), and Q8_0 tiles at 3-4 rows pay 8 multiplies for 3-4 uses,
where 4-row dots with shared loads may win. a 51 ms 2-row pass alone
moves prose above serial (1.61 tokens in 51 + 6.3 ms is 35.6 ms a
token against 40.2).

## 9. long-context attention (L as `VT`, ops.metal)

a step gains 19 ms from 0 to 29.9k and is unmeasured at 60k and 100k.
- measure where it goes: int8 K scan, int8 V, f16 rereads, per visit.
- if the first stage reads V for every key, read K only, then V for keys
  above a weight cut (6 of 18 KB a token).
- a coarser first stage: per block of 64 keys keep a 4-bit or sign
  sketch and a per-block bound; scan int8 only in blocks that can reach
  the cut. first histogram, from a dump at 30k: keys per head with
  weight over 1e-4 of the maximum. gate: under 5% of keys, else drop.
- gate as `VT`: no greedy mismatch, kl within 1e-7 of f16.

## 10. lossy tier (L, each off by default)

each needs its histogram before any kernel work, and a gate of no
greedy mismatch on ref/short and ref/2k with mean kl against the
reference not above the present 1.05e-4 and 9.9e-4 by more than 10%.
- drop light experts: of the 8 chosen, skip those with gate weight under
  a cut `GT` (renormalise or not: test both). histogram gate weight by
  rank over 96 tokens x 48 visits. each expert skipped is 1/8 of 16.1
  ms; two of eight under the cut on average is 4 ms a token.
- attention below Q8_0: requantise q, k, v, o apart to Q6_K or Q5_K in
  a derived file in `~/.cache/gall`. q and k act only through scores,
  so try them first. Q8_0 to Q5_K on all is 12.9 -> about 8.9 ms.
- a two-stage head for greedy: a 2-bit sketch of the head (115 MB, 1.3
  ms), exact Q5_K rows for the best 1,024, full head when the sketch
  margin is under a set number of error deviations. histogram the rank
  of the true argmax in the sketch over 10k positions; gate: never
  outside 1,024. saves up to 2.5 ms a token. not provable, so L.

## 11. prose controller (branch gate-wip)

at first-slot acceptance 0.61 a 2-row pass yields 1.61 tokens for 61 +
6.3 ms: 41.8 ms a token against 40.2, so prose cannot win at today's
costs whatever the controller does; it can only lose less. finish the
controller as a guard (never under 0.99 of serial), and get the gain
from items 6 and 8.

## 12. small-kernel chain (E)

3 ms over floor, about 9 us for each of some 530 dispatches. untried
shapes: fuse across the visit boundary (the last small op of a visit
with the first of the next), and one chain kernel in a single
threadgroup for score, gather, merge at short context where they are
tiny. low odds: three fusions were slower.

## order of work

0, then 1 and 2 (server, one agent), 3 and 4 (prefill, one agent), 6
and 8 (passes, one agent). 5 after bench 0 reports the temperature. 7,
9, 10 after their histograms. file ownership as AGENTS.md; one GPU
user at a time through `/tmp/gall.lock`, so histograms and cpu-only
counts (4, 5's chi-square, 10's sketch rank) run while another agent
holds the GPU.
