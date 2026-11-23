# limits and next steps

a review of the proposed kernel rewrite, with measurements made on 23
November 2026 on the 24 GB M2 MacBook Air (warm, the app's server idle
beside it: rates are about 20% under the cool figures in ENGINE.md).

## the limit model

a token multiplies 4.8 G weights: 3.06 G expert (63.7 M per visit),
1.29 G attention, 0.46 G head. it reads 2.9 GB.

- memory: 2.9 GB at 87 GB/s is 33 ms. no kernel work passes 30 tokens/s
  while a token reads every byte once.
- arithmetic: every 2-4.5 bit format decodes at 140-170 G weights/s
  (IQ3_XXS 149, IQ4_NL 142, NVFP4 142, IQ2_S 150), whatever its decode.
  3.06 G expert weights at 150 is 20 ms, which is the measured 22 ms.
- prefill runs 130 tokens/s, 624 G weight uses/s: the tile path does 4
  times the multiplies of the mat-vec path. the cap is per weight
  decoded, not per multiply.

so serial decode has about 13 ms between today (46 ms) and its floor
(33 ms), and every gain past the floor needs more than one token per
read of the weights.

## measured here

- Q8_0 attention is incompressible. over blocks 0, 7, 12, 20: entropy
  7.56-7.68 bits per weight, 29.5 distinct values per block of 32, every
  block's maximum 127. a lossless smaller format saves at most 4%, so
  the 1.11 GB stays (the lossy RQ path was already removed).
- expert overlap, 96 greedy tokens after `tests/short` and after
  `ref/2k`, all 48 visits:

  | rows in a batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | distinct experts per visit | 8 | 12.6 | 16.3 | 19.4 | 22.2 | 24.8 | 27.2 | 29.5 |

  consecutive tokens share 42-44% (22% in visit 0, 73% in visit 17);
  tokens 4 apart share 35%. the same layer on consecutive passes of one
  token shares 58%, which nothing can use: the passes are sequential.
- the bus has room only beside a compute-bound kernel. the `bw` kernel
  alone: 69 GB/s. with 4 CPU threads streaming 16 GB/s: `bw` 62, total
  78. the slower `bw` shapes (49 and 17 GB/s) lost nothing to the CPU.
- `bw` falls from 70 to 49 to 17 GB/s as a thread's run grows from 1 KB
  to 16 KB to 256 KB: the bytes in flight per threadgroup set the rate,
  so 87 GB/s is the ceiling of one access shape, not of the bus (102.4
  GB/s nominal).

## the proposals reviewed

1. IQ3_XXS as 4-bit codes: no gain. the 4-bit kernels run at 142 G
   weights/s, IQ3_XXS at 149. the same weights would decode no faster
   and read 33% more bytes, for 7.5 GB of memory.
2. a sub-block per thread: tried twice, worth 1 ms at best. skip.
3. int8 activations: not exact, and the multiply is not the limit.
4. the batched verify rewrite: right, and the overlap above sizes it.
   an extra row's floor is its unshared experts (13 ms for row 2, 6 ms
   for row 8, at today's decode rate) plus 3-5 ms. today it costs
   20-54 ms. expected pass times against today's:

   | rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | today, ms | 47 | 67 | 93 | 114 | 150 | 181 | 235 | 244 |
   | tiles, ms | 47 | 66 | 81 | 95 | 107 | 119 | 130 | 140 |

   two rows gain nothing; the gain is in long accepted runs.

## suggestions, by expected gain

### more than one token per read

1. batched verify on tiles (as 4 above): slabs sorted by expert, a slab
   decoded once to halves in threadgroup memory, rows through
   `simdgroup_matrix`; attention, the head and the small ops likewise.
   eight accepted rows then cost 17.5 ms a token (57 tokens/s).
2. drafts that cost nothing. the drafter takes 15-31 ms a block; these
   take none, and the existing controller gates them the same way:
   - forced tokens: under `response_format`, tool-call wrappers and the
     chat template, a grammar that allows one token needs no logits.
     run such runs through prefill (130 tokens/s), no head, no verify.
     today these requests have drafting off.
   - context copy: propose the continuation of the longest suffix match
     in the prompt and the answer so far (code edits, quoted text, JSON
     keys, identifiers). use dspark only where no match exists.
3. a cheap head for the drafter. proposals need not be exact: score a
   shortlist (tokens in context plus the frequent ones, about 16 k rows
   of `P`) in place of 200 k rows, 7 times a block.
4. draft while verifying. the drafter waits for the verified features
   of its last block. test acceptance with its context one block late;
   if it holds, the draft overlaps the verify pass and leaves the
   cycle, and the drafter can move to the CPU (Accelerate) or the
   Neural Engine, which share no cores with the GPU.
5. a tree, not a chain. with a row at 10 ms, verify the drafter's second
   choice at the first uncertain position when its probability passes
   rate x row cost (0.2 at 21 tokens/s).
6. sixteen rows on confident runs: row 8 already shares 71% of its
   experts.

with 1-3, code and mathematics should run at 30-37 tokens/s and copied
or forced spans at 50 or more. prose gains little: a first proposal
accepted 40% of the time breaks even at a 10 ms row.

### serial decode, 46 ms towards 33

7. fewer dispatches. 1,071 dispatches and their barriers cost about
   5 ms (42 ms with no barriers). per visit, 22 dispatches can be 7:
   - q, k, v, rel, gate as one dispatch over 8,576 rows with a row table;
   - a chain on one 2,304-vector (dots, elem, low, ctl; then dots, elem,
     low, ctl, rows, top8) as one threadgroup with threadgroup barriers;
   - post, score, gather, merge as one or two;
   - act inside `up`: a threadgroup takes row r and row r + 1152.
   expect 2-4 ms. the far end is one dispatch per token that takes
   work from an atomic counter; register pressure makes it a risk.
8. find what caps decode at 150 G weights/s before more kernel work.
   run each kernel on one 64 KB slab again and again, so memory is out:
   if Q8_0 also stops near 150, the cap is the convert, multiply and
   `simd_sum` skeleton or the thread launch rate (a thread does 144-576
   weights), and no format or decode change helps.
9. the CPU beside the expert kernels. they leave 30 GB/s of the bus
   unused and the CPU's traffic cost them nothing above. four P cores
   decode perhaps 15-20 G weights/s, 2 ms a token, if a split dispatch
   can wait for them (`MTLSharedEvent`, or the CPU polling a flag the
   GPU writes). measure the cost of 96 command buffer joins first.
10. sample on the GPU and commit the next token before the wait: the
    GPU never idles between tokens (0.4 ms a token; exact at
    temperature 0).
11. the true bus ceiling: `bw` at 8-32 words per thread, and one tensor
    copied to an anonymous buffer against the file mapping. if either
    passes 87 GB/s, the floor moves below 33 ms.

with 7-10 serial decode should reach 38-41 ms (24-26 tokens/s).

### outside the code

12. cooling. throughput falls 10% after 2.5 minutes; a thermal pad to
    the case or a cooled stand keeps the cool rate.

## ruled out

- a smaller exact Q8_0 (entropy above).
- tables of partial dots for the grid formats: 16 KB per input
  position, and a thread has registers only.
- exact pruning of the head: a norm bound on 2,304 dimensions is about
  45 times the typical logit, so it excludes nothing.
- reuse across the three passes of the core layers: each pass needs the
  one before.
