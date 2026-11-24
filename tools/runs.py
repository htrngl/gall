#!/usr/bin/env python3
"""The share of output tokens inside copy runs of 16+, 32+ and 64+
tokens. A token is copied when it equals the token that followed the last
place of the last 12 (else the last 3) tokens before it, in the prompt or
the output so far, as the server's copy drafts find it (serve.rs: Cp).

usage: python3 tools/runs.py PROMPT GEN (token ids, one per line)"""
import sys


def main(prompt, gen):
    p, g = ([int(x) for x in open(f).read().split()] for f in (prompt, gen))
    s, last, hit = p + g, {}, []
    for l in range(1, len(s)):
        e = None
        for k in (tuple(s[l - n:l]) for n in (12, 3) if n <= l):
            e, last[k] = e or last.get(k), l
        if l >= len(p):
            hit.append(e is not None and s[e] == s[l])
    runs, n = [], 0
    for h in hit + [False]:
        n = n + 1 if h else runs.append(n) or 0
    print(f'{len(g)} tokens, {sum(hit)} copied;', ', '.join(
        f'{sum(r for r in runs if r >= m) / len(g):.1%} in runs of {m}+'
        for m in (16, 32, 64)))


if __name__ == '__main__':
    main(*sys.argv[1:]) if len(sys.argv) == 3 else sys.exit(__doc__)
