#!/usr/bin/env python3
"""A long tool session: 10 turns that grow to 50k+ tokens, with repo
files as tool results. Prints the server log and, per turn, prompt
tokens, cached tokens, prefill ms, first token, generated tokens,
tokens/s and the wall time. Requests carry the fields hax sends (no
temperature or other sampler fields), with max_tokens 512 added.

usage: python3 tools/long.py GALL MODEL [serve flags]"""
import json, os, shutil, socket, subprocess, sys, tempfile, threading, time
from http.client import HTTPConnection
from pathlib import Path

R = Path(__file__).resolve().parents[1]
FILE = {'type': 'object', 'properties': {'path': {'type': 'string'}},
        'required': ['path']}
TOOLS = [{'type': 'function', 'function': {'name': n, 'description': d,
          'parameters': FILE}} for n, d in [
    ('read', 'Read a file.'), ('edit', 'Edit a file.'),
    ('write', 'Write a file.'), ('bash', 'Run a shell command.'),
    ('task_wait', 'Wait for a background task.')]]
K = 'content', 'reasoning_content', 'tool_calls'


def main(gall, model, *flags):
    text = ''.join(f.read_text() for f in sorted(R.glob('src/*.rs')))
    parts = [text[i:i + 13000] for i in range(0, 130000, 13000)]
    msgs = [{'role': 'system', 'content': 'You are a coding agent. Use the'
             ' tools to read and change files.\n' + (R / 'AGENTS.md')
             .read_text()},
            {'role': 'user', 'content': 'Read the source and list its bugs.'}]
    with tempfile.TemporaryDirectory() as home, socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        c = Path(home, '.cache/gall')
        c.mkdir(parents=True)
        for f in Path.home().glob('.cache/gall/*.dr'):
            shutil.copy(f, c)
        p = subprocess.Popen([gall, 'serve', model, '--port', str(port),
                              *flags], env=dict(os.environ, HOME=home),
                             stderr=subprocess.PIPE, text=True)
        log = lambda: [print('  ' + l, end='', flush=True) for l in p.stderr]
        threading.Thread(target=log, daemon=True).start()
        try:
            while socket.socket().connect_ex(('127.0.0.1', port)):
                time.sleep(0.05)
            print('turn prompt cached prefill_ms first_s gen t/s total_s')
            for i, part in enumerate(parts):
                h = HTTPConnection('127.0.0.1', port, timeout=3600)
                q = {'model': 'inkling-2-mini', 'messages': msgs,
                     'tools': TOOLS, 'stream': True, 'max_tokens': 512,
                     'stream_options': {'include_usage': True},
                     'return_progress': True}
                t0, t1, m, calls, u, t = time.monotonic(), 0, {}, [], {}, {}
                h.request('POST', '/v1/chat/completions', json.dumps(q))
                for l in h.getresponse():
                    if not l.startswith(b'data: {'):
                        continue
                    e = json.loads(l[6:])
                    u, t = e.get('usage', u), e.get('timings', t)
                    for d in (c['delta'] for c in e['choices']):
                        if any(map(d.get, K)):
                            t1 = t1 or time.monotonic()
                        for k in K[:2]:
                            m[k] = m.get(k, '') + (d.get(k) or '')
                        calls += d.get('tool_calls', [])
                print(i + 1, u['prompt_tokens'], t['cache_n'],
                      round(t['prompt_ms']), round(t1 - t0, 2),
                      t['predicted_n'], round(t['predicted_per_second'], 1),
                      round(time.monotonic() - t0, 2), flush=True)
                msgs.append(dict(m, role='assistant', **(
                    {'tool_calls': calls} if calls else {})))
                msgs += [{'role': 'tool', 'tool_call_id': k['id'],
                          'content': part} for k in calls] or \
                        [{'role': 'user', 'content': 'Read this:\n' + part}]
        finally:
            p.terminate()
            p.wait()


if __name__ == '__main__':
    main(*sys.argv[1:]) if len(sys.argv) > 2 else sys.exit(__doc__)
