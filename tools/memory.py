#!/usr/bin/env python3
"""Peak server memory before the first chat, in prompt reading, in
generation and after the unload. Loads the real model.

usage: python3 tools/memory.py GALL MODEL"""
import ctypes, json, os, shutil, socket, struct, subprocess, sys, tempfile
import threading, time
from http.client import HTTPConnection
from pathlib import Path


def main(gall, model):
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    peak, phase = {}, ['startup']
    with tempfile.TemporaryDirectory() as home, socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        # a fresh snapshot cache, with the drafter table
        c = Path(home, '.cache/gall')
        c.mkdir(parents=True)
        for f in Path.home().glob('.cache/gall/*.dr'):
            shutil.copy(f, c)
        p = subprocess.Popen([gall, 'serve', model, '--port', str(port),
                              '--idle-unload', '1'],
                             env=dict(os.environ, HOME=home),
                             stderr=subprocess.DEVNULL)

        def sample():
            # rusage_info_v0: resident and footprint at offsets 64 and 72
            b = ctypes.create_string_buffer(96)
            while p.poll() is None:
                if lib.proc_pid_rusage(p.pid, 0, b) == 0:
                    r = struct.unpack_from('QQ', b.raw, 64)
                    o = peak.setdefault(phase[0], [0, 0])
                    o[:] = map(max, o, r[::-1])
                time.sleep(0.02)

        def req(m, path, body=None):
            h = HTTPConnection('127.0.0.1', port, timeout=300)
            h.request(m, path, body and json.dumps(body))
            return h.getresponse()

        loaded = lambda: json.load(req('GET', '/health'))['loaded']
        try:
            while socket.socket().connect_ex(('127.0.0.1', port)):
                time.sleep(0.05)
            threading.Thread(target=sample, daemon=True).start()
            time.sleep(0.2)
            sys_ = 'You are a concise assistant. ' + \
                'The orchard has apple trees and pear trees. ' * 70
            q = {'messages': [{'role': 'system', 'content': sys_},
                              {'role': 'user', 'content':
                               'Which two kinds of trees are in the orchard?'}],
                 'stream': True, 'temperature': 0, 'max_tokens': 64}
            phase[0], t = 'prompt', None
            for l in req('POST', '/v1/chat/completions', q):
                if l.startswith(b'data: {'):
                    e = json.loads(l[6:])
                    t = e.get('timings', t)
                    if any(c['delta'].get('content') or
                           c['delta'].get('reasoning_content')
                           for c in e.get('choices', [])):
                        phase[0] = 'generation'
            phase[0] = 'done'
            while loaded():
                time.sleep(0.1)
            phase[0] = 'unloaded'
            time.sleep(0.5)
            peak.pop('done', None)
            print(json.dumps({'timings': t, 'peak': {
                k: {'footprint': f, 'resident': r}
                for k, (f, r) in peak.items()}}, indent=2))
        finally:
            p.terminate()
            p.wait()


if __name__ == '__main__':
    main(*sys.argv[1:]) if len(sys.argv) == 3 else sys.exit(__doc__)
