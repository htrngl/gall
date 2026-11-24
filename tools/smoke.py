#!/usr/bin/env python3
"""HTTP lifecycle on a dummy model (no GPU, no lock).

usage: python3 tools/smoke.py GALL MODEL"""
import json, socket, subprocess, sys, tempfile, time
from http.client import HTTPConnection

KEY = 'TestKey-AbC123'


def main(gall, model):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    def req(m, path, body=None, auth='Bearer ' + KEY, key=None):
        c = HTTPConnection('127.0.0.1', port, timeout=10)
        body = json.dumps(body) if isinstance(body, dict) else body
        h = {'Authorization': auth} if auth else {}
        c.request(m, path, body, dict(h, **{'x-api-key': key} if key else {}))
        r = c.getresponse()
        return r.status, r.read()

    def until(f, t=5):
        end = time.monotonic() + t
        while not f():
            assert time.monotonic() < end, 'timed out'
            time.sleep(0.05)

    loaded = lambda: json.loads(req('GET', '/health', auth=None)[1])['loaded']
    with tempfile.NamedTemporaryFile('w') as k:
        k.write(KEY + '\n')
        k.flush()
        p = subprocess.Popen([gall, 'serve', model, '--dummy', '--port',
                              str(port), '--idle-unload', '1',
                              '--api-key-file', k.name])
        try:
            until(lambda: socket.socket().connect_ex(('127.0.0.1', port)) == 0)
            assert not loaded(), 'the server starts unloaded'
            m = '/v1/models'
            assert req('GET', m, auth=None)[0] == 401
            assert req('GET', m, auth='Bearer ' + KEY.lower())[0] == 401
            assert req('GET', m, auth='bEaReR ' + KEY)[0] == 200
            assert req('POST', '/v1/chat/completions', '{')[0] == 400
            assert req('GET', '/missing')[0] == 404
            q = {'messages': [{'role': 'user', 'content': 'Hello'}],
                 'temperature': 0, 'max_tokens': 32}
            for _ in range(3):  # load, answer, idle unload
                s, b = req('POST', '/v1/chat/completions', q)
                assert s == 200, b
                a = json.loads(b)['choices'][0]['message']['content']
                assert a == 'hello from gall', a
                until(lambda: not loaded())
            s, b = req('POST', '/v1/chat/completions', dict(q, stream=True))
            assert s == 200 and b'data: [DONE]' in b
            q = {'max_tokens': 32, 'messages': q['messages']}
            s, b = req('POST', '/v1/messages', q, auth=None, key=KEY)
            assert s == 200, b
            a = json.loads(b)['content']
            assert a == [{'type': 'text', 'text': 'hello from gall'}], a
            s, b = req('POST', '/v1/messages', dict(q, stream=True))
            assert s == 200 and b'event: message_stop' in b
            assert req('POST', '/park')[0] == 200 and not loaded()
            p.terminate()
            assert p.wait(10) == 0
            print('smoke: ok')
        finally:
            p.kill()


if __name__ == '__main__':
    main(*sys.argv[1:]) if len(sys.argv) == 3 else sys.exit(__doc__)
