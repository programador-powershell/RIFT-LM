#!/usr/bin/env python3
"""Black-box smoke test for the WINNER HTTP server."""
import http.client
import json
import socket
import subprocess
import sys
import time


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


port = free_port()
server = subprocess.Popen([sys.argv[1], "--serve", "--port", str(port)])
try:
    for _ in range(50):
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["status"] == "ok"
            break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("server did not become ready")

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request("GET", "/v1/models")
    response = connection.getresponse()
    assert response.status == 200
    assert json.loads(response.read())["data"][0]["id"] == "winner-synthetic"

    body = json.dumps({"model": "winner-synthetic", "messages": [{"role": "user", "content": "hello"}]})
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    response = connection.getresponse()
    assert response.status == 200
    assert json.loads(response.read())["choices"][0]["message"]["role"] == "assistant"
finally:
    server.terminate()
    server.wait(timeout=5)
