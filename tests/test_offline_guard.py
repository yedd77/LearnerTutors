import socket

import pytest


def test_external_connections_are_blocked():
    with pytest.raises(AssertionError, match="network connection"):
        socket.socket().connect(("8.8.8.8", 53))


def test_http_clients_cannot_escape():
    import requests

    # Literal documentation-range IP: no DNS step, so only the guard can produce this error.
    with pytest.raises(AssertionError, match="network connection"):
        requests.get("http://203.0.113.7/", timeout=2)
