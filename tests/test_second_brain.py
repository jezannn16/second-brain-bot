import socket
import urllib.parse

import pytest

import second_brain
from second_brain import add_document, answer, connect, extract_pdf, fetch_url, search
from telegram_bot import _query_tag, handle_update, parse_allowlist


def test_duplicate_search_tag_citation_delete(tmp_path):
    db = connect(tmp_path / "brain.db")
    ident, created = add_document(db, "note", "Сон", "План: читать книгу каждый день", "личное, привычки")
    assert created
    assert add_document(db, "copy", "different title", "План: читать книгу каждый день")[1] is False
    assert search(db, "читать книгу", "личное")[0]["id"] == ident
    assert answer(db, "читать книгу")["citations"][0]["document_id"] == ident
    db.execute("DELETE FROM documents WHERE id=?", (ident,))
    assert not search(db, "читать книгу")


def test_document_ownership_isolation(tmp_path):
    db = connect(tmp_path / "brain.db")
    add_document(db, "note", "Private", "секретный план запуска", owner_id=100)
    add_document(db, "note", "Private", "секретный план запуска", owner_id=200)
    assert len(search(db, "секретный план", owner_id=100)) == 1
    assert len(search(db, "секретный план", owner_id=200)) == 1
    assert not search(db, "секретный план", owner_id=300)
    assert answer(db, "секретный план", owner_id=300)["citations"] == []


def test_telegram_allowlist_and_user_scoped_search(tmp_path):
    class FakeAPI:
        def __init__(self):
            self.sent = []

        def send(self, chat_id, text):
            self.sent.append((chat_id, text))

    api = FakeAPI()
    pending = {}
    db = str(tmp_path / "brain.db")

    def update(user, text):
        return {"message": {"from": {"id": user}, "chat": {"id": user, "type": "private"}, "text": text}}

    handle_update(update(100, "/note Private | секретный план | test"), api, db, {100, 200}, pending)
    handle_update(update(200, "/search секретный план"), api, db, {100, 200}, pending)
    assert "Ничего не найдено" in api.sent[-1][1]
    assert parse_allowlist("100, 200") == {100, 200}
    with pytest.raises(ValueError):
        parse_allowlist("")
    assert _query_tag("поиск заметок --tag work") == ("поиск заметок", "work")


def test_empty_and_duplicate_text(tmp_path):
    db = connect(tmp_path / "brain.db")
    with pytest.raises(ValueError):
        add_document(db, "x", "y", "  ")
    assert search(db, "!!!") == []
    assert answer(db, "ничего")["citations"] == []


def test_ssrf_rejects_local_before_request():
    with pytest.raises(ValueError, match="public"):
        fetch_url("http://127.0.0.1/private")
    with pytest.raises(ValueError):
        fetch_url("file:///etc/passwd")


def test_http_connection_uses_the_ip_that_was_validated(monkeypatch):
    connected = []

    class FakeSocket:
        def settimeout(self, timeout):
            pass

        def connect(self, address):
            connected.append(address)

        def close(self):
            pass

    class FakeConnection:
        def __init__(self, host, port, timeout):
            self.sock = None
            self.host = host
            self.port = port

        def request(self, method, path, headers):
            assert (method, path) == ("GET", "/docs?q=1")

        def getresponse(self):
            return object()

    monkeypatch.setattr(
        second_brain, "_resolve_public", lambda host, port: (socket.AF_INET, (socket.SOCK_STREAM, 0, ("8.8.8.8", port)))
    )
    monkeypatch.setattr(second_brain.socket, "socket", lambda *args: FakeSocket())
    monkeypatch.setattr(second_brain.http.client, "HTTPConnection", FakeConnection)
    second_brain._open_pinned(urllib.parse.urlsplit("http://public.example/docs?q=1"), 2)
    assert connected == [("8.8.8.8", 80)]


def test_redirect_to_private_address_is_rejected(monkeypatch):
    calls = []
    original_resolver = second_brain._resolve_public

    class FakeResponse:
        status = 302

        def getheader(self, name, default=None):
            return "http://127.0.0.1/private" if name == "Location" else default

        def read(self, size):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def open_checked(parsed, timeout):
        calls.append(parsed.hostname)
        second_brain._resolve_public(parsed.hostname, parsed.port or 80)
        return FakeConnection(), FakeResponse()

    def resolve_for_test(host, port):
        if host == "example.com":
            return socket.AF_INET, (socket.SOCK_STREAM, 0, ("8.8.8.8", port))
        return original_resolver(host, port)

    monkeypatch.setattr(second_brain, "_resolve_public", resolve_for_test)
    monkeypatch.setattr(second_brain, "_open_pinned", open_checked)
    with pytest.raises(ValueError, match="public"):
        fetch_url("http://example.com/start")
    assert calls == ["example.com", "127.0.0.1"]


def test_pdf_requires_pdf_magic():
    with pytest.raises(ValueError, match="not a PDF"):
        extract_pdf(b"plain text")
