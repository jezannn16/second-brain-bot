import socket
import urllib.parse

import pytest

import second_brain
from second_brain import add_document, answer, connect, extract_pdf, fetch_url, search
from telegram_bot import _query_filters, _query_tag, handle_update, parse_allowlist


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


def test_telegram_search_filters_pagination_and_group_privacy(tmp_path):
    class FakeAPI:
        def __init__(self):
            self.sent = []

        def send(self, chat_id, text):
            self.sent.append(text)

    api = FakeAPI()
    db_path = str(tmp_path / "brain.db")
    db = connect(db_path)
    for index in range(6):
        add_document(db, "telegram-note", f"Alpha {index}", f"alpha phrase record {index}", "work", owner_id=50)
    db.commit()
    db.close()

    def update(text, chat_type="private"):
        return {"message": {"from": {"id": 50}, "chat": {"id": 50, "type": chat_type}, "text": text}}

    handle_update(update("/search alpha phrase --tag work --source telegram-note"), api, db_path, {50}, {})
    assert "страница 1" in api.sent[-1] and "/more 1" in api.sent[-1] and "--source" in api.sent[-1]
    handle_update(update("/more 1 alpha phrase --tag work --source telegram-note"), api, db_path, {50}, {})
    assert "страница 2" in api.sent[-1] and "Alpha 5" in api.sent[-1] and "/more 2" not in api.sent[-1]
    handle_update(update("/ask alpha phrase --source telegram-note --after 2026-10-29"), api, db_path, {50}, {})
    assert "telegram-note" in api.sent[-1]

    sent_before_group_message = len(api.sent)
    handle_update(update("/help", "group"), api, db_path, {50}, {})
    assert len(api.sent) == sent_before_group_message

    assert _query_filters('русский запрос --source "https://example.org/a b" --after 2026-10-01')[1]["source"] == "https://example.org/a b"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _query_filters("query --before yesterday")


def test_telegram_large_file_confirmation_and_expiry(tmp_path):
    class FakeAPI:
        def __init__(self):
            self.sent = []

        def send(self, chat_id, text):
            self.sent.append(text)

        def download(self, file_id):
            assert file_id == "file-1"
            return "notes.txt", b"A private uploaded note with sample content"

    api = FakeAPI()
    db_path = str(tmp_path / "brain.db")
    pending = {}
    update = {
        "message": {
            "from": {"id": 70}, "chat": {"id": 70, "type": "private"},
            "document": {"file_id": "file-1", "file_name": "notes.txt", "file_size": 2_500_000},
        }
    }
    handle_update(update, api, db_path, {70}, pending)
    assert pending[70]["file_id"] == "file-1" and "подтвердите" in api.sent[-1].lower()
    handle_update({"message": {"from": {"id": 70}, "chat": {"id": 70, "type": "private"}, "text": "/confirm"}}, api, db_path, {70}, pending)
    db = connect(db_path)
    assert db.execute("select count(*) from documents where owner_id=70").fetchone()[0] == 1
    db.close()
    handle_update(
        {"message": {"from": {"id": 70}, "chat": {"id": 70, "type": "private"}, "text": "/confirm"}},
        api, db_path, {70}, {70: {"file_id": "file-1", "name": "notes.txt", "created_at": 0}},
    )
    assert "ожидающей подтверждения" in api.sent[-1]


def test_malformed_telegram_user_ids_are_ignored(tmp_path):
    class FakeAPI:
        def send(self, chat_id, text):
            raise AssertionError("malformed update must be ignored")

    handle_update(
        {"message": {"from": {"id": "invalid"}, "chat": {"id": 1, "type": "private"}, "text": "/help"}},
        FakeAPI(), str(tmp_path / "brain.db"), {1}, {},
    )


def test_empty_and_duplicate_text(tmp_path):
    db = connect(tmp_path / "brain.db")
    with pytest.raises(ValueError):
        add_document(db, "x", "y", "  ")
    assert search(db, "!!!") == []
    assert answer(db, "ничего")["citations"] == []


def test_search_source_and_inclusive_date_filters(tmp_path):
    db = connect(tmp_path / "brain.db")
    first, _ = add_document(db, "https://docs.example/a", "First", "русский поиск для заметки", "work")
    second, _ = add_document(db, "file.md", "Second", "русский поиск для файла", "work")
    db.execute("UPDATE documents SET created_at='2026-10-01T12:00:00+00:00' WHERE id=?", (first,))
    db.execute("UPDATE documents SET created_at='2026-10-02T12:00:00+00:00' WHERE id=?", (second,))
    db.commit()

    hits = search(db, "русский поиск", "work", source="file.md", created_after="2026-10-02", created_before="2026-10-02")
    assert [item["id"] for item in hits] == [second]
    assert answer(db, "русский поиск", source="https://docs.example/a", created_before="2026-10-01")["citations"][0]["document_id"] == first
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        search(db, "русский", created_after="yesterday")
    with pytest.raises(ValueError, match="on or before"):
        search(db, "русский", created_after="2026-10-03", created_before="2026-10-02")


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


def test_fetch_url_closes_http_connection_without_context_manager(monkeypatch):
    closed = []

    class FakeConnection:
        def close(self):
            closed.append(True)

    class FakeResponse:
        status = 200

        def getheader(self, name, default=None):
            return "text/html; charset=utf-8" if name == "Content-Type" else default

        def read(self, size):
            assert size == second_brain.MAX_BYTES + 1
            return b"<html><title>Test</title><body>Readable page</body></html>"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(second_brain, "_open_pinned", lambda parsed, timeout: (FakeConnection(), FakeResponse()))

    source, content = fetch_url("https://public.example")

    assert source == "https://public.example"
    assert content == "Test Readable page"
    assert closed == [True]


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
        def close(self):
            pass

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
