"""Private local knowledge store with deterministic retrieval and citations."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import sqlite3
import ssl
import sys
import urllib.parse
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from rag import ProviderAnswer, RAGProvider, RAGProviderError, provider_from_env

MAX_BYTES = 8 * 1024 * 1024


def connect(path: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    if os.name == "posix" and str(path) != ":memory:":
        try:
            os.chmod(path, 0o600)
        except OSError:
            db.close()
            raise RuntimeError("Could not restrict the database file to the current user") from None
    db.row_factory = sqlite3.Row
    db.executescript("""
    CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL,
      content_hash TEXT NOT NULL, owner_id INTEGER NOT NULL DEFAULT 0, tags TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'ready', UNIQUE(owner_id,content_hash));
    CREATE INDEX IF NOT EXISTS idx_docs_created ON documents(created_at);
    """)
    return db


def add_document(
    db: sqlite3.Connection, source: str, title: str, content: str, tags: str = "", owner_id: int = 0
) -> tuple[int, bool]:
    content = content.strip()
    if not content:
        raise ValueError("Document is empty")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    found = db.execute("SELECT id FROM documents WHERE content_hash=? AND owner_id=?", (digest, owner_id)).fetchone()
    if found:
        return int(found[0]), False
    cur = db.execute(
        "INSERT INTO documents(source,title,content,content_hash,owner_id,tags,created_at) VALUES(?,?,?,?,?,?,?)",
        (
            source,
            title.strip() or source,
            content,
            digest,
            owner_id,
            ",".join(sorted({t.strip().lstrip("#") for t in tags.split(",") if t.strip()})),
            datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )
    return int(cur.lastrowid), True


def extract_pdf(data: bytes) -> str:
    if len(data) > MAX_BYTES:
        raise ValueError("File exceeds the 8 MiB limit")
    if not data.startswith(b"%PDF-"):
        raise ValueError("File is not a PDF")
    try:
        import io

        from pypdf import PdfReader

        return "\n".join(
            f"[page {index + 1}] {page.extract_text() or ''}"
            for index, page in enumerate(PdfReader(io.BytesIO(data)).pages)
        )
    except ImportError as exc:
        raise ValueError("PDF extraction requires optional dependency: pip install pypdf") from exc
    except Exception as exc:
        raise ValueError("Could not extract text from PDF") from exc


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _resolve_public(host: str, port: int) -> tuple[int, tuple]:
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
        infos = socket.getaddrinfo(str(literal), port, type=socket.SOCK_STREAM)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError("Could not resolve URL host") from exc
    addresses = {ipaddress.ip_address(row[4][0].split("%", 1)[0]) for row in infos}
    if not addresses or any(not ip.is_global for ip in addresses):
        raise ValueError("URL host must resolve only to public IP addresses")
    family, socktype, proto, _, sockaddr = infos[0]
    return family, (socktype, proto, sockaddr)


def _open_pinned(
    parsed: urllib.parse.SplitResult, timeout: float
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    family, (socktype, proto, sockaddr) = _resolve_public(host, port)
    sock = socket.socket(family, socktype, proto)
    sock.settimeout(timeout)
    try:
        # Connect to the exact address validated above; do not resolve the hostname again.
        sock.connect(sockaddr)
        if parsed.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        connection: http.client.HTTPConnection = http.client.HTTPConnection(host, port, timeout=timeout)
        connection.sock = sock
        path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request(
            "GET",
            path,
            headers={
                "Host": parsed.netloc,
                "User-Agent": "PersonalAutomationLab/0.1",
                "Accept": "text/html,text/plain",
                "Connection": "close",
            },
        )
        return connection, connection.getresponse()
    except Exception:
        sock.close()
        raise


def fetch_url(url: str, timeout: float = 8) -> tuple[str, str]:
    """Fetch text from public HTTP(S), pinning each checked DNS result to its socket."""
    current = url
    for _ in range(4):
        parsed = urllib.parse.urlsplit(current)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Only public http(s) URLs without credentials are supported")
        try:
            connection, response = _open_pinned(parsed, timeout)
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise ValueError("Could not fetch URL; check address and connectivity") from exc
        try:
            with response:
                if response.status in (301, 302, 303, 307, 308) and response.getheader("Location"):
                    current = urllib.parse.urljoin(current, response.getheader("Location"))
                    continue
                if response.status < 200 or response.status >= 300:
                    raise ValueError(f"URL fetch failed with HTTP {response.status}")
                content_type = response.getheader("Content-Type", "text/plain").split(";", 1)[0].strip().lower()
                ctype = content_type
                if ctype not in ("text/html", "text/plain"):
                    raise ValueError("URL must return text/html or text/plain")
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("URL content exceeds the 8 MiB limit")
                charset = "utf-8"
                content_type_header = response.getheader("Content-Type", "")
                for part in content_type_header.split(";")[1:]:
                    key, _, value = part.strip().partition("=")
                    if key.lower() == "charset" and value.strip('"'):
                        charset = value.strip('"')
                try:
                    decoded = raw.decode(charset, errors="replace")
                except LookupError as exc:
                    raise ValueError("URL uses an unsupported text encoding") from exc
                if ctype == "text/html":
                    parser = TextExtractor()
                    parser.feed(decoded)
                    decoded = " ".join(parser.parts)
                return current, re.sub(r"\s+", " ", decoded).strip()
        finally:
            connection.close()
    raise ValueError("Too many redirects")


def search(
    db: sqlite3.Connection,
    query: str,
    tag: str = "",
    limit: int = 10,
    owner_id: int = 0,
    offset: int = 0,
    source: str = "",
    created_after: str = "",
    created_before: str = "",
) -> list[dict[str, object]]:
    for value, label in ((created_after, "--after"), (created_before, "--before")):
        if value:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(f"{label} must use YYYY-MM-DD") from exc
    if created_after and created_before and created_after > created_before:
        raise ValueError("--after must be on or before --before")
    words = [w.casefold() for w in re.findall(r"[\w-]+", query) if len(w) > 1]
    if not words:
        return []
    sql = "SELECT * FROM documents WHERE status='ready' AND owner_id=?"
    params: list[object] = [owner_id]
    if source:
        sql += " AND source=?"
        params.append(source)
    if created_after:
        sql += " AND substr(created_at,1,10)>=?"
        params.append(created_after)
    if created_before:
        sql += " AND substr(created_at,1,10)<=?"
        params.append(created_before)
    rows = db.execute(sql + " ORDER BY created_at DESC", params).fetchall()
    results = []
    for row in rows:
        tags = [t.strip() for t in row["tags"].split(",") if t.strip()]
        if tag and tag.casefold() not in {t.casefold() for t in tags}:
            continue
        content = row["content"]
        score = sum(content.casefold().count(word) for word in words)
        if score:
            first = min((content.casefold().find(w) for w in words if w in content.casefold()), default=0)
            page_marker = content.rfind("[page ", 0, first)
            snippet_start = page_marker if page_marker >= 0 else max(0, first - 90)
            results.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "source": row["source"],
                    "version": row["content_hash"][:12],
                    "tags": tags,
                    "score": score,
                    "snippet": content[snippet_start : min(len(content), first + 220)],
                    "offset_start": snippet_start,
                    "offset_end": min(len(content), first + 220),
                    "page": int(page_match.group(1)) if (page_match := re.match(r"\[page (\d+)\]", content[snippet_start:])) else None,
                    "content": content,
                }
            )
    start = max(0, offset)
    return sorted(results, key=lambda x: (-int(x["score"]), int(x["id"])))[start : start + max(1, min(limit, 50))]


def answer(
    db: sqlite3.Connection,
    question: str,
    tag: str = "",
    owner_id: int = 0,
    source: str = "",
    created_after: str = "",
    created_before: str = "",
    provider: RAGProvider | None = None,
) -> dict[str, object]:
    evidence = search(
        db, question, tag, owner_id=owner_id, source=source,
        created_after=created_after, created_before=created_before
    )
    if not evidence:
        return {
            "answer": "В базе знаний не найдено достаточных источников для ответа.",
            "citations": [],
            "mode": "abstained" if provider is not None else "retrieval-only",
        }
    if provider is not None:
        if not _has_sufficient_lexical_evidence(question, evidence):
            return {
                "answer": "В найденных фрагментах недостаточно совпадающих данных для безопасного ответа.",
                "citations": [],
                "mode": "abstained",
            }
        try:
            generated = provider.generate(question, evidence[:5])
            if not isinstance(generated, ProviderAnswer):
                raise RAGProviderError("Optional LLM returned an invalid answer object")
            if (
                not isinstance(generated.answer, str)
                or not isinstance(generated.citation_ids, list)
                or any(type(doc_id) is not int for doc_id in generated.citation_ids)
            ):
                raise RAGProviderError("Optional LLM output did not match the required answer schema")
            answer_text = generated.answer.strip()
            citation_ids = generated.citation_ids
            allowed_ids = {int(item["id"]) for item in evidence}
            if len(citation_ids) != len(set(citation_ids)) or any(doc_id not in allowed_ids for doc_id in citation_ids):
                raise RAGProviderError("Optional LLM cited a source outside the retrieved evidence")
            if not answer_text:
                if citation_ids:
                    raise RAGProviderError("Optional LLM returned citations without an answer")
                return {
                    "answer": "В найденных фрагментах недостаточно данных для ответа.",
                    "citations": [],
                    "mode": "abstained",
                }
            if not citation_ids:
                raise RAGProviderError("Optional LLM answer did not cite any retrieved source")
            cited = [item for item in evidence if int(item["id"]) in citation_ids]
            return {"answer": answer_text, "citations": _citations(cited), "mode": "llm"}
        except RAGProviderError as exc:
            return {
                "answer": f"{exc}. Доступен безопасный retrieval-only режим; найденные фрагменты приведены ниже.",
                "citations": _citations(evidence),
                "mode": "retrieval-fallback",
            }
    return {
        "answer": "Найдены фрагменты, связанные с вопросом. Проверьте цитаты ниже; генеративная модель не подключена.",
        "citations": _citations(evidence),
        "mode": "retrieval-only",
    }


def _citations(evidence: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "document_id": x["id"],
            "source": x["source"],
            "title": x["title"],
            "version": x["version"],
            "passage": x["snippet"],
            "offset_start": x["offset_start"],
            "offset_end": x["offset_end"],
            "page": x["page"],
        }
        for x in evidence
    ]


def _has_sufficient_lexical_evidence(question: str, evidence: list[dict[str, object]]) -> bool:
    terms = set(re.findall(r"[\w-]+", question.casefold()))
    if not terms:
        return False
    required = min(2, len(terms))
    return any(
        sum(term in set(re.findall(r"[\w-]+", str(item.get("content", "")).casefold())) for term in terms)
        >= required
        for item in evidence
    )


def evaluate_retrieval(
    db: sqlite3.Connection, cases: list[dict[str, object]], top_k: int = 5
) -> dict[str, object]:
    """Measure source retrieval, conflicting-source coverage, and no-answer behavior."""
    if not cases:
        raise ValueError("Evaluation set must contain at least one case")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    reciprocal_ranks: list[float] = []
    query_recalls: list[float] = []
    source_recalls: list[float] = []
    answerability_hits = 0
    no_answer_hits = 0
    no_answer_cases = 0
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise TypeError(f"Evaluation case {index} must be an object")
        query_value = case.get("query")
        if not isinstance(query_value, str) or not query_value.strip():
            raise ValueError(f"Evaluation case {index} requires a non-empty query")
        query = query_value.strip()
        if "expected_sources" in case:
            raw_expected = case["expected_sources"]
            if not isinstance(raw_expected, list) or any(not isinstance(source, str) or not source for source in raw_expected):
                raise ValueError(f"Evaluation case {index} expected_sources must be a list of non-empty strings")
            expected_sources = set(raw_expected)
        elif "expected_source" in case:
            raw_expected = case["expected_source"]
            if raw_expected is not None and not isinstance(raw_expected, str):
                raise ValueError(f"Evaluation case {index} expected_source must be a string or null")
            expected_sources = {raw_expected} if raw_expected else set()
        else:
            raise ValueError(f"Evaluation case {index} requires expected_source(s); use [] for no-answer")
        found = search(db, query, limit=top_k)
        found_sources = [str(row["source"]) for row in found]
        answerable = bool(expected_sources)
        retrieved_expected = set(found_sources) & expected_sources
        answerability_hits += bool(found) == answerable
        if not answerable:
            no_answer_cases += 1
            correct_abstention = not found
            no_answer_hits += correct_abstention
            query_recalls.append(float(correct_abstention))
            continue
        rank = next((position for position, source in enumerate(found_sources, 1) if source in expected_sources), None)
        query_recalls.append(len(retrieved_expected) / len(expected_sources))
        source_recalls.append(len(retrieved_expected) / len(expected_sources))
        reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
    count = len(cases)
    return {
        "cases": count,
        "top_k": top_k,
        "recall_at_k": sum(query_recalls) / count,
        "source_recall_at_k": sum(source_recalls) / len(source_recalls) if source_recalls else None,
        "no_answer_accuracy": no_answer_hits / no_answer_cases if no_answer_cases else None,
        "answerability_accuracy": answerability_hits / count,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Local knowledge base")
    p.add_argument("--db", default="second_brain.sqlite3")
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("note")
    x.add_argument("title")
    x.add_argument("text")
    x.add_argument("--tags", default="")
    x = sub.add_parser("file")
    x.add_argument("path")
    x.add_argument("--tags", default="")
    x = sub.add_parser("url")
    x.add_argument("url")
    x.add_argument("--tags", default="")
    x = sub.add_parser("search")
    x.add_argument("query")
    x.add_argument("--tag", default="")
    x.add_argument("--source", default="")
    x.add_argument("--after", dest="created_after", default="")
    x.add_argument("--before", dest="created_before", default="")
    x = sub.add_parser("ask")
    x.add_argument("question")
    x.add_argument("--tag", default="")
    x.add_argument("--source", default="")
    x.add_argument("--after", dest="created_after", default="")
    x.add_argument("--before", dest="created_before", default="")
    sub.add_parser("export")
    x = sub.add_parser("delete")
    x.add_argument("id", type=int)
    x = sub.add_parser("evaluate")
    x.add_argument("file", help="JSON list of {query, expected_source} objects")
    x.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    db = connect(args.db)
    try:
        if args.cmd == "note":
            print(add_document(db, "note", args.title, args.text, args.tags))
        elif args.cmd == "file":
            path = Path(args.path)
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("File exceeds 8 MiB")
            raw = path.read_bytes()
            if path.suffix.lower() == ".pdf":
                content = extract_pdf(raw)
            elif path.suffix.lower() in (".txt", ".md"):
                content = raw.decode("utf-8")
            else:
                raise ValueError("Only .txt, .md, and .pdf are supported")
            print(add_document(db, str(path), path.name, content, args.tags))
        elif args.cmd == "url":
            final, text = fetch_url(args.url)
            print(add_document(db, final, final, text, args.tags))
        elif args.cmd == "search":
            for item in search(
                db, args.query, args.tag, source=args.source,
                created_after=args.created_after, created_before=args.created_before
            ):
                print(f"[{item['id']}] {item['title']} ({item['source']})\n  {item['snippet']}\n")
        elif args.cmd == "ask":
            result = answer(
                db, args.question, args.tag, source=args.source,
                created_after=args.created_after, created_before=args.created_before,
                provider=provider_from_env(),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.cmd == "export":
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT id,source,title,content,tags,created_at,status FROM documents WHERE owner_id=0 ORDER BY id"
                )
            ]
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        elif args.cmd == "delete":
            if db.execute("DELETE FROM documents WHERE id=? AND owner_id=0", (args.id,)).rowcount == 0:
                raise ValueError("Document not found")
            print("Deleted")
        elif args.cmd == "evaluate":
            cases = json.loads(Path(args.file).read_text(encoding="utf-8"))
            if not isinstance(cases, list):
                raise ValueError("Evaluation file must contain a JSON list of objects")
            print(json.dumps(evaluate_retrieval(db, cases, args.top_k), indent=2))
        db.commit()
    except (ValueError, TypeError, OSError, UnicodeError, RAGProviderError) as exc:
        db.rollback()
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
