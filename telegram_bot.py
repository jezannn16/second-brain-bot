"""Allow-listed Telegram long-poll bot; tokens are read only from environment."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from second_brain import MAX_BYTES, add_document, answer, connect, extract_pdf, search

MAX_TELEGRAM_PAGE = 5
CONFIRM_BYTES = 2 * 1024 * 1024


def parse_allowlist(raw: str) -> set[int]:
    try:
        ids = {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("TELEGRAM_ALLOWED_USER_IDS must contain numeric Telegram user IDs") from exc
    if not ids:
        raise ValueError("Set TELEGRAM_ALLOWED_USER_IDS; the bot fails closed without an allowlist")
    return ids


class BotAPI:
    def __init__(self, token: str, timeout: float = 35) -> None:
        if not token or ":" not in token:
            raise ValueError("Set TELEGRAM_BOT_TOKEN from BotFather")
        self._base = f"https://api.telegram.org/bot{token}/"
        self._file_base = f"https://api.telegram.org/file/bot{token}/"
        self._timeout = timeout

    def call(self, method: str, payload: dict[str, object]) -> object:
        body = urllib.parse.urlencode(payload).encode()
        request = urllib.request.Request(
            self._base + method, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                result = json.loads(response.read(2_000_000))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Telegram API request failed; check network and bot settings") from exc
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError("Telegram API rejected the request; check bot settings")
        return result.get("result")

    def download(self, file_id: str) -> tuple[str, bytes]:
        info = self.call("getFile", {"file_id": file_id})
        if not isinstance(info, dict):
            raise TypeError("Telegram did not return file metadata")
        file_path = str(info.get("file_path", ""))
        size = int(info.get("file_size", 0))
        if size < 1 or size > MAX_BYTES:
            raise ValueError("File must be smaller than 8 MiB")
        # Telegram returns a relative file path; never accept a remote URL from the response.
        if not file_path or file_path.startswith("/") or ".." in Path(file_path).parts:
            raise ValueError("Telegram returned an invalid file path")
        try:
            with urllib.request.urlopen(
                self._file_base + urllib.parse.quote(file_path, safe="/"), timeout=self._timeout
            ) as response:
                data = response.read(MAX_BYTES + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("Could not download the Telegram file") from exc
        if len(data) > MAX_BYTES:
            raise ValueError("File exceeds the 8 MiB limit")
        return file_path, data

    def send(self, chat_id: int, text: str) -> None:
        self.call("sendMessage", {"chat_id": chat_id, "text": text[:3900], "disable_web_page_preview": "true"})


def _format_results(rows: list[dict[str, object]], page: int, query: str) -> str:
    if not rows:
        return "Ничего не найдено. Попробуйте другие слова или сохраните источник через /note."
    parts = [f"Найдено: страница {page + 1}"]
    for row in rows:
        parts.append(f"[{row['id']}] {row['title']} — {row['source']} (v{row['version']})\n{row['snippet']}")
    parts.append(f"Следующая страница: /more {page + 1} {query}")
    return "\n\n".join(parts)


def _query_tag(value: str) -> tuple[str, str]:
    parts = value.split()
    if "--tag" not in parts:
        return value.strip(), ""
    index = parts.index("--tag")
    if index + 1 >= len(parts):
        raise ValueError("Формат фильтра: --tag тег")
    tag = parts[index + 1]
    del parts[index : index + 2]
    return " ".join(parts).strip(), tag


def handle_update(
    update: dict[str, object], api: BotAPI, db_path: str, allowed: set[int], pending_files: dict[int, dict[str, object]]
) -> None:
    message = update.get("message")
    if not isinstance(message, dict):
        return
    user = message.get("from")
    chat = message.get("chat")
    if not isinstance(user, dict) or not isinstance(chat, dict):
        return
    user_id = int(user.get("id", 0))
    chat_id = int(chat.get("id", 0))
    if user_id not in allowed:
        return
    if chat.get("type") != "private":
        api.send(chat_id, "Use the bot in a private chat so your sources stay private.")
        return
    text = str(message.get("text", "")).strip()
    command, _, argument = text.partition(" ")
    command = command.split("@", 1)[0].lower()
    db = connect(db_path)
    try:
        if command in ("/start", "/help"):
            api.send(
                chat_id,
                "Second Brain: /note title | text | tag1,tag2; /search query --tag tag; /more page query; /ask question; /url https://…; /delete ID. Также можно прислать TXT/MD/PDF до 8 MiB.",
            )
        elif command == "/note":
            fields = [x.strip() for x in argument.split("|", 2)]
            if len(fields) < 2 or not fields[0] or not fields[1]:
                api.send(chat_id, "Формат: /note заголовок | текст | тег1,тег2")
            else:
                _, created = add_document(
                    db, "telegram-note", fields[0], fields[1], fields[2] if len(fields) == 3 else "", user_id
                )
                db.commit()
                api.send(chat_id, "Заметка сохранена." if created else "Такая заметка уже сохранена.")
        elif command in ("/search", "/more"):
            if command == "/more":
                page_text, _, query = argument.partition(" ")
                try:
                    page = int(page_text)
                    if page < 0:
                        raise ValueError
                except ValueError:
                    api.send(chat_id, "Формат: /more номер_страницы запрос")
                    return
            else:
                page, query = 0, argument
            query, tag = _query_tag(query)
            if not query:
                api.send(chat_id, "Добавьте слова для поиска: /search текст")
                return
            rows = search(
                db, query, tag=tag, limit=MAX_TELEGRAM_PAGE, owner_id=user_id, offset=page * MAX_TELEGRAM_PAGE
            )
            api.send(chat_id, _format_results(rows, page, query))
        elif command == "/ask":
            result = answer(db, argument, owner_id=user_id)
            response = [str(result["answer"])]
            for citation in result["citations"]:
                response.append(
                    f"[{citation['document_id']}] {citation['title']} — {citation['source']} (v{citation['version']})\n{citation['passage']}"
                )
            api.send(chat_id, "\n\n".join(response))
        elif command == "/url":
            if not argument:
                api.send(chat_id, "Формат: /url https://example.org/page")
                return
            from second_brain import fetch_url

            source, content = fetch_url(argument)
            _, created = add_document(db, source, source, content, owner_id=user_id)
            db.commit()
            api.send(chat_id, "URL сохранён." if created else "Такое содержимое уже сохранено.")
        elif command == "/delete":
            try:
                doc_id = int(argument)
            except ValueError:
                api.send(chat_id, "Формат: /delete ID")
                return
            removed = db.execute("DELETE FROM documents WHERE id=? AND owner_id=?", (doc_id, user_id)).rowcount
            db.commit()
            api.send(chat_id, "Источник удалён." if removed else "Источник не найден.")
        elif command == "/confirm":
            pending = pending_files.pop(user_id, None)
            if pending is None or time.time() - float(pending.get("created_at", 0)) > 600:
                api.send(chat_id, "Нет загрузки, ожидающей подтверждения.")
                return
            _save_telegram_file(api, db, user_id, chat_id, pending)
        elif isinstance(message.get("document"), dict):
            document = message["document"]
            name = str(document.get("file_name", "document"))
            size = int(document.get("file_size", 0))
            if Path(name).suffix.lower() not in (".txt", ".md", ".pdf"):
                api.send(chat_id, "Поддерживаются TXT, Markdown и PDF.")
                return
            if size < 1 or size > MAX_BYTES:
                api.send(chat_id, "Файл должен быть не больше 8 MiB.")
                return
            info = {"file_id": str(document.get("file_id", "")), "name": name}
            if size > CONFIRM_BYTES:
                info["created_at"] = time.time()
                api.send(
                    chat_id,
                    "Файл больше 2 MiB. Подтвердите импорт командой /confirm (содержимое будет сохранено локально).",
                )
            else:
                _save_telegram_file(api, db, user_id, chat_id, info)
        else:
            api.send(chat_id, "Команда не распознана. Отправьте /help.")
    except (ValueError, OSError, RuntimeError, UnicodeError) as exc:
        db.rollback()
        api.send(chat_id, f"Не удалось обработать запрос: {exc}")
    finally:
        db.close()


def _save_telegram_file(api: BotAPI, db, user_id: int, chat_id: int, info: dict[str, object]) -> None:
    name = str(info["name"])
    _, data = api.download(str(info["file_id"]))
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        content = extract_pdf(data)
    else:
        content = data.decode("utf-8")
    doc_id, created = add_document(db, f"telegram:{name}", name, content, owner_id=user_id)
    db.commit()
    api.send(chat_id, f"Источник {'сохранён' if created else 'уже есть'}; ID {doc_id}.")


def run() -> int:
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        allowed = parse_allowlist(os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""))
        db_path = os.environ.get("SECOND_BRAIN_DB", "second_brain.sqlite3")
        api = BotAPI(token)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    offset = 0
    pending_files: dict[int, dict[str, object]] = {}
    backoff = 1
    while True:
        try:
            updates = api.call(
                "getUpdates", {"offset": offset, "timeout": 30, "allowed_updates": json.dumps(["message"])}
            )
            for update in updates if isinstance(updates, list) else []:
                if isinstance(update, dict):
                    handle_update(update, api, db_path, allowed, pending_files)
                    offset = max(offset, int(update.get("update_id", 0)) + 1)
            backoff = 1
        except (RuntimeError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == "__main__":
    raise SystemExit(run())
