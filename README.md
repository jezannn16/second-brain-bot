# Second Brain

Local-first notes and document library with deterministic full-text baseline retrieval. Python 3.11+, SQLite, no account or API key required.

## Implemented

- Save notes and import Markdown/text/PDF files (PDF extraction uses optional `pypdf`). 8 MiB input ceiling, extension and PDF signature checks, source/hash/tag/status metadata.
- Import public HTTP(S) HTML/text URLs with timeout, bounded body, content-type check, limited redirects and public-IP validation on each redirect.
- Search across Russian/English text, filter by exact tag, show source snippets, export JSON, delete by id. Duplicate content hashes are skipped.
- `ask` is retrieval-only and returns source passages or a clear no-evidence refusal. There is no LLM, so untrusted source text cannot issue tool actions.
- Optional Telegram long-poll bot for allow-listed users: save/search/ask/delete, source citations, URL ingestion and TXT/MD/PDF uploads; records are isolated by Telegram user ID. Files above 2 MiB require `/confirm`, with an 8 MiB maximum.

## Quick start

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install pytest ruff
# Optional PDF extraction:
python -m pip install pypdf
python second_brain.py --db brain.sqlite3 note 'Reading list' 'Проверить заметки по SQL.' --tags work,sql
python second_brain.py --db brain.sqlite3 search 'заметки SQL' --tag work
python second_brain.py --db brain.sqlite3 ask 'заметки SQL'
python second_brain.py --db brain.sqlite3 export
pytest -q
```

To run Telegram, create a bot with BotFather, copy `.env.example` to `.env`, set the token and numeric user allowlist, then load and run it:

```sh
set -a; . .env; set +a
python telegram_bot.py
```

The bot exits with a configuration error when the allowlist is missing. It polls only; it does not run a public webhook server.

## Limitations / planned

No vector search, LLM generation/evaluation, or screenshot. PDF extraction is optional and scanned/image-only documents may yield no text. URL fetching resolves the hostname, rejects any non-global address, and connects directly to that exact resolved socket address; redirects are limited to three and validated again, while HTTPS retains certificate and hostname verification. Search citations include document ID and content-hash version; PDF extraction prefixes page numbers in extracted text. SQLite stores raw content and Telegram user IDs unencrypted. Search is a simple token-frequency baseline without stemming, date filters or ranking calibration. The bot uses long polling and requires a stable outbound internet connection.

See [demo steps](docs/demo.md) for a synthetic end-to-end walkthrough.
