# Cassette Learning

> A skeuomorphic, conversation-first language tutor. Four "drawers" (chat /
> expression-practice / scenario / material) plus a cassette / loop-player
> frontend. Runs fully offline with **zero credentials**; MIT licensed.

[中文说明](README.md)

## What it is

Cassette Learning is an open-source slice of a larger voice-AI language tutor.
It keeps the parts a learner actually touches:

- **Four drawers** — all share one teaching core (`tutor_agent`):
  - **Chat drawer** — free conversation, the persona remembers you across turns.
  - **Expression-practice drawer** — you give a topic, it drills natural phrasing.
  - **Scenario drawer** — role-play a situation (shop assistant, interview…) in-character.
  - **Material drawer** — paste text / upload audio; it analyzes and turns it into a study pack.
- **Cassette / loop-player frontend** — a retro tape-deck UI for repeating and looping audio,
  the "复读机" metaphor for spaced repetition.
- **A persona with long-term memory** — not a FAQ bot; it has opinions, gets attached to
  topics, and grows a relationship with each user (per-account isolation).

It is **conversation-first**: no fixed scripts, no "ask → must say goodbye → close" flows.
What it says next depends on the live state, not a menu.

## Status: unfinished snapshot (read this first)

> This is a **mid-development snapshot** extracted from a larger product for
> open-sourcing — not a finished product. Below: what is verified vs. what is
> not yet done.

**Verified** (see [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)):
- E0 static: no dangling imports, frontend JS syntax, secret scan 0 hits, no credential files.
- E1 unit: backend `agent` 536 + `test_runner` 431 cases green (14 inherited reds triaged).
- E2 service: full stack up with zero keys; four drawers / material library / wordbook / session all 200.
- E3 browser: four drawers + material import (text→analyze→pack) clicked through, 0 frontend errors.

**Not done / not verified (this is a work in progress, not bug-free)**
1. **Real voice conversation unverified**: the proxy runs text-only without keys; ASR was probed OK but **reply generation never completed** because all model keys were dead (account/quota, not a code defect). Re-run per `docs/ACCEPTANCE.md` §7 once a key is available.
2. **AI material analysis needs a key**: offline, `hooks / top_chunks / summary` are empty (the pack still creates, just without analysis).
3. **Two known minor issues unfixed** (logged in the acceptance report, non-blocking): scenario drawer mislabels as "chat drawer"; material-pack analysis empty without a key.

## Quick start

Requirements: **Python 3.12**, **Node.js 18+**. No API keys needed — the voice/AI
proxy starts in a text-only mode and everything except ASR/TTS/model calls works.

```bash
# 1) backend (FastAPI)
cd backend/app
python -m venv .venv && .venv\Scripts\activate      # Windows
#   macOS/Linux: python3 -m venv .venv && source .venv/bin/activate
pip install -r ../../requirements.txt
WX_DEV_MODE=1 MEMORY_BACKEND=local uvicorn main:app --host 127.0.0.1 --port 8002

# 2) voice / AI proxy (pure-text mode without keys)
cd ../../voice
npm install
node doubao-session-server.mjs            # :3211

# 3) frontend (static page)
cd ../frontend
python -m http.server 8080 --bind 127.0.0.1
```

Open <http://127.0.0.1:8080>. Log in (dev mode, no password), open the library,
create a drawer, import a material — all reachable with zero credentials.

> To enable real ASR/TTS/model replies, set the keys in `.env`
> (see `.env.example`) and restart the proxy. Every key is read from the
> environment only; nothing is hard-coded.

## Project layout

```
backend/app/    FastAPI + tutor_agent teaching core + unit tests
voice/          doubao-session-server.mjs (text mode without keys) + 2 adapters
frontend/       single-page cassette / loop-player UI
docs/           extraction notes + acceptance report (E0–E3)
tools/          frontend JS syntax check
```

## Tests

Backend uses **unittest** (not pytest). The dummy key just keeps the director
module from degrading; tests use a fake client and never hit the network.

```bash
cd backend/app
MEMORY_BACKEND=local CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s agent -p "test_*.py"

MEMORY_BACKEND=local WX_DEV_MODE=1 CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s test_runner -p "test_*.py"
```

Both suites are green (`agent` 536 cases, `test_runner` 431 cases). The 14
inherited reds were triaged one by one — 13 were tests missing a precondition
(feature gate off / proxy not bypassed), 1 asserted an obsolete product rule and
was deleted. See `docs/ACCEPTANCE.md` §1.

Frontend: `py -3.12 tools/check_frontend_js.py`.

## Secrets / privacy

- Hard-coded device/user UUIDs from the parent product were replaced with a
  **runtime-generated random UUID** stored in `localStorage` — source ships no
  real account identity.
- Whole-package scan for `ark-` / `sk-` / real account UUID / author paths /
  LAN IP: **0 hits**. Keys live only in `.env` (git-ignored); the repo has an
  empty `.env.example`.
- The frontend stores audio base64 in `localStorage` (`cassette-state-v2`) —
  existing product behavior; be mindful of the local-storage quota and the
  "voice data on disk" implication when extending it.

## License

MIT — © 2026 WCGG147. The voice-protocol adapters in `voice/` and the 52
original teaching-methodology notes in `backend/knowledge_base/chunks/` are
original content, released under the same license.
