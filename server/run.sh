#!/usr/bin/env bash
# single warm worker: artifacts are big + read-only, one process is enough (README §5)
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
