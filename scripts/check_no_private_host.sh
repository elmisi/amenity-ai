#!/bin/sh
# Fails if a private hostname from the user's local config (the ds4 or ollama
# endpoint — that config lives outside the repo) appears in tracked files.
HOSTS=$(python3 -c '
import json, os
from urllib.parse import urlparse
path = os.path.expanduser("~/.config/amenity-stuff/config.json")
try:
    with open(path) as fh:
        cfg = json.load(fh)
except Exception:
    cfg = {}
for key in ("ds4_base_url", "ollama_base_url"):
    host = urlparse(cfg.get(key) or "").hostname or ""
    if host and host not in ("localhost", "127.0.0.1"):
        print(host)
')
if [ -z "$HOSTS" ]; then
    echo "OK (no private host configured)"
    exit 0
fi
STATUS=0
for HOST in $HOSTS; do
    if git grep -qiF "$HOST" -- . 2>/dev/null; then
        echo "LEAK: private hostname found in tracked files:"
        git grep -inF "$HOST" -- .
        STATUS=1
    fi
done
if [ "$STATUS" -eq 0 ]; then
    echo "OK"
fi
exit "$STATUS"
