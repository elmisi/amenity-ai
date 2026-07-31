#!/bin/sh
# Fails if the user's private ds4 hostname (read from the local config,
# which lives outside the repo) appears anywhere in tracked files.
HOST=$(python3 -c '
import json, os
from urllib.parse import urlparse
path = os.path.expanduser("~/.config/amenity-stuff/config.json")
try:
    with open(path) as fh:
        cfg = json.load(fh)
    print(urlparse(cfg.get("ds4_base_url", "")).hostname or "")
except Exception:
    print("")
')
if [ -z "$HOST" ] || [ "$HOST" = "localhost" ] || [ "$HOST" = "127.0.0.1" ]; then
    echo "OK (no private host configured)"
    exit 0
fi
if git grep -qiF "$HOST" -- . 2>/dev/null; then
    echo "LEAK: private hostname found in tracked files:"
    git grep -inF "$HOST" -- .
    exit 1
fi
echo "OK"
