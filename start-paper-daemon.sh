#!/bin/bash
# Start the pm-bot S1 paper-trading daemon (dry-run, no real orders).
# Encapsulates the env fixes learned 2026-10-19:
#  - PM_BOT_CONFIG: daemon.cities/strategies live in config.toml; without it the
#    daemon refuses to start in dry-run mode ("requires explicit --cities").
#  - Sanitized no_proxy: httpx 0.28.1 crashes parsing the runtime default
#    no_proxy (bracketed IPv6 like [::1] -> "Invalid port: ':1]'").
#    The egress proxy vars (https_proxy etc.) MUST stay: direct egress is blocked.
#  - Deps: runtime restarts can partially wipe .venv; reinstall the small set
#    the daemon needs if the import check fails.
set -e
cd /home/hatch/workspace/pm-bot
export PM_BOT_CONFIG=/home/hatch/workspace/pm-bot/config.toml
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
.venv/bin/python -c "import typer, aiosqlite, pydantic, tomlkit" 2>/dev/null \
  || .venv/bin/pip install -q typer aiosqlite pydantic tomlkit python-dotenv
setsid .venv/bin/python -m pm_bot daemon start --dry-run --bankroll 1000 \
  >> ~/.pm-bot/daemon.log 2>&1 < /dev/null &
echo "paper daemon launched, pid $!"
