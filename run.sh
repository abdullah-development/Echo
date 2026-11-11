#!/bin/sh
# Starts the agent.
#   ./run.sh v1   version 1: fast transcript only (base.en)
#   ./run.sh v2   version 2: hybrid - fast transcript now, accurate correction a few seconds later
#   ./run.sh v3   version 3: Parakeet v3 speech-to-text (most accurate and fast) + jargon fixer
#   ./run.sh v4   version 4: version 3 + text-only sound setting (browser panel)
#   ./run.sh app  version 4 in a compact always-on-top desktop window (Electron, desktop/)
# Extra args are passed through (e.g. ./run.sh v2 --simulate talk.mp3). Default: v1.
set -e
cd "$(dirname "$0")"
[ -x bin/audiotap ] && [ bin/audiotap -nt audiotap/main.swift ] || ./build.sh
if [ "$1" = app ]; then
  cd desktop
  [ -d node_modules ] || npm install
  exec npm start
fi
case "$1" in
  v1|1) shift; pkg=agent_v1 ;;
  v2|2) shift; pkg=agent ;;
  v3|3) shift; pkg=agent_v3 ;;
  v4|4) shift; pkg=agent_v4 ;;
  *) pkg=agent_v1 ;;
esac
case "$pkg" in agent) name="version 2 (hybrid)" ;; agent_v3) name="version 3 (Parakeet)" ;; agent_v4) name="version 4 (assistant)" ;; *) name="version 1" ;; esac
echo "Starting $name..."
exec uv run python -m "$pkg.app" "$@"
