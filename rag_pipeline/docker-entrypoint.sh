# docker-entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

# Who we're going to drop to
APP_USER="${NB_USER:-app}"
APP_UID="${NB_UID:-1000}"
APP_GID="${NB_GID:-1000}"

# Paths we care about (bind mounts land here)
DIRS=(
  /app/data
  /app/data/processed
  /app/data/state
  /app/data/vector_store
  /app/logs
  "$HF_HOME"
  "$TORCH_HOME"
)

# Create + fix ownership/permissions so the app user can write
for d in "${DIRS[@]}"; do
  mkdir -p "$d"
  chown -R "$APP_UID:$APP_GID" "$d"
  chmod -R ug+rwX "$d" || true
done

# Drop privileges and run. If the first argument is an executable command
# (e.g. `make benchmark` runs `python scripts/benchmark_overfetch.py`), run it
# directly; otherwise treat the arguments as main.py subcommands (preprocess,
# index, query, ...).
case "${1:-}" in
  python|python3|bash|sh|pytest)
    exec gosu "$APP_UID:$APP_GID" "$@"
    ;;
  *)
    exec gosu "$APP_UID:$APP_GID" python /app/main.py "$@"
    ;;
esac
