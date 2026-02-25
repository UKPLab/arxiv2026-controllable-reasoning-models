#!/usr/bin/env bash
# ---------------------------------------
# Load environment variables from .env
# into the current shell (no new process).
# ---------------------------------------


if [ ! -f .env ]; then
  echo "❌ .env file not found!"
  return 1 2>/dev/null || exit 1
fi

# Export all variables defined in .env
set -a
source .env
set +a

echo "✅ Environment variables loaded from .env"