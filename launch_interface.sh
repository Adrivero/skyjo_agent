#!/bin/sh

set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Skyjo virtual environment not found at $PYTHON" >&2
    echo "Create it with: uv sync" >&2
    exit 1
fi

cd "$PROJECT_DIR"
exec "$PYTHON" -m interface "$@"
