#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Narou - AI Job Search Assistant"
echo "  First-time setup (macOS / Linux)"
echo "============================================"
echo

cd "$(dirname "$0")"

# Find a Python 3.10+ interpreter
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "")
        if [ -n "$ver" ]; then
            major=${ver%%.*}
            minor=${ver##*.}
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                PY="$cand"
                echo "Using $cand ($ver)"
                break
            fi
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: Python 3.10 or later not found."
    echo
    echo "macOS: install from https://www.python.org/downloads/macos/"
    echo "       or with Homebrew:  brew install python@3.12"
    echo "Linux: sudo apt install python3.12 python3.12-venv    (Debian/Ubuntu)"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
if [ -d .venv ]; then
    echo "    (removing stale .venv)"
    rm -rf .venv
fi
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/4] Upgrading pip and clearing stale index cache..."
python -m pip install --upgrade pip >/dev/null 2>&1 || true
python -m pip cache purge >/dev/null 2>&1 || true

echo "[3/4] Installing dependencies (bypassing any stale pip cache)..."
if ! python -m pip install --no-cache-dir -r requirements.txt; then
    echo
    echo "ERROR: Package install failed."
    echo "If you see 'No matching distribution', your network or mirror may be"
    echo "blocking PyPI. Try from a different network, or set:"
    echo "    export PIP_INDEX_URL=https://pypi.org/simple"
    exit 1
fi

echo "[4/4] Creating data directories..."
mkdir -p data/cache data/labeled data/samples

echo
echo "============================================"
echo "  Setup complete!"
echo
echo "  To start the app, run:  ./run.sh"
echo "============================================"
