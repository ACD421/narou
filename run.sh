#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Narou - AI Job Search Assistant"
echo "  Starting..."
echo "============================================"
echo

cd "$(dirname "$0")"

if [ ! -f .venv/bin/activate ]; then
    echo "ERROR: Virtual environment not found."
    echo "Run ./install.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Opening Narou in your default browser..."
echo
echo "If the browser does not open automatically,"
echo "go to: http://localhost:8501"
echo
echo "Press Ctrl+C in this window to stop the app."
echo "============================================"
echo

exec streamlit run app.py
