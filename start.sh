#!/bin/bash
set -e
pip install -r requirements.txt
exec $(python3 -c "import sys; print(sys.executable)") -m uvicorn app:app --host 0.0.0.0 --port $PORT
