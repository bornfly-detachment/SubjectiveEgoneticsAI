#!/usr/bin/env bash
source ~/llama-factory/venv/bin/activate
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Starting SubjectiveEgoneticsAI..."

# Init DB
python scripts/init_db.py

# Start inference service (background)
echo "Starting inference service on port 8001..."
uvicorn models.inference:app --host 0.0.0.0 --port 8001 --log-level warning &
INFERENCE_PID=$!
echo "Inference PID: $INFERENCE_PID"

# Wait for inference service
sleep 3

# Start main API
echo "Starting main API on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo "API PID: $API_PID"

echo ""
echo "Services running:"
echo "  Main API:        http://localhost:8000"
echo "  Inference:       http://localhost:8001"
echo "  API docs:        http://localhost:8000/docs"
echo ""
echo "Press Ctrl-C to stop all services"

trap "kill $INFERENCE_PID $API_PID 2>/dev/null; echo 'Stopped'" EXIT
wait
