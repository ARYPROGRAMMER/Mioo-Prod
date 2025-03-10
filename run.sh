#!/bin/bash

# Start MongoDB (if not already running)
echo "Ensure MongoDB is running..."

# Start the backend
echo "Starting backend server..."
cd "$(dirname "$0")"
python main.py &
BACKEND_PID=$!

# Start the frontend
echo "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Handle shutdown
function cleanup {
  echo "Shutting down services..."
  kill $BACKEND_PID $FRONTEND_PID
  exit 0
}

trap cleanup INT

# Keep script running
echo "Services started. Press Ctrl+C to stop."
wait
