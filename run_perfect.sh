#!/bin/bash

# WhatNxt - Perfect App Runner
# This script ensures clean startup by killing existing processes and checking ports

echo "🧹 Cleaning up any existing processes..."
pkill -f "python3 app.py" 2>/dev/null || true
pkill -f "http.server" 2>/dev/null || true
sleep 2

echo "🔍 Checking if ports are free..."
if lsof -i :5001 >/dev/null 2>&1; then
    echo "⚠️  Port 5001 is still in use. Force killing..."
    lsof -ti:5001 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is still in use. Force killing..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo "✅ Ports cleared!"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please create it with your API keys."
    echo "   Copy .env.example to .env and add your Gemini API keys."
    exit 1
fi

echo "🚀 Starting WhatNxt App..."

# Start backend
echo "Starting backend on port 5001..."
/usr/local/bin/python3 app.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start! Check the logs above."
    exit 1
fi

echo "✅ Backend started successfully!"

# Start frontend
echo "Starting frontend on port 8000..."
/usr/local/bin/python3 -m http.server 8000 &
FRONTEND_PID=$!

sleep 2

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start! Check the logs above."
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🎉 WhatNxt is running perfectly!"
echo "🌐 Frontend: http://127.0.0.1:8000"
echo "🔧 Backend API: http://127.0.0.1:5001"
echo ""
echo "Press Ctrl+C to stop both servers"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Servers stopped. Goodbye!"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for processes
wait