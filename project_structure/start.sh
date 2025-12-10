#!/bin/bash

# Agent-Centric Insurance Platform Startup Script

echo "🚀 Starting Agent-Centric Insurance Platform..."
echo ""

# Check if XAMPP MySQL is running
echo "Checking MySQL connection..."
if /Applications/XAMPP/xamppfiles/bin/mysql -u root -e "SELECT 1" > /dev/null 2>&1; then
    echo "✅ MySQL is running"
else
    echo "❌ MySQL is not running. Please start XAMPP first."
    exit 1
fi

# Install dependencies if needed
echo ""
echo "Checking Python dependencies..."
if ! python3 -c "import fastapi; import mysql.connector; import pandas" > /dev/null 2>&1; then
    echo "Installing required packages..."
    pip3 install fastapi uvicorn mysql-connector-python pandas pydantic
fi

# Start the API server
echo ""
echo "🔧 Starting API Server on port 8000..."
cd project_structure/api
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "API Server PID: $API_PID"

# Start the frontend server
echo ""
echo "🌐 Starting Frontend Server on port 3000..."
cd ../frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
echo "Frontend Server PID: $FRONTEND_PID"

echo ""
echo "✨ Platform is ready!"
echo ""
echo "📊 Agent Portal: http://localhost:3000/agent.html"
echo "👤 Customer Portal: http://localhost:3000/customer.html"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap "kill $API_PID $FRONTEND_PID; echo 'Servers stopped'; exit" INT
wait
