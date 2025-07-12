#!/bin/bash

# Flexible startup script for OMR Flask application
# This script automatically chooses between development and production servers

set -e

# Default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-5000}
FLASK_DEBUG=${FLASK_DEBUG:-false}

echo "Starting OMR Flask Application..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Debug Mode: $FLASK_DEBUG"
echo "Environment: $FLASK_ENV"

# Check if we're in development or production mode
if [ "$FLASK_DEBUG" = "true" ] || [ "$FLASK_ENV" = "development" ]; then
    echo "Starting in DEVELOPMENT mode with Flask development server..."
    python app.py
else
    echo "Starting in PRODUCTION mode with Waitress..."
    # Check if wsgi.py exists
    if [ -f "wsgi.py" ]; then
        python wsgi.py
    else
        echo "wsgi.py not found, falling back to app.py"
        python app.py
    fi
fi