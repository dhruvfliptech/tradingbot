#!/bin/bash
# RL Service Startup Script
# =========================

set -e

# Configuration
export APP_NAME="rl-service"
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export WORKERS=${WORKERS:-1}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8001}
export RELOAD=${RELOAD:-false}

# Paths
export MODELS_PATH=${MODELS_PATH:-"/app/models"}
export LOGS_PATH=${LOGS_PATH:-"/app/logs"}
export DATA_PATH=${DATA_PATH:-"/app/data"}

# Create directories if they don't exist
mkdir -p "$MODELS_PATH" "$LOGS_PATH" "$DATA_PATH"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check service health
check_health() {
    local max_attempts=30
    local attempt=1
    
    log "Waiting for service to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://$HOST:$PORT/api/v1/health" > /dev/null 2>&1; then
            log "Service is healthy!"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log "Service failed to become healthy after $max_attempts attempts"
    return 1
}

# Function to handle shutdown
shutdown() {
    log "Received shutdown signal, gracefully stopping $APP_NAME..."
    
    # Kill the uvicorn process
    if [ ! -z "$UVICORN_PID" ]; then
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
    fi
    
    log "$APP_NAME stopped"
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

log "Starting $APP_NAME..."
log "Configuration:"
log "  Host: $HOST"
log "  Port: $PORT"
log "  Workers: $WORKERS"
log "  Log Level: $LOG_LEVEL"
log "  Models Path: $MODELS_PATH"
log "  Logs Path: $LOGS_PATH"
log "  Data Path: $DATA_PATH"

# Environment validation
log "Validating environment..."

# Check if Python packages are available
python -c "
import sys
required_packages = ['fastapi', 'uvicorn', 'aiohttp', 'numpy', 'pandas']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing required packages: {missing_packages}')
    sys.exit(1)

print('All required packages are available')
"

if [ $? -ne 0 ]; then
    log "Environment validation failed"
    exit 1
fi

log "Environment validation passed"

# Check external service connectivity (optional)
if [ ! -z "$TRADING_SERVICE_URL" ]; then
    log "Testing connectivity to Trading Service at $TRADING_SERVICE_URL..."
    if curl -f -s --max-time 5 "$TRADING_SERVICE_URL/health" > /dev/null 2>&1; then
        log "Trading Service is reachable"
    else
        log "Warning: Trading Service is not reachable, will use fallback mode"
    fi
fi

if [ ! -z "$DATA_AGGREGATOR_URL" ]; then
    log "Testing connectivity to Data Aggregator at $DATA_AGGREGATOR_URL..."
    if curl -f -s --max-time 5 "$DATA_AGGREGATOR_URL/health" > /dev/null 2>&1; then
        log "Data Aggregator is reachable"
    else
        log "Warning: Data Aggregator is not reachable, will use fallback mode"
    fi
fi

# Pre-start tasks
log "Performing pre-start tasks..."

# Load environment-specific configuration
if [ -f "/app/config/production.env" ]; then
    log "Loading production configuration..."
    set -a
    source /app/config/production.env
    set +a
fi

# Initialize models directory structure
log "Initializing models directory structure..."
mkdir -p "$MODELS_PATH/ensemble"
mkdir -p "$MODELS_PATH/individual"
mkdir -p "$MODELS_PATH/regime_detector"

# Check if pre-trained models exist
if [ -z "$(ls -A $MODELS_PATH)" ]; then
    log "No pre-trained models found, service will start in training mode"
else
    log "Pre-trained models found in $MODELS_PATH"
fi

# Start the RL service
log "Starting RL service with uvicorn..."

if [ "$RELOAD" = "true" ]; then
    # Development mode with reload
    log "Starting in development mode with auto-reload"
    uvicorn integration.rl_service:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --reload \
        --reload-dir /app/integration \
        --access-log &
else
    # Production mode
    log "Starting in production mode"
    uvicorn integration.rl_service:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --access-log &
fi

# Store the PID
UVICORN_PID=$!
log "RL service started with PID $UVICORN_PID"

# Wait for service to be ready
sleep 5

# Health check
if check_health; then
    log "$APP_NAME is running and healthy"
else
    log "$APP_NAME failed to start properly"
    exit 1
fi

# Keep the script running and wait for the uvicorn process
wait "$UVICORN_PID"