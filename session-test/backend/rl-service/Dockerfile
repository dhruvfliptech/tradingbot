# RL Service Dockerfile
# ===================
# Multi-stage build for efficient RL service deployment

# Base stage with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r rlservice && useradd -r -g rlservice rlservice

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY integration/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-mock \
    black \
    flake8 \
    mypy \
    jupyter

# Copy source code
COPY . /app/

# Change ownership
RUN chown -R rlservice:rlservice /app

# Switch to app user
USER rlservice

# Expose port
EXPOSE 8001

# Command for development
CMD ["uvicorn", "integration.rl_service:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY integration/ /app/integration/
COPY agents/ /app/agents/
COPY environment/ /app/environment/
COPY rewards/ /app/rewards/
COPY pretrain/ /app/pretrain/
COPY rl_config.py /app/
COPY __init__.py /app/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs /app/data

# Copy startup script
COPY scripts/start-rl-service.sh /app/start-rl-service.sh
RUN chmod +x /app/start-rl-service.sh

# Change ownership
RUN chown -R rlservice:rlservice /app

# Switch to app user
USER rlservice

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health || exit 1

# Command for production
CMD ["/app/start-rl-service.sh"]

# Lightweight stage for testing
FROM python:3.10-alpine as test

# Install minimal dependencies for testing
RUN apk add --no-cache gcc musl-dev

WORKDIR /app

# Copy requirements and install
COPY integration/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/

# Run tests
CMD ["python", "-m", "pytest", "integration/tests/", "-v"]