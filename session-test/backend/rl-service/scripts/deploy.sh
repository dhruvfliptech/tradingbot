#!/bin/bash
# RL Service Deployment Script
# ===========================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV=${ENV:-"production"}
VERSION=${VERSION:-"latest"}
SKIP_TESTS=${SKIP_TESTS:-false}
FORCE_REBUILD=${FORCE_REBUILD:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command_exists docker; then
        missing_tools+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_tools+=("docker-compose")
    fi
    
    if ! command_exists curl; then
        missing_tools+=("curl")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi
    
    log_success "All prerequisites are met"
}

# Function to validate environment
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check if docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available disk space (at least 2GB)
    local available_space=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 2097152 ]; then  # 2GB in KB
        log_warning "Low disk space detected. At least 2GB is recommended."
    fi
    
    # Check available memory (at least 4GB)
    local available_memory=$(free -m | awk 'NR==2 {print $7}')
    if [ "$available_memory" -lt 4096 ]; then  # 4GB in MB
        log_warning "Low available memory detected. At least 4GB is recommended."
    fi
    
    log_success "Environment validation completed"
}

# Function to run tests
run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping tests (SKIP_TESTS=true)"
        return 0
    fi
    
    log_info "Running tests..."
    
    # Build test image
    docker build -t rl-service:test --target test "$PROJECT_DIR"
    
    # Run tests
    if docker run --rm rl-service:test; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

# Function to build images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    local build_args=""
    if [ "$FORCE_REBUILD" = "true" ]; then
        build_args="--no-cache"
    fi
    
    # Build production image
    log_info "Building production image..."
    docker build $build_args -t rl-service:$VERSION --target production .
    
    # Tag as latest if this is the latest version
    if [ "$VERSION" = "latest" ] || [ "$ENV" = "production" ]; then
        docker tag rl-service:$VERSION rl-service:latest
    fi
    
    log_success "Images built successfully"
}

# Function to setup configuration
setup_configuration() {
    log_info "Setting up configuration for environment: $ENV"
    
    local config_dir="$PROJECT_DIR/config"
    mkdir -p "$config_dir"
    
    # Create environment-specific configuration
    case "$ENV" in
        "development")
            cat > "$config_dir/development.env" << EOF
# Development Environment Configuration
ENV=development
DEBUG=true
LOG_LEVEL=debug
WORKERS=1
RELOAD=true

# External services (development)
TRADING_SERVICE_URL=http://localhost:3000
DATA_AGGREGATOR_URL=http://localhost:3000
ADAPTIVE_THRESHOLD_URL=http://localhost:5000

# A/B Testing
AB_TESTING_ENABLED=false
RL_TRAFFIC_PERCENTAGE=1.0
EOF
            ;;
        "staging")
            cat > "$config_dir/staging.env" << EOF
# Staging Environment Configuration
ENV=staging
DEBUG=false
LOG_LEVEL=info
WORKERS=2
RELOAD=false

# External services (staging)
TRADING_SERVICE_URL=http://backend-staging:3000
DATA_AGGREGATOR_URL=http://backend-staging:3000
ADAPTIVE_THRESHOLD_URL=http://ml-service-staging:5000

# A/B Testing
AB_TESTING_ENABLED=true
RL_TRAFFIC_PERCENTAGE=0.5
EOF
            ;;
        "production")
            cat > "$config_dir/production.env" << EOF
# Production Environment Configuration
ENV=production
DEBUG=false
LOG_LEVEL=warning
WORKERS=4
RELOAD=false

# External services (production)
TRADING_SERVICE_URL=http://backend:3000
DATA_AGGREGATOR_URL=http://backend:3000
ADAPTIVE_THRESHOLD_URL=http://ml-service:5000

# A/B Testing
AB_TESTING_ENABLED=true
RL_TRAFFIC_PERCENTAGE=0.1

# Performance settings
MAX_CONCURRENT_REQUESTS=200
PREDICTION_TIMEOUT_SECONDS=3.0
EOF
            ;;
        *)
            log_error "Unknown environment: $ENV"
            exit 1
            ;;
    esac
    
    log_success "Configuration setup completed for $ENV"
}

# Function to setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    local monitoring_dir="$PROJECT_DIR/monitoring"
    mkdir -p "$monitoring_dir/grafana/dashboards" "$monitoring_dir/grafana/datasources"
    
    # Create Prometheus configuration
    cat > "$monitoring_dir/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rl-service'
    static_configs:
      - targets: ['rl-service:8001']
    metrics_path: '/api/v1/metrics/prometheus'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

    # Create Grafana datasource configuration
    cat > "$monitoring_dir/grafana/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Create basic Grafana dashboard
    cat > "$monitoring_dir/grafana/dashboards/rl-service.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "RL Service Dashboard",
    "tags": ["rl-service"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rl_service_predictions_total[5m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rl_service_prediction_response_time_ms_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    log_success "Monitoring setup completed"
}

# Function to deploy services
deploy_services() {
    log_info "Deploying RL service..."
    
    cd "$PROJECT_DIR"
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose down || true
    
    # Pull external images
    log_info "Pulling external images..."
    docker-compose pull redis postgres prometheus grafana nginx || true
    
    # Start services
    log_info "Starting services..."
    case "$ENV" in
        "development")
            docker-compose up -d rl-service redis
            ;;
        "staging")
            docker-compose up -d rl-service redis postgres prometheus grafana
            ;;
        "production")
            docker-compose up -d
            ;;
    esac
    
    log_success "Services deployed"
}

# Function to verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:8001/api/v1/health" > /dev/null 2>&1; then
            log_success "RL service is healthy and responding"
            break
        fi
        
        log_info "Health check attempt $attempt/$max_attempts..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Service failed to become healthy after $max_attempts attempts"
        log_error "Deployment verification failed"
        
        # Show logs for debugging
        log_info "Recent logs:"
        docker-compose logs --tail=50 rl-service
        
        exit 1
    fi
    
    # Additional service checks
    log_info "Running additional service checks..."
    
    # Check if service responds to status endpoint
    if curl -f -s "http://localhost:8001/api/v1/status" | grep -q "rl-service"; then
        log_success "Status endpoint responding correctly"
    else
        log_warning "Status endpoint not responding as expected"
    fi
    
    # Check metrics endpoint
    if curl -f -s "http://localhost:8001/api/v1/metrics" > /dev/null 2>&1; then
        log_success "Metrics endpoint accessible"
    else
        log_warning "Metrics endpoint not accessible"
    fi
    
    log_success "Deployment verification completed"
}

# Function to show deployment info
show_deployment_info() {
    log_success "RL Service deployment completed successfully!"
    echo
    echo "Service Information:"
    echo "  Environment: $ENV"
    echo "  Version: $VERSION"
    echo "  Service URL: http://localhost:8001"
    echo
    echo "API Endpoints:"
    echo "  Health Check: http://localhost:8001/api/v1/health"
    echo "  Service Status: http://localhost:8001/api/v1/status"
    echo "  API Documentation: http://localhost:8001/docs"
    echo "  Metrics: http://localhost:8001/api/v1/metrics"
    echo
    
    if [ "$ENV" != "development" ]; then
        echo "Monitoring URLs:"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana: http://localhost:3001 (admin/admin)"
        echo
    fi
    
    echo "Useful Commands:"
    echo "  View logs: docker-compose logs -f rl-service"
    echo "  Stop services: docker-compose down"
    echo "  Restart service: docker-compose restart rl-service"
    echo "  Scale service: docker-compose up -d --scale rl-service=3"
    echo
}

# Function to cleanup on error
cleanup_on_error() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose down || true
}

# Main deployment flow
main() {
    log_info "Starting RL Service deployment (Environment: $ENV, Version: $VERSION)"
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Run deployment steps
    check_prerequisites
    validate_environment
    run_tests
    build_images
    setup_configuration
    setup_monitoring
    deploy_services
    verify_deployment
    show_deployment_info
    
    log_success "Deployment completed successfully!"
}

# Script entry point
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENV="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --help)
                echo "RL Service Deployment Script"
                echo
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --env ENV          Environment to deploy (development|staging|production)"
                echo "  --version VERSION  Version tag for the images (default: latest)"
                echo "  --skip-tests       Skip running tests before deployment"
                echo "  --force-rebuild    Force rebuild of Docker images"
                echo "  --help             Show this help message"
                echo
                echo "Environment Variables:"
                echo "  ENV                Environment to deploy"
                echo "  VERSION            Version tag for the images"
                echo "  SKIP_TESTS         Skip tests if set to 'true'"
                echo "  FORCE_REBUILD      Force rebuild if set to 'true'"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    main
fi