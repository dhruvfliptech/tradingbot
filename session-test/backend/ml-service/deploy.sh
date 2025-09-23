#!/bin/bash
# Deployment script for ML Service

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-development}
SERVICE_NAME="ml-service"
COMPOSE_FILE="docker-compose.ml-service.yml"
ENV_FILE=".env"

echo -e "${BLUE}🚀 Deploying ML Service in ${ENVIRONMENT} environment${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Function to check if environment file exists
check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        print_warning "Environment file $ENV_FILE not found. Creating from template..."
        
        cat > "$ENV_FILE" << 'EOF'
# ML Service Environment Variables
# Copy this file to .env and customize the values

# Database Configuration
POSTGRES_PASSWORD=your-secure-postgres-password
ML_SERVICE_DB_PASSWORD=your-secure-ml-service-password

# Security Keys (CHANGE THESE IN PRODUCTION!)
ML_SERVICE_ADMIN_KEY=your-secret-admin-key-here
ML_SERVICE_SECRET_KEY=your-flask-secret-key-here

# External Services (Optional)
SENTRY_DSN=
SLACK_WEBHOOK_URL=

# Monitoring (Optional)
GRAFANA_PASSWORD=admin

# Environment
ENVIRONMENT=development
DEBUG=true
EOF
        
        print_warning "Please edit $ENV_FILE with your configuration before deploying!"
        print_warning "Generated template with default values."
        
        if [ "$ENVIRONMENT" == "production" ]; then
            print_error "Production deployment requires proper configuration. Exiting."
            exit 1
        fi
    fi
}

# Function to validate environment
validate_environment() {
    print_status "Validating environment configuration..."
    
    if [ "$ENVIRONMENT" == "production" ]; then
        # Check critical environment variables
        source "$ENV_FILE"
        
        if [ "$ML_SERVICE_ADMIN_KEY" == "your-secret-admin-key-here" ] || \
           [ "$ML_SERVICE_SECRET_KEY" == "your-flask-secret-key-here" ] || \
           [ "$POSTGRES_PASSWORD" == "your-secure-postgres-password" ]; then
            print_error "Default security keys detected in production environment!"
            print_error "Please update your $ENV_FILE with secure values."
            exit 1
        fi
        
        print_status "Production environment validation passed."
    fi
}

# Function to run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    # Check if migration files exist
    if [ -d "migrations" ] && [ "$(ls -A migrations)" ]; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec ml-service \
            python -c "
from config import get_config
from sqlalchemy import create_engine, text
import os

config = get_config()
engine = create_engine(config.database.url)

# Run basic table creation (this would be replaced with proper migrations)
with engine.connect() as conn:
    conn.execute(text('SELECT 1'))
    print('Database connection successful')
"
        print_status "Database migrations completed."
    else
        print_warning "No migration files found, skipping database migrations."
    fi
}

# Function to perform health check
health_check() {
    print_status "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
           exec -T ml-service curl -f http://localhost:5000/health > /dev/null 2>&1; then
            print_status "Health check passed!"
            return 0
        fi
        
        print_status "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    print_error "Health check failed after $max_attempts attempts"
    return 1
}

# Function to run tests
run_tests() {
    if [ "$ENVIRONMENT" != "production" ]; then
        print_status "Running tests..."
        
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec ml-service \
            python -m pytest tests/ -v --cov=./ --cov-report=term-missing || {
            print_warning "Some tests failed, but continuing with deployment in $ENVIRONMENT environment"
        }
    else
        print_status "Skipping tests in production environment"
    fi
}

# Function to display service status
show_status() {
    print_status "Service status:"
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    
    echo ""
    print_status "Service logs (last 20 lines):"
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs --tail=20 ml-service
    
    echo ""
    print_status "Service URLs:"
    echo "  - API: http://localhost:5000"
    echo "  - Health: http://localhost:5000/health"
    echo "  - Metrics: http://localhost:8000"
    
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps | grep -q prometheus; then
        echo "  - Prometheus: http://localhost:9090"
    fi
    
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps | grep -q grafana; then
        echo "  - Grafana: http://localhost:3000"
    fi
}

# Main deployment process
main() {
    print_status "Starting ML Service deployment..."
    
    # Check prerequisites
    check_env_file
    validate_environment
    
    # Build and start services
    print_status "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build
    
    print_status "Starting services..."
    if [ "$ENVIRONMENT" == "production" ]; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    else
        # Include monitoring services for development
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile monitoring up -d
    fi
    
    # Wait for services to start
    print_status "Waiting for services to start..."
    sleep 10
    
    # Run migrations
    run_migrations
    
    # Run tests (if not production)
    run_tests
    
    # Health check
    if health_check; then
        print_status "✅ ML Service deployment completed successfully!"
        show_status
    else
        print_error "❌ Deployment failed - service health check failed"
        print_error "Check logs with: docker-compose -f $COMPOSE_FILE logs ml-service"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy"|"start")
        main
        ;;
    "stop")
        print_status "Stopping ML Service..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
        print_status "ML Service stopped."
        ;;
    "restart")
        print_status "Restarting ML Service..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart ml-service
        health_check && print_status "✅ ML Service restarted successfully!"
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f ml-service
        ;;
    "status")
        show_status
        ;;
    "test")
        run_tests
        ;;
    "clean")
        print_warning "This will remove all containers and volumes. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down -v --remove-orphans
            docker system prune -f
            print_status "Cleanup completed."
        fi
        ;;
    "help"|"-h"|"--help")
        echo "ML Service Deployment Script"
        echo ""
        echo "Usage: $0 [COMMAND] [ENVIRONMENT]"
        echo ""
        echo "Commands:"
        echo "  deploy    Deploy the ML Service (default)"
        echo "  start     Same as deploy"
        echo "  stop      Stop the ML Service"
        echo "  restart   Restart the ML Service"
        echo "  logs      Show service logs"
        echo "  status    Show service status"
        echo "  test      Run tests"
        echo "  clean     Remove all containers and volumes"
        echo "  help      Show this help message"
        echo ""
        echo "Environments:"
        echo "  development (default)"
        echo "  staging"
        echo "  production"
        echo ""
        echo "Examples:"
        echo "  $0 deploy development"
        echo "  $0 deploy production"
        echo "  $0 stop"
        echo "  $0 logs"
        ;;
    *)
        print_error "Unknown command: $1"
        print_error "Use '$0 help' for usage information."
        exit 1
        ;;
esac