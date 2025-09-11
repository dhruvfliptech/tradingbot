#!/bin/bash

# Trading Bot Deployment Script
# =============================
# Main deployment script with zero-downtime deployment support
# Supports Docker Compose and Kubernetes deployments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/production/deployment"
LOG_FILE="/tmp/trading-bot-deploy-$(date +%Y%m%d-%H%M%S).log"

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
VERSION="${VERSION:-latest}"
REGISTRY_URL="${REGISTRY_URL:-ghcr.io}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-rolling}"
NAMESPACE="${NAMESPACE:-trading-bot}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_HEALTH_CHECK="${SKIP_HEALTH_CHECK:-false}"
TIMEOUT="${TIMEOUT:-600}"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*${NC}" | tee -a "${LOG_FILE}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $*${NC}" | tee -a "${LOG_FILE}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*${NC}" | tee -a "${LOG_FILE}"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENT:
    staging     Deploy to staging environment
    production  Deploy to production environment

OPTIONS:
    -v, --version VERSION           Docker image version (default: latest)
    -r, --registry REGISTRY         Container registry URL (default: ghcr.io)
    -s, --strategy STRATEGY         Deployment strategy: rolling|blue-green|canary (default: rolling)
    -n, --namespace NAMESPACE       Kubernetes namespace (default: trading-bot)
    -d, --dry-run                   Perform a dry run without actual deployment
    --skip-health-check            Skip health checks after deployment
    --timeout SECONDS              Deployment timeout in seconds (default: 600)
    -h, --help                      Show this help message

EXAMPLES:
    $0 staging
    $0 production --version v1.2.3 --strategy blue-green
    $0 staging --dry-run
    
ENVIRONMENT VARIABLES:
    VERSION                 Docker image version
    REGISTRY_URL            Container registry URL
    DEPLOYMENT_STRATEGY     Deployment strategy
    NAMESPACE               Kubernetes namespace
    DRY_RUN                 Enable dry run mode
    SKIP_HEALTH_CHECK       Skip health checks
    TIMEOUT                 Deployment timeout

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY_URL="$2"
                shift 2
                ;;
            -s|--strategy)
                DEPLOYMENT_STRATEGY="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK="true"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    if [[ "${ENVIRONMENT}" != "staging" && "${ENVIRONMENT}" != "production" ]]; then
        error "Invalid environment: ${ENVIRONMENT}. Must be 'staging' or 'production'"
        exit 1
    fi
    
    if [[ "${DEPLOYMENT_STRATEGY}" != "rolling" && "${DEPLOYMENT_STRATEGY}" != "blue-green" && "${DEPLOYMENT_STRATEGY}" != "canary" ]]; then
        error "Invalid deployment strategy: ${DEPLOYMENT_STRATEGY}. Must be 'rolling', 'blue-green', or 'canary'"
        exit 1
    fi
    
    log "Environment: ${ENVIRONMENT}"
    log "Version: ${VERSION}"
    log "Registry: ${REGISTRY_URL}"
    log "Strategy: ${DEPLOYMENT_STRATEGY}"
    log "Namespace: ${NAMESPACE}"
    log "Dry Run: ${DRY_RUN}"
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    # Check required tools
    for tool in kubectl docker helm; do
        if ! command -v "${tool}" &> /dev/null; then
            missing_tools+=("${tool}")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        warn "Namespace ${NAMESPACE} does not exist, creating..."
        if [[ "${DRY_RUN}" == "false" ]]; then
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/00-namespace.yaml"
        fi
    fi
    
    log "Prerequisites check passed"
}

# Load environment configuration
load_config() {
    local config_file="${DEPLOYMENT_DIR}/.env.${ENVIRONMENT}"
    
    if [[ -f "${config_file}" ]]; then
        # shellcheck source=/dev/null
        source "${config_file}"
        log "Loaded configuration from ${config_file}"
    else
        warn "Configuration file ${config_file} not found, using defaults"
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if images exist
    local images=(
        "${REGISTRY_URL}/trading-bot-backend:${VERSION}"
        "${REGISTRY_URL}/trading-bot-ml:${VERSION}"
        "${REGISTRY_URL}/trading-bot-rl:${VERSION}"
    )
    
    for image in "${images[@]}"; do
        if ! docker manifest inspect "${image}" &> /dev/null; then
            error "Image ${image} not found in registry"
            exit 1
        fi
    done
    
    # Check cluster resources
    local node_count
    node_count=$(kubectl get nodes --no-headers | wc -l)
    if [[ "${node_count}" -lt 2 ]]; then
        warn "Only ${node_count} nodes available, consider scaling up for high availability"
    fi
    
    # Check persistent volumes
    if ! kubectl get pvc -n "${NAMESPACE}" &> /dev/null; then
        warn "No persistent volume claims found, deploying storage..."
        if [[ "${DRY_RUN}" == "false" ]]; then
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/03-persistent-volumes.yaml"
        fi
    fi
    
    log "Pre-deployment checks completed"
}

# Deploy configuration
deploy_config() {
    log "Deploying configuration..."
    
    local config_files=(
        "01-configmaps.yaml"
        "02-secrets.yaml"
        "08-rbac.yaml"
    )
    
    for file in "${config_files[@]}"; do
        info "Applying ${file}..."
        if [[ "${DRY_RUN}" == "false" ]]; then
            envsubst < "${DEPLOYMENT_DIR}/kubernetes/${file}" | kubectl apply -f -
        else
            envsubst < "${DEPLOYMENT_DIR}/kubernetes/${file}" | kubectl apply --dry-run=client -f -
        fi
    done
    
    log "Configuration deployed"
}

# Deploy infrastructure services
deploy_infrastructure() {
    log "Deploying infrastructure services..."
    
    local infra_files=(
        "03-persistent-volumes.yaml"
        "04-services.yaml"
        "06-statefulsets.yaml"
    )
    
    for file in "${infra_files[@]}"; do
        info "Applying ${file}..."
        if [[ "${DRY_RUN}" == "false" ]]; then
            envsubst < "${DEPLOYMENT_DIR}/kubernetes/${file}" | kubectl apply -f -
        else
            envsubst < "${DEPLOYMENT_DIR}/kubernetes/${file}" | kubectl apply --dry-run=client -f -
        fi
    done
    
    # Wait for StatefulSets to be ready
    if [[ "${DRY_RUN}" == "false" ]]; then
        kubectl wait --for=condition=ready --timeout="${TIMEOUT}s" statefulset/redis-statefulset -n "${NAMESPACE}" || true
        kubectl wait --for=condition=ready --timeout="${TIMEOUT}s" statefulset/influxdb-statefulset -n "${NAMESPACE}" || true
    fi
    
    log "Infrastructure services deployed"
}

# Deploy application services
deploy_applications() {
    log "Deploying application services using ${DEPLOYMENT_STRATEGY} strategy..."
    
    case "${DEPLOYMENT_STRATEGY}" in
        "rolling")
            deploy_rolling_update
            ;;
        "blue-green")
            deploy_blue_green
            ;;
        "canary")
            deploy_canary
            ;;
        *)
            error "Unknown deployment strategy: ${DEPLOYMENT_STRATEGY}"
            exit 1
            ;;
    esac
}

# Rolling update deployment
deploy_rolling_update() {
    info "Performing rolling update deployment..."
    
    # Update image versions in deployment files
    export VERSION REGISTRY_URL
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        envsubst < "${DEPLOYMENT_DIR}/kubernetes/05-deployments.yaml" | kubectl apply -f -
        
        # Wait for rollout to complete
        kubectl rollout status deployment/backend-deployment -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
        kubectl rollout status deployment/ml-service-deployment -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
        kubectl rollout status deployment/rl-service-deployment -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
        kubectl rollout status deployment/nginx-deployment -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
    else
        envsubst < "${DEPLOYMENT_DIR}/kubernetes/05-deployments.yaml" | kubectl apply --dry-run=client -f -
    fi
    
    log "Rolling update completed"
}

# Blue-green deployment
deploy_blue_green() {
    info "Performing blue-green deployment..."
    
    local current_version
    current_version=$(kubectl get deployment backend-deployment -n "${NAMESPACE}" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "none")
    
    local new_color="green"
    if [[ "${current_version}" == *"green"* ]]; then
        new_color="blue"
    fi
    
    info "Deploying to ${new_color} environment..."
    
    # Create blue-green deployment
    export VERSION REGISTRY_URL COLOR="${new_color}"
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        envsubst < "${DEPLOYMENT_DIR}/kubernetes/05-deployments.yaml" | \
        sed "s/backend-deployment/backend-deployment-${new_color}/g" | \
        sed "s/ml-service-deployment/ml-service-deployment-${new_color}/g" | \
        sed "s/rl-service-deployment/rl-service-deployment-${new_color}/g" | \
        kubectl apply -f -
        
        # Wait for new deployment to be ready
        kubectl wait --for=condition=available --timeout="${TIMEOUT}s" deployment/backend-deployment-"${new_color}" -n "${NAMESPACE}"
        kubectl wait --for=condition=available --timeout="${TIMEOUT}s" deployment/ml-service-deployment-"${new_color}" -n "${NAMESPACE}"
        kubectl wait --for=condition=available --timeout="${TIMEOUT}s" deployment/rl-service-deployment-"${new_color}" -n "${NAMESPACE}"
        
        # Run health checks on new deployment
        if [[ "${SKIP_HEALTH_CHECK}" == "false" ]]; then
            "${SCRIPT_DIR}/health-check.sh" "${ENVIRONMENT}" "${new_color}"
        fi
        
        # Switch traffic to new deployment
        kubectl patch service backend-service -n "${NAMESPACE}" -p "{\"spec\":{\"selector\":{\"color\":\"${new_color}\"}}}"
        kubectl patch service ml-service -n "${NAMESPACE}" -p "{\"spec\":{\"selector\":{\"color\":\"${new_color}\"}}}"
        kubectl patch service rl-service -n "${NAMESPACE}" -p "{\"spec\":{\"selector\":{\"color\":\"${new_color}\"}}}"
        
        # Wait a bit for traffic to stabilize
        sleep 30
        
        # Remove old deployment
        local old_color="blue"
        if [[ "${new_color}" == "blue" ]]; then
            old_color="green"
        fi
        
        kubectl delete deployment backend-deployment-"${old_color}" -n "${NAMESPACE}" --ignore-not-found=true
        kubectl delete deployment ml-service-deployment-"${old_color}" -n "${NAMESPACE}" --ignore-not-found=true
        kubectl delete deployment rl-service-deployment-"${old_color}" -n "${NAMESPACE}" --ignore-not-found=true
    else
        info "Dry run: Would deploy to ${new_color} environment"
    fi
    
    log "Blue-green deployment completed"
}

# Canary deployment
deploy_canary() {
    info "Performing canary deployment..."
    
    local canary_percentage="${CANARY_PERCENTAGE:-10}"
    
    # Deploy canary version
    export VERSION REGISTRY_URL
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Create canary deployment with reduced replicas
        envsubst < "${DEPLOYMENT_DIR}/kubernetes/05-deployments.yaml" | \
        sed "s/backend-deployment/backend-deployment-canary/g" | \
        sed "s/replicas: 3/replicas: 1/g" | \
        kubectl apply -f -
        
        # Wait for canary to be ready
        kubectl wait --for=condition=available --timeout="${TIMEOUT}s" deployment/backend-deployment-canary -n "${NAMESPACE}"
        
        # Monitor canary for errors
        sleep 60
        
        # Check error rates (simplified)
        local error_rate
        error_rate=$(kubectl logs deployment/backend-deployment-canary -n "${NAMESPACE}" --tail=100 | grep -i error | wc -l || echo "0")
        
        if [[ "${error_rate}" -gt 5 ]]; then
            error "High error rate detected in canary deployment, rolling back..."
            kubectl delete deployment backend-deployment-canary -n "${NAMESPACE}"
            exit 1
        fi
        
        # Promote canary to production
        info "Canary validation successful, promoting to production..."
        kubectl delete deployment backend-deployment -n "${NAMESPACE}"
        kubectl patch deployment backend-deployment-canary -n "${NAMESPACE}" -p '{"metadata":{"name":"backend-deployment"}}'
    else
        info "Dry run: Would deploy canary with ${canary_percentage}% traffic"
    fi
    
    log "Canary deployment completed"
}

# Deploy auto-scaling
deploy_autoscaling() {
    log "Deploying auto-scaling configuration..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/07-hpa-autoscaling.yaml"
    else
        kubectl apply --dry-run=client -f "${DEPLOYMENT_DIR}/kubernetes/07-hpa-autoscaling.yaml"
    fi
    
    log "Auto-scaling configured"
}

# Run health checks
run_health_checks() {
    if [[ "${SKIP_HEALTH_CHECK}" == "true" ]]; then
        warn "Skipping health checks"
        return 0
    fi
    
    log "Running health checks..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        "${SCRIPT_DIR}/health-check.sh" "${ENVIRONMENT}"
    else
        info "Dry run: Would run health checks"
    fi
}

# Post-deployment tasks
post_deployment() {
    log "Running post-deployment tasks..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Update deployment history
        kubectl annotate deployment backend-deployment -n "${NAMESPACE}" \
            deployment.kubernetes.io/revision-history-limit=10 \
            deployment.kubernetes.io/deployed-by="${USER:-system}" \
            deployment.kubernetes.io/deployed-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            deployment.kubernetes.io/version="${VERSION}"
        
        # Clean up old ReplicaSets
        kubectl delete replicaset -l app.kubernetes.io/name=trading-bot -n "${NAMESPACE}" \
            --field-selector='status.replicas=0' || true
    fi
    
    log "Post-deployment tasks completed"
}

# Send notifications
send_notifications() {
    local status="$1"
    local message
    
    if [[ "${status}" == "success" ]]; then
        message="🚀 Trading Bot v${VERSION} successfully deployed to ${ENVIRONMENT}!"
    else
        message="🚨 Trading Bot deployment to ${ENVIRONMENT} FAILED!"
    fi
    
    info "${message}"
    
    # Send Slack notification if webhook is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" && "${DRY_RUN}" == "false" ]]; then
        curl -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-type: application/json' \
            --data "{\"text\":\"${message}\"}" \
            --silent || true
    fi
    
    # Send email notification if configured
    if [[ -n "${EMAIL_NOTIFICATIONS:-}" && "${DRY_RUN}" == "false" ]]; then
        echo "${message}" | mail -s "Trading Bot Deployment ${status}" "${EMAIL_NOTIFICATIONS}" || true
    fi
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    
    if [[ ${exit_code} -ne 0 ]]; then
        error "Deployment failed with exit code ${exit_code}"
        send_notifications "failure"
    else
        log "Deployment completed successfully"
        send_notifications "success"
    fi
    
    info "Deployment log saved to: ${LOG_FILE}"
    exit ${exit_code}
}

# Main function
main() {
    trap cleanup EXIT
    
    log "Starting Trading Bot deployment..."
    log "Log file: ${LOG_FILE}"
    
    parse_args "$@"
    validate_environment
    check_prerequisites
    load_config
    pre_deployment_checks
    
    deploy_config
    deploy_infrastructure
    deploy_applications
    deploy_autoscaling
    run_health_checks
    post_deployment
    
    log "🎉 Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"