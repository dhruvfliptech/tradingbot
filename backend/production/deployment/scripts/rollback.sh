#!/bin/bash

# Trading Bot Rollback Script
# ===========================
# Automated rollback procedures with safety checks and notifications

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
LOG_FILE="/tmp/trading-bot-rollback-$(date +%Y%m%d-%H%M%S).log"

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="${NAMESPACE:-trading-bot}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"
ROLLBACK_TO="${ROLLBACK_TO:-previous}"
TIMEOUT="${TIMEOUT:-300}"

# Logging functions
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
    staging     Rollback staging environment
    production  Rollback production environment

OPTIONS:
    -t, --rollback-to REVISION      Rollback to specific revision (default: previous)
    -n, --namespace NAMESPACE       Kubernetes namespace (default: trading-bot)
    -d, --dry-run                   Perform a dry run without actual rollback
    -f, --force                     Force rollback without confirmation prompts
    --timeout SECONDS               Rollback timeout in seconds (default: 300)
    -h, --help                      Show this help message

EXAMPLES:
    $0 staging
    $0 production --rollback-to 3
    $0 production --dry-run
    $0 staging --force

REVISION OPTIONS:
    previous    Rollback to previous revision (default)
    NUMBER      Rollback to specific revision number
    HASH        Rollback to specific git commit hash

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
            -t|--rollback-to)
                ROLLBACK_TO="$2"
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
            -f|--force)
                FORCE="true"
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
    
    log "Environment: ${ENVIRONMENT}"
    log "Namespace: ${NAMESPACE}"
    log "Rollback to: ${ROLLBACK_TO}"
    log "Dry Run: ${DRY_RUN}"
    log "Force: ${FORCE}"
}

# Check prerequisites
check_prerequisites() {
    # Check required tools
    for tool in kubectl; do
        if ! command -v "${tool}" &> /dev/null; then
            error "Missing required tool: ${tool}"
            exit 1
        fi
    done
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        error "Namespace ${NAMESPACE} does not exist"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Get deployment information
get_deployment_info() {
    log "Gathering deployment information..."
    
    # Get current deployments
    local deployments=(
        "backend-deployment"
        "ml-service-deployment"
        "rl-service-deployment"
        "nginx-deployment"
    )
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "${deployment}" -n "${NAMESPACE}" &> /dev/null; then
            local current_revision
            current_revision=$(kubectl rollout history deployment/"${deployment}" -n "${NAMESPACE}" --revision=0 | tail -n 1 | awk '{print $1}')
            
            local current_image
            current_image=$(kubectl get deployment "${deployment}" -n "${NAMESPACE}" -o jsonpath='{.spec.template.spec.containers[0].image}')
            
            info "${deployment}: Revision ${current_revision}, Image: ${current_image}"
        else
            warn "Deployment ${deployment} not found"
        fi
    done
}

# Get rollback revision
get_rollback_revision() {
    local deployment="$1"
    local target_revision
    
    case "${ROLLBACK_TO}" in
        "previous")
            # Get the previous revision
            target_revision=$(kubectl rollout history deployment/"${deployment}" -n "${NAMESPACE}" | tail -n 2 | head -n 1 | awk '{print $1}')
            ;;
        [0-9]*)
            # Specific revision number
            target_revision="${ROLLBACK_TO}"
            ;;
        *)
            error "Invalid rollback target: ${ROLLBACK_TO}"
            exit 1
            ;;
    esac
    
    # Validate revision exists
    if ! kubectl rollout history deployment/"${deployment}" -n "${NAMESPACE}" --revision="${target_revision}" &> /dev/null; then
        error "Revision ${target_revision} not found for ${deployment}"
        exit 1
    fi
    
    echo "${target_revision}"
}

# Confirm rollback
confirm_rollback() {
    if [[ "${FORCE}" == "true" ]]; then
        return 0
    fi
    
    warn "⚠️  ROLLBACK CONFIRMATION REQUIRED"
    echo
    echo "Environment: ${ENVIRONMENT}"
    echo "Namespace: ${NAMESPACE}"
    echo "Rollback to: ${ROLLBACK_TO}"
    echo
    
    # Show what will be rolled back
    echo "Deployments to rollback:"
    local deployments=(
        "backend-deployment"
        "ml-service-deployment"
        "rl-service-deployment"
        "nginx-deployment"
    )
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "${deployment}" -n "${NAMESPACE}" &> /dev/null; then
            local current_revision
            current_revision=$(kubectl rollout history deployment/"${deployment}" -n "${NAMESPACE}" --revision=0 | tail -n 1 | awk '{print $1}')
            
            local target_revision
            target_revision=$(get_rollback_revision "${deployment}")
            
            echo "  - ${deployment}: ${current_revision} → ${target_revision}"
        fi
    done
    
    echo
    if [[ "${ENVIRONMENT}" == "production" ]]; then
        echo "🚨 THIS IS A PRODUCTION ROLLBACK!"
        echo "This action will affect live trading operations."
        echo
    fi
    
    read -rp "Are you sure you want to proceed? (yes/no): " confirm
    
    if [[ "${confirm}" != "yes" ]]; then
        log "Rollback cancelled by user"
        exit 0
    fi
}

# Create pre-rollback backup
create_backup() {
    log "Creating pre-rollback backup..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Run backup script
        if [[ -f "${SCRIPT_DIR}/backup.sh" ]]; then
            "${SCRIPT_DIR}/backup.sh" "${ENVIRONMENT}" --type pre-rollback
        else
            warn "Backup script not found, skipping backup"
        fi
    else
        info "Dry run: Would create pre-rollback backup"
    fi
}

# Stop traffic to affected services
stop_traffic() {
    log "Stopping traffic to services during rollback..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Scale down nginx to stop incoming traffic
        kubectl scale deployment nginx-deployment --replicas=0 -n "${NAMESPACE}"
        
        # Wait for pods to terminate
        kubectl wait --for=delete pod -l app.kubernetes.io/component=nginx -n "${NAMESPACE}" --timeout="${TIMEOUT}s" || true
    else
        info "Dry run: Would stop traffic to services"
    fi
}

# Rollback deployments
rollback_deployments() {
    log "Rolling back deployments..."
    
    local deployments=(
        "backend-deployment"
        "ml-service-deployment"
        "rl-service-deployment"
    )
    
    # Rollback in reverse order to minimize dependencies
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "${deployment}" -n "${NAMESPACE}" &> /dev/null; then
            local target_revision
            target_revision=$(get_rollback_revision "${deployment}")
            
            info "Rolling back ${deployment} to revision ${target_revision}..."
            
            if [[ "${DRY_RUN}" == "false" ]]; then
                kubectl rollout undo deployment/"${deployment}" --to-revision="${target_revision}" -n "${NAMESPACE}"
                
                # Wait for rollback to complete
                kubectl rollout status deployment/"${deployment}" -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
                
                # Verify rollback
                local new_revision
                new_revision=$(kubectl rollout history deployment/"${deployment}" -n "${NAMESPACE}" --revision=0 | tail -n 1 | awk '{print $1}')
                
                if [[ "${new_revision}" != "${target_revision}" ]]; then
                    error "Rollback verification failed for ${deployment}"
                    exit 1
                fi
                
                log "✅ ${deployment} rolled back successfully"
            else
                info "Dry run: Would rollback ${deployment} to revision ${target_revision}"
            fi
        else
            warn "Deployment ${deployment} not found, skipping"
        fi
    done
}

# Rollback nginx (restore traffic)
rollback_nginx() {
    log "Rolling back nginx and restoring traffic..."
    
    if kubectl get deployment nginx-deployment -n "${NAMESPACE}" &> /dev/null; then
        if [[ "${DRY_RUN}" == "false" ]]; then
            # Rollback nginx
            local target_revision
            target_revision=$(get_rollback_revision "nginx-deployment")
            
            kubectl rollout undo deployment/nginx-deployment --to-revision="${target_revision}" -n "${NAMESPACE}"
            kubectl rollout status deployment/nginx-deployment -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
            
            # Scale back up to restore traffic
            kubectl scale deployment nginx-deployment --replicas=2 -n "${NAMESPACE}"
            kubectl wait --for=condition=available deployment/nginx-deployment -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
            
            log "✅ Traffic restored"
        else
            info "Dry run: Would rollback nginx and restore traffic"
        fi
    fi
}

# Health checks after rollback
run_health_checks() {
    log "Running health checks after rollback..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Wait for services to stabilize
        sleep 30
        
        # Run health check script
        if [[ -f "${SCRIPT_DIR}/health-check.sh" ]]; then
            "${SCRIPT_DIR}/health-check.sh" "${ENVIRONMENT}" --timeout 60
        else
            warn "Health check script not found, performing basic checks"
            
            # Basic health checks
            local deployments=(
                "backend-deployment"
                "ml-service-deployment"
                "rl-service-deployment"
                "nginx-deployment"
            )
            
            for deployment in "${deployments[@]}"; do
                if kubectl get deployment "${deployment}" -n "${NAMESPACE}" &> /dev/null; then
                    local ready_replicas
                    ready_replicas=$(kubectl get deployment "${deployment}" -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
                    
                    local desired_replicas
                    desired_replicas=$(kubectl get deployment "${deployment}" -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}')
                    
                    if [[ "${ready_replicas}" -eq "${desired_replicas}" ]]; then
                        info "✅ ${deployment} health check passed"
                    else
                        error "❌ ${deployment} health check failed (${ready_replicas}/${desired_replicas} ready)"
                        exit 1
                    fi
                fi
            done
        fi
    else
        info "Dry run: Would run health checks"
    fi
}

# Update rollback annotations
update_annotations() {
    log "Updating rollback annotations..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        local deployments=(
            "backend-deployment"
            "ml-service-deployment"
            "rl-service-deployment"
            "nginx-deployment"
        )
        
        for deployment in "${deployments[@]}"; do
            if kubectl get deployment "${deployment}" -n "${NAMESPACE}" &> /dev/null; then
                kubectl annotate deployment "${deployment}" -n "${NAMESPACE}" \
                    rollback.kubernetes.io/rolled-back-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    rollback.kubernetes.io/rolled-back-by="${USER:-system}" \
                    rollback.kubernetes.io/rolled-back-to="${ROLLBACK_TO}" \
                    rollback.kubernetes.io/reason="Manual rollback" \
                    --overwrite
            fi
        done
    else
        info "Dry run: Would update rollback annotations"
    fi
}

# Send notifications
send_notifications() {
    local status="$1"
    local message
    
    if [[ "${status}" == "success" ]]; then
        message="🔄 Trading Bot successfully rolled back to ${ROLLBACK_TO} in ${ENVIRONMENT}!"
    else
        message="🚨 Trading Bot rollback FAILED in ${ENVIRONMENT}!"
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
        echo "${message}" | mail -s "Trading Bot Rollback ${status}" "${EMAIL_NOTIFICATIONS}" || true
    fi
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    
    if [[ ${exit_code} -ne 0 ]]; then
        error "Rollback failed with exit code ${exit_code}"
        
        # Attempt emergency recovery
        warn "Attempting emergency recovery..."
        if [[ "${DRY_RUN}" == "false" ]]; then
            # Restore nginx to enable traffic
            kubectl scale deployment nginx-deployment --replicas=2 -n "${NAMESPACE}" || true
        fi
        
        send_notifications "failure"
    else
        log "Rollback completed successfully"
        send_notifications "success"
    fi
    
    info "Rollback log saved to: ${LOG_FILE}"
    exit ${exit_code}
}

# Emergency rollback function
emergency_rollback() {
    warn "🚨 EMERGENCY ROLLBACK INITIATED"
    
    # Force rollback without confirmations
    FORCE="true"
    
    # Rollback to previous version immediately
    ROLLBACK_TO="previous"
    
    # Skip non-critical steps
    log "Performing emergency rollback..."
    
    rollback_deployments
    rollback_nginx
    
    log "Emergency rollback completed"
}

# Main function
main() {
    trap cleanup EXIT
    
    log "Starting Trading Bot rollback..."
    log "Log file: ${LOG_FILE}"
    
    parse_args "$@"
    validate_environment
    check_prerequisites
    get_deployment_info
    
    # Check if this is an emergency rollback
    if [[ "${ROLLBACK_TO}" == "emergency" ]]; then
        emergency_rollback
        return 0
    fi
    
    confirm_rollback
    create_backup
    stop_traffic
    rollback_deployments
    rollback_nginx
    run_health_checks
    update_annotations
    
    log "🎉 Rollback completed successfully!"
}

# Run main function with all arguments
main "$@"