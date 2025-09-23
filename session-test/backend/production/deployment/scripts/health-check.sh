#!/bin/bash

# Trading Bot Health Check Script
# ==============================
# Comprehensive health monitoring for all services

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/trading-bot-health-$(date +%Y%m%d-%H%M%S).log"

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="${NAMESPACE:-trading-bot}"
TIMEOUT="${TIMEOUT:-60}"
RETRIES="${RETRIES:-3}"
WAIT_INTERVAL="${WAIT_INTERVAL:-10}"
VERBOSE="${VERBOSE:-false}"
CHECK_TYPE="${CHECK_TYPE:-all}"

# Health check results
HEALTH_STATUS=()
FAILED_CHECKS=()

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
    staging     Check staging environment
    production  Check production environment

OPTIONS:
    -t, --timeout SECONDS           Health check timeout (default: 60)
    -r, --retries COUNT             Number of retries for failed checks (default: 3)
    -w, --wait SECONDS              Wait interval between retries (default: 10)
    -n, --namespace NAMESPACE       Kubernetes namespace (default: trading-bot)
    -c, --check-type TYPE           Type of checks: all|basic|extended|deep (default: all)
    -v, --verbose                   Enable verbose output
    -h, --help                      Show this help message

CHECK TYPES:
    basic       Basic pod and service health
    extended    Include application-level health checks
    deep        Include performance and integration tests
    all         All checks (basic + extended + deep)

EXAMPLES:
    $0 production
    $0 staging --check-type basic
    $0 production --timeout 120 --verbose

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
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -r|--retries)
                RETRIES="$2"
                shift 2
                ;;
            -w|--wait)
                WAIT_INTERVAL="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--check-type)
                CHECK_TYPE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
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

# Validate settings
validate_settings() {
    local valid_types=("basic" "extended" "deep" "all")
    if [[ ! " ${valid_types[*]} " =~ " ${CHECK_TYPE} " ]]; then
        error "Invalid check type: ${CHECK_TYPE}. Must be one of: ${valid_types[*]}"
        exit 1
    fi
    
    log "Environment: ${ENVIRONMENT}"
    log "Namespace: ${NAMESPACE}"
    log "Check Type: ${CHECK_TYPE}"
    log "Timeout: ${TIMEOUT}s"
    log "Retries: ${RETRIES}"
}

# Check prerequisites
check_prerequisites() {
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is required but not installed"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        error "Namespace ${NAMESPACE} does not exist"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Execute health check with retries
execute_check() {
    local check_name="$1"
    local check_function="$2"
    local attempts=0
    
    while [[ $attempts -lt $RETRIES ]]; do
        attempts=$((attempts + 1))
        
        if [[ "${VERBOSE}" == "true" ]]; then
            info "Running ${check_name} (attempt ${attempts}/${RETRIES})"
        fi
        
        if $check_function; then
            HEALTH_STATUS+=("${check_name}: PASS")
            return 0
        else
            if [[ $attempts -lt $RETRIES ]]; then
                warn "${check_name} failed, retrying in ${WAIT_INTERVAL}s..."
                sleep "${WAIT_INTERVAL}"
            fi
        fi
    done
    
    HEALTH_STATUS+=("${check_name}: FAIL")
    FAILED_CHECKS+=("${check_name}")
    return 1
}

# Basic Kubernetes health checks
check_pods_health() {
    info "Checking pod health..."
    
    local failed_pods=()
    local services=("backend" "ml-service" "rl-service" "nginx" "redis" "influxdb")
    
    for service in "${services[@]}"; do
        local pods
        pods=$(kubectl get pods -l app.kubernetes.io/component="${service}" -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
        
        if [[ -z "${pods}" ]]; then
            if [[ "${service}" == "nginx" || "${service}" == "redis" || "${service}" == "influxdb" ]]; then
                warn "No ${service} pods found (may be expected)"
                continue
            else
                failed_pods+=("${service}")
                continue
            fi
        fi
        
        for pod in ${pods}; do
            local pod_status
            pod_status=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
            
            if [[ "${pod_status}" != "Running" ]]; then
                failed_pods+=("${pod}")
            fi
            
            # Check readiness
            local ready
            ready=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
            
            if [[ "${ready}" != "True" ]]; then
                failed_pods+=("${pod}")
            fi
        done
    done
    
    if [[ ${#failed_pods[@]} -eq 0 ]]; then
        return 0
    else
        error "Failed pods: ${failed_pods[*]}"
        return 1
    fi
}

# Check service endpoints
check_services_health() {
    info "Checking service endpoints..."
    
    local failed_services=()
    local services=("backend-service" "ml-service" "rl-service")
    
    for service in "${services[@]}"; do
        local endpoints
        endpoints=$(kubectl get endpoints "${service}" -n "${NAMESPACE}" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
        
        if [[ -z "${endpoints}" ]]; then
            failed_services+=("${service}")
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        return 0
    else
        error "Services without endpoints: ${failed_services[*]}"
        return 1
    fi
}

# Check deployments status
check_deployments_health() {
    info "Checking deployment status..."
    
    local failed_deployments=()
    local deployments=("backend-deployment" "ml-service-deployment" "rl-service-deployment")
    
    for deployment in "${deployments[@]}"; do
        if ! kubectl get deployment "${deployment}" -n "${NAMESPACE}" &> /dev/null; then
            failed_deployments+=("${deployment}")
            continue
        fi
        
        local ready_replicas
        ready_replicas=$(kubectl get deployment "${deployment}" -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        
        local desired_replicas
        desired_replicas=$(kubectl get deployment "${deployment}" -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
        
        if [[ "${ready_replicas}" -ne "${desired_replicas}" ]]; then
            failed_deployments+=("${deployment}")
        fi
    done
    
    if [[ ${#failed_deployments[@]} -eq 0 ]]; then
        return 0
    else
        error "Failed deployments: ${failed_deployments[*]}"
        return 1
    fi
}

# Check persistent volumes
check_volumes_health() {
    info "Checking persistent volume health..."
    
    local failed_pvcs=()
    local pvcs
    pvcs=$(kubectl get pvc -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    for pvc in ${pvcs}; do
        local status
        status=$(kubectl get pvc "${pvc}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        
        if [[ "${status}" != "Bound" ]]; then
            failed_pvcs+=("${pvc}")
        fi
    done
    
    if [[ ${#failed_pvcs[@]} -eq 0 ]]; then
        return 0
    else
        error "Failed PVCs: ${failed_pvcs[*]}"
        return 1
    fi
}

# Application-level health checks
check_backend_api_health() {
    info "Checking backend API health..."
    
    # Get backend service endpoint
    local backend_service
    backend_service=$(kubectl get service backend-service -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [[ -z "${backend_service}" ]]; then
        error "Backend service not found"
        return 1
    fi
    
    # Create a temporary pod to test the API
    local test_pod="health-check-backend-$(date +%s)"
    
    kubectl run "${test_pod}" \
        --image=curlimages/curl:latest \
        --restart=Never \
        --namespace="${NAMESPACE}" \
        --command -- sleep 60 &>/dev/null
    
    # Wait for pod to be ready
    kubectl wait --for=condition=ready pod/"${test_pod}" -n "${NAMESPACE}" --timeout=30s &>/dev/null
    
    # Test health endpoint
    local health_response
    health_response=$(kubectl exec "${test_pod}" -n "${NAMESPACE}" -- curl -s -o /dev/null -w "%{http_code}" "http://${backend_service}:3000/health" --connect-timeout 10 --max-time 30 2>/dev/null || echo "000")
    
    # Clean up
    kubectl delete pod "${test_pod}" -n "${NAMESPACE}" &>/dev/null
    
    if [[ "${health_response}" == "200" ]]; then
        return 0
    else
        error "Backend API health check failed (HTTP ${health_response})"
        return 1
    fi
}

# Check ML service health
check_ml_service_health() {
    info "Checking ML service health..."
    
    local ml_service
    ml_service=$(kubectl get service ml-service -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [[ -z "${ml_service}" ]]; then
        error "ML service not found"
        return 1
    fi
    
    local test_pod="health-check-ml-$(date +%s)"
    
    kubectl run "${test_pod}" \
        --image=curlimages/curl:latest \
        --restart=Never \
        --namespace="${NAMESPACE}" \
        --command -- sleep 60 &>/dev/null
    
    kubectl wait --for=condition=ready pod/"${test_pod}" -n "${NAMESPACE}" --timeout=30s &>/dev/null
    
    local health_response
    health_response=$(kubectl exec "${test_pod}" -n "${NAMESPACE}" -- curl -s -o /dev/null -w "%{http_code}" "http://${ml_service}:5000/health" --connect-timeout 10 --max-time 30 2>/dev/null || echo "000")
    
    kubectl delete pod "${test_pod}" -n "${NAMESPACE}" &>/dev/null
    
    if [[ "${health_response}" == "200" ]]; then
        return 0
    else
        error "ML service health check failed (HTTP ${health_response})"
        return 1
    fi
}

# Check RL service health
check_rl_service_health() {
    info "Checking RL service health..."
    
    local rl_service
    rl_service=$(kubectl get service rl-service -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [[ -z "${rl_service}" ]]; then
        error "RL service not found"
        return 1
    fi
    
    local test_pod="health-check-rl-$(date +%s)"
    
    kubectl run "${test_pod}" \
        --image=curlimages/curl:latest \
        --restart=Never \
        --namespace="${NAMESPACE}" \
        --command -- sleep 60 &>/dev/null
    
    kubectl wait --for=condition=ready pod/"${test_pod}" -n "${NAMESPACE}" --timeout=30s &>/dev/null
    
    local health_response
    health_response=$(kubectl exec "${test_pod}" -n "${NAMESPACE}" -- curl -s -o /dev/null -w "%{http_code}" "http://${rl_service}:8001/api/v1/health" --connect-timeout 10 --max-time 30 2>/dev/null || echo "000")
    
    kubectl delete pod "${test_pod}" -n "${NAMESPACE}" &>/dev/null
    
    if [[ "${health_response}" == "200" ]]; then
        return 0
    else
        error "RL service health check failed (HTTP ${health_response})"
        return 1
    fi
}

# Check database connectivity
check_database_health() {
    info "Checking database connectivity..."
    
    # Check Redis
    local redis_pod
    redis_pod=$(kubectl get pods -l app.kubernetes.io/component=redis -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "${redis_pod}" ]]; then
        local redis_ping
        redis_ping=$(kubectl exec "${redis_pod}" -n "${NAMESPACE}" -- redis-cli ping 2>/dev/null || echo "FAIL")
        
        if [[ "${redis_ping}" != "PONG" ]]; then
            error "Redis health check failed"
            return 1
        fi
    fi
    
    # Check InfluxDB
    local influxdb_pod
    influxdb_pod=$(kubectl get pods -l app.kubernetes.io/component=influxdb -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "${influxdb_pod}" ]]; then
        local influxdb_ping
        influxdb_ping=$(kubectl exec "${influxdb_pod}" -n "${NAMESPACE}" -- curl -s -o /dev/null -w "%{http_code}" "http://localhost:8086/health" --connect-timeout 5 --max-time 15 2>/dev/null || echo "000")
        
        if [[ "${influxdb_ping}" != "200" ]]; then
            error "InfluxDB health check failed"
            return 1
        fi
    fi
    
    return 0
}

# Check resource utilization
check_resource_utilization() {
    info "Checking resource utilization..."
    
    local high_cpu_pods=()
    local high_memory_pods=()
    
    # Get pod metrics (requires metrics-server)
    if ! kubectl top pods -n "${NAMESPACE}" &>/dev/null; then
        warn "Metrics server not available, skipping resource utilization check"
        return 0
    fi
    
    # Check CPU and memory usage
    while read -r pod cpu memory; do
        [[ "${pod}" == "NAME" ]] && continue
        
        # Extract numeric values
        local cpu_value
        cpu_value=$(echo "${cpu}" | sed 's/m$//' | sed 's/[^0-9]//g')
        
        local memory_value
        memory_value=$(echo "${memory}" | sed 's/Mi$//' | sed 's/Gi$/000/' | sed 's/[^0-9]//g')
        
        # Check thresholds (adjust as needed)
        if [[ "${cpu_value}" -gt 1500 ]]; then  # > 1.5 CPU cores
            high_cpu_pods+=("${pod}")
        fi
        
        if [[ "${memory_value}" -gt 2048 ]]; then  # > 2GB memory
            high_memory_pods+=("${pod}")
        fi
    done < <(kubectl top pods -n "${NAMESPACE}" --no-headers 2>/dev/null)
    
    if [[ ${#high_cpu_pods[@]} -gt 0 ]]; then
        warn "High CPU usage pods: ${high_cpu_pods[*]}"
    fi
    
    if [[ ${#high_memory_pods[@]} -gt 0 ]]; then
        warn "High memory usage pods: ${high_memory_pods[*]}"
    fi
    
    return 0
}

# Deep integration tests
run_integration_tests() {
    info "Running integration tests..."
    
    # Create test pod
    local test_pod="integration-test-$(date +%s)"
    
    kubectl run "${test_pod}" \
        --image=curlimages/curl:latest \
        --restart=Never \
        --namespace="${NAMESPACE}" \
        --command -- sleep 300 &>/dev/null
    
    kubectl wait --for=condition=ready pod/"${test_pod}" -n "${NAMESPACE}" --timeout=60s &>/dev/null
    
    # Test service integration
    local backend_service
    backend_service=$(kubectl get service backend-service -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
    
    # Test basic API endpoints
    local api_tests=(
        "/health"
        "/ready"
        "/metrics"
    )
    
    local failed_tests=()
    
    for endpoint in "${api_tests[@]}"; do
        local response
        response=$(kubectl exec "${test_pod}" -n "${NAMESPACE}" -- curl -s -o /dev/null -w "%{http_code}" "http://${backend_service}:3000${endpoint}" --connect-timeout 10 --max-time 30 2>/dev/null || echo "000")
        
        if [[ "${response}" != "200" ]]; then
            failed_tests+=("${endpoint}")
        fi
    done
    
    # Clean up
    kubectl delete pod "${test_pod}" -n "${NAMESPACE}" &>/dev/null
    
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        return 0
    else
        error "Failed integration tests: ${failed_tests[*]}"
        return 1
    fi
}

# Generate health report
generate_health_report() {
    log "Generating health report..."
    
    local report_file="/tmp/trading-bot-health-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "${report_file}" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "${ENVIRONMENT}",
  "namespace": "${NAMESPACE}",
  "check_type": "${CHECK_TYPE}",
  "overall_status": "$([ ${#FAILED_CHECKS[@]} -eq 0 ] && echo "HEALTHY" || echo "UNHEALTHY")",
  "failed_checks_count": ${#FAILED_CHECKS[@]},
  "total_checks": ${#HEALTH_STATUS[@]},
  "failed_checks": [$(printf '"%s",' "${FAILED_CHECKS[@]}" | sed 's/,$//')]
  "detailed_results": [
$(printf '    "%s",\n' "${HEALTH_STATUS[@]}" | sed 's/,$//')
  ]
}
EOF
    
    info "Health report saved to: ${report_file}"
    
    # Display summary
    echo
    log "=== HEALTH CHECK SUMMARY ==="
    log "Environment: ${ENVIRONMENT}"
    log "Total Checks: ${#HEALTH_STATUS[@]}"
    log "Failed Checks: ${#FAILED_CHECKS[@]}"
    
    if [[ ${#FAILED_CHECKS[@]} -eq 0 ]]; then
        log "🟢 Overall Status: HEALTHY"
    else
        error "🔴 Overall Status: UNHEALTHY"
        error "Failed checks: ${FAILED_CHECKS[*]}"
    fi
    
    echo
    if [[ "${VERBOSE}" == "true" ]]; then
        log "Detailed Results:"
        for result in "${HEALTH_STATUS[@]}"; do
            if [[ "${result}" == *"FAIL"* ]]; then
                error "  ${result}"
            else
                info "  ${result}"
            fi
        done
    fi
}

# Main function
main() {
    log "Starting Trading Bot health checks..."
    
    parse_args "$@"
    validate_settings
    check_prerequisites
    
    # Run health checks based on type
    case "${CHECK_TYPE}" in
        "basic"|"all")
            execute_check "Pod Health" check_pods_health
            execute_check "Service Health" check_services_health
            execute_check "Deployment Health" check_deployments_health
            execute_check "Volume Health" check_volumes_health
            ;&
        
        "extended")
            [[ "${CHECK_TYPE}" == "extended" || "${CHECK_TYPE}" == "all" ]] && {
                execute_check "Backend API Health" check_backend_api_health
                execute_check "ML Service Health" check_ml_service_health
                execute_check "RL Service Health" check_rl_service_health
                execute_check "Database Health" check_database_health
                execute_check "Resource Utilization" check_resource_utilization
            }
            ;&
        
        "deep")
            [[ "${CHECK_TYPE}" == "deep" || "${CHECK_TYPE}" == "all" ]] && {
                execute_check "Integration Tests" run_integration_tests
            }
            ;;
    esac
    
    generate_health_report
    
    # Exit with appropriate code
    if [[ ${#FAILED_CHECKS[@]} -eq 0 ]]; then
        log "🎉 All health checks passed!"
        exit 0
    else
        error "❌ Health checks failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"