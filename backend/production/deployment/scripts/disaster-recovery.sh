#!/bin/bash

# Trading Bot Disaster Recovery Script
# ====================================
# Comprehensive disaster recovery procedures for critical system failures

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
LOG_FILE="/tmp/trading-bot-dr-$(date +%Y%m%d-%H%M%S).log"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-trading-bot}"
DR_TYPE="${DR_TYPE:-full}"
BACKUP_SOURCE="${BACKUP_SOURCE:-latest}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"
TARGET_CLUSTER="${TARGET_CLUSTER:-}"
TIMEOUT="${TIMEOUT:-1800}"  # 30 minutes

# DR Status tracking
DR_STEPS=()
FAILED_STEPS=()
RECOVERY_START_TIME=$(date +%s)

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
    staging     Disaster recovery for staging
    production  Disaster recovery for production

OPTIONS:
    -t, --type TYPE                 DR type: full|partial|data-only|config-only (default: full)
    -s, --source SOURCE             Backup source: latest|DATE|PATH (default: latest)
    -c, --target-cluster CLUSTER   Target cluster for recovery (optional)
    -n, --namespace NAMESPACE       Kubernetes namespace (default: trading-bot)
    -d, --dry-run                   Perform a dry run without actual recovery
    -f, --force                     Force recovery without confirmation
    --timeout SECONDS               Recovery timeout in seconds (default: 1800)
    -h, --help                      Show this help message

DR TYPES:
    full        Complete system recovery (infrastructure + data + applications)
    partial     Application recovery only (assumes infrastructure is healthy)
    data-only   Data recovery only (databases and persistent volumes)
    config-only Configuration recovery only (ConfigMaps, Secrets, RBAC)

EXAMPLES:
    $0 production                           # Full DR for production
    $0 staging --type partial               # Partial recovery for staging
    $0 production --source 20240115-142000  # Recover from specific backup
    $0 production --target-cluster dr-cluster --force

DISASTER SCENARIOS:
    - Complete cluster failure
    - Data corruption or loss
    - Configuration corruption
    - Application deployment failure
    - Multi-region failover

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
            -t|--type)
                DR_TYPE="$2"
                shift 2
                ;;
            -s|--source)
                BACKUP_SOURCE="$2"
                shift 2
                ;;
            -c|--target-cluster)
                TARGET_CLUSTER="$2"
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

# Validate settings
validate_settings() {
    local valid_types=("full" "partial" "data-only" "config-only")
    if [[ ! " ${valid_types[*]} " =~ " ${DR_TYPE} " ]]; then
        error "Invalid DR type: ${DR_TYPE}. Must be one of: ${valid_types[*]}"
        exit 1
    fi
    
    if [[ "${ENVIRONMENT}" != "staging" && "${ENVIRONMENT}" != "production" ]]; then
        error "Invalid environment: ${ENVIRONMENT}. Must be 'staging' or 'production'"
        exit 1
    fi
    
    log "Environment: ${ENVIRONMENT}"
    log "DR Type: ${DR_TYPE}"
    log "Backup Source: ${BACKUP_SOURCE}"
    log "Target Cluster: ${TARGET_CLUSTER:-current}"
    log "Namespace: ${NAMESPACE}"
    log "Dry Run: ${DRY_RUN}"
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    # Check required tools
    for tool in kubectl; do
        if ! command -v "${tool}" &> /dev/null; then
            missing_tools+=("${tool}")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Execute DR step with tracking
execute_dr_step() {
    local step_name="$1"
    local step_function="$2"
    local step_start_time=$(date +%s)
    
    info "Executing DR step: ${step_name}"
    
    if $step_function; then
        local step_end_time=$(date +%s)
        local step_duration=$((step_end_time - step_start_time))
        DR_STEPS+=("${step_name}: SUCCESS (${step_duration}s)")
        log "✅ ${step_name} completed successfully"
        return 0
    else
        DR_STEPS+=("${step_name}: FAILED")
        FAILED_STEPS+=("${step_name}")
        error "❌ ${step_name} failed"
        return 1
    fi
}

# Assess disaster situation
assess_disaster() {
    log "Assessing disaster situation..."
    
    local disaster_assessment=()
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        disaster_assessment+=("CLUSTER_UNREACHABLE")
    fi
    
    # Check namespace existence
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        disaster_assessment+=("NAMESPACE_MISSING")
    fi
    
    # Check core services
    local services=("backend-deployment" "ml-service-deployment" "rl-service-deployment")
    local missing_services=0
    
    for service in "${services[@]}"; do
        if ! kubectl get deployment "${service}" -n "${NAMESPACE}" &> /dev/null; then
            ((missing_services++))
        fi
    done
    
    if [[ ${missing_services} -gt 0 ]]; then
        disaster_assessment+=("SERVICES_MISSING:${missing_services}")
    fi
    
    # Check data stores
    if ! kubectl get statefulset redis-statefulset -n "${NAMESPACE}" &> /dev/null; then
        disaster_assessment+=("REDIS_MISSING")
    fi
    
    if ! kubectl get statefulset influxdb-statefulset -n "${NAMESPACE}" &> /dev/null; then
        disaster_assessment+=("INFLUXDB_MISSING")
    fi
    
    # Check persistent volumes
    local pvc_count
    pvc_count=$(kubectl get pvc -n "${NAMESPACE}" --no-headers 2>/dev/null | wc -l || echo "0")
    
    if [[ ${pvc_count} -eq 0 ]]; then
        disaster_assessment+=("DATA_VOLUMES_MISSING")
    fi
    
    log "Disaster assessment: ${disaster_assessment[*]:-MINIMAL_IMPACT}"
    
    # Determine recovery strategy based on assessment
    if [[ " ${disaster_assessment[*]} " =~ " CLUSTER_UNREACHABLE " ]]; then
        warn "🚨 CLUSTER UNREACHABLE - This requires manual cluster recovery or failover"
        if [[ -z "${TARGET_CLUSTER}" ]]; then
            error "Target cluster must be specified for cluster failover"
            exit 1
        fi
    fi
    
    return 0
}

# Confirm DR operation
confirm_dr_operation() {
    if [[ "${FORCE}" == "true" ]]; then
        return 0
    fi
    
    warn "⚠️  DISASTER RECOVERY CONFIRMATION REQUIRED"
    echo
    echo "Environment: ${ENVIRONMENT}"
    echo "DR Type: ${DR_TYPE}"
    echo "Backup Source: ${BACKUP_SOURCE}"
    echo "Target Cluster: ${TARGET_CLUSTER:-current}"
    echo "Namespace: ${NAMESPACE}"
    echo
    
    if [[ "${ENVIRONMENT}" == "production" ]]; then
        echo "🚨 THIS IS A PRODUCTION DISASTER RECOVERY OPERATION!"
        echo "This will overwrite existing data and configurations."
        echo
    fi
    
    read -rp "Are you absolutely sure you want to proceed? (yes/no): " confirm
    
    if [[ "${confirm}" != "yes" ]]; then
        log "Disaster recovery cancelled by user"
        exit 0
    fi
    
    # Additional confirmation for production
    if [[ "${ENVIRONMENT}" == "production" ]]; then
        read -rp "Type 'DISASTER RECOVERY' to confirm production DR: " prod_confirm
        
        if [[ "${prod_confirm}" != "DISASTER RECOVERY" ]]; then
            log "Production disaster recovery cancelled"
            exit 0
        fi
    fi
}

# Find and validate backup
find_backup() {
    log "Finding backup for recovery..."
    
    local backup_path=""
    
    case "${BACKUP_SOURCE}" in
        "latest")
            # Find latest backup
            local backup_dir="${BACKUP_DIR:-/tmp/trading-bot-backups}"
            if [[ -d "${backup_dir}" ]]; then
                backup_path=$(find "${backup_dir}" -name "trading-bot-${ENVIRONMENT}-*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || echo "")
            fi
            
            # Check S3 if local not found
            if [[ -z "${backup_path}" && -n "${S3_BUCKET:-}" ]]; then
                local latest_s3_backup
                latest_s3_backup=$(aws s3 ls "s3://${S3_BUCKET}/trading-bot/${ENVIRONMENT}/" | sort | tail -1 | awk '{print $4}' || echo "")
                
                if [[ -n "${latest_s3_backup}" ]]; then
                    backup_path="/tmp/${latest_s3_backup}"
                    aws s3 cp "s3://${S3_BUCKET}/trading-bot/${ENVIRONMENT}/${latest_s3_backup}" "${backup_path}"
                fi
            fi
            ;;
        
        20[0-9][0-9][01][0-9][0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9])
            # Specific date format
            local date_backup="trading-bot-${ENVIRONMENT}-full-${BACKUP_SOURCE}"
            
            # Check local first
            if [[ -f "/tmp/trading-bot-backups/${date_backup}.tar.gz" ]]; then
                backup_path="/tmp/trading-bot-backups/${date_backup}.tar.gz"
            elif [[ -n "${S3_BUCKET:-}" ]]; then
                backup_path="/tmp/${date_backup}.tar.gz"
                aws s3 cp "s3://${S3_BUCKET}/trading-bot/${ENVIRONMENT}/${date_backup}.tar.gz" "${backup_path}"
            fi
            ;;
        
        /*)
            # Absolute path provided
            if [[ -f "${BACKUP_SOURCE}" ]]; then
                backup_path="${BACKUP_SOURCE}"
            fi
            ;;
    esac
    
    if [[ -z "${backup_path}" || ! -f "${backup_path}" ]]; then
        error "Backup not found: ${BACKUP_SOURCE}"
        exit 1
    fi
    
    log "Using backup: ${backup_path}"
    
    # Validate backup integrity
    if [[ "${backup_path}" == *.tar.gz ]]; then
        if ! tar -tzf "${backup_path}" > /dev/null 2>&1; then
            error "Backup archive is corrupted: ${backup_path}"
            exit 1
        fi
    fi
    
    # Extract backup if needed
    if [[ "${backup_path}" == *.tar.gz ]]; then
        local extract_dir="/tmp/dr-restore-$(date +%s)"
        mkdir -p "${extract_dir}"
        tar -xzf "${backup_path}" -C "${extract_dir}"
        
        # Find the extracted backup directory
        BACKUP_RESTORE_PATH=$(find "${extract_dir}" -maxdepth 1 -type d -name "trading-bot-*" | head -1)
        
        if [[ -z "${BACKUP_RESTORE_PATH}" ]]; then
            error "Could not find backup contents after extraction"
            exit 1
        fi
    else
        BACKUP_RESTORE_PATH="${backup_path}"
    fi
    
    log "Backup prepared for restore: ${BACKUP_RESTORE_PATH}"
    return 0
}

# Create DR namespace if needed
create_dr_namespace() {
    if [[ "${DR_TYPE}" == "data-only" ]]; then
        return 0
    fi
    
    info "Creating/verifying DR namespace..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Apply namespace configuration
        if [[ -f "${DEPLOYMENT_DIR}/kubernetes/00-namespace.yaml" ]]; then
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/00-namespace.yaml"
        else
            kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
        fi
    else
        info "Dry run: Would create namespace ${NAMESPACE}"
    fi
    
    return 0
}

# Restore configurations
restore_configurations() {
    if [[ "${DR_TYPE}" == "data-only" ]]; then
        return 0
    fi
    
    info "Restoring configurations..."
    
    local config_dir="${BACKUP_RESTORE_PATH}/configs"
    
    if [[ ! -d "${config_dir}" ]]; then
        warn "No configurations found in backup"
        return 0
    fi
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Restore ConfigMaps
        if [[ -f "${config_dir}/configmaps-"*.yaml ]]; then
            kubectl apply -f "${config_dir}"/configmaps-*.yaml
        fi
        
        # Restore Secrets
        if [[ -f "${config_dir}/secrets-"*.yaml ]]; then
            kubectl apply -f "${config_dir}"/secrets-*.yaml
        fi
        
        # Restore RBAC
        if [[ -f "${config_dir}/rbac-"*.yaml ]]; then
            kubectl apply -f "${config_dir}"/rbac-*.yaml
        fi
        
        # Restore ServiceAccounts
        if [[ -f "${config_dir}/serviceaccounts-"*.yaml ]]; then
            kubectl apply -f "${config_dir}"/serviceaccounts-*.yaml
        fi
    else
        info "Dry run: Would restore configurations"
    fi
    
    return 0
}

# Restore persistent volumes
restore_persistent_volumes() {
    if [[ "${DR_TYPE}" == "config-only" ]]; then
        return 0
    fi
    
    info "Restoring persistent volumes..."
    
    local volumes_dir="${BACKUP_RESTORE_PATH}/volumes"
    
    if [[ ! -d "${volumes_dir}" ]]; then
        warn "No volume backups found"
        return 0
    fi
    
    # Apply PVC definitions first
    if [[ -f "${DEPLOYMENT_DIR}/kubernetes/03-persistent-volumes.yaml" ]]; then
        if [[ "${DRY_RUN}" == "false" ]]; then
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/03-persistent-volumes.yaml"
            
            # Wait for PVCs to be bound
            kubectl wait --for=condition=Bound pvc --all -n "${NAMESPACE}" --timeout=300s
        else
            info "Dry run: Would create PVCs"
        fi
    fi
    
    # Restore volume data
    for volume_backup in "${volumes_dir}"/*.tar.gz; do
        [[ ! -f "${volume_backup}" ]] && continue
        
        local pvc_name
        pvc_name=$(basename "${volume_backup}" | sed 's/-[0-9]*-[0-9]*.tar.gz$//')
        
        info "Restoring volume data for PVC: ${pvc_name}"
        
        if [[ "${DRY_RUN}" == "false" ]]; then
            # Create temporary restore pod
            local restore_pod="volume-restore-${pvc_name}-$(date +%s)"
            
            cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${restore_pod}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  containers:
  - name: restore
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: ${pvc_name}
EOF
            
            # Wait for pod to be ready
            kubectl wait --for=condition=ready pod/"${restore_pod}" -n "${NAMESPACE}" --timeout=60s
            
            # Copy and extract backup data
            kubectl cp "${volume_backup}" "${NAMESPACE}/${restore_pod}:/tmp/backup.tar.gz"
            kubectl exec "${restore_pod}" -n "${NAMESPACE}" -- sh -c "cd /data && rm -rf * && tar -xzf /tmp/backup.tar.gz"
            
            # Clean up
            kubectl delete pod "${restore_pod}" -n "${NAMESPACE}"
            
            log "✅ Volume ${pvc_name} restored"
        else
            info "Dry run: Would restore volume ${pvc_name}"
        fi
    done
    
    return 0
}

# Restore databases
restore_databases() {
    if [[ "${DR_TYPE}" == "config-only" ]]; then
        return 0
    fi
    
    info "Restoring databases..."
    
    local db_dir="${BACKUP_RESTORE_PATH}/databases"
    
    if [[ ! -d "${db_dir}" ]]; then
        warn "No database backups found"
        return 0
    fi
    
    # Deploy database infrastructure first
    if [[ "${DR_TYPE}" == "full" ]]; then
        if [[ "${DRY_RUN}" == "false" ]]; then
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/06-statefulsets.yaml"
            
            # Wait for databases to be ready
            kubectl wait --for=condition=ready statefulset/redis-statefulset -n "${NAMESPACE}" --timeout=300s || true
            kubectl wait --for=condition=ready statefulset/influxdb-statefulset -n "${NAMESPACE}" --timeout=300s || true
        else
            info "Dry run: Would deploy database infrastructure"
        fi
    fi
    
    # Restore Redis
    if [[ -f "${db_dir}/redis-dump-"*.rdb ]]; then
        restore_redis "${db_dir}"
    fi
    
    # Restore InfluxDB
    if [[ -d "${db_dir}/influxdb-backup-"* ]]; then
        restore_influxdb "${db_dir}"
    fi
    
    # Restore PostgreSQL
    if [[ -f "${db_dir}/postgresql-dump-"*.sql ]]; then
        restore_postgresql "${db_dir}"
    fi
    
    return 0
}

# Restore Redis database
restore_redis() {
    local db_dir="$1"
    
    info "Restoring Redis database..."
    
    local redis_dump=$(find "${db_dir}" -name "redis-dump-*.rdb" | head -1)
    
    if [[ -z "${redis_dump}" ]]; then
        warn "No Redis dump found"
        return 0
    fi
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Get Redis pod
        local redis_pod
        redis_pod=$(kubectl get pods -l app.kubernetes.io/component=redis -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}')
        
        # Stop Redis temporarily
        kubectl exec "${redis_pod}" -n "${NAMESPACE}" -- redis-cli shutdown nosave || true
        
        # Copy dump file
        kubectl cp "${redis_dump}" "${NAMESPACE}/${redis_pod}:/data/dump.rdb"
        
        # Restart Redis
        kubectl delete pod "${redis_pod}" -n "${NAMESPACE}"
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=redis -n "${NAMESPACE}" --timeout=120s
        
        log "✅ Redis database restored"
    else
        info "Dry run: Would restore Redis from ${redis_dump}"
    fi
}

# Restore InfluxDB database
restore_influxdb() {
    local db_dir="$1"
    
    info "Restoring InfluxDB database..."
    
    local influxdb_backup=$(find "${db_dir}" -name "influxdb-backup-*" -type d | head -1)
    
    if [[ -z "${influxdb_backup}" ]]; then
        warn "No InfluxDB backup found"
        return 0
    fi
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Get InfluxDB pod
        local influxdb_pod
        influxdb_pod=$(kubectl get pods -l app.kubernetes.io/component=influxdb -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}')
        
        # Copy backup to pod
        kubectl cp "${influxdb_backup}" "${NAMESPACE}/${influxdb_pod}:/tmp/restore-backup"
        
        # Restore database
        kubectl exec "${influxdb_pod}" -n "${NAMESPACE}" -- influx restore "/tmp/restore-backup" --bucket trading_bot
        
        # Clean up
        kubectl exec "${influxdb_pod}" -n "${NAMESPACE}" -- rm -rf "/tmp/restore-backup"
        
        log "✅ InfluxDB database restored"
    else
        info "Dry run: Would restore InfluxDB from ${influxdb_backup}"
    fi
}

# Restore PostgreSQL database
restore_postgresql() {
    local db_dir="$1"
    
    info "Restoring PostgreSQL database..."
    
    local pg_dump=$(find "${db_dir}" -name "postgresql-dump-*.sql" | head -1)
    
    if [[ -z "${pg_dump}" ]]; then
        warn "No PostgreSQL dump found"
        return 0
    fi
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Create temporary restore pod
        local restore_pod="postgres-restore-$(date +%s)"
        
        kubectl run "${restore_pod}" \
            --image=postgres:15-alpine \
            --restart=Never \
            --namespace="${NAMESPACE}" \
            --env="DATABASE_URL=${DATABASE_URL}" \
            --command -- sleep 3600
        
        kubectl wait --for=condition=ready pod/"${restore_pod}" -n "${NAMESPACE}" --timeout=60s
        
        # Copy and restore dump
        kubectl cp "${pg_dump}" "${NAMESPACE}/${restore_pod}:/tmp/restore.sql"
        kubectl exec "${restore_pod}" -n "${NAMESPACE}" -- psql "${DATABASE_URL}" -f /tmp/restore.sql
        
        # Clean up
        kubectl delete pod "${restore_pod}" -n "${NAMESPACE}"
        
        log "✅ PostgreSQL database restored"
    else
        info "Dry run: Would restore PostgreSQL from ${pg_dump}"
    fi
}

# Restore applications
restore_applications() {
    if [[ "${DR_TYPE}" == "data-only" || "${DR_TYPE}" == "config-only" ]]; then
        return 0
    fi
    
    info "Restoring applications..."
    
    local app_files=(
        "04-services.yaml"
        "05-deployments.yaml"
        "07-hpa-autoscaling.yaml"
    )
    
    for file in "${app_files[@]}"; do
        if [[ -f "${DEPLOYMENT_DIR}/kubernetes/${file}" ]]; then
            info "Deploying ${file}..."
            
            if [[ "${DRY_RUN}" == "false" ]]; then
                kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/${file}"
            else
                info "Dry run: Would apply ${file}"
            fi
        fi
    done
    
    # Wait for applications to be ready
    if [[ "${DRY_RUN}" == "false" ]]; then
        kubectl wait --for=condition=available deployment --all -n "${NAMESPACE}" --timeout="${TIMEOUT}s"
    fi
    
    return 0
}

# Verify recovery
verify_recovery() {
    info "Verifying disaster recovery..."
    
    if [[ "${DRY_RUN}" == "false" ]]; then
        # Run comprehensive health checks
        if [[ -f "${SCRIPT_DIR}/health-check.sh" ]]; then
            "${SCRIPT_DIR}/health-check.sh" "${ENVIRONMENT}" --check-type extended --timeout 120
        else
            warn "Health check script not found, performing basic verification"
            
            # Basic verification
            kubectl wait --for=condition=available deployment --all -n "${NAMESPACE}" --timeout=300s
            kubectl get pods -n "${NAMESPACE}"
        fi
    else
        info "Dry run: Would verify recovery with health checks"
    fi
    
    return 0
}

# Generate DR report
generate_dr_report() {
    local recovery_end_time=$(date +%s)
    local total_duration=$((recovery_end_time - RECOVERY_START_TIME))
    
    log "Generating disaster recovery report..."
    
    local report_file="/tmp/trading-bot-dr-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "${report_file}" << EOF
{
  "disaster_recovery": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "${ENVIRONMENT}",
    "namespace": "${NAMESPACE}",
    "dr_type": "${DR_TYPE}",
    "backup_source": "${BACKUP_SOURCE}",
    "target_cluster": "${TARGET_CLUSTER:-current}",
    "total_duration_seconds": ${total_duration},
    "dry_run": ${DRY_RUN},
    "overall_status": "$([ ${#FAILED_STEPS[@]} -eq 0 ] && echo "SUCCESS" || echo "FAILED")",
    "failed_steps_count": ${#FAILED_STEPS[@]},
    "total_steps": ${#DR_STEPS[@]}
  },
  "failed_steps": [$(printf '"%s",' "${FAILED_STEPS[@]}" | sed 's/,$//')]
  "detailed_steps": [
$(printf '    "%s",\n' "${DR_STEPS[@]}" | sed 's/,$//')
  ]
}
EOF
    
    info "DR report saved to: ${report_file}"
    
    # Display summary
    echo
    log "=== DISASTER RECOVERY SUMMARY ==="
    log "Environment: ${ENVIRONMENT}"
    log "DR Type: ${DR_TYPE}"
    log "Total Duration: $((total_duration / 60))m $((total_duration % 60))s"
    log "Total Steps: ${#DR_STEPS[@]}"
    log "Failed Steps: ${#FAILED_STEPS[@]}"
    
    if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
        log "🟢 Overall Status: SUCCESS"
    else
        error "🔴 Overall Status: FAILED"
        error "Failed steps: ${FAILED_STEPS[*]}"
    fi
}

# Send notifications
send_notifications() {
    local status="$1"
    local duration="$2"
    local message
    
    if [[ "${status}" == "success" ]]; then
        message="🆘➡️✅ Trading Bot disaster recovery COMPLETED for ${ENVIRONMENT}! Duration: ${duration}"
    else
        message="🆘❌ Trading Bot disaster recovery FAILED for ${ENVIRONMENT}!"
    fi
    
    info "${message}"
    
    # Send Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" && "${DRY_RUN}" == "false" ]]; then
        curl -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-type: application/json' \
            --data "{\"text\":\"${message}\"}" \
            --silent || true
    fi
    
    # Send email notification
    if [[ -n "${EMAIL_NOTIFICATIONS:-}" && "${DRY_RUN}" == "false" ]]; then
        echo "${message}" | mail -s "Trading Bot Disaster Recovery ${status}" "${EMAIL_NOTIFICATIONS}" || true
    fi
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    local recovery_end_time=$(date +%s)
    local total_duration=$((recovery_end_time - RECOVERY_START_TIME))
    local duration_str="$((total_duration / 60))m $((total_duration % 60))s"
    
    if [[ ${exit_code} -ne 0 ]]; then
        error "Disaster recovery failed with exit code ${exit_code}"
        send_notifications "failure" "${duration_str}"
    else
        log "Disaster recovery completed successfully"
        send_notifications "success" "${duration_str}"
    fi
    
    generate_dr_report
    info "DR log saved to: ${LOG_FILE}"
    exit ${exit_code}
}

# Main function
main() {
    trap cleanup EXIT
    
    log "🆘 INITIATING TRADING BOT DISASTER RECOVERY"
    log "Log file: ${LOG_FILE}"
    
    parse_args "$@"
    validate_settings
    check_prerequisites
    
    execute_dr_step "Disaster Assessment" assess_disaster
    confirm_dr_operation
    execute_dr_step "Find Backup" find_backup
    
    # Execute DR steps based on type
    case "${DR_TYPE}" in
        "full")
            execute_dr_step "Create Namespace" create_dr_namespace
            execute_dr_step "Restore Configurations" restore_configurations
            execute_dr_step "Restore Persistent Volumes" restore_persistent_volumes
            execute_dr_step "Restore Databases" restore_databases
            execute_dr_step "Restore Applications" restore_applications
            ;;
        "partial")
            execute_dr_step "Restore Configurations" restore_configurations
            execute_dr_step "Restore Applications" restore_applications
            ;;
        "data-only")
            execute_dr_step "Restore Persistent Volumes" restore_persistent_volumes
            execute_dr_step "Restore Databases" restore_databases
            ;;
        "config-only")
            execute_dr_step "Create Namespace" create_dr_namespace
            execute_dr_step "Restore Configurations" restore_configurations
            ;;
    esac
    
    execute_dr_step "Verify Recovery" verify_recovery
    
    if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
        log "🎉 Disaster recovery completed successfully!"
    else
        error "❌ Disaster recovery completed with failures!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"