#!/bin/bash

# Trading Bot Backup Script
# =========================
# Comprehensive backup solution for databases, configurations, and application data

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
LOG_FILE="/tmp/trading-bot-backup-$(date +%Y%m%d-%H%M%S).log"

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="${NAMESPACE:-trading-bot}"
BACKUP_TYPE="${BACKUP_TYPE:-full}"
BACKUP_DIR="${BACKUP_DIR:-/tmp/trading-bot-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
S3_BUCKET="${S3_BUCKET:-}"
COMPRESS="${COMPRESS:-true}"
ENCRYPT="${ENCRYPT:-true}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Backup timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="trading-bot-${ENVIRONMENT}-${BACKUP_TYPE}-${TIMESTAMP}"

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
    staging     Backup staging environment
    production  Backup production environment

OPTIONS:
    -t, --type TYPE                 Backup type: full|database|configs|volumes (default: full)
    -o, --output DIR                Backup output directory (default: /tmp/trading-bot-backups)
    -r, --retention DAYS            Retention period in days (default: 30)
    -s, --s3-bucket BUCKET          S3 bucket for remote backup storage
    -n, --namespace NAMESPACE       Kubernetes namespace (default: trading-bot)
    --no-compress                   Disable compression
    --no-encrypt                    Disable encryption
    --parallel-jobs JOBS            Number of parallel backup jobs (default: 4)
    -h, --help                      Show this help message

BACKUP TYPES:
    full        Complete backup including databases, configs, and volumes
    database    Database backups only (PostgreSQL, InfluxDB, Redis)
    configs     Configuration backups (ConfigMaps, Secrets)
    volumes     Persistent volume backups
    pre-deploy  Pre-deployment backup (lighter, configs + critical data)
    pre-rollback Pre-rollback backup

EXAMPLES:
    $0 production
    $0 staging --type database
    $0 production --s3-bucket my-backups-bucket
    $0 staging --type configs --output /backup/staging

ENVIRONMENT VARIABLES:
    BACKUP_TYPE         Type of backup to perform
    BACKUP_DIR          Output directory for backups
    S3_BUCKET           S3 bucket for remote storage
    RETENTION_DAYS      Backup retention period
    COMPRESS            Enable/disable compression
    ENCRYPT             Enable/disable encryption

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
                BACKUP_TYPE="$2"
                shift 2
                ;;
            -o|--output)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -s|--s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --no-compress)
                COMPRESS="false"
                shift
                ;;
            --no-encrypt)
                ENCRYPT="false"
                shift
                ;;
            --parallel-jobs)
                PARALLEL_JOBS="$2"
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

# Validate environment and settings
validate_settings() {
    if [[ "${ENVIRONMENT}" != "staging" && "${ENVIRONMENT}" != "production" ]]; then
        error "Invalid environment: ${ENVIRONMENT}. Must be 'staging' or 'production'"
        exit 1
    fi
    
    local valid_types=("full" "database" "configs" "volumes" "pre-deploy" "pre-rollback")
    if [[ ! " ${valid_types[*]} " =~ " ${BACKUP_TYPE} " ]]; then
        error "Invalid backup type: ${BACKUP_TYPE}. Must be one of: ${valid_types[*]}"
        exit 1
    fi
    
    log "Environment: ${ENVIRONMENT}"
    log "Backup Type: ${BACKUP_TYPE}"
    log "Backup Directory: ${BACKUP_DIR}"
    log "Namespace: ${NAMESPACE}"
    log "Retention: ${RETENTION_DAYS} days"
    log "Compression: ${COMPRESS}"
    log "Encryption: ${ENCRYPT}"
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
    
    # Check optional tools
    if [[ "${COMPRESS}" == "true" ]] && ! command -v gzip &> /dev/null; then
        missing_tools+=("gzip")
    fi
    
    if [[ "${ENCRYPT}" == "true" ]] && ! command -v gpg &> /dev/null; then
        missing_tools+=("gpg")
    fi
    
    if [[ -n "${S3_BUCKET}" ]] && ! command -v aws &> /dev/null; then
        missing_tools+=("aws")
    fi
    
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
        error "Namespace ${NAMESPACE} does not exist"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Setup backup directory
setup_backup_dir() {
    local backup_path="${BACKUP_DIR}/${BACKUP_NAME}"
    
    if [[ ! -d "${backup_path}" ]]; then
        mkdir -p "${backup_path}"
        info "Created backup directory: ${backup_path}"
    fi
    
    # Create subdirectories
    mkdir -p "${backup_path}"/{databases,configs,volumes,manifests,logs}
    
    # Set the global backup path
    BACKUP_PATH="${backup_path}"
    
    log "Backup will be stored in: ${BACKUP_PATH}"
}

# Generate backup manifest
generate_manifest() {
    local manifest_file="${BACKUP_PATH}/backup-manifest.json"
    
    cat > "${manifest_file}" << EOF
{
  "backup_info": {
    "name": "${BACKUP_NAME}",
    "environment": "${ENVIRONMENT}",
    "namespace": "${NAMESPACE}",
    "type": "${BACKUP_TYPE}",
    "timestamp": "${TIMESTAMP}",
    "created_by": "${USER:-system}",
    "hostname": "$(hostname)",
    "kubernetes_version": "$(kubectl version --short --client | head -n 1 | awk '{print $3}')"
  },
  "cluster_info": {
    "cluster_name": "$(kubectl config current-context)",
    "server": "$(kubectl cluster-info | head -n 1 | awk -F'https://' '{print $2}' | awk '{print $1}')"
  },
  "components": []
}
EOF
    
    info "Created backup manifest: ${manifest_file}"
}

# Backup databases
backup_databases() {
    if [[ "${BACKUP_TYPE}" != "full" && "${BACKUP_TYPE}" != "database" && "${BACKUP_TYPE}" != "pre-deploy" ]]; then
        return 0
    fi
    
    log "Backing up databases..."
    
    local db_backup_dir="${BACKUP_PATH}/databases"
    
    # Backup Redis
    backup_redis "${db_backup_dir}"
    
    # Backup InfluxDB
    backup_influxdb "${db_backup_dir}"
    
    # Backup PostgreSQL/Supabase (if using managed database)
    backup_postgresql "${db_backup_dir}"
    
    log "Database backups completed"
}

# Backup Redis
backup_redis() {
    local output_dir="$1"
    
    info "Backing up Redis..."
    
    # Check if Redis is running
    if ! kubectl get statefulset redis-statefulset -n "${NAMESPACE}" &> /dev/null; then
        warn "Redis StatefulSet not found, skipping Redis backup"
        return 0
    fi
    
    # Get Redis pod
    local redis_pod
    redis_pod=$(kubectl get pods -l app.kubernetes.io/component=redis -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "${redis_pod}" ]]; then
        warn "Redis pod not found, skipping Redis backup"
        return 0
    fi
    
    # Create Redis backup
    info "Creating Redis backup from pod: ${redis_pod}"
    
    # Save Redis data
    kubectl exec "${redis_pod}" -n "${NAMESPACE}" -- redis-cli BGSAVE
    
    # Wait for backup to complete
    sleep 5
    
    # Copy dump.rdb file
    kubectl cp "${NAMESPACE}/${redis_pod}:/data/dump.rdb" "${output_dir}/redis-dump-${TIMESTAMP}.rdb"
    
    # Get Redis info
    kubectl exec "${redis_pod}" -n "${NAMESPACE}" -- redis-cli INFO > "${output_dir}/redis-info-${TIMESTAMP}.txt"
    
    log "✅ Redis backup completed"
}

# Backup InfluxDB
backup_influxdb() {
    local output_dir="$1"
    
    info "Backing up InfluxDB..."
    
    # Check if InfluxDB is running
    if ! kubectl get statefulset influxdb-statefulset -n "${NAMESPACE}" &> /dev/null; then
        warn "InfluxDB StatefulSet not found, skipping InfluxDB backup"
        return 0
    fi
    
    # Get InfluxDB pod
    local influxdb_pod
    influxdb_pod=$(kubectl get pods -l app.kubernetes.io/component=influxdb -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "${influxdb_pod}" ]]; then
        warn "InfluxDB pod not found, skipping InfluxDB backup"
        return 0
    fi
    
    info "Creating InfluxDB backup from pod: ${influxdb_pod}"
    
    # Export InfluxDB data
    kubectl exec "${influxdb_pod}" -n "${NAMESPACE}" -- influx backup "/tmp/influxdb-backup-${TIMESTAMP}" --bucket trading_bot
    
    # Copy backup files
    kubectl cp "${NAMESPACE}/${influxdb_pod}:/tmp/influxdb-backup-${TIMESTAMP}" "${output_dir}/influxdb-backup-${TIMESTAMP}"
    
    # Clean up temporary backup on pod
    kubectl exec "${influxdb_pod}" -n "${NAMESPACE}" -- rm -rf "/tmp/influxdb-backup-${TIMESTAMP}"
    
    log "✅ InfluxDB backup completed"
}

# Backup PostgreSQL (external database)
backup_postgresql() {
    local output_dir="$1"
    
    if [[ -z "${DATABASE_URL:-}" ]]; then
        info "No PostgreSQL DATABASE_URL configured, skipping PostgreSQL backup"
        return 0
    fi
    
    info "Backing up PostgreSQL database..."
    
    # Create a temporary pod for database backup
    local backup_pod="postgres-backup-${TIMESTAMP}"
    
    kubectl run "${backup_pod}" \
        --image=postgres:15-alpine \
        --restart=Never \
        --namespace="${NAMESPACE}" \
        --env="DATABASE_URL=${DATABASE_URL}" \
        --command -- sleep 3600
    
    # Wait for pod to be ready
    kubectl wait --for=condition=ready pod/"${backup_pod}" -n "${NAMESPACE}" --timeout=60s
    
    # Perform backup
    kubectl exec "${backup_pod}" -n "${NAMESPACE}" -- pg_dump "${DATABASE_URL}" > "${output_dir}/postgresql-dump-${TIMESTAMP}.sql"
    
    # Clean up
    kubectl delete pod "${backup_pod}" -n "${NAMESPACE}"
    
    log "✅ PostgreSQL backup completed"
}

# Backup configurations
backup_configs() {
    if [[ "${BACKUP_TYPE}" == "volumes" ]]; then
        return 0
    fi
    
    log "Backing up configurations..."
    
    local config_backup_dir="${BACKUP_PATH}/configs"
    
    # Backup ConfigMaps
    info "Backing up ConfigMaps..."
    kubectl get configmaps -n "${NAMESPACE}" -o yaml > "${config_backup_dir}/configmaps-${TIMESTAMP}.yaml"
    
    # Backup Secrets (excluding system secrets)
    info "Backing up Secrets..."
    kubectl get secrets -n "${NAMESPACE}" -l app.kubernetes.io/name=trading-bot -o yaml > "${config_backup_dir}/secrets-${TIMESTAMP}.yaml"
    
    # Backup ServiceAccounts
    info "Backing up ServiceAccounts..."
    kubectl get serviceaccounts -n "${NAMESPACE}" -o yaml > "${config_backup_dir}/serviceaccounts-${TIMESTAMP}.yaml"
    
    # Backup RBAC
    info "Backing up RBAC..."
    kubectl get roles,rolebindings -n "${NAMESPACE}" -o yaml > "${config_backup_dir}/rbac-${TIMESTAMP}.yaml"
    
    # Backup NetworkPolicies
    info "Backing up NetworkPolicies..."
    kubectl get networkpolicies -n "${NAMESPACE}" -o yaml > "${config_backup_dir}/networkpolicies-${TIMESTAMP}.yaml" 2>/dev/null || true
    
    log "✅ Configuration backup completed"
}

# Backup persistent volumes
backup_volumes() {
    if [[ "${BACKUP_TYPE}" != "full" && "${BACKUP_TYPE}" != "volumes" ]]; then
        return 0
    fi
    
    log "Backing up persistent volumes..."
    
    local volume_backup_dir="${BACKUP_PATH}/volumes"
    
    # Get all PVCs in the namespace
    local pvcs
    pvcs=$(kubectl get pvc -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "${pvcs}" ]]; then
        warn "No persistent volume claims found"
        return 0
    fi
    
    # Backup each PVC
    for pvc in ${pvcs}; do
        backup_pvc "${pvc}" "${volume_backup_dir}"
    done
    
    log "✅ Persistent volume backup completed"
}

# Backup individual PVC
backup_pvc() {
    local pvc_name="$1"
    local output_dir="$2"
    
    info "Backing up PVC: ${pvc_name}"
    
    # Create a backup pod for this PVC
    local backup_pod="volume-backup-${pvc_name}-${TIMESTAMP}"
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${backup_pod}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  containers:
  - name: backup
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
    kubectl wait --for=condition=ready pod/"${backup_pod}" -n "${NAMESPACE}" --timeout=60s
    
    # Create tar archive of the volume
    kubectl exec "${backup_pod}" -n "${NAMESPACE}" -- tar -czf "/tmp/${pvc_name}-${TIMESTAMP}.tar.gz" -C /data .
    
    # Copy the archive
    kubectl cp "${NAMESPACE}/${backup_pod}:/tmp/${pvc_name}-${TIMESTAMP}.tar.gz" "${output_dir}/${pvc_name}-${TIMESTAMP}.tar.gz"
    
    # Clean up
    kubectl delete pod "${backup_pod}" -n "${NAMESPACE}"
    
    info "✅ PVC ${pvc_name} backed up"
}

# Backup Kubernetes manifests
backup_manifests() {
    if [[ "${BACKUP_TYPE}" == "volumes" ]]; then
        return 0
    fi
    
    log "Backing up Kubernetes manifests..."
    
    local manifest_backup_dir="${BACKUP_PATH}/manifests"
    
    # Backup all resources in the namespace
    local resources=(
        "deployments"
        "statefulsets"
        "services"
        "ingresses"
        "horizontalpodautoscalers"
        "poddisruptionbudgets"
    )
    
    for resource in "${resources[@]}"; do
        info "Backing up ${resource}..."
        kubectl get "${resource}" -n "${NAMESPACE}" -o yaml > "${manifest_backup_dir}/${resource}-${TIMESTAMP}.yaml" 2>/dev/null || true
    done
    
    log "✅ Kubernetes manifests backed up"
}

# Backup application logs
backup_logs() {
    if [[ "${BACKUP_TYPE}" != "full" && "${BACKUP_TYPE}" != "pre-rollback" ]]; then
        return 0
    fi
    
    log "Backing up application logs..."
    
    local log_backup_dir="${BACKUP_PATH}/logs"
    
    # Get pods
    local pods
    pods=$(kubectl get pods -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "${pods}" ]]; then
        warn "No pods found for log backup"
        return 0
    fi
    
    # Backup logs from each pod
    for pod in ${pods}; do
        info "Backing up logs from pod: ${pod}"
        kubectl logs "${pod}" -n "${NAMESPACE}" --all-containers=true > "${log_backup_dir}/${pod}-${TIMESTAMP}.log" 2>/dev/null || true
        
        # Also get previous logs if available
        kubectl logs "${pod}" -n "${NAMESPACE}" --previous --all-containers=true > "${log_backup_dir}/${pod}-previous-${TIMESTAMP}.log" 2>/dev/null || true
    done
    
    log "✅ Application logs backed up"
}

# Compress backup
compress_backup() {
    if [[ "${COMPRESS}" != "true" ]]; then
        return 0
    fi
    
    log "Compressing backup..."
    
    local compressed_file="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    
    # Create compressed archive
    tar -czf "${compressed_file}" -C "${BACKUP_DIR}" "${BACKUP_NAME}"
    
    # Remove uncompressed directory
    rm -rf "${BACKUP_PATH}"
    
    # Update backup path
    BACKUP_PATH="${compressed_file}"
    
    log "✅ Backup compressed to: ${BACKUP_PATH}"
}

# Encrypt backup
encrypt_backup() {
    if [[ "${ENCRYPT}" != "true" ]]; then
        return 0
    fi
    
    log "Encrypting backup..."
    
    # Check if GPG key is available
    local gpg_key="${GPG_BACKUP_KEY:-trading-bot-backup@example.com}"
    
    if ! gpg --list-keys "${gpg_key}" &> /dev/null; then
        warn "GPG key ${gpg_key} not found, skipping encryption"
        return 0
    fi
    
    local encrypted_file="${BACKUP_PATH}.gpg"
    
    # Encrypt the backup
    gpg --trust-model always --encrypt --recipient "${gpg_key}" --output "${encrypted_file}" "${BACKUP_PATH}"
    
    # Remove unencrypted file
    rm -f "${BACKUP_PATH}"
    
    # Update backup path
    BACKUP_PATH="${encrypted_file}"
    
    log "✅ Backup encrypted to: ${BACKUP_PATH}"
}

# Upload to S3
upload_to_s3() {
    if [[ -z "${S3_BUCKET}" ]]; then
        return 0
    fi
    
    log "Uploading backup to S3..."
    
    local s3_key="trading-bot/${ENVIRONMENT}/$(basename "${BACKUP_PATH}")"
    
    # Upload backup
    aws s3 cp "${BACKUP_PATH}" "s3://${S3_BUCKET}/${s3_key}"
    
    # Set lifecycle policy for automatic cleanup
    aws s3api put-object-tagging \
        --bucket "${S3_BUCKET}" \
        --key "${s3_key}" \
        --tagging "TagSet=[{Key=Environment,Value=${ENVIRONMENT}},{Key=RetentionDays,Value=${RETENTION_DAYS}}]"
    
    log "✅ Backup uploaded to: s3://${S3_BUCKET}/${s3_key}"
}

# Clean up old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Clean up local backups
    find "${BACKUP_DIR}" -name "trading-bot-${ENVIRONMENT}-*" -type f -mtime +"${RETENTION_DAYS}" -delete 2>/dev/null || true
    find "${BACKUP_DIR}" -name "trading-bot-${ENVIRONMENT}-*" -type d -mtime +"${RETENTION_DAYS}" -exec rm -rf {} + 2>/dev/null || true
    
    # Clean up S3 backups if configured
    if [[ -n "${S3_BUCKET}" ]]; then
        aws s3 ls "s3://${S3_BUCKET}/trading-bot/${ENVIRONMENT}/" | while read -r line; do
            local date_str=$(echo "${line}" | awk '{print $1}')
            local file_name=$(echo "${line}" | awk '{print $4}')
            
            # Calculate age
            local file_date=$(date -d "${date_str}" +%s 2>/dev/null || echo "0")
            local cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%s)
            
            if [[ "${file_date}" -lt "${cutoff_date}" && -n "${file_name}" ]]; then
                info "Deleting old S3 backup: ${file_name}"
                aws s3 rm "s3://${S3_BUCKET}/trading-bot/${ENVIRONMENT}/${file_name}"
            fi
        done
    fi
    
    log "✅ Old backups cleaned up"
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    local verification_passed=true
    
    # Check if backup file exists
    if [[ ! -f "${BACKUP_PATH}" && ! -d "${BACKUP_PATH}" ]]; then
        error "Backup file/directory does not exist: ${BACKUP_PATH}"
        verification_passed=false
    fi
    
    # Check file size
    if [[ -f "${BACKUP_PATH}" ]]; then
        local file_size
        file_size=$(stat -f%z "${BACKUP_PATH}" 2>/dev/null || stat -c%s "${BACKUP_PATH}" 2>/dev/null || echo "0")
        
        if [[ "${file_size}" -lt 1024 ]]; then
            error "Backup file is too small: ${file_size} bytes"
            verification_passed=false
        else
            info "Backup size: $(numfmt --to=iec "${file_size}")"
        fi
    fi
    
    # Verify compressed archive
    if [[ "${BACKUP_PATH}" == *.tar.gz && "${verification_passed}" == "true" ]]; then
        if ! tar -tzf "${BACKUP_PATH}" > /dev/null 2>&1; then
            error "Backup archive is corrupted"
            verification_passed=false
        fi
    fi
    
    # Verify encryption
    if [[ "${BACKUP_PATH}" == *.gpg && "${verification_passed}" == "true" ]]; then
        if ! gpg --list-packets "${BACKUP_PATH}" > /dev/null 2>&1; then
            error "Backup encryption is invalid"
            verification_passed=false
        fi
    fi
    
    if [[ "${verification_passed}" == "true" ]]; then
        log "✅ Backup verification passed"
    else
        error "❌ Backup verification failed"
        exit 1
    fi
}

# Send notifications
send_notifications() {
    local status="$1"
    local backup_size="$2"
    local message
    
    if [[ "${status}" == "success" ]]; then
        message="📦 Trading Bot backup completed successfully for ${ENVIRONMENT}! Size: ${backup_size}, Type: ${BACKUP_TYPE}"
    else
        message="🚨 Trading Bot backup FAILED for ${ENVIRONMENT}!"
    fi
    
    info "${message}"
    
    # Send Slack notification if webhook is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-type: application/json' \
            --data "{\"text\":\"${message}\"}" \
            --silent || true
    fi
    
    # Send email notification if configured
    if [[ -n "${EMAIL_NOTIFICATIONS:-}" ]]; then
        echo "${message}" | mail -s "Trading Bot Backup ${status}" "${EMAIL_NOTIFICATIONS}" || true
    fi
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    
    # Calculate backup size
    local backup_size="Unknown"
    if [[ -f "${BACKUP_PATH}" ]]; then
        local file_size
        file_size=$(stat -f%z "${BACKUP_PATH}" 2>/dev/null || stat -c%s "${BACKUP_PATH}" 2>/dev/null || echo "0")
        backup_size=$(numfmt --to=iec "${file_size}")
    fi
    
    if [[ ${exit_code} -ne 0 ]]; then
        error "Backup failed with exit code ${exit_code}"
        send_notifications "failure" "${backup_size}"
    else
        log "Backup completed successfully"
        send_notifications "success" "${backup_size}"
    fi
    
    info "Backup log saved to: ${LOG_FILE}"
    exit ${exit_code}
}

# Main function
main() {
    trap cleanup EXIT
    
    log "Starting Trading Bot backup..."
    log "Log file: ${LOG_FILE}"
    
    parse_args "$@"
    validate_settings
    check_prerequisites
    setup_backup_dir
    generate_manifest
    
    case "${BACKUP_TYPE}" in
        "full")
            backup_databases
            backup_configs
            backup_volumes
            backup_manifests
            backup_logs
            ;;
        "database")
            backup_databases
            ;;
        "configs")
            backup_configs
            backup_manifests
            ;;
        "volumes")
            backup_volumes
            ;;
        "pre-deploy")
            backup_configs
            backup_manifests
            backup_databases
            ;;
        "pre-rollback")
            backup_configs
            backup_manifests
            backup_logs
            ;;
    esac
    
    compress_backup
    encrypt_backup
    verify_backup
    upload_to_s3
    cleanup_old_backups
    
    log "🎉 Backup completed successfully!"
    log "Backup location: ${BACKUP_PATH}"
}

# Run main function with all arguments
main "$@"