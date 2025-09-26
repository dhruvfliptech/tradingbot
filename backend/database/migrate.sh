#!/bin/bash
# Database Migration Script for Supabase
# Usage: ./migrate.sh [up|down|status]

# Load environment variables
source ../.env

SUPABASE_URL=${SUPABASE_URL}
SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
MIGRATION_DIR="./migrations"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run migrations
run_migration() {
    local file=$1
    local direction=$2

    echo -e "${YELLOW}Running migration: $file${NC}"

    # Execute via Supabase SQL editor API
    npx supabase db push --file "$MIGRATION_DIR/$file"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Migration $file completed${NC}"
        # Record migration in migrations table
        record_migration "$file" "$direction"
    else
        echo -e "${RED}✗ Migration $file failed${NC}"
        exit 1
    fi
}

# Record migration status
record_migration() {
    local file=$1
    local direction=$2

    if [ "$direction" = "up" ]; then
        echo "INSERT INTO schema_migrations (version, executed_at) VALUES ('$file', NOW());" | npx supabase db push
    else
        echo "DELETE FROM schema_migrations WHERE version = '$file';" | npx supabase db push
    fi
}

# Check migration status
check_status() {
    echo -e "${YELLOW}Current migration status:${NC}"
    echo "SELECT version, executed_at FROM schema_migrations ORDER BY executed_at DESC;" | npx supabase db push
}

# Main execution
case "$1" in
    up)
        for file in $(ls $MIGRATION_DIR/*.sql | sort); do
            filename=$(basename "$file")
            run_migration "$filename" "up"
        done
        ;;
    down)
        # Implement rollback logic if needed
        echo -e "${YELLOW}Rollback not implemented yet${NC}"
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {up|down|status}"
        exit 1
        ;;
esac