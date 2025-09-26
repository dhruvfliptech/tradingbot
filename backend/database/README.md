# Database Schema Management

## Overview
This project uses Supabase (PostgreSQL) with a comprehensive schema for trading operations.

## Migration Structure
```
migrations/
├── 000_migrations_table.sql     # Migration tracking
├── 001_trading_tables.sql       # Core trading schema
├── 002_backtesting_tables.sql   # Backtesting schema
└── 003_indexes_performance.sql  # Performance optimizations (future)
```

## Running Migrations

### Automatic (via script)
```bash
# Run all pending migrations
./migrate.sh up

# Check migration status
./migrate.sh status
```

### Manual (via Supabase Dashboard)
1. Go to Supabase SQL Editor
2. Copy contents of migration file
3. Execute SQL
4. Record in schema_migrations table

## Schema Features

### Security
- Row Level Security (RLS) on all tables
- UUID primary keys
- Encrypted API key storage
- Audit trails

### Performance
- Strategic indexes on frequently queried columns
- Materialized views for analytics (planned)
- Partitioning for large tables (future)

### Data Integrity
- Foreign key constraints
- Check constraints for business rules
- Trigger-based validations
- Cascading deletes where appropriate

## Best Practices

### Adding New Migrations
1. Name files with incrementing numbers: `004_feature_name.sql`
2. Include both UP and DOWN migrations (comments for rollback)
3. Test in development first
4. Add meaningful comments

### Migration Guidelines
- Keep migrations atomic and reversible
- Never modify existing migrations in production
- Use transactions for multi-table changes
- Document breaking changes

## Backup Strategy
Supabase provides automatic backups:
- Point-in-time recovery (7 days on Pro)
- Daily backups retained for 30 days
- Manual snapshots before major changes

## Monitoring
- Use Supabase Dashboard for query performance
- Monitor slow queries via pg_stat_statements
- Set up alerts for failed migrations

## Future Enhancements
1. **Partitioning**: Partition trades/orders tables by month
2. **Archiving**: Move old data to archive tables
3. **Read Replicas**: For analytics queries
4. **Caching Layer**: Redis for frequently accessed data