# Database Migrations

This project uses Alembic for database migrations. Alembic is a lightweight database migration tool for SQLAlchemy.

## Running Migrations

To apply all pending migrations:

```bash
cd /path/to/backend
alembic upgrade head
```

## Creating New Migrations

When you change your SQLAlchemy models, you need to create new migrations:

```bash
# Generate a migration automatically
alembic revision --autogenerate -m "Description of your changes"

# Or create a blank migration
alembic revision -m "Description of your changes"
```

## Rolling Back Migrations

To roll back the most recent migration:

```bash
alembic downgrade -1
```

To roll back to a specific migration, use its revision ID:

```bash
alembic downgrade revision_id
```

## Checking Migration Status

To see which migrations have been applied:

```bash
alembic history --verbose
```

To see current revision:

```bash
alembic current
```

## Migration Files

Migration files are stored in the `versions` directory. Each file contains:

- `upgrade()`: Function that applies the changes
- `downgrade()`: Function that reverts the changes

## Best Practices

1. Always review auto-generated migrations before applying them
2. Test migrations on development databases before production
3. Backup your database before applying migrations in production
4. Keep migrations small and focused on specific changes 