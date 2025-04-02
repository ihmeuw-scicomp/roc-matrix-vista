# Database Migrations

This directory contains database migrations for the application using Alembic.

## Migration History

The migrations should be applied in the following order:

1. `01_add_raw_data_to_roc_analysis.py` - Adds columns for storing raw data (true_labels, predicted_probs)
2. `b3bc2a22b90c_add_unlabeled_predictions_column.py` - Adds the unlabeled_predictions column and creates tables if they don't exist

## Running Migrations

To run migrations:

```bash
# Navigate to the backend directory
cd backend

# Apply all migrations
alembic upgrade head

# Generate a new migration (after changing models)
alembic revision --autogenerate -m "Description of changes"
```

## Troubleshooting

If you encounter issues with migrations:

1. Check the current database version: `alembic current`
2. View migration history: `alembic history`
3. If needed, stamp the database with a specific version: `alembic stamp <revision_id>`

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