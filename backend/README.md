# Backend Service

This directory contains the backend service for the ROC Matrix Vista application.

## Project Structure

```
backend/
├── src/backend/            # Main application code
│   ├── api/                # API endpoints and routers
│   ├── db/                 # Database connection and setup
│   ├── models/             # SQLAlchemy models
│   ├── repositories/       # Data access layer
│   ├── routes/             # Route definitions
│   ├── schemas/            # Pydantic schemas for validation
│   ├── services/           # Business logic
│   ├── utils/              # Utility functions
│   ├── config.py           # Application configuration
│   ├── main.py             # Application entry point
│   └── __init__.py
├── migrations/             # Alembic database migrations
│   └── versions/           # Migration scripts
├── alembic.ini             # Alembic configuration
├── pyproject.toml          # Poetry project definition
├── poetry.lock             # Poetry dependency lock file
├── requirements.txt        # pip requirements (alternative to Poetry)
└── setup.py                # Package setup for installation
```

## Getting Started

### Installation

1. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. Setup environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Database Migrations

See the [migrations README](migrations/README.md) for detailed information on running and managing database migrations.

### Running the Application

```bash
# Using Poetry
poetry run python -m src.backend.main

# Or directly
python -m src.backend.main
```

## Development

- The application uses FastAPI for the web framework
- SQLAlchemy for ORM
- Alembic for database migrations
- Pydantic for data validation

### Adding New Features

1. Create/update models in `src/backend/models/`
2. Create/update schemas in `src/backend/schemas/`
3. Implement business logic in `src/backend/services/`
4. Add API endpoints in `src/backend/api/`
5. Generate migrations: `alembic revision --autogenerate -m "Description"` 