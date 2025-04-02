import sqlite3
import os
from sqlalchemy import create_engine, text

# Get the database path from the environment or use a default
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///roc_data.db')

# Create a connection to the database
if DATABASE_URL.startswith('sqlite'):
    # Extract the file path from the SQLAlchemy URL
    db_path = DATABASE_URL.replace('sqlite:///', '')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the column exists
    cursor.execute("PRAGMA table_info(roc_analyses)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Add the column if it doesn't exist
    if 'unlabeled_predictions' not in columns:
        print("Adding 'unlabeled_predictions' column to roc_analyses table...")
        cursor.execute("ALTER TABLE roc_analyses ADD COLUMN unlabeled_predictions JSON")
        conn.commit()
        print("Column added successfully!")
    else:
        print("Column 'unlabeled_predictions' already exists.")
    
    # Close the connection
    conn.close()
else:
    # For non-SQLite databases, use SQLAlchemy
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        # Different syntax for different database types could be added here
        print("Adding 'unlabeled_predictions' column to roc_analyses table...")
        connection.execute(text("ALTER TABLE roc_analyses ADD COLUMN unlabeled_predictions JSON"))
        connection.commit()
        print("Column added successfully!")

print("Migration completed.") 