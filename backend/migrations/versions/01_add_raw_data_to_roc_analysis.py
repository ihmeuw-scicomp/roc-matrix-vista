"""Add raw data to roc analysis

Revision ID: 01_add_raw_data_to_roc_analysis
Revises: 
Create Date: 2023-04-02 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '01_add_raw_data_to_roc_analysis'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add columns for storing raw data in the roc_analyses table."""
    # Check if the columns already exist before adding them
    from sqlalchemy.engine.reflection import Inspector
    from sqlalchemy import create_engine
    
    config = op.get_context().config
    url = config.get_main_option('sqlalchemy.url')
    engine = create_engine(url)
    inspector = Inspector.from_engine(engine)
    
    # Make sure the table exists
    if 'roc_analyses' in inspector.get_table_names():
        # Check if columns already exist
        columns = [col['name'] for col in inspector.get_columns('roc_analyses')]
        
        if 'true_labels' not in columns:
            op.add_column('roc_analyses', sa.Column('true_labels', sa.JSON(), nullable=True))
        
        if 'predicted_probs' not in columns:
            op.add_column('roc_analyses', sa.Column('predicted_probs', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Remove the raw data columns."""
    op.drop_column('roc_analyses', 'predicted_probs')
    op.drop_column('roc_analyses', 'true_labels') 