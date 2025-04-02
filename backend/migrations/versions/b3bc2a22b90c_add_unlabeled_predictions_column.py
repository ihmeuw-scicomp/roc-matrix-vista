"""Add unlabeled_predictions column

Revision ID: b3bc2a22b90c
Revises: 01_add_raw_data_to_roc_analysis
Create Date: 2025-04-02 13:34:29.919094

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import create_engine
import os


# revision identifiers, used by Alembic.
revision: str = 'b3bc2a22b90c'
down_revision: Union[str, None] = '01_add_raw_data_to_roc_analysis'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Check if the tables already exist
    config = op.get_context().config
    url = config.get_main_option('sqlalchemy.url')
    engine = create_engine(url)
    inspector = Inspector.from_engine(engine)
    
    # Check if the roc_analyses table exists
    if 'roc_analyses' not in inspector.get_table_names():
        # Create the roc_analyses table with all columns
        op.create_table('roc_analyses',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(), nullable=False),
            sa.Column('description', sa.String(), nullable=True),
            sa.Column('auc_score', sa.Float(), nullable=False),
            sa.Column('default_threshold', sa.Float(), nullable=False),
            sa.Column('roc_curve_data', sa.JSON(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('true_labels', sa.JSON(), nullable=True),
            sa.Column('predicted_probs', sa.JSON(), nullable=True),
            sa.Column('unlabeled_predictions', sa.JSON(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_roc_analyses_id'), 'roc_analyses', ['id'], unique=False)
    else:
        # Check if the unlabeled_predictions column exists
        columns = [col['name'] for col in inspector.get_columns('roc_analyses')]
        if 'unlabeled_predictions' not in columns:
            # Add only the unlabeled_predictions column to the existing table
            op.add_column('roc_analyses', sa.Column('unlabeled_predictions', sa.JSON(), nullable=True))
    
    # Check if the confusion_matrices table exists
    if 'confusion_matrices' not in inspector.get_table_names():
        # Create the confusion_matrices table
        op.create_table('confusion_matrices',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('roc_analysis_id', sa.Integer(), nullable=True),
            sa.Column('threshold', sa.Float(), nullable=False),
            sa.Column('true_positives', sa.Integer(), nullable=False),
            sa.Column('false_positives', sa.Integer(), nullable=False),
            sa.Column('true_negatives', sa.Integer(), nullable=False),
            sa.Column('false_negatives', sa.Integer(), nullable=False),
            sa.Column('precision', sa.Float(), nullable=False),
            sa.Column('recall', sa.Float(), nullable=False),
            sa.Column('f1_score', sa.Float(), nullable=False),
            sa.Column('accuracy', sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(['roc_analysis_id'], ['roc_analyses.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_confusion_matrices_id'), 'confusion_matrices', ['id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Check if the unlabeled_predictions column exists
    config = op.get_context().config
    url = config.get_main_option('sqlalchemy.url')
    engine = create_engine(url)
    inspector = Inspector.from_engine(engine)
    
    # Check if the tables exist
    if 'confusion_matrices' in inspector.get_table_names():
        op.drop_index(op.f('ix_confusion_matrices_id'), table_name='confusion_matrices')
        op.drop_table('confusion_matrices')
    
    if 'roc_analyses' in inspector.get_table_names():
        # Check if the unlabeled_predictions column exists
        columns = [col['name'] for col in inspector.get_columns('roc_analyses')]
        if 'unlabeled_predictions' in columns:
            # Drop only the unlabeled_predictions column
            op.drop_column('roc_analyses', 'unlabeled_predictions')
        else:
            # If we're dropping the whole table
            op.drop_index(op.f('ix_roc_analyses_id'), table_name='roc_analyses')
            op.drop_table('roc_analyses')
