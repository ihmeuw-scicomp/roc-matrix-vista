"""add raw data to roc analysis

Revision ID: [alembic will generate this]
Revises: [your previous revision]
Create Date: [alembic will generate this]

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = '[alembic will generate this]'
down_revision = '[your previous revision]'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('roc_analysis', sa.Column('true_labels', JSON, nullable=True))
    op.add_column('roc_analysis', sa.Column('predicted_probs', JSON, nullable=True))


def downgrade():
    op.drop_column('roc_analysis', 'predicted_probs')
    op.drop_column('roc_analysis', 'true_labels')
