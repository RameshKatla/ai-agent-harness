"""Associate persisted messages with an agent run and their order."""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("run_id", sa.Uuid(), nullable=True))
    op.add_column("messages", sa.Column("position", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_messages_run_id", "messages", "agent_runs", ["run_id"], ["id"], ondelete="CASCADE",
    )
    op.create_unique_constraint("uq_messages_run_position", "messages", ["run_id", "position"])


def downgrade() -> None:
    op.drop_constraint("uq_messages_run_position", "messages", type_="unique")
    op.drop_constraint("fk_messages_run_id", "messages", type_="foreignkey")
    op.drop_column("messages", "position")
    op.drop_column("messages", "run_id")
