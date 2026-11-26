"""Index the latest run lookup for each user's thread."""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_agent_runs_thread_latest", "agent_runs", ["user_id", "thread_id", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_runs_thread_latest", table_name="agent_runs")
