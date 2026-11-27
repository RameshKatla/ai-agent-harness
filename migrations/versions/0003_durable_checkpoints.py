"""Create the LangGraph PostgreSQL saver schema through upstream revision 9."""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TABLE checkpoint_migrations (v INTEGER PRIMARY KEY)")
    op.execute("""
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT, type TEXT,
            checkpoint JSONB NOT NULL, metadata JSONB NOT NULL DEFAULT '{}',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )
    """)
    op.execute("""
        CREATE TABLE checkpoint_blobs (
            thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
            channel TEXT NOT NULL, version TEXT NOT NULL, type TEXT NOT NULL, blob BYTEA,
            PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
        )
    """)
    op.execute("""
        CREATE TABLE checkpoint_writes (
            thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL, idx INTEGER NOT NULL,
            channel TEXT NOT NULL, type TEXT, blob BYTEA NOT NULL, task_path TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        )
    """)
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        op.create_index(f"{table}_thread_id_idx", table, ["thread_id"])
    op.execute("INSERT INTO checkpoint_migrations (v) SELECT generate_series(0, 9)")


def downgrade() -> None:
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"):
        op.drop_table(table)
