"""Database session and engine management."""

import sqlite3
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.base import Base

settings = get_settings()

# Engine creation with SQLite thread safety setting
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
)


# Enforce foreign key constraints in SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite connections."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides an active database session for a request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables and apply safe additive column migrations."""
    # Ensure all models are imported so Base metadata is populated
    import app.db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Apply safe additive schema migration for SQLite if tables already existed
    if settings.DATABASE_URL.startswith("sqlite"):
        with engine.connect() as conn:
            from sqlalchemy import inspect, text

            inspector = inspect(conn)
            tables = inspector.get_table_names()

            if "claim_evidence" in tables:
                ce_cols = [c["name"] for c in inspector.get_columns("claim_evidence")]
                if "explanation" not in ce_cols:
                    conn.execute(text("ALTER TABLE claim_evidence ADD COLUMN explanation TEXT"))
                    conn.commit()

            if "sources" in tables:
                src_cols = [c["name"] for c in inspector.get_columns("sources")]
                if "published_date" not in src_cols:
                    conn.execute(text("ALTER TABLE sources ADD COLUMN published_date VARCHAR(100)"))
                if "source_type" not in src_cols:
                    conn.execute(text("ALTER TABLE sources ADD COLUMN source_type VARCHAR(50)"))
                if "retrieval_method" not in src_cols:
                    conn.execute(text("ALTER TABLE sources ADD COLUMN retrieval_method VARCHAR(50)"))
                conn.commit()

            if "reports" in tables:
                rep_cols = [c["name"] for c in inspector.get_columns("reports")]
                if "correlation_id" not in rep_cols:
                    conn.execute(text("ALTER TABLE reports ADD COLUMN correlation_id VARCHAR(36)"))
                if "message_id" not in rep_cols:
                    conn.execute(text("ALTER TABLE reports ADD COLUMN message_id VARCHAR(255)"))
                conn.commit()

            if "whatsapp_processed_messages" in tables:
                wpm_cols = [c["name"] for c in inspector.get_columns("whatsapp_processed_messages")]
                if "correlation_id" not in wpm_cols:
                    conn.execute(text("ALTER TABLE whatsapp_processed_messages ADD COLUMN correlation_id VARCHAR(36)"))
                conn.commit()

