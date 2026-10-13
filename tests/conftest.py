import os
import sqlite3
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure deterministic mock mode for automated test suite
os.environ["SEARCH_PROVIDER"] = "mock"
os.environ["WHATSAPP_VERIFY_TOKEN"] = "satya_test_verify_token"
os.environ["WHATSAPP_APP_SECRET"] = "satya_test_secret_12345"

from app.config import Settings, get_settings
get_settings.cache_clear()

from app.db.base import Base
from app.db.session import get_db
from app.main import app

# Test SQLite in-memory engine with StaticPool to keep schema in-memory across connections
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(test_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite connections."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_test_settings() -> Settings:
    """Provide testing environment settings."""
    return Settings(
        APP_NAME="SATYA AI Test",
        APP_ENV="test",
        DEBUG=True,
        DATABASE_URL=TEST_DATABASE_URL,
        SEARCH_PROVIDER="mock",
        LLM_PROVIDER="none",
        WHATSAPP_VERIFY_TOKEN="satya_test_verify_token",
        WHATSAPP_APP_SECRET="satya_test_secret_12345",
        WHATSAPP_PHONE_NUMBER_ID="10987654321",
        WHATSAPP_ACCESS_TOKEN="EAA_TEST_TOKEN_123",
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables in the test database for the session."""
    import app.db.models  # noqa: F401

    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a clean transactional database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    if transaction.is_active:
        transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient with overridden database and settings dependencies."""

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = get_test_settings

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
