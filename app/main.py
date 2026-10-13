"""FastAPI application entry point for SATYA AI."""
# Reload trigger: 2026-10-12T22:32:00

from contextlib import asynccontextmanager
import logging
import time
import uuid
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.health import router as health_router
from app.api.router import api_router
from app.config import get_settings
from app.db.session import init_db
from app.logging_config import setup_logging

settings = get_settings()

# Initialize structured logging
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown routines."""
    logger.info(
        "Starting SATYA AI service",
        extra={"extra_data": {"environment": settings.APP_ENV, "version": __version__}},
    )
    # Initialize database tables on startup
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as exc:
        logger.error(f"Failed to initialize database: {exc}", exc_info=True)
        raise exc

    yield

    logger.info("Shutting down SATYA AI service")


app = FastAPI(
    title=settings.APP_NAME,
    version=__version__,
    description=(
        "Evidence-first WhatsApp fact-checking backend for SATYA AI. "
        "Provides cautious verdicts, structured reports, claims extraction, "
        "and source evidence tracking."
    ),
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID and Access Logging Middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Assigns unique request ID, records duration, and logs requests safely."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.perf_counter()
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )

    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"

        logger.info(
            f"Request completed: {request.method} {request.url.path} with status {response.status_code}",
            extra={"request_id": request_id, "extra_data": {"duration_seconds": process_time}},
        )
        return response
    except Exception as exc:
        process_time = time.perf_counter() - start_time
        logger.error(
            f"Unhandled exception during {request.method} {request.url.path}: {exc}",
            exc_info=True,
            extra={"request_id": request_id},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected server error occurred. Please try again later.",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )


# Safe Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic validation errors cleanly without exposing internals."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    raw_errors = exc.errors()
    safe_errors = []
    for err in raw_errors:
        err_dict = dict(err)
        if "ctx" in err_dict and isinstance(err_dict["ctx"], dict):
            err_dict["ctx"] = {k: str(v) for k, v in err_dict["ctx"].items()}
        safe_errors.append(err_dict)

    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {safe_errors}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT if hasattr(status, "HTTP_422_UNPROCESSABLE_CONTENT") else 422,
        content={
            "error": "Validation Error",
            "message": "The request payload failed schema validation.",
            "detail": safe_errors,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent JSON formatting."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "detail": exc.detail,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


from app.api.whatsapp import router as whatsapp_router

# Root health endpoint (GET /health)
app.include_router(health_router)

# Direct root webhook routes (e.g. GET/POST /webhook/whatsapp)
app.include_router(whatsapp_router)

# Mount API router under prefix (e.g. /api/verify, /api/whatsapp/webhook)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Root endpoint welcoming clients."""
    return {
        "service": settings.APP_NAME,
        "version": __version__,
        "status": "online",
        "docs_url": "/docs" if settings.DEBUG else "disabled",
    }
