"""API router aggregator for SATYA AI."""

from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.verify import router as verify_router
from app.api.whatsapp import router as whatsapp_router

api_router = APIRouter()

# Include verification and webhook endpoints
api_router.include_router(verify_router)
api_router.include_router(health_router)
api_router.include_router(whatsapp_router)
