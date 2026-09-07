"""
ElevateBox AI Voice Agent — FastAPI application entry point.

Startup sequence:
  1. Load and validate all settings
  2. Configure structured logging
  3. Connect to Redis
  4. Create/update the Vapi assistant
  5. Register all routes
  6. Start serving

The application is fully async throughout.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1 import calls, webhooks
from app.core.config import get_settings
from app.core.exceptions import VoiceAgentError, WebhookAuthError
from app.core.logging import configure_logging, get_logger
from app.core.security import add_timing_header, limiter
from app.services.state import close_redis, get_state_manager
from app.services.vapi_agent import get_vapi_client

logger = get_logger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Startup: initialise all resources.
    Shutdown: gracefully close all connections.
    """
    settings = get_settings()
    configure_logging(log_level=settings.log_level, is_production=settings.is_production)

    logger.info(
        "starting_elevateox_voice_agent",
        env=settings.app_env,
        base_url=settings.app_base_url,
    )

    # ── Startup ──────────────────────────────────────────────────────────────

    # 1. Redis
    try:
        state_mgr = await get_state_manager()
        healthy = await state_mgr.health_check()
        if healthy:
            logger.info("redis_connected")
        else:
            logger.error("redis_health_check_failed")
    except Exception as exc:
        logger.error("redis_connection_failed", error=str(exc))
        # Don't crash on Redis failure — degrade gracefully

    # 2. Vapi assistant setup
    try:
        vapi = get_vapi_client()
        assistant_id = await vapi.create_or_update_assistant()
        # Store assistant_id in app state for use by call endpoints
        app.state.vapi_assistant_id = assistant_id
        logger.info("vapi_assistant_ready", assistant_id=assistant_id)
    except Exception as exc:
        logger.error("vapi_setup_failed", error=str(exc))
        # Don't crash — the call endpoint will retry

    logger.info("application_startup_complete")

    yield  # Application is now running

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("shutting_down")
    await close_redis()
    logger.info("shutdown_complete")


# ─── App factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ElevateBox AI Voice Agent",
        description=(
            "Production-grade AI outbound voice sales agent. "
            "Calls leads, qualifies them in Telugu/Hindi/English, "
            "and sends WhatsApp follow-ups mid-call."
        ),
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ── CORS — restrict in production ─────────────────────────────────────────
    allowed_origins = ["*"] if not settings.is_production else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Timing middleware ─────────────────────────────────────────────────────
    app.middleware("http")(add_timing_header)

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(WebhookAuthError)
    async def webhook_auth_error_handler(
        request: Request, exc: WebhookAuthError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": exc.message},
        )

    @app.exception_handler(VoiceAgentError)
    async def voice_agent_error_handler(
        request: Request, exc: VoiceAgentError
    ) -> JSONResponse:
        logger.error("voice_agent_error", message=exc.message, detail=exc.detail)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": exc.message},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", path=str(request.url))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(calls.router, prefix="/api/v1")
    app.include_router(webhooks.router, prefix="/api/v1/webhooks")

    # ── Health endpoints ──────────────────────────────────────────────────────
    @app.get("/health", tags=["health"], summary="Health check")
    async def health() -> JSONResponse:
        state_mgr = await get_state_manager()
        redis_ok = await state_mgr.health_check()
        return JSONResponse({
            "status": "healthy" if redis_ok else "degraded",
            "redis": "ok" if redis_ok else "unavailable",
            "version": "1.0.0",
        })

    @app.get("/", tags=["health"], summary="Root")
    async def root() -> JSONResponse:
        return JSONResponse({"service": "ElevateBox Voice Agent", "status": "running"})

    return app


# ── Application instance ──────────────────────────────────────────────────────
app = create_app()
