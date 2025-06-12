"""
Application FastAPI pour le service d'authentification
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager

from ..shared.config import get_auth_config
from ..shared.middleware import CommonMiddleware
from ..shared.events import EventBusFactory, AuthEventHandler
from ..shared.cache import CacheFactory
from ..shared.utils import LoggerFactory, HealthChecker

from .routes import auth_router, users_router
from .auth import AuthManager
from .database import AuthDatabase


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    # Startup
    logger = LoggerFactory.get_logger("auth_service")
    logger.info("🚀 Démarrage du service d'authentification")
    
    # Initialiser les composants
    config = get_auth_config()
    
    # Base de données
    auth_db = AuthDatabase(config.get_database_url())
    await auth_db.connect()
    app.state.auth_db = auth_db
    
    # Cache
    cache = CacheFactory.create_session_cache(config.get_redis_url())
    await cache.connect()
    app.state.cache = cache
    
    # Event Bus
    event_bus = EventBusFactory.create("redis", redis_url=config.get_redis_url())
    await event_bus.start()
    
    # Event Handler
    event_handler = AuthEventHandler("auth_service")
    await event_bus.subscribe(
        [EventType.USER_AUTHENTICATED, EventType.USER_REGISTERED],
        event_handler
    )
    app.state.event_bus = event_bus
    
    # Auth Manager
    auth_manager = AuthManager(auth_db, cache, event_bus)
    app.state.auth_manager = auth_manager
    
    # Health Checker
    health_checker = HealthChecker("auth_service")
    health_checker.add_check("database", lambda: auth_db.health_check())
    health_checker.add_check("cache", lambda: cache.redis_client.ping())
    app.state.health_checker = health_checker
    
    logger.info("✅ Service d'authentification démarré")
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt du service d'authentification")
    await event_bus.stop()
    await cache.disconnect()
    await auth_db.disconnect()


def create_auth_app() -> FastAPI:
    """Créer l'application FastAPI pour le service d'authentification"""
    
    app = FastAPI(
        title="DoctorPy Auth Service",
        description="Service d'authentification et d'autorisation",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Configuration
    config = get_auth_config()
    
    # Middleware
    CommonMiddleware.setup_middleware(app, config)
    
    # Routes
    app.include_router(auth_router, prefix="/auth", tags=["authentication"])
    app.include_router(users_router, prefix="/users", tags=["users"])
    
    # Endpoints globaux
    @app.get("/")
    async def root():
        return {
            "service": "DoctorPy Auth Service",
            "version": "2.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Endpoint de vérification de santé détaillé"""
        health_checker = app.state.health_checker
        return await health_checker.run_checks()
    
    return app


# Point d'entrée pour développement
if __name__ == "__main__":
    import uvicorn
    
    app = create_auth_app()
    config = get_auth_config()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.service_port,
        log_level=config.monitoring.log_level.lower(),
        reload=config.is_development()
    )