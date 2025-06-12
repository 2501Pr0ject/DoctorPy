"""
Application FastAPI pour le service RAG
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from ..shared.config import get_service_config
from ..shared.middleware import CommonMiddleware
from ..shared.events import EventBusFactory, EventType
from ..shared.cache import CacheFactory
from ..shared.utils import LoggerFactory, HealthChecker

from .routes import rag_router, admin_router
from .rag_manager import RAGManager
from .models import RAGServiceConfig, RAGServiceError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    logger = LoggerFactory.get_logger("rag_service")
    
    try:
        # Startup
        logger.info("🚀 Démarrage du service RAG")
        
        # Configuration
        config = RAGServiceConfig(
            port=get_service_config().get("rag", {}).get("port", 8002),
            host=get_service_config().get("rag", {}).get("host", "0.0.0.0"),
            vector_store_path=get_service_config().get("rag", {}).get("vector_store_path", "./vector_stores/chroma_db")
        )
        app.state.config = config
        
        # Cache IA spécialisé
        cache = CacheFactory.create_ai_cache(
            redis_url=get_service_config().get("redis_url", "redis://localhost:6379")
        )
        await cache.connect()
        app.state.cache = cache
        logger.info("✅ Cache IA connecté")
        
        # Event Bus
        event_bus = EventBusFactory.create(
            "redis", 
            redis_url=get_service_config().get("redis_url", "redis://localhost:6379")
        )
        await event_bus.start()
        app.state.event_bus = event_bus
        logger.info("✅ Event Bus connecté")
        
        # RAG Manager
        try:
            rag_manager = RAGManager(config, cache, event_bus)
            app.state.rag_manager = rag_manager
            logger.info("✅ RAG Manager initialisé")
        except RAGServiceError as e:
            logger.error(f"❌ Erreur initialisation RAG Manager: {e.message}")
            raise
        
        # Health Checker
        health_checker = HealthChecker("rag_service")
        app.state.health_checker = health_checker
        
        # Enregistrer le service
        await event_bus.publish(
            EventType.SERVICE_STARTED,
            {
                "service_name": "rag_service",
                "port": config.port,
                "timestamp": "now",
                "version": "1.0.0"
            }
        )
        
        logger.info(f"🎯 Service RAG démarré sur le port {config.port}")
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur critique au démarrage: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Arrêt du service RAG")
        
        try:
            # Nettoyer les ressources
            if hasattr(app.state, 'event_bus'):
                await app.state.event_bus.publish(
                    EventType.SERVICE_STOPPED,
                    {
                        "service_name": "rag_service",
                        "timestamp": "now"
                    }
                )
                await app.state.event_bus.stop()
            
            if hasattr(app.state, 'cache'):
                await app.state.cache.disconnect()
                
        except Exception as e:
            logger.error(f"⚠️ Erreur arrêt propre: {e}")
        
        logger.info("✅ Service RAG arrêté proprement")


# Créer l'application FastAPI
def create_app() -> FastAPI:
    """Créer et configurer l'application FastAPI"""
    
    app = FastAPI(
        title="DoctorPy RAG Service",
        description="Service RAG (Retrieval-Augmented Generation) pour DoctorPy",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # À configurer selon l'environnement
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware communs
    common_middleware = CommonMiddleware()
    app.add_middleware(type(common_middleware), instance=common_middleware)
    
    # Routes
    app.include_router(rag_router)
    app.include_router(admin_router)
    
    # Route racine
    @app.get("/")
    async def root():
        return {
            "service": "DoctorPy RAG Service",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs"
        }
    
    # Health check global
    @app.get("/health")
    async def health_check(request: Request):
        try:
            rag_manager = request.app.state.rag_manager
            health_data = await rag_manager.get_health_status()
            return health_data
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Gestionnaire d'erreurs global
    @app.exception_handler(RAGServiceError)
    async def rag_service_error_handler(request: Request, exc: RAGServiceError):
        logger = LoggerFactory.get_logger("rag_service")
        logger.error(f"❌ Erreur service RAG: {exc.message}")
        
        return {
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    
    return app


# Point d'entrée principal
if __name__ == "__main__":
    import uvicorn
    
    # Configuration du logging
    logger = LoggerFactory.get_logger("rag_service")
    
    try:
        # Créer l'application
        app = create_app()
        
        # Configuration depuis l'environnement ou défaut
        config = RAGServiceConfig()
        
        logger.info(f"🌟 Lancement du service RAG sur {config.host}:{config.port}")
        
        # Lancer le serveur
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        raise