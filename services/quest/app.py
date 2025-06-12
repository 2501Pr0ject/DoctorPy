"""
Application FastAPI pour le service Quest
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

from .routes import quest_router, user_router, admin_router
from .quest_manager import QuestManager
from .models import QuestServiceConfig, QuestServiceError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    logger = LoggerFactory.get_logger("quest_service")
    
    try:
        # Startup
        logger.info("🚀 Démarrage du service Quest")
        
        # Configuration
        config = QuestServiceConfig(
            port=get_service_config().get("quest", {}).get("port", 8004),
            host=get_service_config().get("quest", {}).get("host", "0.0.0.0"),
            database_url=get_service_config().get("quest", {}).get("database_url", "sqlite:///./data/databases/quests.db")
        )
        app.state.config = config
        
        # Cache de session pour les progressions
        cache = CacheFactory.create_session_cache(
            redis_url=get_service_config().get("redis_url", "redis://localhost:6379")
        )
        await cache.connect()
        app.state.cache = cache
        logger.info("✅ Cache de session connecté")
        
        # Event Bus
        event_bus = EventBusFactory.create(
            "redis", 
            redis_url=get_service_config().get("redis_url", "redis://localhost:6379")
        )
        await event_bus.start()
        app.state.event_bus = event_bus
        logger.info("✅ Event Bus connecté")
        
        # Quest Manager
        try:
            quest_manager = QuestManager(config, cache, event_bus)
            app.state.quest_manager = quest_manager
            logger.info("✅ Quest Manager initialisé")
        except QuestServiceError as e:
            logger.error(f"❌ Erreur initialisation Quest Manager: {e.message}")
            raise
        
        # Health Checker
        health_checker = HealthChecker("quest_service")
        app.state.health_checker = health_checker
        
        # Enregistrer le service
        await event_bus.publish(
            EventType.SERVICE_STARTED,
            {
                "service_name": "quest_service",
                "port": config.port,
                "timestamp": "now",
                "version": "1.0.0",
                "features": [
                    "quest_management",
                    "gamification",
                    "progress_tracking",
                    "achievements",
                    "leaderboard"
                ]
            }
        )
        
        logger.info(f"🎮 Service Quest démarré sur le port {config.port}")
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur critique au démarrage: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Arrêt du service Quest")
        
        try:
            # Nettoyer les ressources
            if hasattr(app.state, 'event_bus'):
                await app.state.event_bus.publish(
                    EventType.SERVICE_STOPPED,
                    {
                        "service_name": "quest_service",
                        "timestamp": "now"
                    }
                )
                await app.state.event_bus.stop()
            
            if hasattr(app.state, 'cache'):
                await app.state.cache.disconnect()
                
        except Exception as e:
            logger.error(f"⚠️ Erreur arrêt propre: {e}")
        
        logger.info("✅ Service Quest arrêté proprement")


# Créer l'application FastAPI
def create_app() -> FastAPI:
    """Créer et configurer l'application FastAPI"""
    
    app = FastAPI(
        title="DoctorPy Quest Service",
        description="Service de gamification et gestion des quêtes pour DoctorPy",
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
    app.include_router(quest_router)
    app.include_router(user_router)
    app.include_router(admin_router)
    
    # Route racine
    @app.get("/")
    async def root():
        return {
            "service": "DoctorPy Quest Service",
            "version": "1.0.0",
            "status": "running",
            "features": [
                "Quest Management",
                "Gamification System", 
                "Progress Tracking",
                "Achievement System",
                "Leaderboard",
                "User Analytics"
            ],
            "docs": "/docs"
        }
    
    # Health check global
    @app.get("/health")
    async def health_check(request: Request):
        try:
            quest_manager = request.app.state.quest_manager
            health_data = await quest_manager.get_health_status()
            return health_data
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Route de statistiques publiques
    @app.get("/stats/public")
    async def public_stats(request: Request):
        """Statistiques publiques (sans authentification)"""
        try:
            quest_manager = request.app.state.quest_manager
            
            return {
                "total_quests": len(quest_manager.quests),
                "total_categories": len(set(q.category for q in quest_manager.quests.values())),
                "total_users": len(quest_manager.user_stats),
                "total_completions": len([
                    p for p in quest_manager.progress.values() 
                    if p.status.value == "completed"
                ]),
                "categories_available": [category.value for category in quest_manager.quests.values()],
                "service_uptime": "running"
            }
        except Exception as e:
            logger = LoggerFactory.get_logger("quest_service")
            logger.error(f"❌ Erreur stats publiques: {e}")
            return {"error": "Stats indisponibles"}
    
    # Gestionnaire d'erreurs global
    @app.exception_handler(QuestServiceError)
    async def quest_service_error_handler(request: Request, exc: QuestServiceError):
        logger = LoggerFactory.get_logger("quest_service")
        logger.error(f"❌ Erreur service Quest: {exc.message}")
        
        return {
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    
    # Middleware pour logging des requêtes importantes
    @app.middleware("http")
    async def log_quest_activities(request: Request, call_next):
        logger = LoggerFactory.get_logger("quest_activities")
        
        # Logger les activités importantes
        if request.url.path.startswith("/api/v1/quests/start"):
            logger.info(f"🎯 Nouvelle quête démarrée depuis {request.client.host}")
        elif request.url.path.startswith("/api/v1/quests/submit"):
            logger.info(f"📝 Réponse soumise depuis {request.client.host}")
        elif "leaderboard" in request.url.path:
            logger.info(f"🏆 Consultation du leaderboard depuis {request.client.host}")
        
        response = await call_next(request)
        return response
    
    return app


# Point d'entrée principal
if __name__ == "__main__":
    import uvicorn
    
    # Configuration du logging
    logger = LoggerFactory.get_logger("quest_service")
    
    try:
        # Créer l'application
        app = create_app()
        
        # Configuration depuis l'environnement ou défaut
        config = QuestServiceConfig()
        
        logger.info(f"🌟 Lancement du service Quest sur {config.host}:{config.port}")
        logger.info("🎮 Fonctionnalités: Quêtes, Gamification, Achievements, Leaderboard")
        
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