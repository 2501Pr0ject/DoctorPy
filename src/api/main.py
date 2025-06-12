# src/api/main.py
"""
Application FastAPI principale pour l'assistant pédagogique
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import asyncio
from datetime import datetime

from src.core.config import get_settings
from src.core.logger import setup_logging
from src.core.database import init_database
from src.agents.state_manager import initialize_state_manager, periodic_cleanup
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import chat, quests, users, admin
from src.core.exceptions import (
    ValidationError, 
    AuthenticationError, 
    NotFoundError,
    InternalServerError
)

# Configuration du logging
setup_logging()
logger = logging.getLogger(__name__)

# Configuration
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application"""
    logger.info("🚀 Démarrage de l'assistant pédagogique")
    
    try:
        # Initialiser la base de données
        await init_database()
        logger.info("✅ Base de données initialisée")
        
        # Initialiser le gestionnaire d'états
        state_manager = await initialize_state_manager(
            checkpoint_path=settings.STATE_MANAGER_CHECKPOINT_PATH
        )
        app.state.state_manager = state_manager
        logger.info("✅ Gestionnaire d'états initialisé")
        
        # Démarrer la tâche de nettoyage périodique
        cleanup_task = asyncio.create_task(periodic_cleanup())
        app.state.cleanup_task = cleanup_task
        logger.info("✅ Tâche de nettoyage démarrée")
        
        # Initialiser d'autres services si nécessaire
        # - Vector store
        # - Ollama client
        # - Cache Redis, etc.
        
        logger.info("🎉 Application prête !")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage: {e}")
        raise
    
    finally:
        # Nettoyage lors de l'arrêt
        logger.info("🛑 Arrêt de l'application")
        
        # Annuler les tâches en cours
        if hasattr(app.state, 'cleanup_task'):
            app.state.cleanup_task.cancel()
            try:
                await app.state.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Sauvegarder les sessions actives
        if hasattr(app.state, 'state_manager'):
            await app.state.state_manager.cleanup_expired_sessions()
        
        logger.info("✅ Arrêt propre terminé")


# Création de l'application FastAPI
app = FastAPI(
    title="Assistant Pédagogique IA",
    description="API pour l'assistant d'apprentissage avec IA et agents spécialisés",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)


# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Middleware de sécurité
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Middleware personnalisés
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)


# Gestionnaires d'exceptions personnalisés
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    logger.warning(f"Erreur de validation: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": exc.message,
            "details": exc.details if hasattr(exc, 'details') else None
        }
    )


@app.exception_handler(AuthenticationError)
async def auth_exception_handler(request, exc):
    logger.warning(f"Erreur d'authentification: {exc.message}")
    return JSONResponse(
        status_code=401,
        content={
            "error": "Authentication Error",
            "message": exc.message
        }
    )


@app.exception_handler(NotFoundError)
async def not_found_exception_handler(request, exc):
    logger.info(f"Ressource non trouvée: {exc.message}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": exc.message
        }
    )


@app.exception_handler(InternalServerError)
async def internal_server_exception_handler(request, exc):
    logger.error(f"Erreur interne: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Une erreur interne s'est produite"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Erreur non gérée: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Une erreur inattendue s'est produite"
        }
    )


# Routes de base
@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Assistant Pédagogique IA - API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_url": "/docs" if settings.ENVIRONMENT == "development" else None
    }


@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    try:
        # Vérifier les composants critiques
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Vérifier le gestionnaire d'états
        if hasattr(app.state, 'state_manager'):
            stats = await app.state.state_manager.get_system_stats()
            health_status["components"]["state_manager"] = {
                "status": "healthy",
                "active_sessions": stats["active_sessions"]
            }
        else:
            health_status["components"]["state_manager"] = {
                "status": "unhealthy",
                "error": "State manager not initialized"
            }
            health_status["status"] = "degraded"
        
        # Vérifier la base de données
        try:
            from src.core.database import get_db_session
            async with get_db_session() as session:
                await session.execute("SELECT 1")
            health_status["components"]["database"] = {"status": "healthy"}
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            },
            status_code=503
        )


@app.get("/metrics")
async def get_metrics():
    """Métriques de l'application"""
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": 0,  # À implémenter
            "total_requests": 0,  # À implémenter avec un middleware de comptage
            "active_sessions": 0,
            "system_stats": {}
        }
        
        if hasattr(app.state, 'state_manager'):
            system_stats = await app.state.state_manager.get_system_stats()
            metrics["active_sessions"] = system_stats["active_sessions"]
            metrics["system_stats"] = system_stats
        
        return metrics
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")


# Inclusion des routes
app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["Chat"]
)

app.include_router(
    quests.router,
    prefix="/api/v1/quests",
    tags=["Quêtes"]
)

app.include_router(
    users.router,
    prefix="/api/v1/users",
    tags=["Utilisateurs"]
)

app.include_router(
    admin.router,
    prefix="/api/v1/admin",
    tags=["Administration"]
)


# Fonction utilitaire pour accéder au state manager
def get_state_manager():
    """Dépendance pour obtenir le gestionnaire d'états"""
    if not hasattr(app.state, 'state_manager'):
        raise HTTPException(
            status_code=503,
            detail="Gestionnaire d'états non disponible"
        )
    return app.state.state_manager


# Point d'entrée pour le développement
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )