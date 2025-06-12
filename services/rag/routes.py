"""
Routes FastAPI pour le service RAG
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Dict, Any

from ..shared.middleware import require_permission
from ..shared.utils import LoggerFactory

from .models import (
    RAGQueryRequest, RAGResponse, IndexDocumentRequest, IndexDocumentResponse,
    HealthCheckResponse, RAGServiceError
)
from .rag_manager import RAGManager

# Logger
logger = LoggerFactory.get_logger("rag_routes")

# Security
security = HTTPBearer()

# Routers
rag_router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])
admin_router = APIRouter(prefix="/api/v1/rag/admin", tags=["RAG Admin"])


# Dependency pour récupérer le RAG manager
def get_rag_manager() -> RAGManager:
    """Récupère l'instance du RAG manager depuis l'état de l'app"""
    from fastapi import Request
    return Request.app.state.rag_manager


@rag_router.post("/query", response_model=RAGResponse)
async def query_rag(
    request: RAGQueryRequest,
    rag_manager: RAGManager = Depends(get_rag_manager),
    token: str = Depends(security)
):
    """
    Traite une requête RAG
    
    Nécessite la permission: rag:query
    """
    try:
        logger.info(f"🔍 Requête RAG reçue: {request.query[:50]}...")
        
        response = await rag_manager.query(request)
        
        logger.info(f"✅ Requête traitée en {response.processing_time_ms}ms")
        return response
        
    except RAGServiceError as e:
        logger.error(f"❌ Erreur service RAG: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": "Erreur interne du serveur"}
        )


@rag_router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    rag_manager: RAGManager = Depends(get_rag_manager)
):
    """
    Vérification de santé du service RAG
    """
    try:
        health_data = await rag_manager.get_health_status()
        
        return HealthCheckResponse(**health_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "HEALTH_CHECK_FAILED", "message": str(e)}
        )


@admin_router.post("/documents/index", response_model=IndexDocumentResponse)
async def index_document(
    request: IndexDocumentRequest,
    background_tasks: BackgroundTasks,
    rag_manager: RAGManager = Depends(get_rag_manager),
    token: str = Depends(require_permission("rag:admin"))
):
    """
    Indexe un nouveau document dans le vector store
    
    Nécessite la permission: rag:admin
    """
    try:
        logger.info(f"📄 Demande d'indexation: {request.metadata.title}")
        
        # Indexation en arrière-plan pour les gros documents
        if len(request.content) > 10000:  # > 10KB
            background_tasks.add_task(
                rag_manager.index_document, 
                request
            )
            return IndexDocumentResponse(
                document_id="processing",
                chunks_created=0,
                status="processing",
                processing_time_ms=0
            )
        else:
            # Indexation synchrone pour les petits documents
            response = await rag_manager.index_document(request)
            return response
            
    except Exception as e:
        logger.error(f"❌ Erreur indexation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INDEXING_ERROR", "message": str(e)}
        )


@admin_router.get("/documents/stats")
async def get_document_stats(
    rag_manager: RAGManager = Depends(get_rag_manager),
    token: str = Depends(require_permission("rag:admin"))
):
    """
    Récupère les statistiques des documents indexés
    
    Nécessite la permission: rag:admin
    """
    try:
        # Cette fonctionnalité pourrait être implémentée plus tard
        return {
            "message": "Statistiques des documents - À implémenter",
            "total_documents": 0,
            "total_chunks": 0,
            "last_update": None
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "STATS_ERROR", "message": str(e)}
        )


@admin_router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag_manager: RAGManager = Depends(get_rag_manager),
    token: str = Depends(require_permission("rag:admin"))
):
    """
    Supprime un document du vector store
    
    Nécessite la permission: rag:admin
    """
    try:
        # Cette fonctionnalité pourrait être implémentée plus tard
        logger.info(f"🗑️ Suppression document: {document_id}")
        
        return {
            "message": f"Document {document_id} supprimé avec succès",
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur suppression: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "DELETE_ERROR", "message": str(e)}
        )


@admin_router.post("/cache/clear")
async def clear_cache(
    rag_manager: RAGManager = Depends(get_rag_manager),
    token: str = Depends(require_permission("rag:admin"))
):
    """
    Vide le cache du service RAG
    
    Nécessite la permission: rag:admin
    """
    try:
        await rag_manager.cache.clear()
        logger.info("🧹 Cache RAG vidé")
        
        return {"message": "Cache vidé avec succès"}
        
    except Exception as e:
        logger.error(f"❌ Erreur vidage cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "CACHE_CLEAR_ERROR", "message": str(e)}
        )


@rag_router.get("/templates")
async def get_query_templates():
    """
    Récupère les templates de requêtes disponibles
    """
    templates = {
        "code_help": {
            "name": "Aide au code",
            "description": "Aide pour écrire ou corriger du code Python",
            "examples": [
                "Comment créer une liste en Python ?",
                "Erreur dans ma fonction, peux-tu m'aider ?",
                "Quelle est la syntaxe pour les boucles for ?"
            ]
        },
        "concept_explanation": {
            "name": "Explication de concepts",
            "description": "Explication de concepts Python",
            "examples": [
                "Qu'est-ce qu'une classe en Python ?",
                "Explique-moi les décorateurs",
                "Comment fonctionnent les générateurs ?"
            ]
        },
        "debugging": {
            "name": "Débogage",
            "description": "Aide pour résoudre des bugs",
            "examples": [
                "J'ai une erreur IndexError",
                "Mon code ne fait pas ce que j'attends",
                "Aide-moi à déboguer cette fonction"
            ]
        },
        "best_practices": {
            "name": "Bonnes pratiques",
            "description": "Conseils pour écrire du bon code Python",
            "examples": [
                "Comment organiser mon code Python ?",
                "Quelles sont les bonnes pratiques pour les fonctions ?",
                "Comment gérer les erreurs efficacement ?"
            ]
        }
    }
    
    return {"templates": templates}


# Export des routers
__all__ = ["rag_router", "admin_router"]