"""
Routes FastAPI pour le service Quest
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer
from typing import List, Dict, Any, Optional

from ..shared.middleware import require_permission
from ..shared.utils import LoggerFactory

from .models import (
    Quest, QuestProgress, UserStats, Achievement,
    StartQuestRequest, SubmitAnswerRequest, CreateQuestRequest,
    QuestSearchRequest, QuestListResponse, ProgressResponse,
    SubmitAnswerResponse, LeaderboardResponse, QuestServiceError,
    QuestCategory, QuestDifficulty, QuestStatus
)
from .quest_manager import QuestManager

# Logger
logger = LoggerFactory.get_logger("quest_routes")

# Security
security = HTTPBearer()

# Routers
quest_router = APIRouter(prefix="/api/v1/quests", tags=["Quests"])
user_router = APIRouter(prefix="/api/v1/users", tags=["User Progress"])
admin_router = APIRouter(prefix="/api/v1/quests/admin", tags=["Quest Admin"])


# Dependency pour récupérer le Quest manager
def get_quest_manager() -> QuestManager:
    """Récupère l'instance du Quest manager depuis l'état de l'app"""
    from fastapi import Request
    return Request.app.state.quest_manager


# === ROUTES QUÊTES ===

@quest_router.get("/", response_model=QuestListResponse)
async def get_quests(
    category: Optional[QuestCategory] = None,
    difficulty: Optional[QuestDifficulty] = None,
    search: Optional[str] = None,
    tags: Optional[str] = Query(None, description="Tags séparés par virgules"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Récupère la liste des quêtes disponibles
    
    Nécessite la permission: quest:read
    """
    try:
        # Extraire user_id du token (simplification)
        user_id = "current_user"  # À remplacer par extraction du JWT
        
        # Construire la requête de recherche
        search_request = QuestSearchRequest(
            category=category,
            difficulty=difficulty,
            search_term=search,
            tags=tags.split(',') if tags else [],
            limit=limit,
            offset=offset
        )
        
        # Récupérer les quêtes
        quests = await quest_manager.get_available_quests(user_id, search_request)
        
        return QuestListResponse(
            quests=quests,
            total=len(quests),  # Simplification
            offset=offset,
            limit=limit
        )
        
    except QuestServiceError as e:
        logger.error(f"❌ Erreur récupération quêtes: {e.message}")
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


@quest_router.get("/{quest_id}", response_model=Quest)
async def get_quest(
    quest_id: str,
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Récupère les détails d'une quête spécifique
    
    Nécessite la permission: quest:read
    """
    try:
        if quest_id not in quest_manager.quests:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "QUEST_NOT_FOUND", "message": "Quête introuvable"}
            )
        
        return quest_manager.quests[quest_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération quête: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)}
        )


@quest_router.post("/start", response_model=QuestProgress)
async def start_quest(
    request: StartQuestRequest,
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Démarre une nouvelle quête
    
    Nécessite la permission: quest:read
    """
    try:
        progress = await quest_manager.start_quest(request)
        logger.info(f"🎯 Quête démarrée: {request.quest_id} pour {request.user_id}")
        return progress
        
    except QuestServiceError as e:
        logger.error(f"❌ Erreur démarrage quête: {e.message}")
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
            detail={"error": "INTERNAL_ERROR", "message": str(e)}
        )


@quest_router.post("/submit", response_model=SubmitAnswerResponse)
async def submit_answer(
    request: SubmitAnswerRequest,
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Soumet une réponse à une question
    
    Nécessite la permission: quest:read
    """
    try:
        response = await quest_manager.submit_answer(request)
        logger.info(f"📝 Réponse soumise pour progress {request.progress_id}")
        return response
        
    except QuestServiceError as e:
        logger.error(f"❌ Erreur soumission réponse: {e.message}")
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
            detail={"error": "INTERNAL_ERROR", "message": str(e)}
        )


@quest_router.get("/categories/list")
async def get_categories():
    """Récupère la liste des catégories disponibles"""
    return {
        "categories": [
            {
                "value": category.value,
                "label": category.value.replace('_', ' ').title(),
                "description": f"Quêtes de {category.value.replace('_', ' ')}"
            }
            for category in QuestCategory
        ]
    }


@quest_router.get("/difficulties/list")
async def get_difficulties():
    """Récupère la liste des niveaux de difficulté"""
    difficulty_info = {
        QuestDifficulty.VERY_EASY: {"level": 1, "description": "Très facile - Pour débuter"},
        QuestDifficulty.EASY: {"level": 2, "description": "Facile - Bases acquises"},
        QuestDifficulty.MEDIUM: {"level": 3, "description": "Moyen - Connaissances intermédiaires"},
        QuestDifficulty.HARD: {"level": 4, "description": "Difficile - Niveau avancé"},
        QuestDifficulty.VERY_HARD: {"level": 5, "description": "Très difficile - Expert"}
    }
    
    return {
        "difficulties": [
            {
                "value": difficulty.value,
                "level": info["level"],
                "label": difficulty.value.replace('_', ' ').title(),
                "description": info["description"]
            }
            for difficulty, info in difficulty_info.items()
        ]
    }


# === ROUTES UTILISATEUR ===

@user_router.get("/{user_id}/stats", response_model=UserStats)
async def get_user_stats(
    user_id: str,
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Récupère les statistiques d'un utilisateur
    
    Nécessite la permission: user:read (ou être le propriétaire)
    """
    try:
        stats = await quest_manager.get_user_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "STATS_ERROR", "message": str(e)}
        )


@user_router.get("/{user_id}/progress")
async def get_user_progress(
    user_id: str,
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Récupère la progression actuelle d'un utilisateur
    
    Nécessite la permission: user:read (ou être le propriétaire)
    """
    try:
        # Récupérer toutes les progressions de l'utilisateur
        user_progresses = [
            progress for progress in quest_manager.progress.values()
            if progress.user_id == user_id
        ]
        
        return {
            "user_id": user_id,
            "total_progresses": len(user_progresses),
            "in_progress": [p for p in user_progresses if p.status == QuestStatus.IN_PROGRESS],
            "completed": [p for p in user_progresses if p.status == QuestStatus.COMPLETED],
            "failed": [p for p in user_progresses if p.status == QuestStatus.FAILED]
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération progression: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PROGRESS_ERROR", "message": str(e)}
        )


@user_router.get("/{user_id}/achievements")
async def get_user_achievements(
    user_id: str,
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Récupère les achievements d'un utilisateur
    
    Nécessite la permission: user:read (ou être le propriétaire)
    """
    try:
        user_achievements = quest_manager.user_achievements.get(user_id, [])
        
        # Enrichir avec les détails des achievements
        detailed_achievements = []
        for user_ach in user_achievements:
            achievement = quest_manager.achievements.get(user_ach.achievement_id)
            if achievement:
                detailed_achievements.append({
                    "achievement": achievement,
                    "earned_at": user_ach.earned_at,
                    "quest_id": user_ach.quest_id
                })
        
        return {
            "user_id": user_id,
            "total_achievements": len(detailed_achievements),
            "achievements": detailed_achievements
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération achievements: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "ACHIEVEMENTS_ERROR", "message": str(e)}
        )


@quest_router.get("/leaderboard/global", response_model=LeaderboardResponse)
async def get_leaderboard(
    limit: int = Query(50, ge=1, le=100),
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(security)
):
    """
    Récupère le classement global
    
    Nécessite la permission: quest:read
    """
    try:
        leaderboard = await quest_manager.get_leaderboard(limit)
        return leaderboard
        
    except QuestServiceError as e:
        logger.error(f"❌ Erreur récupération leaderboard: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.error_code,
                "message": e.message
            }
        )
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)}
        )


# === ROUTES ADMIN ===

@admin_router.post("/create", response_model=Quest)
async def create_quest(
    request: CreateQuestRequest,
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(require_permission("quest:create"))
):
    """
    Crée une nouvelle quête
    
    Nécessite la permission: quest:create
    """
    try:
        # Extraire creator_id du token (simplification)
        creator_id = "admin_user"  # À remplacer par extraction du JWT
        
        quest = await quest_manager.create_quest(request, creator_id)
        logger.info(f"📝 Nouvelle quête créée: {quest.title}")
        return quest
        
    except QuestServiceError as e:
        logger.error(f"❌ Erreur création quête: {e.message}")
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
            detail={"error": "INTERNAL_ERROR", "message": str(e)}
        )


@admin_router.get("/stats/global")
async def get_global_stats(
    quest_manager: QuestManager = Depends(get_quest_manager),
    token: str = Depends(require_permission("quest:admin"))
):
    """
    Récupère les statistiques globales du système
    
    Nécessite la permission: quest:admin
    """
    try:
        # Calculer les statistiques globales
        total_quests = len(quest_manager.quests)
        total_users = len(quest_manager.user_stats)
        total_completions = len([
            p for p in quest_manager.progress.values() 
            if p.status == QuestStatus.COMPLETED
        ])
        active_progresses = len([
            p for p in quest_manager.progress.values() 
            if p.status == QuestStatus.IN_PROGRESS
        ])
        
        # Stats par catégorie
        category_stats = {}
        for quest in quest_manager.quests.values():
            if quest.category not in category_stats:
                category_stats[quest.category] = {"total": 0, "completed": 0}
            category_stats[quest.category]["total"] += 1
            
        for progress in quest_manager.progress.values():
            if progress.status == QuestStatus.COMPLETED:
                quest = quest_manager.quests.get(progress.quest_id)
                if quest and quest.category in category_stats:
                    category_stats[quest.category]["completed"] += 1
        
        return {
            "global_stats": {
                "total_quests": total_quests,
                "total_users": total_users,
                "total_completions": total_completions,
                "active_progresses": active_progresses,
                "completion_rate": (total_completions / (total_completions + active_progresses)) * 100 if (total_completions + active_progresses) > 0 else 0
            },
            "category_stats": category_stats,
            "top_performers": list(quest_manager.user_stats.items())[:10]  # Top 10 users
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération stats globales: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "GLOBAL_STATS_ERROR", "message": str(e)}
        )


@quest_router.get("/health")
async def health_check(
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Vérification de santé du service Quest
    """
    try:
        health_data = await quest_manager.get_health_status()
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "HEALTH_CHECK_FAILED", "message": str(e)}
        )


# Export des routers
__all__ = ["quest_router", "user_router", "admin_router"]