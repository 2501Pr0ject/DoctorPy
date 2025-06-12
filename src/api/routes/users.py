# src/api/routes/users.py
"""
Routes API pour la gestion des utilisateurs et de leurs progrès
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, date
from enum import Enum

from src.core.exceptions import ValidationError, NotFoundError, AuthenticationError
from src.api.middleware.auth import get_current_user, get_optional_user
from src.models.schemas import UserBase, UserCreate, UserUpdate
from src.core.database import get_db_session
from src.models import User, UserProgress, UserQuest

logger = logging.getLogger(__name__)

router = APIRouter()


# Modèles Pydantic
class UserSkill(BaseModel):
    name: str
    level: float = Field(..., ge=0, le=1, description="Niveau de compétence (0-1)")
    xp: int = Field(default=0, ge=0)
    last_updated: datetime


class UserStats(BaseModel):
    total_xp: int
    level: str
    quests_completed: int
    quests_in_progress: int
    total_sessions: int
    total_time_spent: int  # en secondes
    average_score: float
    streak_days: int
    badges: List[str]


class UserProfile(BaseModel):
    id: int
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    learning_objectives: List[str] = Field(default_factory=list)
    preferred_difficulty: Optional[str] = None
    preferred_language: str = Field(default="fr")
    timezone: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    stats: UserStats
    skills: List[UserSkill]


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = Field(None, max_length=255)
    learning_objectives: Optional[List[str]] = None
    preferred_difficulty: Optional[str] = Field(None, regex="^(easy|medium|hard)$")
    preferred_language: Optional[str] = Field(None, max_length=5)
    timezone: Optional[str] = Field(None, max_length=50)


class UserProgressResponse(BaseModel):
    skill_name: str
    current_level: float
    xp_gained: int
    progress_history: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    recommendations: List[str]


class LearningPathItem(BaseModel):
    id: int
    type: str  # quest, tutorial, exercise
    title: str
    description: str
    difficulty: str
    estimated_duration: int
    status: str
    order: int


class LearningPath(BaseModel):
    id: int
    name: str
    description: str
    total_items: int
    completed_items: int
    estimated_total_duration: int
    items: List[LearningPathItem]
    created_at: datetime


class UserAchievement(BaseModel):
    id: int
    badge_name: str
    title: str
    description: str
    icon_url: Optional[str] = None
    earned_at: datetime
    category: str


class DashboardData(BaseModel):
    user_stats: UserStats
    recent_quests: List[Dict[str, Any]]
    current_learning_path: Optional[LearningPath] = None
    recent_achievements: List[UserAchievement]
    weekly_progress: Dict[str, Any]
    recommendations: List[str]


# Routes principales
@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: UserBase = Depends(get_current_user)
):
    """
    Récupère le profil de l'utilisateur connecté
    """
    try:
        async with get_db_session() as session:
            # Récupérer l'utilisateur complet
            user = await session.get(User, current_user.id)
            if not user:
                raise NotFoundError("Utilisateur non trouvé")
            
            # Récupérer les statistiques
            stats = await _get_user_stats(session, current_user.id)
            
            # Récupérer les compétences
            skills = await _get_user_skills(session, current_user.id)
            
            return UserProfile(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                bio=user.bio,
                avatar_url=user.avatar_url,
                learning_objectives=user.learning_objectives or [],
                preferred_difficulty=user.preferred_difficulty,
                preferred_language=user.preferred_language or "fr",
                timezone=user.timezone,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login,
                stats=stats,
                skills=skills
            )
            
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du profil: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du profil")


@router.put("/me", response_model=UserProfile)
async def update_current_user_profile(
    profile_update: UserProfileUpdate,
    current_user: UserBase = Depends(get_current_user)
):
    """
    Met à jour le profil de l'utilisateur connecté
    """
    try:
        async with get_db_session() as session:
            user = await session.get(User, current_user.id)
            if not user:
                raise NotFoundError("Utilisateur non trouvé")
            
            # Mettre à jour les champs modifiés
            update_data = profile_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(user, field, value)
            
            user.updated_at = datetime.utcnow()
            await session.commit()
            
            # Retourner le profil mis à jour
            return await get_current_user_profile(current_user)
            
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du profil: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")


@router.get("/me/dashboard", response_model=DashboardData)
async def get_user_dashboard(
    current_user: UserBase = Depends(get_current_user)
):
    """
    Récupère les données du tableau de bord utilisateur
    """
    try:
        async with get_db_session() as session:
            # Statistiques utilisateur
            stats = await _get_user_stats(session, current_user.id)
            
            # Quêtes récentes
            recent_quests = await _get_recent_quests(session, current_user.id, limit=5)
            
            # Parcours d'apprentissage actuel
            current_path = await _get_current_learning_path(session, current_user.id)
            
            # Succès récents
            recent_achievements = await _get_recent_achievements(session, current_user.id, limit=3)
            
            # Progrès hebdomadaire
            weekly_progress = await _get_weekly_progress(session, current_user.id)
            
            # Recommandations
            recommendations = await _generate_recommendations(session, current_user.id)
            
            return DashboardData(
                user_stats=stats,
                recent_quests=recent_quests,
                current_learning_path=current_path,
                recent_achievements=recent_achievements,
                weekly_progress=weekly_progress,
                recommendations=recommendations
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du tableau de bord: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du tableau de bord")


@router.get("/me/progress", response_model=List[UserProgressResponse])
async def get_user_progress(
    skill_name: Optional[str] = Query(None, description="Filtrer par compétence"),
    current_user: UserBase = Depends(get_current_user)
):
    """
    Récupère la progression détaillée de l'utilisateur
    """
    try:
        async with get_db_session() as session:
            progress_data = []
            
            # Récupérer les compétences
            query = """
                SELECT skill_name, skill_level, xp_gained, updated_at
                FROM user_progress 
                WHERE user_id = ?
            """
            params = [current_user.id]
            
            if skill_name:
                query += " AND skill_name = ?"
                params.append(skill_name)
            
            result = await session.execute(query, params)
            skills = result.fetchall()
            
            for skill in skills:
                # Récupérer l'historique de progression
                history_query = """
                    SELECT skill_level, xp_gained, updated_at
                    FROM user_progress_history 
                    WHERE user_id = ? AND skill_name = ?
                    ORDER BY updated_at DESC
                    LIMIT 10
                """
                history_result = await session.execute(history_query, [current_user.id, skill[0]])
                history = [
                    {
                        "level": row[0],
                        "xp": row[1],
                        "date": row[2]
                    }
                    for row in history_result.fetchall()
                ]
                
                # Récupérer les jalons
                milestones = await _get_skill_milestones(skill[0], skill[1])
                
                # Générer des recommandations
                recommendations = await _get_skill_recommendations(skill[0], skill[1])
                
                progress_data.append(UserProgressResponse(
                    skill_name=skill[0],
                    current_level=skill[1],
                    xp_gained=skill[2],
                    progress_history=history,
                    milestones=milestones,
                    recommendations=recommendations
                ))
            
            return progress_data
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des progrès: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des progrès")


@router.get("/me/achievements", response_model=List[UserAchievement])
async def get_user_achievements(
    current_user: UserBase = Depends(get_current_user)
):
    """
    Récupère les succès de l'utilisateur
    """
    try:
        async with get_db_session() as session:
            query = """
                SELECT ub.id, ub.badge_name, b.title, b.description, 
                       b.icon_url, ub.earned_at, b.category
                FROM user_badges ub
                JOIN badges b ON ub.badge_name = b.name
                WHERE ub.user_id = ?
                ORDER BY ub.earned_at DESC
            """
            result = await session.execute(query, [current_user.id])
            
            achievements = []
            for row in result.fetchall():
                achievements.append(UserAchievement(
                    id=row[0],
                    badge_name=row[1],
                    title=row[2],
                    description=row[3],
                    icon_url=row[4],
                    earned_at=row[5],
                    category=row[6]
                ))
            
            return achievements
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des succès: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des succès")


@router.get("/me/learning-paths", response_model=List[LearningPath])
async def get_user_learning_paths(
    current_user: UserBase = Depends(get_current_user)
):
    """
    Récupère les parcours d'apprentissage de l'utilisateur
    """
    try:
        async with get_db_session() as session:
            paths = await _get_user_learning_paths(session, current_user.id)
            return paths
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des parcours: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des parcours")


@router.post("/me/learning-paths/{path_id}/enroll")
async def enroll_in_learning_path(
    path_id: int,
    current_user: UserBase = Depends(get_current_user)
):
    """
    Inscrit l'utilisateur à un parcours d'apprentissage
    """
    try:
        async with get_db_session() as session:
            # Vérifier que le parcours existe
            path_query = "SELECT id, name FROM learning_paths WHERE id = ?"
            result = await session.execute(path_query, [path_id])
            path = result.fetchone()
            
            if not path:
                raise NotFoundError(f"Parcours {path_id} non trouvé")
            
            # Vérifier si déjà inscrit
            enrollment_query = """
                SELECT id FROM user_learning_paths 
                WHERE user_id = ? AND learning_path_id = ?
            """
            existing = await session.execute(enrollment_query, [current_user.id, path_id])
            
            if existing.fetchone():
                raise ValidationError("Déjà inscrit à ce parcours")
            
            # Inscrire l'utilisateur
            insert_query = """
                INSERT INTO user_learning_paths (user_id, learning_path_id, enrolled_at, status)
                VALUES (?, ?, ?, 'active')
            """
            await session.execute(insert_query, [current_user.id, path_id, datetime.utcnow()])
            await session.commit()
            
            return {
                "message": f"Inscription au parcours '{path[1]}' réussie",
                "path_id": path_id
            }
            
    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de l'inscription au parcours: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'inscription")


@router.get("/me/sessions")
async def get_user_sessions(
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_user: UserBase = Depends(get_current_user)
):
    """
    Récupère l'historique des sessions de l'utilisateur
    """
    try:
        async with get_db_session() as session:
            query = """
                SELECT session_id, duration, interactions, agents_used, created_at
                FROM user_sessions 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            result = await session.execute(query, [current_user.id, limit, offset])
            
            sessions = []
            for row in result.fetchall():
                sessions.append({
                    "session_id": row[0],
                    "duration": row[1],
                    "interactions": row[2],
                    "agents_used": row[3].split(",") if row[3] else [],
                    "created_at": row[4]
                })
            
            return {
                "sessions": sessions,
                "total": len(sessions),
                "limit": limit,
                "offset": offset
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des sessions")


@router.post("/me/settings/preferences")
async def update_user_preferences(
    preferences: Dict[str, Any],
    current_user: UserBase = Depends(get_current_user)
):
    """
    Met à jour les préférences utilisateur
    """
    try:
        async with get_db_session() as session:
            # Stocker les préférences dans une table dédiée ou dans le profil utilisateur
            query = """
                INSERT INTO user_preferences (user_id, preferences, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences = ?,
                    updated_at = ?
            """
            preferences_json = json.dumps(preferences)
            now = datetime.utcnow()
            
            await session.execute(query, [
                current_user.id, preferences_json, now,
                preferences_json, now
            ])
            await session.commit()
            
            return {
                "message": "Préférences mises à jour avec succès",
                "preferences": preferences
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des préférences: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")


@router.get("/me/analytics")
async def get_user_analytics(
    period: str = Query("month", regex="^(week|month|year)$"),
    current_user: UserBase = Depends(get_current_user)
):
    """
    Récupère les analytiques détaillées de l'utilisateur
    """
    try:
        async with get_db_session() as session:
            analytics = await _get_user_analytics(session, current_user.id, period)
            return analytics
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des analytiques: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des analytiques")


@router.delete("/me")
async def delete_user_account(
    confirmation: str = Query(..., description="Tapez 'DELETE' pour confirmer"),
    current_user: UserBase = Depends(get_current_user)
):
    """
    Supprime le compte utilisateur
    """
    if confirmation != "DELETE":
        raise HTTPException(status_code=400, detail="Confirmation invalide")
    
    try:
        async with get_db_session() as session:
            # Marquer comme supprimé au lieu de supprimer définitivement
            user = await session.get(User, current_user.id)
            if user:
                user.is_active = False
                user.deleted_at = datetime.utcnow()
                await session.commit()
            
            return {"message": "Compte supprimé avec succès"}
            
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du compte: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression")


# Fonctions utilitaires privées
async def _get_user_stats(session, user_id: int) -> UserStats:
    """Calcule les statistiques utilisateur"""
    try:
        # Total XP
        xp_query = "SELECT COALESCE(SUM(xp_gained), 0) FROM user_progress WHERE user_id = ?"
        xp_result = await session.execute(xp_query, [user_id])
        total_xp = xp_result.fetchone()[0]
        
        # Niveau basé sur l'XP
        level = "beginner"
        if total_xp > 1000:
            level = "intermediate"
        if total_xp > 5000:
            level = "advanced"
        if total_xp > 10000:
            level = "expert"
        
        # Quêtes
        quest_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress
            FROM user_quests WHERE user_id = ?
        """
        quest_result = await session.execute(quest_query, [user_id])
        quest_stats = quest_result.fetchone()
        
        # Sessions
        session_query = """
            SELECT COUNT(*), COALESCE(SUM(duration), 0)
            FROM user_sessions WHERE user_id = ?
        """
        session_result = await session.execute(session_query, [user_id])
        session_stats = session_result.fetchone()
        
        # Score moyen (placeholder)
        average_score = 0.75
        
        # Série de jours (placeholder)
        streak_days = 5
        
        # Badges
        badge_query = "SELECT badge_name FROM user_badges WHERE user_id = ?"
        badge_result = await session.execute(badge_query, [user_id])
        badges = [row[0] for row in badge_result.fetchall()]
        
        return UserStats(
            total_xp=total_xp,
            level=level,
            quests_completed=quest_stats[1] if quest_stats else 0,
            quests_in_progress=quest_stats[2] if quest_stats else 0,
            total_sessions=session_stats[0] if session_stats else 0,
            total_time_spent=session_stats[1] if session_stats else 0,
            average_score=average_score,
            streak_days=streak_days,
            badges=badges
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des stats: {e}")
        return UserStats(
            total_xp=0, level="beginner", quests_completed=0,
            quests_in_progress=0, total_sessions=0, total_time_spent=0,
            average_score=0.0, streak_days=0, badges=[]
        )


async def _get_user_skills(session, user_id: int) -> List[UserSkill]:
    """Récupère les compétences utilisateur"""
    try:
        query = """
            SELECT skill_name, skill_level, xp_gained, updated_at
            FROM user_progress WHERE user_id = ?
            ORDER BY skill_level DESC
        """
        result = await session.execute(query, [user_id])
        
        skills = []
        for row in result.fetchall():
            skills.append(UserSkill(
                name=row[0],
                level=row[1],
                xp=row[2],
                last_updated=row[3]
            ))
        
        return skills
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des compétences: {e}")
        return []


async def _get_recent_quests(session, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Récupère les quêtes récentes"""
    try:
        query = """
            SELECT uq.id, q.title, q.category, uq.status, uq.progress_percentage, uq.updated_at
            FROM user_quests uq
            JOIN quests q ON uq.quest_id = q.id
            WHERE uq.user_id = ?
            ORDER BY uq.updated_at DESC
            LIMIT ?
        """
        result = await session.execute(query, [user_id, limit])
        
        quests = []
        for row in result.fetchall():
            quests.append({
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "status": row[3],
                "progress_percentage": row[4],
                "updated_at": row[5]
            })
        
        return quests
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des quêtes récentes: {e}")
        return []


async def _get_current_learning_path(session, user_id: int) -> Optional[LearningPath]:
    """Récupère le parcours d'apprentissage actuel"""
    try:
        # Implémentation placeholder
        return None
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du parcours: {e}")
        return None


async def _get_recent_achievements(session, user_id: int, limit: int = 3) -> List[UserAchievement]:
    """Récupère les succès récents"""
    try:
        query = """
            SELECT ub.id, ub.badge_name, b.title, b.description, 
                   b.icon_url, ub.earned_at, b.category
            FROM user_badges ub
            JOIN badges b ON ub.badge_name = b.name
            WHERE ub.user_id = ?
            ORDER BY ub.earned_at DESC
            LIMIT ?
        """
        result = await session.execute(query, [user_id, limit])
        
        achievements = []
        for row in result.fetchall():
            achievements.append(UserAchievement(
                id=row[0],
                badge_name=row[1],
                title=row[2],
                description=row[3],
                icon_url=row[4],
                earned_at=row[5],
                category=row[6]
            ))
        
        return achievements
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des succès: {e}")
        return []


async def _get_weekly_progress(session, user_id: int) -> Dict[str, Any]:
    """Calcule les progrès hebdomadaires"""
    try:
        # Implémentation placeholder
        return {
            "xp_gained": 250,
            "quests_completed": 2,
            "time_spent": 3600,  # en secondes
            "daily_breakdown": [
                {"day": "Lundi", "xp": 50, "time": 900},
                {"day": "Mardi", "xp": 75, "time": 1200},
                # ... autres jours
            ]
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des progrès hebdomadaires: {e}")
        return {}


async def _generate_recommendations(session, user_id: int) -> List[str]:
    """Génère des recommandations personnalisées"""
    try:
        # Algorithme simple de recommandation
        recommendations = [
            "Pratiquer les boucles Python avec de nouveaux exercices",
            "Explorer les concepts avancés de programmation orientée objet",
            "Commencer un projet de data science"
        ]
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de recommandations: {e}")
        return []


async def _get_skill_milestones(skill_name: str, current_level: float) -> List[Dict[str, Any]]:
    """Récupère les jalons pour une compétence"""
    milestones = [
        {"level": 0.25, "title": "Débutant", "achieved": current_level >= 0.25},
        {"level": 0.5, "title": "Intermédiaire", "achieved": current_level >= 0.5},
        {"level": 0.75, "title": "Avancé", "achieved": current_level >= 0.75},
        {"level": 1.0, "title": "Expert", "achieved": current_level >= 1.0}
    ]
    
    return milestones


async def _get_skill_recommendations(skill_name: str, current_level: float) -> List[str]:
    """Génère des recommandations pour une compétence"""
    if current_level < 0.3:
        return [f"Pratiquer les bases de {skill_name}", f"Suivre un tutoriel sur {skill_name}"]
    elif current_level < 0.7:
        return [f"Approfondir {skill_name} avec des projets", f"Explorer les concepts avancés de {skill_name}"]
    else:
        return [f"Maîtriser {skill_name} avec des défis complexes", f"Enseigner {skill_name} à d'autres"]


async def _get_user_learning_paths(session, user_id: int) -> List[LearningPath]:
    """Récupère les parcours d'apprentissage de l'utilisateur"""
    # Implémentation placeholder
    return []


async def _get_user_analytics(session, user_id: int, period: str) -> Dict[str, Any]:
    """Calcule les analytiques utilisateur pour une période"""
    # Implémentation placeholder
    return {
        "period": period,
        "total_time": 7200,
        "sessions_count": 15,
        "average_session_duration": 480,
        "skills_improved": ["python", "algorithms"],
        "best_performance_day": "Tuesday",
        "progress_trend": "increasing"
    }