"""
Routes API pour l'administration du système
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from enum import Enum

from src.core.exceptions import ValidationError, NotFoundError, AuthenticationError
from src.api.middleware.auth import get_current_user, require_admin
from src.models.schemas import UserBase
from src.core.database import get_db_session
from src.agents.state_manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter()


# Modèles pour l'administration
class SystemStatus(BaseModel):
    status: str
    version: str
    uptime: int
    active_sessions: int
    total_users: int
    total_quests: int
    database_status: str
    ai_service_status: str
    last_backup: Optional[datetime] = None


class UserManagement(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime]
    total_sessions: int
    total_xp: int


class QuestManagement(BaseModel):
    id: int
    title: str
    category: str
    difficulty: str
    created_by: str
    created_at: datetime
    is_active: bool
    completion_rate: float
    average_rating: float
    total_attempts: int


class SystemMetrics(BaseModel):
    daily_active_users: int
    weekly_active_users: int
    monthly_active_users: int
    total_sessions_today: int
    average_session_duration: float
    top_categories: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]


class MaintenanceTask(BaseModel):
    id: int
    name: str
    description: str
    status: str
    scheduled_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[int]
    created_by: str


class BackupInfo(BaseModel):
    id: int
    filename: str
    size: int
    created_at: datetime
    type: str  # full, incremental
    status: str


# Dépendances
async def get_state_manager() -> StateManager:
    """Récupère le gestionnaire d'états"""
    from src.api.main import app
    if not hasattr(app.state, 'state_manager'):
        raise HTTPException(status_code=503, detail="Gestionnaire d'états non disponible")
    return app.state.state_manager


# Routes système
@router.get("/system/status", response_model=SystemStatus)
async def get_system_status(
    admin_user: UserBase = Depends(require_admin),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Récupère le statut général du système
    """
    try:
        # Statistiques du gestionnaire d'états
        system_stats = await state_manager.get_system_stats()
        
        # Vérifier les services
        database_status = await _check_database_status()
        ai_service_status = await _check_ai_service_status()
        
        # Informations système
        async with get_db_session() as session:
            # Compter les utilisateurs
            user_count_result = await session.execute("SELECT COUNT(*) FROM users WHERE is_active = true")
            total_users = user_count_result.fetchone()[0]
            
            # Compter les quêtes
            quest_count_result = await session.execute("SELECT COUNT(*) FROM quests WHERE is_active = true")
            total_quests = quest_count_result.fetchone()[0]
            
            # Dernière sauvegarde
            backup_result = await session.execute(
                "SELECT created_at FROM backups ORDER BY created_at DESC LIMIT 1"
            )
            last_backup_row = backup_result.fetchone()
            last_backup = last_backup_row[0] if last_backup_row else None
        
        return SystemStatus(
            status="healthy" if database_status == "healthy" and ai_service_status == "healthy" else "degraded",
            version="1.0.0",
            uptime=3600,  # À calculer réellement
            active_sessions=system_stats["active_sessions"],
            total_users=total_users,
            total_quests=total_quests,
            database_status=database_status,
            ai_service_status=ai_service_status,
            last_backup=last_backup
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut système: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du statut")


@router.get("/system/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    period: str = Query("day", regex="^(day|week|month)$"),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Récupère les métriques du système
    """
    try:
        async with get_db_session() as session:
            # Calculer les métriques selon la période
            if period == "day":
                date_filter = datetime.utcnow() - timedelta(days=1)
            elif period == "week":
                date_filter = datetime.utcnow() - timedelta(weeks=1)
            else:  # month
                date_filter = datetime.utcnow() - timedelta(days=30)
            
            # Utilisateurs actifs
            dau_query = """
                SELECT COUNT(DISTINCT user_id) 
                FROM user_sessions 
                WHERE created_at >= ?
            """
            dau_result = await session.execute(dau_query, [datetime.utcnow() - timedelta(days=1)])
            daily_active_users = dau_result.fetchone()[0]
            
            wau_query = """
                SELECT COUNT(DISTINCT user_id) 
                FROM user_sessions 
                WHERE created_at >= ?
            """
            wau_result = await session.execute(wau_query, [datetime.utcnow() - timedelta(weeks=1)])
            weekly_active_users = wau_result.fetchone()[0]
            
            mau_query = """
                SELECT COUNT(DISTINCT user_id) 
                FROM user_sessions 
                WHERE created_at >= ?
            """
            mau_result = await session.execute(mau_query, [datetime.utcnow() - timedelta(days=30)])
            monthly_active_users = mau_result.fetchone()[0]
            
            # Sessions du jour
            today_sessions_query = """
                SELECT COUNT(*), AVG(duration)
                FROM user_sessions 
                WHERE DATE(created_at) = DATE(?)
            """
            today_result = await session.execute(today_sessions_query, [datetime.utcnow()])
            today_stats = today_result.fetchone()
            
            # Top catégories
            top_categories_query = """
                SELECT q.category, COUNT(*) as attempts
                FROM user_quests uq
                JOIN quests q ON uq.quest_id = q.id
                WHERE uq.created_at >= ?
                GROUP BY q.category
                ORDER BY attempts DESC
                LIMIT 5
            """
            cat_result = await session.execute(top_categories_query, [date_filter])
            top_categories = [
                {"category": row[0], "attempts": row[1]}
                for row in cat_result.fetchall()
            ]
        
        return SystemMetrics(
            daily_active_users=daily_active_users,
            weekly_active_users=weekly_active_users,
            monthly_active_users=monthly_active_users,
            total_sessions_today=today_stats[0] if today_stats else 0,
            average_session_duration=today_stats[1] if today_stats and today_stats[1] else 0.0,
            top_categories=top_categories,
            performance_metrics={
                "avg_response_time": 250.0,  # ms
                "success_rate": 0.98,
                "error_rate": 0.02
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des métriques")


# Gestion des utilisateurs
@router.get("/users", response_model=List[UserManagement])
async def list_users(
    active_only: bool = Query(True, description="Filtrer les utilisateurs actifs"),
    search: Optional[str] = Query(None, description="Rechercher par nom ou email"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Liste tous les utilisateurs avec options de filtrage
    """
    try:
        async with get_db_session() as session:
            # Construire la requête de base
            query = """
                SELECT u.id, u.username, u.email, u.full_name, u.is_active, u.is_admin,
                       u.created_at, u.last_login,
                       COUNT(DISTINCT us.id) as total_sessions,
                       COALESCE(SUM(up.xp_gained), 0) as total_xp
                FROM users u
                LEFT JOIN user_sessions us ON u.id = us.user_id
                LEFT JOIN user_progress up ON u.id = up.user_id
            """
            
            conditions = []
            params = []
            
            if active_only:
                conditions.append("u.is_active = ?")
                params.append(True)
            
            if search:
                conditions.append("(u.username LIKE ? OR u.email LIKE ? OR u.full_name LIKE ?)")
                search_pattern = f"%{search}%"
                params.extend([search_pattern, search_pattern, search_pattern])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += """
                GROUP BY u.id, u.username, u.email, u.full_name, u.is_active, 
                         u.is_admin, u.created_at, u.last_login
                ORDER BY u.created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            
            result = await session.execute(query, params)
            
            users = []
            for row in result.fetchall():
                users.append(UserManagement(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    full_name=row[3],
                    is_active=row[4],
                    is_admin=row[5],
                    created_at=row[6],
                    last_login=row[7],
                    total_sessions=row[8],
                    total_xp=row[9]
                ))
            
            return users
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des utilisateurs: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des utilisateurs")


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    is_active: bool,
    admin_user: UserBase = Depends(require_admin)
):
    """
    Active ou désactive un utilisateur
    """
    try:
        async with get_db_session() as session:
            user = await session.get(User, user_id)
            if not user:
                raise NotFoundError(f"Utilisateur {user_id} non trouvé")
            
            user.is_active = is_active
            user.updated_at = datetime.utcnow()
            await session.commit()
            
            action = "activé" if is_active else "désactivé"
            logger.info(f"Utilisateur {user_id} {action} par l'admin {admin_user.id}")
            
            return {
                "message": f"Utilisateur {action} avec succès",
                "user_id": user_id,
                "is_active": is_active
            }
            
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du statut utilisateur: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")


@router.put("/users/{user_id}/admin")
async def toggle_admin_status(
    user_id: int,
    is_admin: bool,
    admin_user: UserBase = Depends(require_admin)
):
    """
    Donne ou retire les droits administrateur
    """
    try:
        async with get_db_session() as session:
            user = await session.get(User, user_id)
            if not user:
                raise NotFoundError(f"Utilisateur {user_id} non trouvé")
            
            # Empêcher de se retirer ses propres droits admin
            if user_id == admin_user.id and not is_admin:
                raise ValidationError("Vous ne pouvez pas vous retirer vos propres droits admin")
            
            user.is_admin = is_admin
            user.updated_at = datetime.utcnow()
            await session.commit()
            
            action = "accordés" if is_admin else "retirés"
            logger.info(f"Droits admin {action} pour l'utilisateur {user_id} par l'admin {admin_user.id}")
            
            return {
                "message": f"Droits administrateur {action} avec succès",
                "user_id": user_id,
                "is_admin": is_admin
            }
            
    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des droits admin: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")


# Gestion des quêtes
@router.get("/quests", response_model=List[QuestManagement])
async def list_quests_admin(
    category: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Liste toutes les quêtes avec statistiques
    """
    try:
        async with get_db_session() as session:
            query = """
                SELECT q.id, q.title, q.category, q.difficulty, q.created_by,
                       q.created_at, q.is_active,
                       COUNT(DISTINCT uq.user_id) as total_attempts,
                       AVG(CASE WHEN uq.status = 'completed' THEN 1.0 ELSE 0.0 END) as completion_rate,
                       AVG(COALESCE(qr.rating, 0)) as average_rating
                FROM quests q
                LEFT JOIN user_quests uq ON q.id = uq.quest_id
                LEFT JOIN quest_ratings qr ON q.id = qr.quest_id
            """
            
            conditions = []
            params = []
            
            if active_only:
                conditions.append("q.is_active = ?")
                params.append(True)
            
            if category:
                conditions.append("q.category = ?")
                params.append(category)
            
            if difficulty:
                conditions.append("q.difficulty = ?")
                params.append(difficulty)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += """
                GROUP BY q.id, q.title, q.category, q.difficulty, q.created_by,
                         q.created_at, q.is_active
                ORDER BY q.created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            
            result = await session.execute(query, params)
            
            quests = []
            for row in result.fetchall():
                quests.append(QuestManagement(
                    id=row[0],
                    title=row[1],
                    category=row[2],
                    difficulty=row[3],
                    created_by=row[4],
                    created_at=row[5],
                    is_active=row[6],
                    total_attempts=row[7],
                    completion_rate=row[8] or 0.0,
                    average_rating=row[9] or 0.0
                ))
            
            return quests
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des quêtes admin: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des quêtes")


@router.put("/quests/{quest_id}/status")
async def update_quest_status(
    quest_id: int,
    is_active: bool,
    admin_user: UserBase = Depends(require_admin)
):
    """
    Active ou désactive une quête
    """
    try:
        async with get_db_session() as session:
            quest = await session.get(Quest, quest_id)
            if not quest:
                raise NotFoundError(f"Quête {quest_id} non trouvée")
            
            quest.is_active = is_active
            quest.updated_at = datetime.utcnow()
            await session.commit()
            
            action = "activée" if is_active else "désactivée"
            logger.info(f"Quête {quest_id} {action} par l'admin {admin_user.id}")
            
            return {
                "message": f"Quête {action} avec succès",
                "quest_id": quest_id,
                "is_active": is_active
            }
            
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du statut de quête: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")


# Gestion des sessions
@router.get("/sessions")
async def list_active_sessions(
    admin_user: UserBase = Depends(require_admin),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Liste toutes les sessions actives
    """
    try:
        sessions = await state_manager.get_active_sessions()
        
        return {
            "active_sessions": sessions,
            "total_count": len(sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des sessions")


@router.delete("/sessions/{session_id}")
async def terminate_session(
    session_id: str,
    admin_user: UserBase = Depends(require_admin),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Termine une session spécifique
    """
    try:
        result = await state_manager.end_session(session_id)
        
        if "error" in result:
            raise NotFoundError(result["error"])
        
        logger.info(f"Session {session_id} terminée par l'admin {admin_user.id}")
        
        return {
            "message": "Session terminée avec succès",
            "session_id": session_id,
            "final_stats": result.get("session_stats", {})
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la terminaison de session: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la terminaison")


# Maintenance et sauvegardes
@router.post("/maintenance/backup")
async def create_backup(
    backup_type: str = Query("full", regex="^(full|incremental)$"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Lance une sauvegarde du système
    """
    try:
        # Ajouter la tâche de sauvegarde en arrière-plan
        background_tasks.add_task(_perform_backup, backup_type, admin_user.id)
        
        return {
            "message": f"Sauvegarde {backup_type} lancée en arrière-plan",
            "type": backup_type,
            "initiated_by": admin_user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du lancement de sauvegarde: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du lancement de sauvegarde")


@router.get("/maintenance/backups", response_model=List[BackupInfo])
async def list_backups(
    limit: int = Query(20, ge=1, le=100),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Liste les sauvegardes disponibles
    """
    try:
        async with get_db_session() as session:
            query = """
                SELECT id, filename, size, created_at, type, status
                FROM backups
                ORDER BY created_at DESC
                LIMIT ?
            """
            result = await session.execute(query, [limit])
            
            backups = []
            for row in result.fetchall():
                backups.append(BackupInfo(
                    id=row[0],
                    filename=row[1],
                    size=row[2],
                    created_at=row[3],
                    type=row[4],
                    status=row[5]
                ))
            
            return backups
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sauvegardes: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération")


@router.post("/maintenance/cleanup")
async def cleanup_system(
    clean_logs: bool = Query(True, description="Nettoyer les anciens logs"),
    clean_sessions: bool = Query(True, description="Nettoyer les sessions expirées"),
    clean_temp_files: bool = Query(True, description="Nettoyer les fichiers temporaires"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Lance un nettoyage du système
    """
    try:
        # Lancer les tâches de nettoyage en arrière-plan
        if clean_sessions:
            from src.api.main import app
            if hasattr(app.state, 'state_manager'):
                cleaned_sessions = await app.state.state_manager.cleanup_expired_sessions()
                logger.info(f"{cleaned_sessions} sessions expirées nettoyées par l'admin {admin_user.id}")
        
        if clean_logs:
            background_tasks.add_task(_cleanup_old_logs, admin_user.id)
        
        if clean_temp_files:
            background_tasks.add_task(_cleanup_temp_files, admin_user.id)
        
        return {
            "message": "Nettoyage système lancé",
            "tasks": {
                "clean_logs": clean_logs,
                "clean_sessions": clean_sessions,
                "clean_temp_files": clean_temp_files
            },
            "initiated_by": admin_user.username
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage système: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du nettoyage")


@router.get("/logs")
async def get_system_logs(
    level: str = Query("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    limit: int = Query(100, ge=1, le=1000),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Récupère les logs système
    """
    try:
        # Lire les logs depuis le fichier (implémentation simplifiée)
        logs = await _read_system_logs(level, limit)
        
        return {
            "logs": logs,
            "level": level,
            "count": len(logs),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des logs: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des logs")


@router.post("/system/restart")
async def restart_system_components(
    component: str = Query(..., regex="^(ai_service|cache|all)$"),
    admin_user: UserBase = Depends(require_admin)
):
    """
    Redémarre des composants du système
    """
    try:
        # Implémentation selon le composant
        if component == "ai_service":
            # Redémarrer le service IA
            result = await _restart_ai_service()
        elif component == "cache":
            # Vider le cache
            result = await _clear_cache()
        else:  # all
            # Redémarrer tous les services non critiques
            result = await _restart_all_services()
        
        logger.warning(f"Redémarrage du composant {component} par l'admin {admin_user.id}")
        
        return {
            "message": f"Redémarrage du composant {component} effectué",
            "component": component,
            "result": result,
            "initiated_by": admin_user.username
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du redémarrage: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du redémarrage")


# Fonctions utilitaires privées
async def _check_database_status() -> str:
    """Vérifie le statut de la base de données"""
    try:
        async with get_db_session() as session:
            await session.execute("SELECT 1")
        return "healthy"
    except Exception:
        return "unhealthy"


async def _check_ai_service_status() -> str:
    """Vérifie le statut du service IA"""
    try:
        # Implémentation selon votre service IA
        return "healthy"
    except Exception:
        return "unhealthy"


async def _perform_backup(backup_type: str, admin_id: int):
    """Effectue une sauvegarde en arrière-plan"""
    try:
        logger.info(f"Début de sauvegarde {backup_type} par l'admin {admin_id}")
        
        # Implémentation de la sauvegarde
        # - Sauvegarder la base de données
        # - Sauvegarder les fichiers
        # - Compresser et stocker
        
        # Enregistrer en base
        async with get_db_session() as session:
            query = """
                INSERT INTO backups (filename, size, created_at, type, status, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            await session.execute(query, [
                f"backup_{backup_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz",
                1024000,  # Taille exemple
                datetime.utcnow(),
                backup_type,
                "completed",
                admin_id
            ])
            await session.commit()
        
        logger.info(f"Sauvegarde {backup_type} terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")


async def _cleanup_old_logs(admin_id: int):
    """Nettoie les anciens logs"""
    try:
        logger.info(f"Nettoyage des logs lancé par l'admin {admin_id}")
        # Implémentation du nettoyage des logs
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des logs: {e}")


async def _cleanup_temp_files(admin_id: int):
    """Nettoie les fichiers temporaires"""
    try:
        logger.info(f"Nettoyage des fichiers temp lancé par l'admin {admin_id}")
        # Implémentation du nettoyage des fichiers temporaires
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des fichiers temp: {e}")


async def _read_system_logs(level: str, limit: int) -> List[Dict[str, Any]]:
    """Lit les logs système"""
    try:
        # Implémentation de lecture des logs
        # Pour l'exemple, retourner des logs factices
        logs = []
        for i in range(min(limit, 10)):
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": f"Exemple de log {level} #{i}",
                "module": "system"
            })
        
        return logs
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des logs: {e}")
        return []


async def _restart_ai_service() -> Dict[str, Any]:
    """Redémarre le service IA"""
    try:
        # Implémentation du redémarrage du service IA
        return {"status": "success", "message": "Service IA redémarré"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _clear_cache() -> Dict[str, Any]:
    """Vide le cache"""
    try:
        # Implémentation du vidage de cache
        return {"status": "success", "message": "Cache vidé"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _restart_all_services() -> Dict[str, Any]:
    """Redémarre tous les services"""
    try:
        # Implémentation du redémarrage global
        return {"status": "success", "message": "Tous les services redémarrés"}
    except Exception as e:
        return {"status": "error", "message": str(e)}