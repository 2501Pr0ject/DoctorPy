# src/agents/state_manager_simple.py
"""
Version simplifiée du gestionnaire d'états pour les tests
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, TypedDict, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# TYPES & ENUMS
# =============================================================================

class ConversationMode(str, Enum):
    """Modes de conversation disponibles"""
    FREE_CHAT = "free_chat"
    GUIDED_QUEST = "guided_quest"
    CODE_REVIEW = "code_review"
    EXPLANATION = "explanation"
    EVALUATION = "evaluation"
    HELP_REQUEST = "help_request"


class SessionStatus(str, Enum):
    """Statuts de session"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"


class ConversationState(TypedDict):
    """État de la conversation"""
    session_id: str
    user_id: int
    mode: ConversationMode
    current_agent: Optional[str]
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    quest_progress: Optional[Dict[str, Any]]
    user_profile: Optional[Dict[str, Any]]
    last_activity: datetime
    metadata: Dict[str, Any]


@dataclass
class SessionMetrics:
    """Métriques de session"""
    session_id: str
    duration_seconds: int
    message_count: int
    agents_used: List[str]
    quests_completed: int
    code_executed: int
    errors_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None


@dataclass
class UserSession:
    """Session utilisateur complète"""
    session_id: str
    user_id: int
    status: SessionStatus
    mode: ConversationMode
    state: ConversationState
    metrics: SessionMetrics
    created_at: datetime
    last_activity: datetime
    expires_at: datetime


# =============================================================================
# GESTIONNAIRE D'ÉTATS SIMPLIFIÉ
# =============================================================================

class SimpleStateManager:
    """
    Version simplifiée du gestionnaire d'états pour les tests
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or "./data/checkpoints"
        self.max_sessions = 100
        self.session_timeout = timedelta(minutes=30)
        
        # Sessions actives en mémoire
        self.active_sessions: Dict[str, UserSession] = {}
        
        # Créer le dossier de checkpoint
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("SimpleStateManager initialisé")
    
    async def create_session(
        self, 
        user_id: int, 
        mode: ConversationMode = ConversationMode.FREE_CHAT,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Crée une nouvelle session utilisateur"""
        try:
            # Nettoyer les sessions expirées
            await self._cleanup_expired_sessions()
            
            # Limiter le nombre de sessions
            if len(self.active_sessions) >= self.max_sessions:
                oldest_session_id = min(
                    self.active_sessions.keys(),
                    key=lambda s: self.active_sessions[s].last_activity
                )
                await self.end_session(oldest_session_id)
            
            # Générer ID unique
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # État initial
            initial_state = ConversationState(
                session_id=session_id,
                user_id=user_id,
                mode=mode,
                current_agent=None,
                messages=[],
                context=initial_context or {},
                quest_progress=None,
                user_profile={"user_id": user_id},
                last_activity=now,
                metadata={"created_at": now.isoformat()}
            )
            
            # Métriques initiales
            metrics = SessionMetrics(
                session_id=session_id,
                duration_seconds=0,
                message_count=0,
                agents_used=[],
                quests_completed=0,
                code_executed=0,
                errors_count=0,
                created_at=now
            )
            
            # Créer la session
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                status=SessionStatus.ACTIVE,
                mode=mode,
                state=initial_state,
                metrics=metrics,
                created_at=now,
                last_activity=now,
                expires_at=now + self.session_timeout
            )
            
            # Stocker en mémoire
            self.active_sessions[session_id] = session
            
            logger.info(f"Session créée: {session_id} pour utilisateur {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de session: {e}")
            raise Exception(f"Impossible de créer la session: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Récupère une session par ID"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Vérifier l'expiration
                if datetime.utcnow() > session.expires_at:
                    await self.end_session(session_id)
                    return None
                
                return session
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de session {session_id}: {e}")
            return None
    
    async def process_message(
        self, 
        session_id: str, 
        message: str, 
        message_type: str = "human"
    ) -> Dict[str, Any]:
        """Traite un message dans une session"""
        try:
            session = await self.get_session(session_id)
            if not session:
                raise Exception(f"Session {session_id} non trouvée")
            
            # Mettre à jour l'activité
            await self.update_session_activity(session_id)
            
            # Ajouter le message à l'état
            message_obj = {
                "type": message_type,
                "content": message,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {}
            }
            session.state["messages"].append(message_obj)
            session.metrics.message_count += 1
            
            # Simuler une réponse simple
            response_content = f"Reçu: {message}"
            response_msg = {
                "type": "ai",
                "content": response_content,
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "tutor",
                "metadata": {}
            }
            session.state["messages"].append(response_msg)
            
            logger.info(f"Message traité dans session {session_id}")
            return {
                "content": response_content,
                "type": "ai",
                "agent": "tutor",
                "session_id": session_id,
                "timestamp": response_msg["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].metrics.errors_count += 1
            raise
    
    async def update_session_activity(self, session_id: str):
        """Met à jour l'activité d'une session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            now = datetime.utcnow()
            
            session.last_activity = now
            session.expires_at = now + self.session_timeout
            session.state["last_activity"] = now
            
            # Mettre à jour les métriques
            session.metrics.duration_seconds = int(
                (now - session.created_at).total_seconds()
            )
    
    async def end_session(self, session_id: str, reason: str = "user_request"):
        """Termine une session"""
        try:
            if session_id not in self.active_sessions:
                return
            
            session = self.active_sessions[session_id]
            now = datetime.utcnow()
            
            # Finaliser les métriques
            session.status = SessionStatus.COMPLETED
            session.metrics.completed_at = now
            session.metrics.duration_seconds = int(
                (now - session.created_at).total_seconds()
            )
            
            # Retirer de la mémoire
            del self.active_sessions[session_id]
            
            logger.info(f"Session terminée: {session_id} ({reason})")
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de session {session_id}: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Nettoie les sessions expirées"""
        try:
            now = datetime.utcnow()
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if now > session.expires_at
            ]
            
            for session_id in expired_sessions:
                await self.end_session(session_id, "expired")
            
            if expired_sessions:
                logger.info(f"Sessions expirées nettoyées: {len(expired_sessions)}")
                
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des sessions: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques système"""
        try:
            await self._cleanup_expired_sessions()
            
            active_count = len(self.active_sessions)
            total_messages = sum(s.metrics.message_count for s in self.active_sessions.values())
            
            return {
                "active_sessions": active_count,
                "max_sessions": self.max_sessions,
                "total_messages_today": total_messages,
                "session_timeout_minutes": self.session_timeout.total_seconds() / 60,
                "checkpoint_path": str(self.checkpoint_path)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {"error": str(e)}


# Export des classes principales
__all__ = [
    "SimpleStateManager",
    "ConversationState", 
    "ConversationMode",
    "SessionStatus",
    "UserSession",
    "SessionMetrics"
]