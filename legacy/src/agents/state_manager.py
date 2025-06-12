# src/agents/state_manager.py
"""
Gestionnaire d'états pour LangGraph - Orchestration des agents et des conversations
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, TypedDict, Union, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from pathlib import Path

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Imports conditionnels pour éviter les erreurs
try:
    from src.core.config import get_settings
    settings = get_settings()
except:
    # Configuration par défaut en cas d'erreur
    class DefaultSettings:
        STATE_MANAGER_CHECKPOINT_PATH = "./data/checkpoints"
        MAX_CONCURRENT_SESSIONS = 100
        SESSION_TIMEOUT_MINUTES = 30
    settings = DefaultSettings()
from src.core.database import get_db_session
from src.core.exceptions import ValidationError, NotFoundError
from src.agents.base_agents import AgentType

logger = logging.getLogger(__name__)


# =============================================================================
# TYPES & ENUMS
# =============================================================================

class ConversationMode(str, Enum):
    """Modes de conversation disponibles"""
    FREE_CHAT = "free_chat"          # Conversation libre
    GUIDED_QUEST = "guided_quest"    # Quête guidée
    CODE_REVIEW = "code_review"      # Révision de code
    EXPLANATION = "explanation"      # Explication de concept
    EVALUATION = "evaluation"        # Évaluation/test
    HELP_REQUEST = "help_request"    # Demande d'aide


class SessionStatus(str, Enum):
    """Statuts de session"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"


class NodeType(str, Enum):
    """Types de nœuds dans le graphe"""
    ROUTER = "router"               # Routage des messages
    TUTOR = "tutor"                # Agent tuteur
    QUEST_MANAGER = "quest_manager" # Gestionnaire de quêtes
    CODE_EVALUATOR = "code_evaluator" # Évaluateur de code
    CONTEXT_RETRIEVER = "context_retriever" # Récupérateur de contexte


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

class ConversationState(TypedDict):
    """État de la conversation LangGraph"""
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
# GESTIONNAIRE D'ÉTATS PRINCIPAL
# =============================================================================

class StateManager:
    """
    Gestionnaire d'états centralisé pour LangGraph et les sessions utilisateur
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or settings.STATE_MANAGER_CHECKPOINT_PATH
        self.max_sessions = settings.MAX_CONCURRENT_SESSIONS
        self.session_timeout = timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
        
        # Sessions actives en mémoire
        self.active_sessions: Dict[str, UserSession] = {}
        
        # Checkpoint SQLite pour persistance
        self.checkpointer = None
        self._initialize_checkpointer()
        
        # Graphe LangGraph
        self.conversation_graph = None
        self._build_conversation_graph()
        
        logger.info("StateManager initialisé")
    
    def _initialize_checkpointer(self):
        """Initialise le système de checkpoint"""
        try:
            checkpoint_dir = Path(self.checkpoint_path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            db_path = checkpoint_dir / "checkpoints.db"
            self.checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
            
            logger.info(f"Checkpointer SQLite initialisé: {db_path}")
            
        except Exception as e:
            logger.warning(f"Impossible d'initialiser SQLite, fallback vers MemorySaver: {e}")
            self.checkpointer = MemorySaver()
    
    def _build_conversation_graph(self):
        """Construit le graphe de conversation LangGraph"""
        from langgraph.graph import StateGraph
        
        graph = StateGraph(ConversationState)
        
        # Ajouter les nœuds
        graph.add_node("router", self._route_message)
        graph.add_node("tutor", self._handle_tutor)
        graph.add_node("quest_manager", self._handle_quest)
        graph.add_node("code_evaluator", self._handle_code)
        graph.add_node("context_retriever", self._retrieve_context)
        
        # Définir les arêtes
        graph.add_edge("router", "tutor")
        graph.add_edge("router", "quest_manager")
        graph.add_edge("router", "code_evaluator")
        graph.add_edge("context_retriever", "tutor")
        
        # Point d'entrée
        graph.set_entry_point("router")
        
        # Compiler le graphe avec checkpoint
        self.conversation_graph = graph.compile(checkpointer=self.checkpointer)
        
        logger.info("Graphe de conversation LangGraph construit")
    
    # =========================================================================
    # GESTION DES SESSIONS
    # =========================================================================
    
    async def create_session(
        self, 
        user_id: int, 
        mode: ConversationMode = ConversationMode.FREE_CHAT,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Crée une nouvelle session utilisateur"""
        try:
            # Vérifier la limite de sessions
            await self._cleanup_expired_sessions()
            
            if len(self.active_sessions) >= self.max_sessions:
                # Fermer la session la plus ancienne
                oldest_session_id = min(
                    self.active_sessions.keys(),
                    key=lambda s: self.active_sessions[s].last_activity
                )
                await self.end_session(oldest_session_id)
            
            # Générer ID unique
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Récupérer le profil utilisateur
            user_profile = await self._get_user_profile(user_id)
            
            # État initial
            initial_state = ConversationState(
                session_id=session_id,
                user_id=user_id,
                mode=mode,
                current_agent=None,
                messages=[],
                context=initial_context or {},
                quest_progress=None,
                user_profile=user_profile,
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
            
            # Enregistrer en base
            await self._persist_session(session)
            
            logger.info(f"Session créée: {session_id} pour utilisateur {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de session: {e}")
            raise ValidationError(f"Impossible de créer la session: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Récupère une session par ID"""
        try:
            # Vérifier en mémoire d'abord
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Vérifier l'expiration
                if datetime.utcnow() > session.expires_at:
                    await self.end_session(session_id)
                    return None
                
                return session
            
            # Sinon, chercher en base
            return await self._load_session_from_db(session_id)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de session {session_id}: {e}")
            return None
    
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
            
            # Sauvegarder en base
            await self._persist_session_final(session, reason)
            
            # Retirer de la mémoire
            del self.active_sessions[session_id]
            
            logger.info(f"Session terminée: {session_id} ({reason})")
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de session {session_id}: {e}")
    
    # =========================================================================
    # TRAITEMENT DES MESSAGES
    # =========================================================================
    
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
                raise NotFoundError(f"Session {session_id} non trouvée")
            
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
            
            # Traiter via LangGraph
            config = {"configurable": {"thread_id": session_id}}
            
            result = await self.conversation_graph.ainvoke(
                session.state,
                config=config
            )
            
            # Mettre à jour l'état
            session.state.update(result)
            
            # Extraire la réponse
            response = self._extract_response(result)
            
            logger.info(f"Message traité dans session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].metrics.errors_count += 1
            raise
    
    # =========================================================================
    # NŒUDS DU GRAPHE LANGGRAPH
    # =========================================================================
    
    async def _route_message(self, state: ConversationState) -> ConversationState:
        """Nœud de routage - détermine quel agent doit traiter le message"""
        try:
            last_message = state["messages"][-1] if state["messages"] else None
            
            if not last_message:
                state["current_agent"] = "tutor"
                return state
            
            content = last_message["content"].lower()
            
            # Logique de routage simple
            if any(keyword in content for keyword in ["code", "programme", "script"]):
                state["current_agent"] = "code_evaluator"
            elif any(keyword in content for keyword in ["quête", "exercice", "défi"]):
                state["current_agent"] = "quest_manager"
            elif any(keyword in content for keyword in ["aide", "help", "expliquer"]):
                state["current_agent"] = "tutor"
            else:
                state["current_agent"] = "tutor"
            
            return state
            
        except Exception as e:
            logger.error(f"Erreur dans le routeur: {e}")
            state["current_agent"] = "tutor"
            return state
    
    async def _handle_tutor(self, state: ConversationState) -> ConversationState:
        """Nœud tuteur - gestion des conversations générales"""
        try:
            from src.agents.tutor_agent import TutorAgent
            
            tutor = TutorAgent()
            response = await tutor.process_message(state)
            
            # Ajouter la réponse aux messages
            response_msg = {
                "type": "ai",
                "content": response["content"],
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "tutor",
                "metadata": response.get("metadata", {})
            }
            state["messages"].append(response_msg)
            
            # Mettre à jour le contexte
            if "context_update" in response:
                state["context"].update(response["context_update"])
            
            return state
            
        except Exception as e:
            logger.error(f"Erreur dans l'agent tuteur: {e}")
            # Réponse de fallback
            error_msg = {
                "type": "ai",
                "content": "Désolé, je rencontre une difficulté. Pouvez-vous reformuler votre question ?",
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "tutor",
                "metadata": {"error": str(e)}
            }
            state["messages"].append(error_msg)
            return state
    
    async def _handle_quest(self, state: ConversationState) -> ConversationState:
        """Nœud gestionnaire de quêtes"""
        try:
            from src.agents.quest_generator import QuestGeneratorAgent
            
            quest_manager = QuestGeneratorAgent()
            response = await quest_manager.process_message(state)
            
            # Ajouter la réponse
            response_msg = {
                "type": "ai",
                "content": response["content"],
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "quest_manager",
                "metadata": response.get("metadata", {})
            }
            state["messages"].append(response_msg)
            
            # Mettre à jour le progrès des quêtes
            if "quest_progress" in response:
                state["quest_progress"] = response["quest_progress"]
            
            return state
            
        except Exception as e:
            logger.error(f"Erreur dans le gestionnaire de quêtes: {e}")
            error_msg = {
                "type": "ai",
                "content": "Erreur lors du traitement de la quête.",
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "quest_manager",
                "metadata": {"error": str(e)}
            }
            state["messages"].append(error_msg)
            return state
    
    async def _handle_code(self, state: ConversationState) -> ConversationState:
        """Nœud évaluateur de code"""
        try:
            from src.agents.code_evaluator import CodeEvaluatorAgent
            
            evaluator = CodeEvaluatorAgent()
            response = await evaluator.process_message(state)
            
            # Ajouter la réponse
            response_msg = {
                "type": "ai",
                "content": response["content"],
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "code_evaluator",
                "metadata": response.get("metadata", {})
            }
            state["messages"].append(response_msg)
            
            # Mettre à jour les métriques si code exécuté
            if response.get("code_executed", False):
                session_id = state["session_id"]
                if session_id in self.active_sessions:
                    self.active_sessions[session_id].metrics.code_executed += 1
            
            return state
            
        except Exception as e:
            logger.error(f"Erreur dans l'évaluateur de code: {e}")
            error_msg = {
                "type": "ai",
                "content": "Erreur lors de l'évaluation du code.",
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "code_evaluator",
                "metadata": {"error": str(e)}
            }
            state["messages"].append(error_msg)
            return state
    
    async def _retrieve_context(self, state: ConversationState) -> ConversationState:
        """Nœud de récupération de contexte RAG"""
        try:
            from src.rag.retriever import DocumentRetriever
            
            retriever = DocumentRetriever()
            last_message = state["messages"][-1]["content"] if state["messages"] else ""
            
            # Récupérer le contexte pertinent
            context_docs = await retriever.get_relevant_documents(last_message)
            
            # Ajouter au contexte
            state["context"]["retrieved_docs"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", 0.0)
                }
                for doc in context_docs
            ]
            
            return state
            
        except Exception as e:
            logger.error(f"Erreur dans la récupération de contexte: {e}")
            return state
    
    # =========================================================================
    # MÉTHODES UTILITAIRES
    # =========================================================================
    
    def _extract_response(self, state: ConversationState) -> Dict[str, Any]:
        """Extrait la réponse formatée de l'état"""
        last_message = state["messages"][-1] if state["messages"] else None
        
        if not last_message:
            return {"content": "Aucune réponse générée", "type": "error"}
        
        return {
            "content": last_message["content"],
            "type": last_message["type"],
            "agent": last_message.get("agent"),
            "metadata": last_message.get("metadata", {}),
            "session_id": state["session_id"],
            "timestamp": last_message["timestamp"]
        }
    
    async def _get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Récupère le profil utilisateur"""
        try:
            async with get_db_session() as session:
                query = """
                    SELECT username, email, full_name, preferred_difficulty, 
                           learning_objectives, created_at
                    FROM users WHERE id = ?
                """
                result = await session.execute(query, [user_id])
                user_data = result.fetchone()
                
                if user_data:
                    return {
                        "username": user_data[0],
                        "email": user_data[1],
                        "full_name": user_data[2],
                        "preferred_difficulty": user_data[3],
                        "learning_objectives": json.loads(user_data[4] or "[]"),
                        "created_at": user_data[5]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du profil: {e}")
            return None
    
    async def _persist_session(self, session: UserSession):
        """Persiste une session en base de données"""
        try:
            async with get_db_session() as db_session:
                query = """
                    INSERT INTO user_sessions (
                        session_id, user_id, status, mode, state_data, 
                        created_at, last_activity, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        status = ?, last_activity = ?, state_data = ?
                """
                state_data = json.dumps(session.state, default=str)
                
                await db_session.execute(query, [
                    session.session_id, session.user_id, session.status.value,
                    session.mode.value, state_data, session.created_at,
                    session.last_activity, session.expires_at,
                    session.status.value, session.last_activity, state_data
                ])
                await db_session.commit()
                
        except Exception as e:
            logger.error(f"Erreur lors de la persistance de session: {e}")
    
    async def _persist_session_final(self, session: UserSession, reason: str):
        """Persiste les données finales d'une session"""
        try:
            async with get_db_session() as db_session:
                # Mettre à jour la session
                await db_session.execute("""
                    UPDATE user_sessions 
                    SET status = ?, completed_at = ?, end_reason = ?
                    WHERE session_id = ?
                """, [
                    session.status.value, 
                    session.metrics.completed_at,
                    reason,
                    session.session_id
                ])
                
                # Enregistrer les métriques
                await db_session.execute("""
                    INSERT INTO session_metrics (
                        session_id, duration_seconds, message_count, agents_used,
                        quests_completed, code_executed, errors_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        duration_seconds = ?, message_count = ?, agents_used = ?,
                        quests_completed = ?, code_executed = ?, errors_count = ?
                """, [
                    session.session_id, session.metrics.duration_seconds,
                    session.metrics.message_count, ",".join(session.metrics.agents_used),
                    session.metrics.quests_completed, session.metrics.code_executed,
                    session.metrics.errors_count, session.metrics.created_at,
                    session.metrics.duration_seconds, session.metrics.message_count,
                    ",".join(session.metrics.agents_used), session.metrics.quests_completed,
                    session.metrics.code_executed, session.metrics.errors_count
                ])
                
                await db_session.commit()
                
        except Exception as e:
            logger.error(f"Erreur lors de la persistance finale: {e}")
    
    async def _load_session_from_db(self, session_id: str) -> Optional[UserSession]:
        """Charge une session depuis la base de données"""
        try:
            async with get_db_session() as db_session:
                query = """
                    SELECT user_id, status, mode, state_data, created_at, 
                           last_activity, expires_at
                    FROM user_sessions WHERE session_id = ?
                """
                result = await db_session.execute(query, [session_id])
                session_data = result.fetchone()
                
                if not session_data:
                    return None
                
                # Reconstituer la session
                state = json.loads(session_data[3])
                
                # Charger les métriques
                metrics_query = """
                    SELECT duration_seconds, message_count, agents_used,
                           quests_completed, code_executed, errors_count
                    FROM session_metrics WHERE session_id = ?
                """
                metrics_result = await db_session.execute(metrics_query, [session_id])
                metrics_data = metrics_result.fetchone()
                
                if metrics_data:
                    metrics = SessionMetrics(
                        session_id=session_id,
                        duration_seconds=metrics_data[0],
                        message_count=metrics_data[1],
                        agents_used=metrics_data[2].split(",") if metrics_data[2] else [],
                        quests_completed=metrics_data[3],
                        code_executed=metrics_data[4],
                        errors_count=metrics_data[5],
                        created_at=session_data[4]
                    )
                else:
                    metrics = SessionMetrics(
                        session_id=session_id,
                        duration_seconds=0,
                        message_count=0,
                        agents_used=[],
                        quests_completed=0,
                        code_executed=0,
                        errors_count=0,
                        created_at=session_data[4]
                    )
                
                session = UserSession(
                    session_id=session_id,
                    user_id=session_data[0],
                    status=SessionStatus(session_data[1]),
                    mode=ConversationMode(session_data[2]),
                    state=state,
                    metrics=metrics,
                    created_at=session_data[4],
                    last_activity=session_data[5],
                    expires_at=session_data[6]
                )
                
                return session
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de session: {e}")
            return None
    
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
    
    async def cleanup_expired_sessions(self):
        """Méthode publique pour nettoyer les sessions expirées"""
        await self._cleanup_expired_sessions()
    
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


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

async def initialize_state_manager(checkpoint_path: Optional[str] = None) -> StateManager:
    """Initialise le gestionnaire d'états"""
    try:
        manager = StateManager(checkpoint_path)
        logger.info("StateManager initialisé avec succès")
        return manager
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du StateManager: {e}")
        raise


async def periodic_cleanup():
    """Tâche de nettoyage périodique des sessions expirées"""
    while True:
        try:
            # Cette fonction sera appelée depuis l'application principale
            await asyncio.sleep(300)  # 5 minutes
            
        except asyncio.CancelledError:
            logger.info("Tâche de nettoyage annulée")
            break
        except Exception as e:
            logger.error(f"Erreur dans la tâche de nettoyage: {e}")
            await asyncio.sleep(60)  # Attendre 1 minute avant de réessayer


# Export des classes principales
__all__ = [
    "StateManager",
    "ConversationState", 
    "ConversationMode",
    "SessionStatus",
    "UserSession",
    "SessionMetrics",
    "initialize_state_manager",
    "periodic_cleanup"
]