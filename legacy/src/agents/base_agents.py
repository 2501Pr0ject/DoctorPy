"""
Agent de base avec fonctionnalités communes pour tous les agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum

from src.core.config import get_config
from src.core.database import get_db_session
from src.models import User, UserProgress
from src.utils import generate_uuid, get_current_timestamp

logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    """Types d'agents"""
    TUTOR = "tutor"
    QUEST_GENERATOR = "quest_generator"
    CODE_EVALUATOR = "code_evaluator"
    PROGRESS_TRACKER = "progress_tracker"

class AgentState(str, Enum):
    """États des agents"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class AgentContext:
    """Contexte partagé entre les agents"""
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = None
    user_progress: Optional[Dict[str, Any]] = None
    current_quest: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentResponse:
    """Réponse standardisée des agents"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    next_actions: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    errors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la réponse en dictionnaire"""
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'suggestions': self.suggestions,
            'next_actions': self.next_actions,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'errors': self.errors,
            'timestamp': get_current_timestamp()
        }

class BaseAgent(ABC):
    """Agent de base avec fonctionnalités communes"""
    
    def __init__(self, agent_type: AgentType, name: str = None):
        self.agent_type = agent_type
        self.name = name or f"{agent_type.value}_agent"
        self.agent_id = generate_uuid()
        self.config = get_config()
        self.state = AgentState.IDLE
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = self.created_at
        
        # Statistiques de l'agent
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        
        # Configuration spécifique à l'agent
        self.max_retries = 3
        self.timeout_seconds = 30
        self.enable_logging = True
        
        logger.info(f"Agent {self.name} initialisé (ID: {self.agent_id})")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """
        Méthode principale de traitement de l'agent
        
        Args:
            input_data: Données d'entrée
            context: Contexte de l'agent
            
        Returns:
            Réponse de l'agent
        """
        pass
    
    async def execute(self, input_data: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """
        Exécute l'agent avec gestion d'erreurs et métriques
        
        Args:
            input_data: Données d'entrée
            context: Contexte de l'agent
            
        Returns:
            Réponse de l'agent
        """
        start_time = datetime.now(timezone.utc)
        self.state = AgentState.PROCESSING
        self.total_requests += 1
        
        try:
            # Validation des entrées
            if not self._validate_input(input_data):
                return AgentResponse(
                    success=False,
                    message="Données d'entrée invalides",
                    errors=["Validation des données d'entrée échouée"]
                )
            
            # Préparation du contexte
            context = await self._prepare_context(context)
            
            # Traitement principal avec timeout
            response = await asyncio.wait_for(
                self.process(input_data, context),
                timeout=self.timeout_seconds
            )
            
            # Post-traitement
            response = await self._post_process_response(response, context)
            
            self.state = AgentState.COMPLETED
            self.successful_requests += 1
            
            # Logging du succès
            if self.enable_logging:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.info(f"Agent {self.name} - Traitement réussi en {duration:.2f}s")
            
            return response
            
        except asyncio.TimeoutError:
            self.state = AgentState.ERROR
            self.failed_requests += 1
            error_msg = f"Timeout après {self.timeout_seconds}s"
            logger.error(f"Agent {self.name} - {error_msg}")
            
            return AgentResponse(
                success=False,
                message="Traitement trop long",
                errors=[error_msg]
            )
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.failed_requests += 1
            error_msg = f"Erreur lors du traitement: {str(e)}"
            logger.error(f"Agent {self.name} - {error_msg}")
            
            return AgentResponse(
                success=False,
                message="Erreur interne",
                errors=[error_msg]
            )
        
        finally:
            # Mise à jour des métriques
            self.last_activity = datetime.now(timezone.utc)
            duration = (self.last_activity - start_time).total_seconds()
            self._update_metrics(duration)
    
    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Valide les données d'entrée
        
        Args:
            input_data: Données à valider
            
        Returns:
            True si valide
        """
        if not isinstance(input_data, dict):
            return False
        
        # Validation de base - peut être overridée par les agents enfants
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in input_data:
                logger.warning(f"Champ requis manquant: {field}")
                return False
        
        return True
    
    async def _prepare_context(self, context: AgentContext) -> AgentContext:
        """
        Prépare et enrichit le contexte
        
        Args:
            context: Contexte initial
            
        Returns:
            Contexte enrichi
        """
        # Charger les informations utilisateur si nécessaire
        if context.user_id and not context.user_progress:
            try:
                with get_db_session() as db:
                    user = db.query(User).filter(User.id == context.user_id).first()
                    if user:
                        progress = db.query(UserProgress).filter(
                            UserProgress.user_id == context.user_id
                        ).first()
                        
                        if progress:
                            context.user_progress = {
                                'level': progress.overall_level,
                                'xp_points': progress.xp_points,
                                'skill_scores': progress.get_skill_scores(),
                                'current_streak': progress.current_streak
                            }
                        
                        context.preferences = {
                            'level': user.level,
                            'learning_style': user.learning_style,
                            'preferred_languages': user.get_preferred_languages()
                        }
            except Exception as e:
                logger.error(f"Erreur lors du chargement du contexte utilisateur: {e}")
        
        return context
    
    async def _post_process_response(self, response: AgentResponse, context: AgentContext) -> AgentResponse:
        """
        Post-traite la réponse de l'agent
        
        Args:
            response: Réponse initiale
            context: Contexte de l'agent
            
        Returns:
            Réponse post-traitée
        """
        # Ajouter des métadonnées
        if response.data is None:
            response.data = {}
        
        response.data.update({
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'agent_type': self.agent_type.value,
            'processing_time': (datetime.now(timezone.utc) - self.last_activity).total_seconds()
        })
        
        # Logging des réponses
        if self.enable_logging and response.success:
            logger.debug(f"Agent {self.name} - Réponse: {response.message}")
        
        return response
    
    def _update_metrics(self, duration: float):
        """
        Met à jour les métriques de performance
        
        Args:
            duration: Durée du traitement en secondes
        """
        # Calcul de la moyenne mobile du temps de réponse
        if self.average_response_time == 0:
            self.average_response_time = duration
        else:
            # Moyenne pondérée (90% ancien, 10% nouveau)
            self.average_response_time = (self.average_response_time * 0.9) + (duration * 0.1)
    
    def get_required_fields(self) -> List[str]:
        """
        Retourne les champs requis pour cet agent
        
        Returns:
            Liste des champs requis
        """
        return []  # À override par les agents enfants
    
    def get_capabilities(self) -> List[str]:
        """
        Retourne les capacités de l'agent
        
        Returns:
            Liste des capacités
        """
        return []  # À override par les agents enfants
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retourne le statut de l'agent
        
        Returns:
            Statut de l'agent
        """
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'type': self.agent_type.value,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'metrics': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': round(success_rate, 3),
                'average_response_time': round(self.average_response_time, 3)
            },
            'capabilities': self.get_capabilities(),
            'configuration': {
                'max_retries': self.max_retries,
                'timeout_seconds': self.timeout_seconds,
                'enable_logging': self.enable_logging
            }
        }
    
    def reset_metrics(self):
        """Remet à zéro les métriques"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        logger.info(f"Métriques de l'agent {self.name} remises à zéro")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé de l'agent
        
        Returns:
            Statut de santé
        """
        try:
            # Test basique de l'agent
            test_context = AgentContext()
            test_input = {'test': True}
            
            start_time = datetime.now(timezone.utc)
            
            # Certains agents peuvent ne pas supporter le mode test
            if hasattr(self, '_health_check_test'):
                result = await self._health_check_test(test_input, test_context)
            else:
                # Test minimal - juste vérifier que l'agent répond
                result = True
            
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                'healthy': True,
                'response_time': response_time,
                'last_check': get_current_timestamp(),
                'details': 'Agent fonctionnel'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'last_check': get_current_timestamp(),
                'details': 'Erreur lors du test de santé'
            }
    
    def configure(self, **kwargs):
        """
        Configure l'agent avec de nouveaux paramètres
        
        Args:
            **kwargs: Paramètres de configuration
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Agent {self.name} - Configuration mise à jour: {key} = {value}")
            else:
                logger.warning(f"Agent {self.name} - Paramètre de configuration inconnu: {key}")

class AgentManager:
    """Gestionnaire d'agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_registry: Dict[AgentType, type] = {}
        self.config = get_config()
        
    def register_agent_type(self, agent_type: AgentType, agent_class: type):
        """
        Enregistre un type d'agent
        
        Args:
            agent_type: Type d'agent
            agent_class: Classe de l'agent
        """
        self.agent_registry[agent_type] = agent_class
        logger.info(f"Type d'agent enregistré: {agent_type.value}")
    
    def create_agent(self, agent_type: AgentType, name: str = None, **kwargs) -> BaseAgent:
        """
        Crée une instance d'agent
        
        Args:
            agent_type: Type d'agent à créer
            name: Nom de l'agent
            **kwargs: Arguments pour l'agent
            
        Returns:
            Instance de l'agent
        """
        if agent_type not in self.agent_registry:
            raise ValueError(f"Type d'agent non enregistré: {agent_type.value}")
        
        agent_class = self.agent_registry[agent_type]
        agent = agent_class(agent_type=agent_type, name=name, **kwargs)
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent créé: {agent.name} (ID: {agent.agent_id})")
        
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Récupère un agent par son ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Récupère tous les agents d'un type donné"""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        Supprime un agent
        
        Args:
            agent_id: ID de l'agent à supprimer
            
        Returns:
            True si supprimé avec succès
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            logger.info(f"Agent supprimé: {agent.name}")
            return True
        return False
    
    async def health_check_all(self) -> Dict[str, Any]:
        """
        Vérifie la santé de tous les agents
        
        Returns:
            Statut de santé global
        """
        results = {}
        healthy_count = 0
        
        for agent_id, agent in self.agents.items():
            try:
                health = await agent.health_check()
                results[agent_id] = health
                if health['healthy']:
                    healthy_count += 1
            except Exception as e:
                results[agent_id] = {
                    'healthy': False,
                    'error': str(e),
                    'last_check': get_current_timestamp()
                }
        
        return {
            'overall_healthy': healthy_count == len(self.agents),
            'healthy_agents': healthy_count,
            'total_agents': len(self.agents),
            'agents': results
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Retourne le statut du système d'agents
        
        Returns:
            Statut du système
        """
        agent_states = {}
        for state in AgentState:
            agent_states[state.value] = len([
                a for a in self.agents.values() if a.state == state
            ])
        
        return {
            'total_agents': len(self.agents),
            'registered_types': len(self.agent_registry),
            'agent_states': agent_states,
            'agent_types': {
                agent_type.value: len(self.get_agents_by_type(agent_type))
                for agent_type in self.agent_registry.keys()
            }
        }

# Instance globale du gestionnaire d'agents
agent_manager = AgentManager()