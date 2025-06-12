# src/agents/state_manager.py
"""
Gestionnaire d'états LangGraph - Orchestration des agents pédagogiques
"""

from typing import Dict, Any, List, Optional, Union, Callable, TypedDict
import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
import json
import re
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from src.agents.base_agent import BaseAgent, AgentContext, AgentResponse, AgentType, agent_manager
from src.agents.tutor_agent import TutorAgent
from src.agents.quest_generator import QuestGeneratorAgent
from src.agents.code_evaluator import CodeEvaluatorAgent
from src.core.database import get_db_session
from src.models import User, UserProgress, UserQuest
from src.utils import generate_uuid, get_current_timestamp

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """États de conversation possibles"""
    INITIAL = "initial"
    CHATTING = "chatting"
    QUEST_SELECTION = "quest_selection"
    QUEST_ACTIVE = "quest_active"
    CODE_EVALUATION = "code_evaluation"
    FEEDBACK_REVIEW = "feedback_review"
    QUEST_GENERATION = "quest_generation"
    HELP_MODE = "help_mode"
    PROGRESS_REVIEW = "progress_review"
    LEARNING_PATH = "learning_path"
    COMPLETED = "completed"
    ERROR = "error"

class AgentAction(str, Enum):
    """Actions possibles des agents"""
    RESPOND_TO_QUERY = "respond_to_query"
    GENERATE_QUEST = "generate_quest"
    EVALUATE_CODE = "evaluate_code"
    PROVIDE_HINT = "provide_hint"
    CHECK_PROGRESS = "check_progress"
    SUGGEST_NEXT_STEP = "suggest_next_step"
    CREATE_LEARNING_PATH = "create_learning_path"
    HANDLE_ERROR = "handle_error"
    END_SESSION = "end_session"

class WorkflowState(TypedDict):
    """État du workflow LangGraph"""
    # Contexte utilisateur
    user_id: Optional[int]
    session_id: str
    conversation_state: str
    
    # Données de conversation
    messages: List[Dict[str, Any]]
    current_query: str
    last_response: str
    
    # Contexte pédagogique
    current_quest_id: Optional[int]
    current_step: int
    user_level: str
    learning_objectives: List[str]
    user_skills: Dict[str, float]  # skill -> level (0-1)
    
    # États des agents
    agent_responses: Dict[str, Dict[str, Any]]
    last_agent_used: Optional[str]
    pending_actions: List[str]
    
    # Métadonnées
    session_start_time: str
    last_activity_time: str
    total_interactions: int
    
    # Flags de contrôle
    needs_code_evaluation: bool
    needs_quest_generation: bool
    needs_progress_review: bool
    awaiting_user_input: bool
    error_occurred: bool
    
    # Données temporaires
    temp_data: Dict[str, Any]

@dataclass
class ConversationFlow:
    """Définit un flux de conversation"""
    name: str
    description: str
    entry_conditions: List[Callable[[WorkflowState], bool]]
    states: List[ConversationState]
    transitions: Dict[str, List[str]]
    default_agent: AgentType
    priority: int = 1

class StateManager:
    """Gestionnaire d'états principal utilisant LangGraph"""
    
    def __init__(self, checkpoint_path: str = "data/checkpoints/state_manager.db"):
        self.checkpoint_path = checkpoint_path
        self.graph = None
        
        # Créer le répertoire s'il n'existe pas
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_path}")
        
        # Agents disponibles
        self.agents: Dict[AgentType, BaseAgent] = {}
        
        # Flux de conversation prédéfinis
        self.conversation_flows: List[ConversationFlow] = []
        
        # Historique des sessions
        self.active_sessions: Dict[str, WorkflowState] = {}
        
        # Configuration
        self.max_session_duration = 3600  # 1 heure
        self.max_interactions_per_session = 100
        self.auto_save_frequency = 10  # Sauvegarder toutes les 10 interactions
        
        # Patterns de reconnaissance
        self.code_patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',         # Inline code
            r'def\s+\w+\s*\(',  # Fonction Python
            r'class\s+\w+\s*:', # Classe Python
            r'for\s+\w+\s+in',  # Boucle for
            r'if\s+.+:',        # Condition if
        ]
        
        self._initialize_agents()
        self._define_conversation_flows()
        self._build_graph()
        
        logger.info("Gestionnaire d'états initialisé avec LangGraph")
    
    def _initialize_agents(self):
        """Initialise les agents disponibles"""
        try:
            # Créer les instances d'agents
            self.agents[AgentType.TUTOR] = agent_manager.create_agent(
                AgentType.TUTOR, "main_tutor"
            )
            self.agents[AgentType.QUEST_GENERATOR] = agent_manager.create_agent(
                AgentType.QUEST_GENERATOR, "quest_creator"
            )
            self.agents[AgentType.CODE_EVALUATOR] = agent_manager.create_agent(
                AgentType.CODE_EVALUATOR, "code_assessor"
            )
            
            logger.info(f"Agents initialisés: {list(self.agents.keys())}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des agents: {e}")
            raise
    
    def _define_conversation_flows(self):
        """Définit les flux de conversation possibles"""
        
        # Flux principal: Chat avec le tuteur
        self.conversation_flows.append(ConversationFlow(
            name="main_chat",
            description="Conversation principale avec le tuteur",
            entry_conditions=[
                lambda state: state["conversation_state"] in ["initial", "chatting"],
                lambda state: not state["needs_code_evaluation"],
                lambda state: not state["needs_quest_generation"]
            ],
            states=[ConversationState.INITIAL, ConversationState.CHATTING],
            transitions={
                "initial": ["chatting", "quest_selection", "help_mode"],
                "chatting": ["chatting", "code_evaluation", "quest_selection", "completed"]
            },
            default_agent=AgentType.TUTOR,
            priority=1
        ))
        
        # Flux d'évaluation de code
        self.conversation_flows.append(ConversationFlow(
            name="code_evaluation",
            description="Évaluation et feedback sur le code",
            entry_conditions=[
                lambda state: state["needs_code_evaluation"],
                lambda state: "code" in state.get("temp_data", {}) or self._contains_code(state["current_query"])
            ],
            states=[ConversationState.CODE_EVALUATION, ConversationState.FEEDBACK_REVIEW],
            transitions={
                "code_evaluation": ["feedback_review", "chatting"],
                "feedback_review": ["chatting", "code_evaluation", "quest_selection"]
            },
            default_agent=AgentType.CODE_EVALUATOR,
            priority=3
        ))
        
        # Flux de génération de quête
        self.conversation_flows.append(ConversationFlow(
            name="quest_generation",
            description="Génération de nouvelles quêtes",
            entry_conditions=[
                lambda state: state["needs_quest_generation"],
                lambda state: state["conversation_state"] in ["quest_selection", "quest_generation"]
            ],
            states=[ConversationState.QUEST_GENERATION, ConversationState.QUEST_SELECTION],
            transitions={
                "quest_generation": ["quest_selection", "quest_active"],
                "quest_selection": ["quest_active", "chatting", "quest_generation"]
            },
            default_agent=AgentType.QUEST_GENERATOR,
            priority=2
        ))
        
        # Flux de quête active
        self.conversation_flows.append(ConversationFlow(
            name="active_quest",
            description="Quête en cours d'exécution",
            entry_conditions=[
                lambda state: state["current_quest_id"] is not None,
                lambda state: state["conversation_state"] == "quest_active"
            ],
            states=[ConversationState.QUEST_ACTIVE],
            transitions={
                "quest_active": ["quest_active", "code_evaluation", "feedback_review", "completed", "chatting"]
            },
            default_agent=AgentType.TUTOR,
            priority=4
        ))
        
        # Flux de révision des progrès
        self.conversation_flows.append(ConversationFlow(
            name="progress_review",
            description="Révision et analyse des progrès",
            entry_conditions=[
                lambda state: state["needs_progress_review"],
                lambda state: any(word in state["current_query"].lower() 
                               for word in ["progrès", "progress", "statistiques", "bilan"])
            ],
            states=[ConversationState.PROGRESS_REVIEW, ConversationState.LEARNING_PATH],
            transitions={
                "progress_review": ["learning_path", "chatting", "quest_selection"],
                "learning_path": ["quest_generation", "chatting"]
            },
            default_agent=AgentType.TUTOR,
            priority=2
        ))
    
    def _build_graph(self):
        """Construit le graphe LangGraph"""
        
        # Créer le graphe d'état
        workflow = StateGraph(WorkflowState)
        
        # Ajouter les nœuds pour chaque état
        workflow.add_node("router", self._route_conversation)
        workflow.add_node("tutor_chat", self._handle_tutor_interaction)
        workflow.add_node("code_evaluation", self._handle_code_evaluation)
        workflow.add_node("quest_generation", self._handle_quest_generation)
        workflow.add_node("quest_management", self._handle_quest_management)
        workflow.add_node("progress_review", self._handle_progress_review)
        workflow.add_node("error_handler", self._handle_error)
        workflow.add_node("session_manager", self._manage_session)
        
        # Définir le point d'entrée
        workflow.set_entry_point("router")
        
        # Ajouter les transitions conditionnelles
        workflow.add_conditional_edges(
            "router",
            self._determine_next_node,
            {
                "tutor_chat": "tutor_chat",
                "code_evaluation": "code_evaluation",
                "quest_generation": "quest_generation",
                "quest_management": "quest_management",
                "progress_review": "progress_review",
                "error_handler": "error_handler",
                "end": END
            }
        )
        
        # Transitions depuis chaque nœud vers le router ou la fin
        for node in ["tutor_chat", "code_evaluation", "quest_generation", 
                    "quest_management", "progress_review"]:
            workflow.add_conditional_edges(
                node,
                self._check_continuation,
                {
                    "continue": "router",
                    "session_manager": "session_manager",
                    "error": "error_handler",
                    "end": END
                }
            )
        
        # Transitions depuis error_handler et session_manager
        workflow.add_conditional_edges(
            "error_handler",
            lambda state: "continue" if not state["error_occurred"] else "end",
            {"continue": "router", "end": END}
        )
        
        workflow.add_conditional_edges(
            "session_manager",
            lambda state: "end",
            {"end": END}
        )
        
        # Compiler le graphe
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("Graphe LangGraph construit avec succès")
    
    async def start_session(self, user_id: Optional[int] = None,
                          initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Démarre une nouvelle session de conversation
        
        Args:
            user_id: ID de l'utilisateur (optionnel)
            initial_context: Contexte initial (optionnel)
            
        Returns:
            ID de session généré
        """
        session_id = generate_uuid()
        
        # Créer l'état initial
        initial_state = WorkflowState(
            user_id=user_id,
            session_id=session_id,
            conversation_state=ConversationState.INITIAL.value,
            messages=[],
            current_query="",
            last_response="",
            current_quest_id=None,
            current_step=0,
            user_level="beginner",
            learning_objectives=[],
            user_skills={},
            agent_responses={},
            last_agent_used=None,
            pending_actions=[],
            session_start_time=get_current_timestamp(),
            last_activity_time=get_current_timestamp(),
            total_interactions=0,
            needs_code_evaluation=False,
            needs_quest_generation=False,
            needs_progress_review=False,
            awaiting_user_input=True,
            error_occurred=False,
            temp_data=initial_context or {}
        )
        
        # Charger le contexte utilisateur si disponible
        if user_id:
            user_context = await self._load_user_context(user_id)
            initial_state.update(user_context)
        
        # Enregistrer la session
        self.active_sessions[session_id] = initial_state
        
        # Message de bienvenue
        welcome_message = await self._generate_welcome_message(initial_state)
        initial_state["last_response"] = welcome_message
        initial_state["messages"].append({
            "role": "assistant",
            "content": welcome_message,
            "type": "welcome",
            "timestamp": get_current_timestamp()
        })
        
        logger.info(f"Session {session_id} démarrée pour l'utilisateur {user_id}")
        return session_id
    
    async def process_user_input(self, session_id: str, user_input: str,
                                input_type: str = "text") -> Dict[str, Any]:
        """
        Traite une entrée utilisateur dans le contexte d'une session
        
        Args:
            session_id: ID de la session
            user_input: Entrée de l'utilisateur
            input_type: Type d'entrée (text, code, file, etc.)
            
        Returns:
            Réponse du système
        """
        if session_id not in self.active_sessions:
            return {
                "error": "Session non trouvée",
                "session_id": session_id
            }
        
        try:
            # Récupérer l'état actuel
            current_state = self.active_sessions[session_id].copy()
            
            # Mettre à jour avec la nouvelle entrée
            current_state["current_query"] = user_input
            current_state["last_activity_time"] = get_current_timestamp()
            current_state["total_interactions"] += 1
            current_state["awaiting_user_input"] = False
            
            # Ajouter le message utilisateur
            current_state["messages"].append({
                "role": "user",
                "content": user_input,
                "type": input_type,
                "timestamp": get_current_timestamp()
            })
            
            # Analyser l'entrée pour déterminer les actions nécessaires
            await self._analyze_user_input(current_state, user_input, input_type)
            
            # Traiter avec le graphe LangGraph
            config = {"configurable": {"thread_id": session_id}}
            
            result = await self.graph.ainvoke(current_state, config)
            
            # Mettre à jour la session
            self.active_sessions[session_id] = result
            
            # Préparer la réponse
            response = {
                "session_id": session_id,
                "response": result.get("last_response", ""),
                "conversation_state": result.get("conversation_state"),
                "suggestions": result.get("temp_data", {}).get("suggestions", []),
                "next_actions": result.get("pending_actions", []),
                "quest_status": self._get_quest_status(result),
                "user_progress": await self._get_user_progress_summary(result.get("user_id")),
                "user_skills": result.get("user_skills", {}),
                "metadata": {
                    "agent_used": result.get("last_agent_used"),
                    "total_interactions": result.get("total_interactions", 0),
                    "session_duration": self._calculate_session_duration(result)
                }
            }
            
            # Sauvegarder périodiquement
            if result["total_interactions"] % self.auto_save_frequency == 0:
                await self._save_session_state(session_id, result)
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'entrée: {e}")
            return {
                "error": f"Erreur de traitement: {str(e)}",
                "session_id": session_id
            }
    
    async def _analyze_user_input(self, state: WorkflowState, user_input: str, input_type: str):
        """Analyse l'entrée utilisateur pour déterminer les actions nécessaires"""
        
        query_lower = user_input.lower()
        
        # Détection de code
        if input_type == "code" or self._contains_code(user_input):
            state["needs_code_evaluation"] = True
            state["temp_data"]["code"] = self._extract_code_from_query(user_input)
        
        # Détection de demande de quête
        quest_keywords = ["quête", "quest", "exercice", "défi", "challenge", "mission"]
        if any(keyword in query_lower for keyword in quest_keywords):
            state["needs_quest_generation"] = True
        
        # Détection de demande de progrès
        progress_keywords = ["progrès", "progress", "statistiques", "bilan", "niveau", "compétences"]
        if any(keyword in query_lower for keyword in progress_keywords):
            state["needs_progress_review"] = True
        
        # Détection d'aide
        help_keywords = ["aide", "help", "indice", "hint", "comment", "expliquer"]
        if any(keyword in query_lower for keyword in help_keywords):
            state["temp_data"]["needs_help"] = True
        
        # Analyse du sentiment et de l'intention
        state["temp_data"]["user_sentiment"] = self._analyze_sentiment(user_input)
        state["temp_data"]["query_intent"] = self._detect_intent(user_input)
    
    def _contains_code(self, text: str) -> bool:
        """Vérifie si le texte contient du code"""
        return any(re.search(pattern, text, re.MULTILINE) for pattern in self.code_patterns)
    
    def _extract_code_from_query(self, query: str) -> str:
        """Extrait le code d'une requête"""
        # Recherche de blocs de code
        code_block_match = re.search(r'```(?:python)?\s*([\s\S]*?)```', query)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Recherche de code inline
        inline_code_matches = re.findall(r'`([^`]+)`', query)
        if inline_code_matches:
            return '\n'.join(inline_code_matches)
        
        # Recherche de patterns Python
        for pattern in self.code_patterns[2:]:  # Skip les patterns de markdown
            match = re.search(pattern, query, re.MULTILINE)
            if match:
                # Extraire quelques lignes autour du match
                lines = query.split('\n')
                match_line = next((i for i, line in enumerate(lines) if pattern in line), -1)
                if match_line != -1:
                    start = max(0, match_line - 2)
                    end = min(len(lines), match_line + 3)
                    return '\n'.join(lines[start:end])
        
        return ""
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyse simple du sentiment"""
        positive_words = ["merci", "super", "génial", "parfait", "excellent", "bien"]
        negative_words = ["problème", "erreur", "bug", "difficile", "compliqué", "aide"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _detect_intent(self, text: str) -> str:
        """Détecte l'intention de l'utilisateur"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["qu'est-ce", "comment", "pourquoi", "expliquer"]):
            return "question"
        elif any(word in text_lower for word in ["faire", "créer", "générer", "montrer"]):
            return "request"
        elif any(word in text_lower for word in ["erreur", "problème", "bug", "aide"]):
            return "help"
        elif any(word in text_lower for word in ["merci", "ok", "d'accord"]):
            return "acknowledgment"
        else:
            return "general"
    
    async def _route_conversation(self, state: WorkflowState) -> WorkflowState:
        """Node router - Détermine le flux de conversation approprié"""
        
        # Analyser l'état actuel pour déterminer le meilleur flux
        suitable_flows = []
        
        for flow in self.conversation_flows:
            try:
                if all(condition(state) for condition in flow.entry_conditions):
                    suitable_flows.append(flow)
            except Exception as e:
                logger.warning(f"Erreur lors de l'évaluation du flux {flow.name}: {e}")
        
        # Sélectionner le flux le plus prioritaire
        if suitable_flows:
            selected_flow = max(suitable_flows, key=lambda f: f.priority)
            state["temp_data"]["selected_flow"] = selected_flow.name
            state["temp_data"]["flow_agent"] = selected_flow.default_agent.value
            
            logger.debug(f"Flux sélectionné: {selected_flow.name}")
        else:
            # Flux par défaut: chat avec le tuteur
            state["temp_data"]["selected_flow"] = "main_chat"
            state["temp_data"]["flow_agent"] = AgentType.TUTOR.value
        
        return state
    
    def _determine_next_node(self, state: WorkflowState) -> str:
        """Détermine le prochain nœud à exécuter"""
        
        if state["error_occurred"]:
            return "error_handler"
        
        # Vérifier si la session doit se terminer
        if self._should_end_session(state):
            return "end"
        
        # Router selon le flux sélectionné et les besoins
        if state["needs_code_evaluation"]:
            return "code_evaluation"
        elif state["needs_quest_generation"]:
            return "quest_generation"
        elif state["needs_progress_review"]:
            return "progress_review"
        elif state["current_quest_id"]:
            return "quest_management"
        else:
            return "tutor_chat"
    
    async def _handle_tutor_interaction(self, state: WorkflowState) -> WorkflowState:
        """Node tuteur - Gère l'interaction avec l'agent tuteur"""
        
        try:
            # Préparer le contexte pour le tuteur
            context = await self._build_agent_context(state)
            
            # Déterminer le sujet principal
            subject = self._detect_subject(state["current_query"])
            
            # Préparer les données d'entrée
            input_data = {
                "question": state["current_query"],
                "subject": subject,
                "difficulty": self._determine_difficulty(state),
                "context": {
                    "user_level": state["user_level"],
                    "recent_interactions": state["messages"][-5:] if state["messages"] else [],
                    "user_skills": state["user_skills"],
                    "learning_objectives": state["learning_objectives"]
                }
            }
            
            # Ajouter contexte d'aide si nécessaire
            if state.get("temp_data", {}).get("needs_help"):
                input_data["help_mode"] = True
            
            # Appeler l'agent tuteur
            tutor_agent = self.agents[AgentType.TUTOR]
            response = await tutor_agent.execute(input_data, context)
            
            # Mettre à jour l'état
            state["last_response"] = response.message
            state["last_agent_used"] = AgentType.TUTOR.value
            state["agent_responses"][AgentType.TUTOR.value] = response.to_dict()
            state["conversation_state"] = ConversationState.CHATTING.value
            
            # Ajouter la réponse aux messages
            state["messages"].append({
                "role": "assistant",
                "content": response.message,
                "agent": AgentType.TUTOR.value,
                "timestamp": get_current_timestamp(),
                "confidence": response.confidence,
                "subject": subject
            })
            
            # Traiter les suggestions
            if response.suggestions:
                state["temp_data"]["suggestions"] = response.suggestions
            
            # Traiter les actions suivantes
            if response.next_actions:
                state["pending_actions"].extend([
                    action.get("action", "") for action in response.next_actions
                ])
            
            # Mettre à jour les compétences si des indices sont présents
            if response.data and "skill_updates" in response.data:
                state["user_skills"].update(response.data["skill_updates"])
            
            state["awaiting_user_input"] = True
            
        except Exception as e:
            logger.error(f"Erreur dans l'interaction tuteur: {e}")
            state["error_occurred"] = True
            state["temp_data"]["error_message"] = str(e)
        
        return state
    
    def _detect_subject(self, query: str) -> str:
        """Détecte le sujet principal de la requête"""
        query_lower = query.lower()
        
        subjects = {
            "python": ["python", "variable", "fonction", "classe", "liste", "dictionnaire"],
            "javascript": ["javascript", "js", "node", "react", "html", "css"],
            "data_science": ["data", "pandas", "numpy", "matplotlib", "analyse"],
            "web": ["web", "html", "css", "javascript", "serveur"],
            "algorithms": ["algorithme", "tri", "recherche", "complexité"],
            "general": []
        }
        
        for subject, keywords in subjects.items():
            if any(keyword in query_lower for keyword in keywords):
                return subject
        
        return "python"  # Défaut
    
    def _determine_difficulty(self, state: WorkflowState) -> str:
        """Détermine le niveau de difficulté approprié"""
        user_level = state["user_level"]
        user_skills = state["user_skills"]
        
        if user_level == "beginner" and not user_skills:
            return "easy"
        elif user_level == "intermediate" or any(skill > 0.5 for skill in user_skills.values()):
            return "medium"
        elif user_level == "advanced" or any(skill > 0.8 for skill in user_skills.values()):
            return "hard"
        else:
            return "auto"
    
    async def _handle_code_evaluation(self, state: WorkflowState) -> WorkflowState:
        """Node évaluation - Gère l'évaluation de code"""
        
        try:
            # Récupérer le code à évaluer
            code = state.get("temp_data", {}).get("code")
            if not code:
                code = self._extract_code_from_query(state["current_query"])
            
            if not code:
                state["last_response"] = "Je n'ai pas trouvé de code à évaluer. Pouvez-vous partager votre code entre des balises ```python et ``` ?"
                state["needs_code_evaluation"] = False
                return state
            
            # Préparer le contexte
            context = await self._build_agent_context(state)
            
            # Préparer les données d'entrée
            input_data = {
                "code": code,
                "exercise_type": state.get("temp_data", {}).get("exercise_type", "general"),
                "context": {
                    "user_level": state["user_level"],
                    "user_skills": state["user_skills"],
                    "current_quest": state.get("current_quest_id")
                }
            }
            
            # Ajouter les cas de test s'ils existent
            if "test_cases" in state.get("temp_data", {}):
                input_data["test_cases"] = state["temp_data"]["test_cases"]
            
            # Appeler l'agent évaluateur
            evaluator_agent = self.agents[AgentType.CODE_EVALUATOR]
            response = await evaluator_agent.execute(input_data, context)
            
            # Mettre à jour l'état
            state["last_response"] = response.message
            state["last_agent_used"] = AgentType.CODE_EVALUATOR.value
            state["agent_responses"][AgentType.CODE_EVALUATOR.value] = response.to_dict()
            state["conversation_state"] = ConversationState.CODE_EVALUATION.value
            state["needs_code_evaluation"] = False
            
            # Stocker les résultats détaillés
            if response.data:
                state["temp_data"]["evaluation_results"] = response.data
                state["temp_data"]["code_score"] = response.data.get("overall_score", 0)
                
                # Mettre à jour les compétences basées sur l'évaluation
                skill_updates = response.data.get("skill_assessment", {})
                for skill, level in skill_updates.items():
                    current_level = state["user_skills"].get(skill, 0)
                    # Moyenne pondérée pour une progression graduelle
                    new_level = (current_level * 0.7) + (level * 0.3)
                    state["user_skills"][skill] = min(1.0, new_level)
            
            # Ajouter la réponse aux messages
            state["messages"].append({
                "role": "assistant", 
                "content": response.message,
                "agent": AgentType.CODE_EVALUATOR.value,
                "timestamp": get_current_timestamp(),
                "data": response.data,
                "code_evaluated": code[:100] + "..." if len(code) > 100 else code
            })
            
            # Suggestions d'amélioration
            if response.suggestions:
                state["temp_data"]["suggestions"] = response.suggestions
            
            # Nettoyer les données temporaires
            state["temp_data"].pop("code", None)
            state["awaiting_user_input"] = True
            
        except Exception as e:
            logger.error(f"Erreur dans l'évaluation de code: {e}")
            state["error_occurred"] = True
            state["temp_data"]["error_message"] = str(e)
        
        return state
    
    async def _handle_quest_generation(self, state: WorkflowState) -> WorkflowState:
        """Node génération - Gère la génération de quêtes"""
        
        try:
            # Analyser la requête pour extraire les paramètres
            quest_params = self._extract_quest_parameters(state["current_query"], state)
            
            # Préparer le contexte
            context = await self._build_agent_context(state)
            
            # Préparer les données d'entrée
            input_data = {
                "category": quest_params.get("category", "python"),
                "difficulty": quest_params.get("difficulty", self._determine_difficulty(state)),
                "topic": quest_params.get("topic", ""),
                "type": quest_params.get("type", "coding"),
                "context": {
                    "user_level": state["user_level"],
                    "user_skills": state["user_skills"],
                    "learning_objectives": state["learning_objectives"],
                    "completed_quests": await self._get_completed_quests(state["user_id"])
                }
            }
            
            # Appeler l'agent générateur
            generator_agent = self.agents[AgentType.QUEST_GENERATOR]
            response = await generator_agent.execute(input_data, context)
            
            # Mettre à jour l'état
            state["last_response"] = response.message
            state["last_agent_used"] = AgentType.QUEST_GENERATOR.value
            state["agent_responses"][AgentType.QUEST_GENERATOR.value] = response.to_dict()
            state["conversation_state"] = ConversationState.QUEST_GENERATION.value
            state["needs_quest_generation"] = False
            
            # Stocker la quête générée
            if response.data and response.data.get("quest_id"):
                state["temp_data"]["generated_quest_id"] = response.data["quest_id"]
                state["temp_data"]["quest_data"] = response.data
                
                # Proposer d'activer la quête
                state["temp_data"]["suggestions"] = [
                    {
                        "text": "Commencer cette quête",
                        "action": "start_quest",
                        "quest_id": response.data["quest_id"]
                    },
                    {
                        "text": "Générer une autre quête",
                        "action": "generate_quest"
                    },
                    {
                        "text": "Retour au chat",
                        "action": "return_chat"
                    }
                ]
            
            # Ajouter la réponse aux messages
            state["messages"].append({
                "role": "assistant",
                "content": response.message, 
                "agent": AgentType.QUEST_GENERATOR.value,
                "timestamp": get_current_timestamp(),
                "data": response.data
            })
            
            state["awaiting_user_input"] = True
            
        except Exception as e:
            logger.error(f"Erreur dans la génération de quête: {e}")
            state["error_occurred"] = True
            state["temp_data"]["error_message"] = str(e)
        
        return state
    
    def _extract_quest_parameters(self, query: str, state: WorkflowState) -> Dict[str, Any]:
        """Extrait les paramètres de quête de la requête utilisateur"""
        params = {}
        query_lower = query.lower()
        
        # Détection de catégorie
        categories = {
            "python": ["python", "programmation", "code"],
            "web": ["web", "html", "css", "javascript"],
            "data": ["data", "données", "analyse", "pandas"],
            "algorithms": ["algorithme", "tri", "recherche"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                params["category"] = category
                break
        
        # Détection de difficulté
        if any(word in query_lower for word in ["facile", "débutant", "simple"]):
            params["difficulty"] = "easy"
        elif any(word in query_lower for word in ["difficile", "avancé", "complexe"]):
            params["difficulty"] = "hard"
        elif any(word in query_lower for word in ["moyen", "intermédiaire"]):
            params["difficulty"] = "medium"
        
        # Détection de type
        if any(word in query_lower for word in ["quiz", "question", "qcm"]):
            params["type"] = "quiz"
        elif any(word in query_lower for word in ["projet", "application"]):
            params["type"] = "project"
        else:
            params["type"] = "coding"
        
        # Extraction du sujet spécifique
        topics = ["fonction", "classe", "liste", "dictionnaire", "boucle", "condition"]
        for topic in topics:
            if topic in query_lower:
                params["topic"] = topic
                break
        
        return params
    
    async def _handle_quest_management(self, state: WorkflowState) -> WorkflowState:
        """Node gestion quête - Gère les quêtes actives"""
        
        try:
            quest_id = state["current_quest_id"]
            if not quest_id:
                state["conversation_state"] = ConversationState.CHATTING.value
                return state
            
            # Charger les informations de la quête
            quest_info = await self._load_quest_info(quest_id, state["user_id"])
            
            if not quest_info:
                state["current_quest_id"] = None
                state["conversation_state"] = ConversationState.CHATTING.value
                state["last_response"] = "Quête non trouvée. Retour au mode chat normal."
                return state
            
            # Déterminer l'action selon la requête
            query_lower = state["current_query"].lower()
            response_message = ""
            
            if any(word in query_lower for word in ["aide", "indice", "hint", "help"]):
                # Fournir une aide
                response_message = self._generate_quest_hint(quest_info, state["current_step"])
                
            elif any(word in query_lower for word in ["suivant", "next", "continuer"]):
                # Passer à l'étape suivante
                response_message = await self._advance_quest_step(state, quest_info)
                
            elif any(word in query_lower for word in ["quitter", "stop", "abandonner"]):
                # Quitter la quête
                state["current_quest_id"] = None
                state["current_step"] = 0
                state["conversation_state"] = ConversationState.CHATTING.value
                response_message = "Quête abandonnée. Vous pouvez en commencer une nouvelle quand vous voulez !"
                
            elif any(word in query_lower for word in ["statut", "état", "progression"]):
                # Afficher le statut de la quête
                response_message = self._get_quest_progress_message(quest_info, state["current_step"])
                
            else:
                # Interaction normale avec le tuteur dans le contexte de la quête
                context = await self._build_agent_context(state)
                context.current_quest = quest_info
                
                input_data = {
                    "question": state["current_query"],
                    "subject": quest_info.get("category", "python"),
                    "difficulty": quest_info.get("difficulty", "auto"),
                    "context": {
                        "quest_context": quest_info,
                        "current_step": state["current_step"],
                        "user_skills": state["user_skills"]
                    }
                }
                
                tutor_agent = self.agents[AgentType.TUTOR]
                response = await tutor_agent.execute(input_data, context)
                response_message = response.message
                
                # Vérifier si l'utilisateur a fourni une solution
                if self._contains_code(state["current_query"]):
                    state["needs_code_evaluation"] = True
                    state["temp_data"]["code"] = self._extract_code_from_query(state["current_query"])
                    state["temp_data"]["quest_context"] = quest_info
            
            # Mettre à jour l'état
            state["last_response"] = response_message
            state["conversation_state"] = ConversationState.QUEST_ACTIVE.value
            
            # Ajouter la réponse aux messages
            state["messages"].append({
                "role": "assistant",
                "content": response_message,
                "context": "quest_management",
                "quest_id": quest_id,
                "quest_step": state["current_step"],
                "timestamp": get_current_timestamp()
            })
            
            state["awaiting_user_input"] = True
            
        except Exception as e:
            logger.error(f"Erreur dans la gestion de quête: {e}")
            state["error_occurred"] = True
            state["temp_data"]["error_message"] = str(e)
        
        return state
    
    async def _handle_progress_review(self, state: WorkflowState) -> WorkflowState:
        """Node révision progrès - Analyse et présente les progrès de l'utilisateur"""
        
        try:
            # Préparer le contexte
            context = await self._build_agent_context(state)
            
            # Récupérer les données de progrès
            progress_data = await self._compile_progress_data(state["user_id"], state)
            
            # Préparer les données d'entrée pour le tuteur
            input_data = {
                "question": "Analyse mes progrès et donne-moi des recommandations",
                "subject": "progress_analysis",
                "difficulty": "auto",
                "context": {
                    "progress_data": progress_data,
                    "user_skills": state["user_skills"],
                    "learning_objectives": state["learning_objectives"],
                    "session_stats": {
                        "total_interactions": state["total_interactions"],
                        "session_duration": self._calculate_session_duration(state)
                    }
                }
            }
            
            # Utiliser le tuteur pour analyser les progrès
            tutor_agent = self.agents[AgentType.TUTOR]
            response = await tutor_agent.execute(input_data, context)
            
            # Ajouter des recommandations spécifiques
            recommendations = self._generate_learning_recommendations(progress_data, state["user_skills"])
            
            enhanced_message = f"{response.message}\n\n🎯 **Recommandations personnalisées:**\n"
            for i, rec in enumerate(recommendations[:3], 1):
                enhanced_message += f"{i}. {rec}\n"
            
            # Mettre à jour l'état
            state["last_response"] = enhanced_message
            state["last_agent_used"] = AgentType.TUTOR.value
            state["conversation_state"] = ConversationState.PROGRESS_REVIEW.value
            state["needs_progress_review"] = False
            
            # Stocker les données de progrès
            state["temp_data"]["progress_analysis"] = progress_data
            state["temp_data"]["recommendations"] = recommendations
            
            # Suggestions d'actions
            state["temp_data"]["suggestions"] = [
                {
                    "text": "Créer un plan d'apprentissage personnalisé",
                    "action": "create_learning_path"
                },
                {
                    "text": "Commencer une quête adaptée à mon niveau",
                    "action": "generate_quest"
                },
                {
                    "text": "Voir mes statistiques détaillées",
                    "action": "detailed_stats"
                }
            ]
            
            # Ajouter la réponse aux messages
            state["messages"].append({
                "role": "assistant",
                "content": enhanced_message,
                "agent": "progress_reviewer",
                "timestamp": get_current_timestamp(),
                "data": progress_data
            })
            
            state["awaiting_user_input"] = True
            
        except Exception as e:
            logger.error(f"Erreur dans la révision des progrès: {e}")
            state["error_occurred"] = True
            state["temp_data"]["error_message"] = str(e)
        
        return state
    
    async def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Node erreur - Gère les erreurs du système"""
        
        error_message = state.get("temp_data", {}).get("error_message", "Une erreur inconnue s'est produite")
        
        # Analyser le type d'erreur pour une réponse appropriée
        if "timeout" in error_message.lower():
            user_message = "Désolé, le traitement prend plus de temps que prévu. Pouvez-vous réessayer ?"
        elif "connection" in error_message.lower():
            user_message = "Problème de connexion détecté. Vérifiez votre connexion et réessayez."
        elif "not found" in error_message.lower():
            user_message = "L'élément demandé n'a pas été trouvé. Pouvez-vous vérifier votre demande ?"
        else:
            user_message = "Désolé, j'ai rencontré un problème technique. Pouvez-vous reformuler votre demande ?"
        
        # Logger l'erreur détaillée
        logger.error(f"Erreur dans la session {state['session_id']}: {error_message}")
        
        # Réinitialiser les flags d'erreur et d'état
        state["error_occurred"] = False
        state["temp_data"].pop("error_message", None)
        
        # Réinitialiser les flags de besoin
        state["needs_code_evaluation"] = False
        state["needs_quest_generation"] = False
        state["needs_progress_review"] = False
        
        # Revenir à un état stable
        state["conversation_state"] = ConversationState.CHATTING.value
        state["last_response"] = user_message
        
        # Ajouter le message d'erreur
        state["messages"].append({
            "role": "assistant",
            "content": user_message,
            "type": "error_recovery",
            "timestamp": get_current_timestamp()
        })
        
        # Proposer des actions de récupération
        state["temp_data"]["suggestions"] = [
            {
                "text": "Recommencer",
                "action": "restart"
            },
            {
                "text": "Poser une nouvelle question",
                "action": "new_question"
            },
            {
                "text": "Voir l'aide",
                "action": "help"
            }
        ]
        
        state["awaiting_user_input"] = True
        
        return state
    
    async def _manage_session(self, state: WorkflowState) -> WorkflowState:
        """Node session - Gère le cycle de vie des sessions"""
        
        # Calculer les statistiques de session
        session_stats = {
            "duration": self._calculate_session_duration(state),
            "interactions": state["total_interactions"],
            "agents_used": list(set(state["agent_responses"].keys())),
            "skills_practiced": list(state["user_skills"].keys()),
            "quests_completed": state.get("temp_data", {}).get("quests_completed", 0)
        }
        
        # Sauvegarder l'état final
        await self._save_session_state(state["session_id"], state)
        
        # Mettre à jour les statistiques utilisateur
        if state["user_id"]:
            await self._update_user_statistics(state, session_stats)
        
        # Générer un message de fin de session
        end_message = self._generate_session_summary(session_stats)
        state["last_response"] = end_message
        
        # Ajouter le message de fin
        state["messages"].append({
            "role": "assistant",
            "content": end_message,
            "type": "session_end",
            "timestamp": get_current_timestamp(),
            "session_stats": session_stats
        })
        
        # Nettoyer les données temporaires
        state["temp_data"] = {"session_stats": session_stats}
        state["conversation_state"] = ConversationState.COMPLETED.value
        
        logger.info(f"Session {state['session_id']} terminée après {state['total_interactions']} interactions")
        
        return state
    
    def _check_continuation(self, state: WorkflowState) -> str:
        """Vérifie si la conversation doit continuer"""
        
        if state["error_occurred"]:
            return "error"
        
        if self._should_end_session(state):
            return "session_manager"
        
        if state["awaiting_user_input"]:
            return "end"  # Attendre la prochaine entrée utilisateur
        
        return "continue"
    
    def _should_end_session(self, state: WorkflowState) -> bool:
        """Détermine si la session doit se terminer"""
        
        # Vérifier la durée de session
        start_time = datetime.fromisoformat(state["session_start_time"].replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        session_duration = (current_time - start_time).total_seconds()
        
        if session_duration > self.max_session_duration:
            return True
        
        # Vérifier le nombre d'interactions
        if state["total_interactions"] >= self.max_interactions_per_session:
            return True
        
        # Vérifier les indicateurs explicites de fin
        if state["conversation_state"] == ConversationState.COMPLETED.value:
            return True
        
        # Vérifier les mots-clés de fin dans la requête
        if state["current_query"]:
            end_keywords = ["au revoir", "bye", "quit", "exit", "fin", "stop"]
            if any(keyword in state["current_query"].lower() for keyword in end_keywords):
                return True
        
        return False
    
    # Méthodes utilitaires
    
    async def _build_agent_context(self, state: WorkflowState) -> AgentContext:
        """Construit le contexte pour les agents"""
        return AgentContext(
            user_id=state["user_id"],
            session_id=state["session_id"],
            conversation_history=state["messages"],
            current_quest=await self._load_quest_info(state["current_quest_id"], state["user_id"]) if state["current_quest_id"] else None,
            user_progress=await self._get_user_progress_summary(state["user_id"]),
            metadata={
                "user_level": state["user_level"],
                "user_skills": state["user_skills"],
                "learning_objectives": state["learning_objectives"],
                "session_duration": self._calculate_session_duration(state)
            }
        )
    
    async def _load_user_context(self, user_id: int) -> Dict[str, Any]:
        """Charge le contexte utilisateur depuis la base de données"""
        try:
            async with get_db_session() as session:
                # Charger l'utilisateur
                user = await session.get(User, user_id)
                if not user:
                    return {}
                
                # Charger les progrès
                progress = await session.execute(
                    "SELECT * FROM user_progress WHERE user_id = ?", (user_id,)
                )
                progress_data = progress.fetchall()
                
                # Compiler les compétences
                user_skills = {}
                for row in progress_data:
                    if row.skill_name and row.skill_level:
                        user_skills[row.skill_name] = row.skill_level
                
                return {
                    "user_level": user.level or "beginner",
                    "learning_objectives": user.learning_objectives or [],
                    "user_skills": user_skills
                }
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du contexte utilisateur: {e}")
            return {}
    
    async def _load_quest_info(self, quest_id: Optional[int], user_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """Charge les informations d'une quête"""
        if not quest_id:
            return None
            
        try:
            async with get_db_session() as session:
                # Charger la quête
                quest = await session.get(UserQuest, quest_id)
                if not quest:
                    return None
                
                return {
                    "id": quest.id,
                    "title": quest.title,
                    "description": quest.description,
                    "category": quest.category,
                    "difficulty": quest.difficulty,
                    "steps": quest.steps or [],
                    "current_step": quest.current_step or 0,
                    "status": quest.status,
                    "created_at": quest.created_at
                }
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la quête: {e}")
            return None
    
    def _generate_quest_hint(self, quest_info: Dict[str, Any], current_step: int) -> str:
        """Génère un indice pour la quête actuelle"""
        steps = quest_info.get("steps", [])
        if current_step < len(steps):
            step = steps[current_step]
            hint = step.get("hint", "Réfléchissez à la logique nécessaire pour cette étape.")
            return f"💡 **Indice pour l'étape {current_step + 1}:** {hint}"
        else:
            return "Vous avez terminé toutes les étapes disponibles ! Félicitations !"
    
    async def _advance_quest_step(self, state: WorkflowState, quest_info: Dict[str, Any]) -> str:
        """Avance à l'étape suivante de la quête"""
        steps = quest_info.get("steps", [])
        current_step = state["current_step"]
        
        if current_step < len(steps) - 1:
            state["current_step"] += 1
            next_step = steps[state["current_step"]]
            return f"🎯 **Étape {state['current_step'] + 1}:** {next_step.get('description', 'Nouvelle étape disponible!')}"
        else:
            # Quête terminée
            state["current_quest_id"] = None
            state["current_step"] = 0
            state["conversation_state"] = ConversationState.CHATTING.value
            
            # Mettre à jour les compétences
            if quest_info.get("skills_reward"):
                for skill, bonus in quest_info["skills_reward"].items():
                    current = state["user_skills"].get(skill, 0)
                    state["user_skills"][skill] = min(1.0, current + bonus)
            
            return "🎉 **Félicitations !** Vous avez terminé cette quête ! Vos compétences ont été mises à jour."
    
    def _get_quest_progress_message(self, quest_info: Dict[str, Any], current_step: int) -> str:
        """Génère un message de progression de quête"""
        steps = quest_info.get("steps", [])
        total_steps = len(steps)
        progress_percent = int((current_step / total_steps) * 100) if total_steps > 0 else 0
        
        return f"""📊 **Progression de la quête: {quest_info['title']}**
        
🎯 Étape actuelle: {current_step + 1}/{total_steps}
📈 Progression: {progress_percent}%
🏷️ Difficulté: {quest_info.get('difficulty', 'N/A')}
📝 Description: {quest_info.get('description', 'Aucune description')}

Tapez 'suivant' pour continuer ou 'aide' pour un indice."""
    
    async def _compile_progress_data(self, user_id: Optional[int], state: WorkflowState) -> Dict[str, Any]:
        """Compile les données de progrès de l'utilisateur"""
        progress = {
            "skills": state["user_skills"],
            "session_stats": {
                "interactions": state["total_interactions"],
                "duration": self._calculate_session_duration(state),
                "agents_used": len(set(state["agent_responses"].keys()))
            },
            "learning_objectives": state["learning_objectives"],
            "conversation_summary": {
                "total_messages": len(state["messages"]),
                "code_evaluations": len([m for m in state["messages"] if m.get("agent") == "CODE_EVALUATOR"]),
                "questions_asked": len([m for m in state["messages"] if m.get("role") == "user"])
            }
        }
        
        if user_id:
            try:
                async with get_db_session() as session:
                    # Charger statistiques supplémentaires depuis la DB
                    quests = await session.execute(
                        "SELECT COUNT(*) as total, SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed FROM user_quests WHERE user_id = ?",
                        (user_id,)
                    )
                    quest_stats = quests.fetchone()
                    
                    progress["quest_stats"] = {
                        "total_quests": quest_stats[0] if quest_stats else 0,
                        "completed_quests": quest_stats[1] if quest_stats else 0
                    }
                    
            except Exception as e:
                logger.error(f"Erreur lors de la compilation des progrès: {e}")
        
        return progress
    
    def _generate_learning_recommendations(self, progress_data: Dict[str, Any], skills: Dict[str, float]) -> List[str]:
        """Génère des recommandations d'apprentissage personnalisées"""
        recommendations = []
        
        # Analyser les compétences
        if not skills:
            recommendations.append("Commencer par les bases de Python avec une quête débutant")
        else:
            # Trouver les compétences faibles
            weak_skills = {k: v for k, v in skills.items() if v < 0.3}
            strong_skills = {k: v for k, v in skills.items() if v > 0.7}
            
            if weak_skills:
                skill_name = list(weak_skills.keys())[0]
                recommendations.append(f"Renforcer vos bases en {skill_name} avec des exercices ciblés")
            
            if strong_skills:
                skill_name = list(strong_skills.keys())[0]
                recommendations.append(f"Approfondir {skill_name} avec des défis plus avancés")
            
            # Recommandations selon les stats
            if progress_data.get("conversation_summary", {}).get("code_evaluations", 0) < 3:
                recommendations.append("Pratiquer l'écriture de code avec plus d'exercices")
        
        # Recommandations générales
        session_interactions = progress_data.get("session_stats", {}).get("interactions", 0)
        if session_interactions > 20:
            recommendations.append("Prendre une pause et revenir plus tard pour consolider")
        elif session_interactions < 5:
            recommendations.append("Explorer davantage les fonctionnalités disponibles")
        
        return recommendations[:5]  # Limiter à 5 recommandations
    
    def _calculate_session_duration(self, state: WorkflowState) -> int:
        """Calcule la durée de session en secondes"""
        try:
            start_time = datetime.fromisoformat(state["session_start_time"].replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            return int((current_time - start_time).total_seconds())
        except:
            return 0
    
    def _generate_session_summary(self, session_stats: Dict[str, Any]) -> str:
        """Génère un résumé de fin de session"""
        duration_mins = session_stats["duration"] // 60
        
        return f"""🎓 **Résumé de votre session d'apprentissage**
        
⏱️ Durée: {duration_mins} minutes
💬 Interactions: {session_stats["interactions"]}
🤖 Agents utilisés: {', '.join(session_stats["agents_used"])}
🎯 Compétences pratiquées: {', '.join(session_stats["skills_practiced"])}
🏆 Quêtes complétées: {session_stats["quests_completed"]}

Merci d'avoir utilisé l'assistant pédagogique ! À bientôt ! 👋"""
    
    async def _generate_welcome_message(self, state: WorkflowState) -> str:
        """Génère un message de bienvenue personnalisé"""
        if state["user_id"]:
            # Utilisateur connu
            user_context = await self._load_user_context(state["user_id"])
            level = user_context.get("user_level", "débutant")
            skills = user_context.get("user_skills", {})
            
            if skills:
                skill_summary = f"Vos compétences actuelles: {', '.join(skills.keys())}"
            else:
                skill_summary = "Prêt à commencer votre apprentissage"
            
            return f"""👋 **Ravi de vous revoir !**
            
📚 Niveau: {level}
🎯 {skill_summary}

Que souhaitez-vous apprendre aujourd'hui ? Je peux vous aider avec :
• Des explications de concepts
• L'évaluation de votre code
• La génération d'exercices personnalisés
• Le suivi de vos progrès

Posez-moi une question ou dites-moi ce que vous aimeriez faire !"""
        else:
            # Nouvel utilisateur
            return """👋 **Bienvenue dans votre assistant pédagogique !**
            
Je suis là pour vous accompagner dans votre apprentissage de la programmation. Voici ce que je peux faire pour vous :

🎓 **Tuteur personnalisé** - Répondre à vos questions et expliquer les concepts
🔍 **Évaluateur de code** - Analyser votre code et vous donner des conseils
🎮 **Générateur de quêtes** - Créer des exercices adaptés à votre niveau
📈 **Suivi des progrès** - Vous aider à voir votre évolution

Pour commencer, vous pouvez :
• Me poser une question sur la programmation
• Me montrer du code pour que je l'évalue
• Me demander de créer un exercice pour vous

Qu'est-ce qui vous intéresse le plus ?"""
    
    def _get_quest_status(self, state: WorkflowState) -> Optional[Dict[str, Any]]:
        """Retourne le statut de la quête active"""
        if not state["current_quest_id"]:
            return None
            
        return {
            "quest_id": state["current_quest_id"],
            "current_step": state["current_step"],
            "status": "active"
        }
    
    async def _get_user_progress_summary(self, user_id: Optional[int]) -> Dict[str, Any]:
        """Retourne un résumé des progrès utilisateur"""
        if not user_id:
            return {"total_xp": 0, "level": "beginner", "badges": []}
            
        try:
            async with get_db_session() as session:
                # Charger les données de base
                user = await session.get(User, user_id)
                if not user:
                    return {"total_xp": 0, "level": "beginner", "badges": []}
                
                # Calculer XP total et niveau
                progress_query = await session.execute(
                    "SELECT SUM(xp_gained) as total_xp FROM user_progress WHERE user_id = ?",
                    (user_id,)
                )
                total_xp = progress_query.fetchone()[0] or 0
                
                # Déterminer le niveau basé sur l'XP
                level = "beginner"
                if total_xp > 1000:
                    level = "intermediate"
                if total_xp > 5000:
                    level = "advanced"
                if total_xp > 10000:
                    level = "expert"
                
                # Charger les badges
                badges_query = await session.execute(
                    "SELECT badge_name FROM user_badges WHERE user_id = ?",
                    (user_id,)
                )
                badges = [row[0] for row in badges_query.fetchall()]
                
                return {
                    "total_xp": total_xp,
                    "level": level,
                    "badges": badges,
                    "quests_completed": len([b for b in badges if "quest" in b.lower()])
                }
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul des progrès: {e}")
            return {"total_xp": 0, "level": "beginner", "badges": []}
    
    async def _get_completed_quests(self, user_id: Optional[int]) -> List[Dict[str, Any]]:
        """Retourne la liste des quêtes complétées"""
        if not user_id:
            return []
            
        try:
            async with get_db_session() as session:
                quests_query = await session.execute(
                    "SELECT id, title, category, difficulty FROM user_quests WHERE user_id = ? AND status = 'completed'",
                    (user_id,)
                )
                
                return [
                    {
                        "id": row[0],
                        "title": row[1],
                        "category": row[2],
                        "difficulty": row[3]
                    }
                    for row in quests_query.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des quêtes: {e}")
            return []
    
    async def _save_session_state(self, session_id: str, state: WorkflowState):
        """Sauvegarde l'état de session"""
        try:
            # Créer une version sérialisable de l'état
            serializable_state = {
                "session_id": state["session_id"],
                "user_id": state["user_id"],
                "conversation_state": state["conversation_state"],
                "total_interactions": state["total_interactions"],
                "user_skills": state["user_skills"],
                "learning_objectives": state["learning_objectives"],
                "session_start_time": state["session_start_time"],
                "last_activity_time": state["last_activity_time"]
            }
            
            # Sauvegarder dans un fichier JSON pour backup
            session_file = Path(f"data/sessions/{session_id}.json")
            session_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_state, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"État de session {session_id} sauvegardé")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de session: {e}")
    
    async def _update_user_statistics(self, state: WorkflowState, session_stats: Dict[str, Any]):
        """Met à jour les statistiques utilisateur en base"""
        if not state["user_id"]:
            return
            
        try:
            async with get_db_session() as session:
                # Mettre à jour les compétences
                for skill_name, skill_level in state["user_skills"].items():
                    await session.execute(
                        """
                        INSERT INTO user_progress (user_id, skill_name, skill_level, xp_gained, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, skill_name) DO UPDATE SET
                            skill_level = ?,
                            xp_gained = xp_gained + ?,
                            updated_at = ?
                        """,
                        (
                            state["user_id"], skill_name, skill_level, 
                            int(skill_level * 100), get_current_timestamp(),
                            skill_level, int(skill_level * 10), get_current_timestamp()
                        )
                    )
                
                # Mettre à jour les statistiques de session
                await session.execute(
                    """
                    INSERT INTO user_sessions (user_id, session_id, duration, interactions, agents_used, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state["user_id"], state["session_id"],
                        session_stats["duration"], session_stats["interactions"],
                        json.dumps(session_stats["agents_used"]), get_current_timestamp()
                    )
                )
                
                await session.commit()
                logger.debug(f"Statistiques utilisateur {state['user_id']} mises à jour")
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des statistiques: {e}")
    
    # Méthodes publiques pour l'interface
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retourne les informations d'une session"""
        if session_id not in self.active_sessions:
            return None
            
        state = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "user_id": state["user_id"],
            "conversation_state": state["conversation_state"],
            "total_interactions": state["total_interactions"],
            "session_duration": self._calculate_session_duration(state),
            "current_quest": self._get_quest_status(state),
            "user_skills": state["user_skills"],
            "last_activity": state["last_activity_time"]
        }
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """Termine une session manuellement"""
        if session_id not in self.active_sessions:
            return {"error": "Session non trouvée"}
            
        state = self.active_sessions[session_id]
        
        # Forcer la fin de session
        state["conversation_state"] = ConversationState.COMPLETED.value
        
        # Traiter avec le gestionnaire de session
        config = {"configurable": {"thread_id": session_id}}
        final_state = await self.graph.ainvoke(state, config)
        
        # Nettoyer la session active
        del self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "status": "ended",
            "final_message": final_state.get("last_response", "Session terminée"),
            "session_stats": final_state.get("temp_data", {}).get("session_stats", {})
        }
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Retourne la liste des sessions actives"""
        sessions = []
        for session_id, state in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "user_id": state["user_id"],
                "conversation_state": state["conversation_state"],
                "total_interactions": state["total_interactions"],
                "duration": self._calculate_session_duration(state),
                "last_activity": state["last_activity_time"]
            })
        return sessions
    
    async def cleanup_expired_sessions(self):
        """Nettoie les sessions expirées"""
        current_time = datetime.now(timezone.utc)
        expired_sessions = []
        
        for session_id, state in self.active_sessions.items():
            try:
                last_activity = datetime.fromisoformat(state["last_activity_time"].replace('Z', '+00:00'))
                if (current_time - last_activity).total_seconds() > self.max_session_duration:
                    expired_sessions.append(session_id)
            except:
                expired_sessions.append(session_id)  # Session corrompue
        
        # Terminer les sessions expirées
        for session_id in expired_sessions:
            try:
                await self.end_session(session_id)
                logger.info(f"Session expirée {session_id} nettoyée")
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage de session {session_id}: {e}")
                # Forcer la suppression en cas d'erreur
                self.active_sessions.pop(session_id, None)
        
        return len(expired_sessions)
    
    def get_conversation_flows(self) -> List[Dict[str, Any]]:
        """Retourne les flux de conversation disponibles"""
        return [
            {
                "name": flow.name,
                "description": flow.description,
                "states": [state.value for state in flow.states],
                "default_agent": flow.default_agent.value,
                "priority": flow.priority
            }
            for flow in self.conversation_flows
        ]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système"""
        return {
            "active_sessions": len(self.active_sessions),
            "agents_available": list(self.agents.keys()),
            "conversation_flows": len(self.conversation_flows),
            "checkpoint_path": self.checkpoint_path,
            "max_session_duration": self.max_session_duration,
            "max_interactions_per_session": self.max_interactions_per_session,
            "auto_save_frequency": self.auto_save_frequency
        }


# Instance globale du gestionnaire d'états
state_manager = StateManager()


# Fonction utilitaire pour l'initialisation
async def initialize_state_manager(checkpoint_path: str = None) -> StateManager:
    """
    Initialise le gestionnaire d'états avec configuration optionnelle
    
    Args:
        checkpoint_path: Chemin vers le fichier de checkpoints (optionnel)
        
    Returns:
        Instance du gestionnaire d'états
    """
    global state_manager
    
    if checkpoint_path:
        state_manager = StateManager(checkpoint_path)
    
    # Nettoyer les sessions expirées au démarrage
    cleaned = await state_manager.cleanup_expired_sessions()
    if cleaned > 0:
        logger.info(f"{cleaned} sessions expirées nettoyées au démarrage")
    
    return state_manager


# Décorateurs pour la gestion des erreurs
def handle_session_errors(func):
    """Décorateur pour gérer les erreurs de session"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Erreur dans {func.__name__}: {e}")
            return {
                "error": f"Erreur système: {str(e)}",
                "function": func.__name__
            }
    return wrapper


# Tâche de nettoyage périodique
async def periodic_cleanup():
    """Tâche de nettoyage périodique des sessions"""
    while True:
        try:
            await asyncio.sleep(1800)  # 30 minutes
            cleaned = await state_manager.cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"Nettoyage périodique: {cleaned} sessions supprimées")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage périodique: {e}")


if __name__ == "__main__":
    # Test du gestionnaire d'états
    async def test_state_manager():
        # Initialiser
        manager = await initialize_state_manager()
        
        # Créer une session de test
        session_id = await manager.start_session()
        print(f"Session créée: {session_id}")
        
        # Test d'interaction
        response = await manager.process_user_input(
            session_id, 
            "Bonjour ! Peux-tu m'expliquer les listes en Python ?"
        )
        print(f"Réponse: {response['response']}")
        
        # Test d'évaluation de code
        code_response = await manager.process_user_input(
            session_id,
            "```python\ndef addition(a, b):\n    return a + b\n```",
            "code"
        )
        print(f"Évaluation: {code_response['response']}")
        
        # Terminer la session
        end_result = await manager.end_session(session_id)
        print(f"Session terminée: {end_result}")
    
    # Exécuter le test
    asyncio.run(test_state_manager())