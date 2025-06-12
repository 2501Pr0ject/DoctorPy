# src/api/routes/quests.py
"""
Routes API pour la gestion des quêtes pédagogiques
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from enum import Enum

from src.agents.state_manager import StateManager
from src.core.exceptions import ValidationError, NotFoundError
from src.api.middleware.auth import get_current_user
from src.models.schemas import UserBase
from src.quests.quest_manager import QuestManager
from src.core.database import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter()


# Énumérations
class QuestDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestCategory(str, Enum):
    PYTHON_BASICS = "python_basics"
    PYTHON_INTERMEDIATE = "python_intermediate"
    PYTHON_ADVANCED = "python_advanced"
    DATA_SCIENCE = "data_science"
    WEB_DEVELOPMENT = "web"
    ALGORITHMS = "algorithms"


class QuestStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


# Modèles Pydantic
class QuestStep(BaseModel):
    id: int
    title: str
    description: str
    instructions: str
    hint: Optional[str] = None
    expected_output: Optional[str] = None
    test_cases: Optional[List[Dict[str, Any]]] = None
    order: int


class QuestBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    category: QuestCategory
    difficulty: QuestDifficulty
    estimated_duration: int = Field(..., gt=0, description="Durée estimée en minutes")
    learning_objectives: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class QuestCreate(QuestBase):
    steps: List[Dict[str, Any]] = Field(..., min_items=1)
    rewards: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class QuestResponse(QuestBase):
    id: int
    steps: List[QuestStep]
    status: QuestStatus
    progress_percentage: float
    current_step: int
    total_steps: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_rating: Optional[int] = None
    rewards: Dict[str, Any] = Field(default_factory=dict)


class QuestListItem(BaseModel):
    id: int
    title: str
    description: str
    category: QuestCategory
    difficulty: QuestDifficulty
    estimated_duration: int
    status: QuestStatus
    progress_percentage: float
    created_at: datetime
    tags: List[str] = Field(default_factory=list)


class QuestGenerationRequest(BaseModel):
    category: Optional[QuestCategory] = None
    difficulty: Optional[QuestDifficulty] = None
    topic: Optional[str] = Field(None, max_length=100)
    learning_objectives: List[str] = Field(default_factory=list)
    custom_requirements: Optional[str] = Field(None, max_length=500)
    session_id: Optional[str] = None


class QuestStartRequest(BaseModel):
    session_id: Optional[str] = None


class QuestSubmissionRequest(BaseModel):
    step_id: int
    user_code: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class QuestSubmissionResponse(BaseModel):
    step_id: int
    is_correct: bool
    score: float
    feedback: str
    hints: List[str] = Field(default_factory=list)
    next_step: Optional[QuestStep] = None
    quest_completed: bool = False


class QuestProgressUpdate(BaseModel):
    current_step: int
    progress_percentage: float
    status: QuestStatus
    time_spent: Optional[int] = None  # en secondes


class QuestRatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = Field(None, max_length=500)


# Dépendances
async def get_state_manager() -> StateManager:
    """Récupère le gestionnaire d'états"""
    from src.api.main import app
    if not hasattr(app.state, 'state_manager'):
        raise HTTPException(status_code=503, detail="Gestionnaire d'états non disponible")
    return app.state.state_manager


async def get_quest_manager() -> QuestManager:
    """Récupère le gestionnaire de quêtes"""
    # À implémenter selon votre architecture
    return QuestManager()


# Routes principales
@router.get("/", response_model=List[QuestListItem])
async def list_quests(
    category: Optional[QuestCategory] = Query(None, description="Filtrer par catégorie"),
    difficulty: Optional[QuestDifficulty] = Query(None, description="Filtrer par difficulté"),
    status: Optional[QuestStatus] = Query(None, description="Filtrer par statut"),
    tags: Optional[str] = Query(None, description="Filtrer par tags (séparés par des virgules)"),
    limit: int = Query(20, ge=1, le=100, description="Nombre de quêtes à retourner"),
    offset: int = Query(0, ge=0, description="Décalage pour la pagination"),
    current_user: Optional[UserBase] = Depends(get_current_user),
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Liste les quêtes disponibles avec filtres optionnels
    """
    try:
        logger.info(f"Récupération de la liste des quêtes pour l'utilisateur {current_user.id if current_user else 'anonyme'}")
        
        # Construire les filtres
        filters = {}
        if category:
            filters["category"] = category.value
        if difficulty:
            filters["difficulty"] = difficulty.value
        if status and current_user:
            filters["status"] = status.value
            filters["user_id"] = current_user.id
        if tags:
            filters["tags"] = [tag.strip() for tag in tags.split(",")]
        
        # Récupérer les quêtes
        quests = await quest_manager.list_quests(
            filters=filters,
            limit=limit,
            offset=offset,
            user_id=current_user.id if current_user else None
        )
        
        # Convertir en format API
        quest_items = []
        for quest in quests:
            quest_items.append(QuestListItem(
                id=quest["id"],
                title=quest["title"],
                description=quest["description"],
                category=QuestCategory(quest["category"]),
                difficulty=QuestDifficulty(quest["difficulty"]),
                estimated_duration=quest["estimated_duration"],
                status=QuestStatus(quest.get("status", "not_started")),
                progress_percentage=quest.get("progress_percentage", 0.0),
                created_at=quest["created_at"],
                tags=quest.get("tags", [])
            ))
        
        return quest_items
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des quêtes: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des quêtes")


@router.post("/generate", response_model=QuestResponse)
async def generate_quest(
    request: QuestGenerationRequest,
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Génère une nouvelle quête personnalisée
    """
    try:
        logger.info(f"Génération d'une quête pour l'utilisateur {current_user.id if current_user else 'anonyme'}")
        
        # Si une session est fournie, utiliser le state manager
        if request.session_id:
            if request.session_id not in state_manager.active_sessions:
                raise NotFoundError(f"Session {request.session_id} non trouvée")
            
            # Préparer la requête pour le générateur de quêtes
            generation_prompt = f"Génère une quête "
            if request.category:
                generation_prompt += f"de catégorie {request.category.value} "
            if request.difficulty:
                generation_prompt += f"de difficulté {request.difficulty.value} "
            if request.topic:
                generation_prompt += f"sur le sujet {request.topic} "
            if request.custom_requirements:
                generation_prompt += f"avec les exigences: {request.custom_requirements}"
            
            # Traiter avec le state manager
            result = await state_manager.process_user_input(
                session_id=request.session_id,
                user_input=generation_prompt,
                input_type="text"
            )
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            # Récupérer la quête générée depuis les données temporaires
            session_state = state_manager.active_sessions[request.session_id]
            quest_data = session_state.get("temp_data", {}).get("quest_data")
            
            if not quest_data:
                raise HTTPException(status_code=500, detail="Échec de la génération de quête")
            
        else:
            # Génération directe sans session
            quest_manager = await get_quest_manager()
            quest_data = await quest_manager.generate_quest(
                category=request.category.value if request.category else None,
                difficulty=request.difficulty.value if request.difficulty else "auto",
                topic=request.topic,
                learning_objectives=request.learning_objectives,
                user_id=current_user.id if current_user else None
            )
        
        # Construire la réponse
        steps = []
        for i, step_data in enumerate(quest_data.get("steps", [])):
            steps.append(QuestStep(
                id=i,
                title=step_data.get("title", f"Étape {i+1}"),
                description=step_data.get("description", ""),
                instructions=step_data.get("instructions", ""),
                hint=step_data.get("hint"),
                expected_output=step_data.get("expected_output"),
                test_cases=step_data.get("test_cases"),
                order=i
            ))
        
        quest_response = QuestResponse(
            id=quest_data.get("id", 0),
            title=quest_data.get("title", "Quête générée"),
            description=quest_data.get("description", ""),
            category=QuestCategory(quest_data.get("category", "python_basics")),
            difficulty=QuestDifficulty(quest_data.get("difficulty", "medium")),
            estimated_duration=quest_data.get("estimated_duration", 30),
            learning_objectives=quest_data.get("learning_objectives", []),
            prerequisites=quest_data.get("prerequisites", []),
            tags=quest_data.get("tags", []),
            steps=steps,
            status=QuestStatus.NOT_STARTED,
            progress_percentage=0.0,
            current_step=0,
            total_steps=len(steps),
            created_at=datetime.utcnow(),
            rewards=quest_data.get("rewards", {})
        )
        
        return quest_response
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la génération de quête: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la génération de quête")


@router.get("/{quest_id}", response_model=QuestResponse)
async def get_quest(
    quest_id: int,
    current_user: Optional[UserBase] = Depends(get_current_user),
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Récupère une quête spécifique par son ID
    """
    try:
        quest_data = await quest_manager.get_quest(
            quest_id=quest_id,
            user_id=current_user.id if current_user else None
        )
        
        if not quest_data:
            raise NotFoundError(f"Quête {quest_id} non trouvée")
        
        # Construire la réponse (similar à generate_quest)
        steps = []
        for i, step_data in enumerate(quest_data.get("steps", [])):
            steps.append(QuestStep(
                id=step_data.get("id", i),
                title=step_data.get("title", f"Étape {i+1}"),
                description=step_data.get("description", ""),
                instructions=step_data.get("instructions", ""),
                hint=step_data.get("hint"),
                expected_output=step_data.get("expected_output"),
                test_cases=step_data.get("test_cases"),
                order=i
            ))
        
        return QuestResponse(
            id=quest_data["id"],
            title=quest_data["title"],
            description=quest_data["description"],
            category=QuestCategory(quest_data["category"]),
            difficulty=QuestDifficulty(quest_data["difficulty"]),
            estimated_duration=quest_data["estimated_duration"],
            learning_objectives=quest_data.get("learning_objectives", []),
            prerequisites=quest_data.get("prerequisites", []),
            tags=quest_data.get("tags", []),
            steps=steps,
            status=QuestStatus(quest_data.get("status", "not_started")),
            progress_percentage=quest_data.get("progress_percentage", 0.0),
            current_step=quest_data.get("current_step", 0),
            total_steps=len(steps),
            created_at=quest_data["created_at"],
            updated_at=quest_data.get("updated_at"),
            completed_at=quest_data.get("completed_at"),
            user_rating=quest_data.get("user_rating"),
            rewards=quest_data.get("rewards", {})
        )
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la quête {quest_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de la quête")


@router.post("/{quest_id}/start")
async def start_quest(
    quest_id: int,
    request: QuestStartRequest,
    current_user: UserBase = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager),
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Démarre une quête pour l'utilisateur
    """
    try:
        logger.info(f"Démarrage de la quête {quest_id} pour l'utilisateur {current_user.id}")
        
        # Vérifier que la quête existe
        quest_data = await quest_manager.get_quest(quest_id)
        if not quest_data:
            raise NotFoundError(f"Quête {quest_id} non trouvée")
        
        # Démarrer la quête en base
        await quest_manager.start_quest(quest_id, current_user.id)
        
        # Si une session est fournie, activer la quête dans la session
        if request.session_id:
            if request.session_id not in state_manager.active_sessions:
                raise NotFoundError(f"Session {request.session_id} non trouvée")
            
            session_state = state_manager.active_sessions[request.session_id]
            session_state["current_quest_id"] = quest_id
            session_state["current_step"] = 0
            session_state["conversation_state"] = "quest_active"
            
        return {
            "message": "Quête démarrée avec succès",
            "quest_id": quest_id,
            "current_step": 0,
            "session_id": request.session_id
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de la quête {quest_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du démarrage de la quête")


@router.post("/{quest_id}/submit", response_model=QuestSubmissionResponse)
async def submit_quest_step(
    quest_id: int,
    request: QuestSubmissionRequest,
    current_user: UserBase = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Soumet une réponse pour une étape de quête
    """
    try:
        logger.info(f"Soumission pour la quête {quest_id}, étape {request.step_id}")
        
        # Si une session est fournie, utiliser l'évaluateur de code via le state manager
        if request.session_id:
            if request.session_id not in state_manager.active_sessions:
                raise NotFoundError(f"Session {request.session_id} non trouvée")
            
            session_state = state_manager.active_sessions[request.session_id]
            session_state["needs_code_evaluation"] = True
            session_state["temp_data"]["code"] = request.user_code
            session_state["temp_data"]["quest_context"] = {"quest_id": quest_id, "step_id": request.step_id}
            
            # Traiter l'évaluation
            result = await state_manager.process_user_input(
                session_id=request.session_id,
                user_input=f"Voici ma solution pour l'étape {request.step_id}:\n```python\n{request.user_code}\n```",
                input_type="code"
            )
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            # Récupérer les résultats d'évaluation
            evaluation_results = session_state.get("temp_data", {}).get("evaluation_results", {})
            
        else:
            # Évaluation directe sans session
            # À implémenter selon votre architecture
            evaluation_results = {"overall_score": 0.5, "is_correct": False}
        
        # Construire la réponse
        is_correct = evaluation_results.get("overall_score", 0) >= 0.7
        score = evaluation_results.get("overall_score", 0)
        
        response = QuestSubmissionResponse(
            step_id=request.step_id,
            is_correct=is_correct,
            score=score,
            feedback=evaluation_results.get("feedback", "Code évalué"),
            hints=evaluation_results.get("suggestions", []),
            quest_completed=False  # À déterminer selon la logique métier
        )
        
        return response
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la soumission: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la soumission")


@router.put("/{quest_id}/progress")
async def update_quest_progress(
    quest_id: int,
    progress: QuestProgressUpdate,
    current_user: UserBase = Depends(get_current_user),
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Met à jour la progression d'une quête
    """
    try:
        await quest_manager.update_progress(
            quest_id=quest_id,
            user_id=current_user.id,
            current_step=progress.current_step,
            progress_percentage=progress.progress_percentage,
            status=progress.status.value,
            time_spent=progress.time_spent
        )
        
        return {
            "message": "Progression mise à jour avec succès",
            "quest_id": quest_id,
            "current_step": progress.current_step,
            "progress_percentage": progress.progress_percentage
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de progression: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")


@router.post("/{quest_id}/rate")
async def rate_quest(
    quest_id: int,
    rating: QuestRatingRequest,
    current_user: UserBase = Depends(get_current_user),
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Note une quête
    """
    try:
        await quest_manager.rate_quest(
            quest_id=quest_id,
            user_id=current_user.id,
            rating=rating.rating,
            feedback=rating.feedback
        )
        
        return {
            "message": "Note enregistrée avec succès",
            "quest_id": quest_id,
            "rating": rating.rating
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la notation: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la notation")


@router.get("/{quest_id}/analytics")
async def get_quest_analytics(
    quest_id: int,
    current_user: UserBase = Depends(get_current_user),
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Récupère les analytiques d'une quête pour l'utilisateur
    """
    try:
        analytics = await quest_manager.get_user_quest_analytics(
            quest_id=quest_id,
            user_id=current_user.id
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des analytiques: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des analytiques")


@router.delete("/{quest_id}")
async def abandon_quest(
    quest_id: int,
    current_user: UserBase = Depends(get_current_user),
    quest_manager: QuestManager = Depends(get_quest_manager)
):
    """
    Abandonne une quête
    """
    try:
        await quest_manager.abandon_quest(quest_id, current_user.id)
        
        return {
            "message": "Quête abandonnée avec succès",
            "quest_id": quest_id
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'abandon de la quête: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'abandon")