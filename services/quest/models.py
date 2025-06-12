"""
Modèles de données pour le service Quest
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, validator, Field
import uuid


class QuestStatus(str, Enum):
    """Statuts des quêtes"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"


class QuestDifficulty(str, Enum):
    """Niveaux de difficulté"""
    VERY_EASY = "very_easy"      # 1
    EASY = "easy"                # 2
    MEDIUM = "medium"            # 3
    HARD = "hard"                # 4
    VERY_HARD = "very_hard"      # 5


class QuestCategory(str, Enum):
    """Catégories de quêtes"""
    PYTHON_BASICS = "python_basics"
    PYTHON_INTERMEDIATE = "python_intermediate"
    PYTHON_ADVANCED = "python_advanced"
    DATA_SCIENCE = "data_science"
    WEB_DEVELOPMENT = "web_development"
    ALGORITHMS = "algorithms"
    DATABASES = "databases"
    MACHINE_LEARNING = "machine_learning"
    DEVOPS = "devops"


class QuestionType(str, Enum):
    """Types de questions"""
    MULTIPLE_CHOICE = "multiple_choice"
    CODE_COMPLETION = "code_completion"
    CODE_DEBUG = "code_debug"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    EXPLANATION = "explanation"


class AchievementType(str, Enum):
    """Types d'achievements"""
    QUEST_COMPLETION = "quest_completion"
    STREAK = "streak"
    SKILL_MASTERY = "skill_mastery"
    SPEED_COMPLETION = "speed_completion"
    PERFECT_SCORE = "perfect_score"
    HELP_OTHERS = "help_others"


class Question(BaseModel):
    """Question individuelle dans une quête"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_type: QuestionType
    question_text: str
    code_snippet: Optional[str] = None
    options: Optional[List[str]] = None  # Pour multiple choice
    correct_answer: Union[str, List[str], int]
    explanation: str
    points: int = 10
    time_limit_seconds: Optional[int] = None
    hints: List[str] = []
    
    @validator('question_text')
    def validate_question_text(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("La question doit contenir au moins 10 caractères")
        return v.strip()


class Quest(BaseModel):
    """Quête complète"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    category: QuestCategory
    difficulty: QuestDifficulty
    status: QuestStatus = QuestStatus.DRAFT
    
    # Contenu pédagogique
    questions: List[Question] = []
    total_points: int = 0
    estimated_time_minutes: int = 30
    
    # Métadonnées
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    
    # Gamification
    prerequisites: List[str] = []  # IDs des quêtes prérequises
    unlocks: List[str] = []  # IDs des quêtes débloquées
    tags: List[str] = []
    
    # Statistiques
    completion_count: int = 0
    average_score: float = 0.0
    average_time_minutes: float = 0.0
    
    class Config:
        use_enum_values = True


class QuestProgress(BaseModel):
    """Progression d'un utilisateur sur une quête"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    quest_id: str
    status: QuestStatus
    
    # Progression
    current_question_index: int = 0
    answers: Dict[str, Any] = {}  # question_id -> answer
    score: int = 0
    max_possible_score: int = 0
    
    # Temps
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    time_spent_seconds: int = 0
    
    # Aide utilisée
    hints_used: List[str] = []
    help_requested: int = 0
    
    class Config:
        use_enum_values = True


class Achievement(BaseModel):
    """Achievement/Badge gagné par un utilisateur"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    icon: str
    achievement_type: AchievementType
    
    # Conditions
    required_quests: List[str] = []
    required_score: Optional[int] = None
    required_streak: Optional[int] = None
    required_category: Optional[QuestCategory] = None
    
    # Récompenses
    points_reward: int = 0
    badge_color: str = "gold"
    
    class Config:
        use_enum_values = True


class UserAchievement(BaseModel):
    """Achievement débloqué par un utilisateur"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    achievement_id: str
    earned_at: datetime = Field(default_factory=datetime.now)
    quest_id: Optional[str] = None  # Quête qui a déclenché l'achievement


class UserStats(BaseModel):
    """Statistiques détaillées d'un utilisateur"""
    user_id: str
    
    # Progression générale
    total_quests_completed: int = 0
    total_points: int = 0
    current_level: int = 1
    current_streak: int = 0
    longest_streak: int = 0
    
    # Par catégorie
    category_progress: Dict[QuestCategory, Dict[str, int]] = {}
    
    # Temps
    total_time_spent_minutes: int = 0
    average_quest_time_minutes: float = 0.0
    
    # Achievements
    total_achievements: int = 0
    recent_achievements: List[str] = []  # IDs des derniers achievements
    
    # Performance
    average_score_percentage: float = 0.0
    perfect_scores: int = 0
    
    # Activité
    last_active: datetime = Field(default_factory=datetime.now)
    days_active: int = 0
    
    class Config:
        use_enum_values = True


# Requêtes API

class StartQuestRequest(BaseModel):
    """Requête pour démarrer une quête"""
    quest_id: str
    user_id: str


class SubmitAnswerRequest(BaseModel):
    """Requête pour soumettre une réponse"""
    progress_id: str
    question_id: str
    answer: Union[str, List[str], int]
    time_spent_seconds: int = 0


class CreateQuestRequest(BaseModel):
    """Requête pour créer une nouvelle quête"""
    title: str
    description: str
    category: QuestCategory
    difficulty: QuestDifficulty
    questions: List[Question]
    estimated_time_minutes: int = 30
    prerequisites: List[str] = []
    tags: List[str] = []
    
    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Le titre doit contenir au moins 5 caractères")
        return v.strip()


class QuestSearchRequest(BaseModel):
    """Requête de recherche de quêtes"""
    category: Optional[QuestCategory] = None
    difficulty: Optional[QuestDifficulty] = None
    status: Optional[QuestStatus] = None
    tags: List[str] = []
    search_term: Optional[str] = None
    limit: int = 20
    offset: int = 0


# Réponses API

class QuestListResponse(BaseModel):
    """Réponse avec liste de quêtes"""
    quests: List[Quest]
    total: int
    offset: int
    limit: int


class ProgressResponse(BaseModel):
    """Réponse de progression"""
    progress: QuestProgress
    current_question: Optional[Question] = None
    is_completed: bool = False
    next_question_index: Optional[int] = None


class AnswerFeedback(BaseModel):
    """Feedback sur une réponse"""
    is_correct: bool
    points_earned: int
    explanation: str
    correct_answer: Optional[Union[str, List[str], int]] = None
    hints_available: List[str] = []


class SubmitAnswerResponse(BaseModel):
    """Réponse après soumission d'une réponse"""
    feedback: AnswerFeedback
    progress: QuestProgress
    quest_completed: bool = False
    achievements_unlocked: List[Achievement] = []


class LeaderboardEntry(BaseModel):
    """Entrée du classement"""
    user_id: str
    username: str
    total_points: int
    quests_completed: int
    current_level: int
    rank: int


class LeaderboardResponse(BaseModel):
    """Réponse du classement"""
    entries: List[LeaderboardEntry]
    user_rank: Optional[int] = None
    total_users: int


@dataclass
class QuestServiceConfig:
    """Configuration du service Quest"""
    port: int = 8004
    host: str = "0.0.0.0"
    database_url: str = "sqlite:///./data/databases/quests.db"
    
    # Gamification
    points_per_correct_answer: int = 10
    points_per_quest_completion: int = 50
    streak_bonus_multiplier: float = 1.5
    time_bonus_threshold_percent: float = 0.8  # Si terminé en moins de 80% du temps estimé
    
    # Performance
    max_concurrent_users: int = 100
    cache_ttl_seconds: int = 1800  # 30 minutes
    
    # Achievements
    enable_achievements: bool = True
    achievement_check_interval_minutes: int = 5


class QuestServiceError(Exception):
    """Exception personnalisée du service Quest"""
    def __init__(self, message: str, error_code: str = "QUEST_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)