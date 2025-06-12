# src/models/schemas.py
"""
Schémas Pydantic pour l'API et validation des données
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, EmailStr, validator, Field, root_validator

# Import des enums depuis les modèles
from .user import UserLevel, UserRole, LearningStyle
from .quest import QuestStatus, QuestDifficulty, QuestCategory, QuestionType, UserQuestStatus
from .progress import ProgressType, SkillCategory, LearningMetricType

# ===== SCHÉMAS DE BASE =====

class TimestampMixin(BaseModel):
    """Mixin pour les timestamps"""
    created_at: datetime
    updated_at: Optional[datetime] = None

class PaginationParams(BaseModel):
    """Paramètres de pagination"""
    page: int = Field(1, ge=1, description="Numéro de page")
    size: int = Field(20, ge=1, le=100, description="Taille de page")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

class PaginatedResponse(BaseModel):
    """Réponse paginée générique"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    
    @validator('pages', pre=True, always=True)
    def calculate_pages(cls, v, values):
        total = values.get('total', 0)
        size = values.get('size', 20)
        return (total + size - 1) // size if size > 0 else 0

class APIResponse(BaseModel):
    """Réponse API standard"""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

# ===== SCHÉMAS UTILISATEUR =====

class UserStatsSchema(BaseModel):
    """Schéma statistiques utilisateur"""
    total_sessions: int
    total_quests_completed: int
    total_study_time: int  # en minutes
    current_streak: int
    longest_streak: int
    average_session_duration: float  # en minutes
    quests_this_week: int
    study_time_this_week: int
    level_progress: float  # pourcentage vers le niveau suivant
    achievements_count: int
    completion_rate: float
    accuracy_rate: float

# ===== SCHÉMAS QUÊTE =====

class QuestBaseSchema(BaseModel):
    """Schéma de base pour les quêtes"""
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    short_description: Optional[str] = Field(None, max_length=500)
    category: QuestCategory
    difficulty: QuestDifficulty
    level: UserLevel
    tags: Optional[List[str]] = []
    learning_objectives: List[str] = Field(..., min_items=1)
    prerequisites: Optional[List[str]] = []
    estimated_duration: int = Field(30, ge=5, le=480)  # 5 min à 8h
    passing_score: float = Field(0.7, ge=0.0, le=1.0)
    xp_reward: int = Field(100, ge=0)

class QuestCreateSchema(QuestBaseSchema):
    """Schéma création de quête"""
    slug: Optional[str] = None
    
    @validator('slug', pre=True, always=True)
    def generate_slug(cls, v, values):
        if not v and 'title' in values:
            from src.utils.helpers import slugify
            return slugify(values['title'])
        return v

class QuestUpdateSchema(BaseModel):
    """Schéma mise à jour de quête"""
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, min_length=10)
    short_description: Optional[str] = Field(None, max_length=500)
    category: Optional[QuestCategory] = None
    difficulty: Optional[QuestDifficulty] = None
    level: Optional[UserLevel] = None
    tags: Optional[List[str]] = None
    learning_objectives: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None
    estimated_duration: Optional[int] = Field(None, ge=5, le=480)
    passing_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    xp_reward: Optional[int] = Field(None, ge=0)
    status: Optional[QuestStatus] = None

class QuestResponseSchema(QuestBaseSchema, TimestampMixin):
    """Schéma réponse de quête"""
    id: int
    uuid: str
    slug: str
    status: QuestStatus
    version: str
    created_by: Optional[int] = None
    total_steps: int = 0
    total_questions: int = 0
    total_attempts: int = 0
    total_completions: int = 0
    average_score: float = 0.0
    average_duration: int = 0
    difficulty_rating: Optional[float] = None
    badge_id: Optional[str] = None
    published_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class QuestSummarySchema(BaseModel):
    """Schéma résumé de quête"""
    id: int
    title: str
    short_description: Optional[str]
    category: QuestCategory
    difficulty: QuestDifficulty
    level: UserLevel
    estimated_duration: int
    xp_reward: int
    completion_rate: float
    average_rating: Optional[float] = None
    is_completed: Optional[bool] = None  # Pour un utilisateur spécifique
    user_progress: Optional[float] = None  # Progression utilisateur

# ===== SCHÉMAS ÉTAPE DE QUÊTE =====

class QuestStepBaseSchema(BaseModel):
    """Schéma de base pour les étapes"""
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    step_type: str = Field("content", regex="^(content|question|coding|review)$")
    is_optional: bool = False
    content: Optional[str] = None
    code_template: Optional[str] = None
    expected_output: Optional[str] = None
    resources: Optional[List[Dict[str, str]]] = []
    hints: Optional[List[str]] = []
    validation_rules: Optional[Dict[str, Any]] = {}
    max_attempts: int = Field(3, ge=1, le=10)

class QuestStepCreateSchema(QuestStepBaseSchema):
    """Schéma création d'étape"""
    order: int = Field(..., ge=1)

class QuestStepUpdateSchema(BaseModel):
    """Schéma mise à jour d'étape"""
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, min_length=10)
    step_type: Optional[str] = Field(None, regex="^(content|question|coding|review)$")
    is_optional: Optional[bool] = None
    content: Optional[str] = None
    code_template: Optional[str] = None
    expected_output: Optional[str] = None
    resources: Optional[List[Dict[str, str]]] = None
    hints: Optional[List[str]] = None
    validation_rules: Optional[Dict[str, Any]] = None
    max_attempts: Optional[int] = Field(None, ge=1, le=10)
    order: Optional[int] = Field(None, ge=1)

class QuestStepResponseSchema(QuestStepBaseSchema, TimestampMixin):
    """Schéma réponse d'étape"""
    id: int
    quest_id: int
    order: int
    
    class Config:
        from_attributes = True

# ===== SCHÉMAS QUESTION =====

class QuestionBaseSchema(BaseModel):
    """Schéma de base pour les questions"""
    question_text: str = Field(..., min_length=10)
    question_type: QuestionType
    choices: Optional[List[str]] = []
    correct_answer: str
    explanation: Optional[str] = None
    points: int = Field(1, ge=1, le=10)
    time_limit: Optional[int] = Field(None, ge=10, le=3600)  # 10s à 1h
    shuffle_choices: bool = True
    case_sensitive: bool = False

class QuestionCreateSchema(QuestionBaseSchema):
    """Schéma création de question"""
    
    @validator('choices')
    def validate_choices_for_mcq(cls, v, values):
        if values.get('question_type') == QuestionType.MULTIPLE_CHOICE and len(v) < 2:
            raise ValueError('Les QCM doivent avoir au moins 2 choix')
        return v

class QuestionUpdateSchema(BaseModel):
    """Schéma mise à jour de question"""
    question_text: Optional[str] = Field(None, min_length=10)
    question_type: Optional[QuestionType] = None
    choices: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    points: Optional[int] = Field(None, ge=1, le=10)
    time_limit: Optional[int] = Field(None, ge=10, le=3600)
    shuffle_choices: Optional[bool] = None
    case_sensitive: Optional[bool] = None

class QuestionResponseSchema(QuestionBaseSchema, TimestampMixin):
    """Schéma réponse de question"""
    id: int
    step_id: int
    
    class Config:
        from_attributes = True

# ===== SCHÉMAS PROGRESSION =====

class UserProgressSchema(BaseModel):
    """Schéma progression utilisateur"""
    overall_level: UserLevel
    xp_points: int
    level_progress: float
    skill_scores: Dict[str, float]
    total_quests_attempted: int
    total_quests_completed: int
    total_study_time: int
    current_streak: int
    longest_streak: int
    completion_rate: float
    code_success_rate: float
    
    class Config:
        from_attributes = True

class DailyProgressSchema(BaseModel):
    """Schéma progression quotidienne"""
    date: date
    study_time_minutes: int
    quests_completed: int
    questions_answered: int
    correct_answers: int
    xp_earned: int
    accuracy_rate: float
    
    class Config:
        from_attributes = True

class SkillAssessmentSchema(BaseModel):
    """Schéma évaluation de compétence"""
    skill_category: SkillCategory
    skill_name: str
    score: float
    max_score: float = 100.0
    assessment_type: str
    response_time: Optional[int] = None
    attempts_count: int = 1
    difficulty_level: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommendations: List[str] = []
    
    class Config:
        from_attributes = True

class LearningSessionSchema(BaseModel):
    """Schéma session d'apprentissage"""
    session_type: str
    focus_area: Optional[str] = None
    difficulty_level: Optional[str] = None
    duration_minutes: int
    tasks_completed: int = 0
    tasks_attempted: int = 0
    average_accuracy: float = 0.0
    xp_earned: int = 0
    engagement_score: Optional[float] = None
    self_assessment: Optional[int] = Field(None, ge=1, le=5)
    difficulty_rating: Optional[int] = Field(None, ge=1, le=5)
    enjoyment_rating: Optional[int] = Field(None, ge=1, le=5)
    notes: Optional[str] = None
    
    class Config:
        from_attributes = True

# ===== SCHÉMAS RÉPONSE UTILISATEUR =====

class UserAnswerSchema(BaseModel):
    """Schéma réponse utilisateur"""
    question_id: int
    answer_text: str
    is_correct: bool
    points_earned: int = 0
    attempt_number: int = 1
    time_taken: Optional[int] = None
    hint_used: bool = False

class UserAnswerCreateSchema(BaseModel):
    """Schéma création réponse"""
    question_id: int
    answer_text: str
    time_taken: Optional[int] = None
    hint_used: bool = False

class QuizSubmissionSchema(BaseModel):
    """Schéma soumission de quiz"""
    step_id: int
    answers: List[UserAnswerCreateSchema]
    total_time: Optional[int] = None

# ===== SCHÉMAS QUEST UTILISATEUR =====

class UserQuestSchema(BaseModel):
    """Schéma quête utilisateur"""
    quest_id: int
    status: UserQuestStatus
    current_step: int = 0
    progress_percentage: float = 0.0
    score: float = 0.0
    attempts: int = 0
    time_spent: int = 0
    hints_used: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UserQuestStartSchema(BaseModel):
    """Schéma démarrage de quête"""
    quest_id: int

class UserQuestProgressSchema(BaseModel):
    """Schéma progression dans une quête"""
    step_number: int
    step_score: float = 0.0
    time_spent: int = 0  # minutes
    user_code: Optional[str] = None
    user_notes: Optional[str] = None

# ===== SCHÉMAS DASHBOARD =====

class DashboardStatsSchema(BaseModel):
    """Schéma statistiques tableau de bord"""
    total_users: int
    active_users_today: int
    total_quests: int
    quests_completed_today: int
    average_session_duration: float
    top_categories: List[Dict[str, Any]]

class UserDashboardSchema(BaseModel):
    """Schéma tableau de bord utilisateur"""
    user: UserResponseSchema
    progress: UserProgressSchema
    daily_stats: List[DailyProgressSchema]
    recent_quests: List[QuestSummarySchema]
    achievements: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    study_calendar: Dict[str, int]  # Date -> minutes

class LearningInsightsSchema(BaseModel):
    """Schéma insights d'apprentissage"""
    strengths: List[str]
    improvement_areas: List[str]
    recommendations: List[str]
    learning_patterns: Dict[str, Any]
    predictions: Dict[str, Any]
    overall_score: float
    skill_breakdown: Dict[str, float]

# ===== SCHÉMAS AUTHENTIFICATION =====

class LoginSchema(BaseModel):
    """Schéma connexion"""
    username_or_email: str
    password: str
    remember_me: bool = False

class TokenSchema(BaseModel):
    """Schéma token"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class LoginResponseSchema(BaseModel):
    """Schéma réponse connexion"""
    user: UserResponseSchema
    token: TokenSchema

class PasswordResetSchema(BaseModel):
    """Schéma réinitialisation mot de passe"""
    email: EmailStr

class PasswordResetConfirmSchema(BaseModel):
    """Schéma confirmation réinitialisation"""
    token: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Les mots de passe ne correspondent pas')
        return v

class ChangePasswordSchema(BaseModel):
    """Schéma changement mot de passe"""
    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Les mots de passe ne correspondent pas')
        return v

# ===== SCHÉMAS RECHERCHE ET FILTRES =====

class QuestFilterSchema(BaseModel):
    """Schéma filtres de quête"""
    category: Optional[QuestCategory] = None
    difficulty: Optional[QuestDifficulty] = None
    level: Optional[UserLevel] = None
    status: Optional[QuestStatus] = None
    tags: Optional[List[str]] = []
    min_duration: Optional[int] = None
    max_duration: Optional[int] = None
    completed: Optional[bool] = None  # Pour filtrer selon la completion utilisateur

class QuestSearchSchema(BaseModel):
    """Schéma recherche de quête"""
    query: Optional[str] = None
    filters: Optional[QuestFilterSchema] = None
    sort_by: Optional[str] = Field("created_at", regex="^(title|difficulty|duration|rating|created_at|updated_at)$")
    sort_order: Optional[str] = Field("desc", regex="^(asc|desc)$")
    pagination: PaginationParams = PaginationParams()

class UserSearchSchema(BaseModel):
    """Schéma recherche utilisateur"""
    query: Optional[str] = None
    level: Optional[UserLevel] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    min_streak: Optional[int] = None
    pagination: PaginationParams = PaginationParams()

# ===== SCHÉMAS ANALYTICS =====

class AnalyticsTimeRange(BaseModel):
    """Schéma plage temporelle pour analytics"""
    start_date: date
    end_date: date
    
    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('La date de fin doit être après la date de début')
        return v

class LearningAnalyticsSchema(BaseModel):
    """Schéma analytics d'apprentissage"""
    period_type: str
    period_start: date
    period_end: date
    total_study_time: int
    average_session_duration: float
    sessions_count: int
    active_days: int
    consistency_score: float
    quests_completed: int
    average_quest_score: float
    accuracy_rate: float
    improvement_rate: float
    xp_gained: int
    skills_improved: int
    achievements_unlocked: int
    learning_velocity: float
    retention_rate: float
    
    class Config:
        from_attributes = True

class SystemAnalyticsSchema(BaseModel):
    """Schéma analytics système"""
    total_users: int
    active_users: int
    new_users_this_month: int
    total_quests: int
    total_completions: int
    average_completion_rate: float
    most_popular_categories: List[Dict[str, Any]]
    user_retention_rates: Dict[str, float]
    system_performance: Dict[str, Any]

# ===== SCHÉMAS NOTIFICATIONS =====

class NotificationSchema(BaseModel):
    """Schéma notification"""
    type: str
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    priority: str = Field("normal", regex="^(low|normal|high|urgent)$")
    is_read: bool = False
    created_at: datetime

class NotificationCreateSchema(BaseModel):
    """Schéma création notification"""
    user_id: int
    type: str
    title: str = Field(..., max_length=200)
    message: str = Field(..., max_length=1000)
    data: Optional[Dict[str, Any]] = None
    priority: str = Field("normal", regex="^(low|normal|high|urgent)$")

# ===== SCHÉMAS EXPORT/IMPORT =====

class ExportRequestSchema(BaseModel):
    """Schéma demande d'export"""
    export_type: str = Field(..., regex="^(user_data|progress|quests|analytics)$")
    format: str = Field("json", regex="^(json|csv|pdf)$")
    include_personal_data: bool = True
    date_range: Optional[AnalyticsTimeRange] = None

class ImportQuestSchema(BaseModel):
    """Schéma import de quête"""
    quest_data: QuestCreateSchema
    steps: List[QuestStepCreateSchema]
    questions: List[QuestionCreateSchema]
    validate_only: bool = False

# ===== SCHÉMAS CONFIGURATION =====

class SystemConfigSchema(BaseModel):
    """Schéma configuration système"""
    maintenance_mode: bool = False
    registration_enabled: bool = True
    max_concurrent_users: int = 1000
    session_timeout_minutes: int = 60
    password_policy: Dict[str, Any]
    notification_settings: Dict[str, Any]
    analytics_enabled: bool = True

class UserPreferencesSchema(BaseModel):
    """Schéma préférences utilisateur"""
    theme: str = Field("light", regex="^(light|dark|auto)$")
    language: str = Field("fr", regex="^(fr|en|es)$")
    notifications_enabled: bool = True
    email_notifications: bool = True
    reminder_time: Optional[str] = Field(None, regex="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    timezone: str = "UTC"
    accessibility: Dict[str, Any] = {}

# ===== SCHÉMAS VALIDATION =====

class ValidationResultSchema(BaseModel):
    """Schéma résultat de validation"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

class CodeValidationSchema(BaseModel):
    """Schéma validation de code"""
    code: str
    language: str = "python"
    strict_mode: bool = False

class CodeExecutionResultSchema(BaseModel):
    """Schéma résultat d'exécution de code"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    test_results: Optional[List[Dict[str, Any]]] = None

# ===== SCHÉMAS GAMIFICATION =====

class AchievementSchema(BaseModel):
    """Schéma achievement"""
    achievement_id: str
    title: str
    description: str
    icon_url: Optional[str] = None
    category: str
    target_value: Optional[int] = None
    xp_reward: int = 0
    rarity: str = Field("common", regex="^(common|uncommon|rare|epic|legendary)$")

class UserAchievementSchema(BaseModel):
    """Schéma achievement utilisateur"""
    achievement: AchievementSchema
    progress: float = 0.0
    current_value: int = 0
    is_unlocked: bool = False
    unlocked_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class LeaderboardEntrySchema(BaseModel):
    """Schéma entrée classement"""
    user_id: int
    username: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    score: float
    rank: int
    level: UserLevel
    badge: Optional[str] = None

class LeaderboardSchema(BaseModel):
    """Schéma classement"""
    type: str  # xp, quests, streak, skills
    period: str  # daily, weekly, monthly, all_time
    entries: List[LeaderboardEntrySchema]
    user_rank: Optional[int] = None
    last_updated: datetimeBaseSchema(BaseModel):
    """Schéma de base utilisateur"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=1000)
    level: UserLevel = UserLevel.BEGINNER
    learning_style: LearningStyle = LearningStyle.VISUAL

class UserCreateSchema(UserBaseSchema):
    """Schéma création utilisateur"""
    password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Les mots de passe ne correspondent pas')
        return v

class UserUpdateSchema(BaseModel):
    """Schéma mise à jour utilisateur"""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=1000)
    level: Optional[UserLevel] = None
    learning_style: Optional[LearningStyle] = None
    avatar_url: Optional[str] = None

class UserResponseSchema(UserBaseSchema, TimestampMixin):
    """Schéma réponse utilisateur"""
    id: int
    uuid: str
    is_active: bool
    is_verified: bool
    role: UserRole
    avatar_url: Optional[str] = None
    last_login: Optional[datetime] = None
    total_sessions: int = 0
    total_quests_completed: int = 0
    total_study_time: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    
    class Config:
        from_attributes = True

class User