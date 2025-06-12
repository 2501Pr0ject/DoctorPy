"""
Modèles pour les quêtes pédagogiques
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import json
import uuid

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import event, CheckConstraint

from pydantic import BaseModel, validator, Field
from pydantic.types import Json

from .user import Base  # Utiliser la même base que le modèle User

class QuestStatus(str, Enum):
    """Statuts des quêtes"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    UNDER_REVIEW = "under_review"

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
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    CODE_COMPLETION = "code_completion"
    CODE_WRITING = "code_writing"
    CODE_DEBUGGING = "code_debugging"
    MATCHING = "matching"
    ORDERING = "ordering"

class UserQuestStatus(str, Enum):
    """Statuts des quêtes pour les utilisateurs"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    LOCKED = "locked"

# ===== MODÈLES SQLALCHEMY =====

class Quest(Base):
    """Modèle principal des quêtes"""
    
    __tablename__ = "quests"
    
    # Identifiants
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    slug = Column(String(200), unique=True, nullable=False, index=True)
    
    # Métadonnées de base
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    short_description = Column(String(500))
    
    # Classification
    category = Column(String(50), nullable=False, index=True)
    difficulty = Column(String(20), nullable=False, index=True)
    level = Column(String(20), nullable=False, index=True)  # beginner, intermediate, advanced
    tags = Column(Text)  # JSON array des tags
    
    # Contenu pédagogique
    learning_objectives = Column(Text, nullable=False)  # JSON array
    prerequisites = Column(Text)  # JSON array des prérequis
    estimated_duration = Column(Integer, default=30)  # minutes
    
    # Structure de la quête
    total_steps = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    passing_score = Column(Float, default=0.7)  # Score minimum pour réussir
    
    # Métadonnées de création
    created_by = Column(Integer, ForeignKey("users.id"))
    status = Column(String(20), default=QuestStatus.DRAFT.value)
    version = Column(String(10), default="1.0")
    
    # Analytics
    total_attempts = Column(Integer, default=0)
    total_completions = Column(Integer, default=0)
    average_score = Column(Float, default=0.0)
    average_duration = Column(Integer, default=0)  # minutes
    difficulty_rating = Column(Float)  # Rating donné par les utilisateurs
    
    # Gamification
    xp_reward = Column(Integer, default=100)
    badge_id = Column(String(100))  # Badge à débloquer
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    published_at = Column(DateTime)
    
    # Relations
    steps = relationship("QuestStep", back_populates="quest", cascade="all, delete-orphan", order_by="QuestStep.order")
    user_quests = relationship("UserQuest", back_populates="quest", cascade="all, delete-orphan")
    creator = relationship("User", foreign_keys=[created_by])
    
    # Contraintes
    __table_args__ = (
        CheckConstraint('passing_score >= 0.0 AND passing_score <= 1.0', name='valid_passing_score'),
        CheckConstraint('estimated_duration > 0', name='positive_duration'),
        Index('idx_quest_category_difficulty', 'category', 'difficulty'),
        Index('idx_quest_level_status', 'level', 'status'),
    )
    
    def __repr__(self):
        return f"<Quest(id={self.id}, title='{self.title}', category='{self.category}')>"
    
    def get_tags(self) -> List[str]:
        """Retourne la liste des tags"""
        if self.tags:
            try:
                return json.loads(self.tags)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_tags(self, tags: List[str]):
        """Définit les tags"""
        self.tags = json.dumps(tags)
    
    def get_learning_objectives(self) -> List[str]:
        """Retourne les objectifs d'apprentissage"""
        if self.learning_objectives:
            try:
                return json.loads(self.learning_objectives)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_learning_objectives(self, objectives: List[str]):
        """Définit les objectifs d'apprentissage"""
        self.learning_objectives = json.dumps(objectives)
    
    def get_prerequisites(self) -> List[str]:
        """Retourne les prérequis"""
        if self.prerequisites:
            try:
                return json.loads(self.prerequisites)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_prerequisites(self, prerequisites: List[str]):
        """Définit les prérequis"""
        self.prerequisites = json.dumps(prerequisites)
    
    def calculate_completion_rate(self) -> float:
        """Calcule le taux de completion"""
        if self.total_attempts > 0:
            return self.total_completions / self.total_attempts
        return 0.0
    
    def update_analytics(self, score: float, duration: int, completed: bool):
        """Met à jour les analytics de la quête"""
        self.total_attempts += 1
        
        if completed:
            self.total_completions += 1
        
        # Mise à jour du score moyen
        if self.total_attempts == 1:
            self.average_score = score
        else:
            self.average_score = ((self.average_score * (self.total_attempts - 1)) + score) / self.total_attempts
        
        # Mise à jour de la durée moyenne
        if self.total_attempts == 1:
            self.average_duration = duration
        else:
            self.average_duration = ((self.average_duration * (self.total_attempts - 1)) + duration) / self.total_attempts
    
    def is_accessible_for_user(self, user_level: str) -> bool:
        """Vérifie si la quête est accessible pour un niveau utilisateur"""
        level_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
        quest_level = level_order.get(self.level, 1)
        user_level_num = level_order.get(user_level, 1)
        
        # Permet d'accéder aux quêtes de son niveau et des niveaux inférieurs
        return quest_level <= user_level_num + 1  # +1 pour permettre un peu de challenge

class QuestStep(Base):
    """Étapes d'une quête"""
    
    __tablename__ = "quest_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    quest_id = Column(Integer, ForeignKey("quests.id"), nullable=False)
    
    # Ordre et organisation
    order = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Type d'étape
    step_type = Column(String(20), default="content")  # content, question, coding, review
    is_optional = Column(Boolean, default=False)
    
    # Contenu
    content = Column(Text)  # Contenu textuel/markdown
    code_template = Column(Text)  # Template de code
    expected_output = Column(Text)  # Sortie attendue
    
    # Ressources
    resources = Column(Text)  # JSON array des ressources (liens, docs)
    hints = Column(Text)  # JSON array des indices
    
    # Validation
    validation_rules = Column(Text)  # JSON des règles de validation
    max_attempts = Column(Integer, default=3)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    quest = relationship("Quest", back_populates="steps")
    questions = relationship("Question", back_populates="step", cascade="all, delete-orphan")
    user_step_progress = relationship("UserStepProgress", back_populates="step", cascade="all, delete-orphan")
    
    # Contraintes
    __table_args__ = (
        Index('idx_quest_step_order', 'quest_id', 'order'),
    )
    
    def __repr__(self):
        return f"<QuestStep(id={self.id}, quest_id={self.quest_id}, order={self.order}, title='{self.title}')>"
    
    def get_resources(self) -> List[Dict[str, str]]:
        """Retourne les ressources"""
        if self.resources:
            try:
                return json.loads(self.resources)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_resources(self, resources: List[Dict[str, str]]):
        """Définit les ressources"""
        self.resources = json.dumps(resources)
    
    def get_hints(self) -> List[str]:
        """Retourne les indices"""
        if self.hints:
            try:
                return json.loads(self.hints)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_hints(self, hints: List[str]):
        """Définit les indices"""
        self.hints = json.dumps(hints)
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Retourne les règles de validation"""
        if self.validation_rules:
            try:
                return json.loads(self.validation_rules)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_validation_rules(self, rules: Dict[str, Any]):
        """Définit les règles de validation"""
        self.validation_rules = json.dumps(rules)

class Question(Base):
    """Questions dans les étapes"""
    
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    step_id = Column(Integer, ForeignKey("quest_steps.id"), nullable=False)
    
    # Contenu de la question
    question_text = Column(Text, nullable=False)
    question_type = Column(String(20), nullable=False)
    
    # Réponses et choix
    choices = Column(Text)  # JSON array pour QCM
    correct_answer = Column(Text, nullable=False)
    explanation = Column(Text)
    
    # Configuration
    points = Column(Integer, default=1)
    time_limit = Column(Integer)  # secondes
    shuffle_choices = Column(Boolean, default=True)
    case_sensitive = Column(Boolean, default=False)
    
    # Métadonnées
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    step = relationship("QuestStep", back_populates="questions")
    user_answers = relationship("UserAnswer", back_populates="question", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Question(id={self.id}, step_id={self.step_id}, type='{self.question_type}')>"
    
    def get_choices(self) -> List[str]:
        """Retourne les choix pour les QCM"""
        if self.choices:
            try:
                return json.loads(self.choices)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_choices(self, choices: List[str]):
        """Définit les choix"""
        self.choices = json.dumps(choices)
    
    def validate_answer(self, user_answer: str) -> bool:
        """Valide la réponse utilisateur"""
        if not self.case_sensitive:
            user_answer = user_answer.lower().strip()
            correct = self.correct_answer.lower().strip()
        else:
            user_answer = user_answer.strip()
            correct = self.correct_answer.strip()
        
        if self.question_type == QuestionType.MULTIPLE_CHOICE.value:
            return user_answer == correct
        elif self.question_type == QuestionType.TRUE_FALSE.value:
            return user_answer.lower() in ['true', 'false', 'vrai', 'faux'] and user_answer == correct
        elif self.question_type == QuestionType.SHORT_ANSWER.value:
            # Pour les réponses courtes, permettre une certaine flexibilité
            return user_answer == correct or user_answer in correct.split('|')  # Multiple bonnes réponses séparées par |
        
        return user_answer == correct

class UserQuest(Base):
    """Progression des utilisateurs dans les quêtes"""
    
    __tablename__ = "user_quests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    quest_id = Column(Integer, ForeignKey("quests.id"), nullable=False)
    
    # Statut et progression
    status = Column(String(20), default=UserQuestStatus.NOT_STARTED.value)
    current_step = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    
    # Scores et performance
    score = Column(Float, default=0.0)
    max_score = Column(Float, default=0.0)
    attempts = Column(Integer, default=0)
    hints_used = Column(Integer, default=0)
    
    # Temps
    time_spent = Column(Integer, default=0)  # minutes
    estimated_remaining_time = Column(Integer)
    
    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    last_activity = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    user = relationship("User")
    quest = relationship("Quest", back_populates="user_quests")
    step_progress = relationship("UserStepProgress", back_populates="user_quest", cascade="all, delete-orphan")
    
    # Contraintes
    __table_args__ = (
        Index('idx_user_quest_unique', 'user_id', 'quest_id', unique=True),
        Index('idx_user_quest_status', 'user_id', 'status'),
        CheckConstraint('progress_percentage >= 0.0 AND progress_percentage <= 1.0', name='valid_progress'),
    )
    
    def __repr__(self):
        return f"<UserQuest(id={self.id}, user_id={self.user_id}, quest_id={self.quest_id}, status='{self.status}')>"
    
    def start_quest(self):
        """Démarre la quête"""
        if self.status == UserQuestStatus.NOT_STARTED.value:
            self.status = UserQuestStatus.IN_PROGRESS.value
            self.started_at = datetime.now(timezone.utc)
            self.last_activity = datetime.now(timezone.utc)
    
    def complete_quest(self, final_score: float):
        """Complète la quête"""
        self.status = UserQuestStatus.COMPLETED.value
        self.completed_at = datetime.now(timezone.utc)
        self.progress_percentage = 1.0
        self.score = final_score
        self.last_activity = datetime.now(timezone.utc)
    
    def fail_quest(self):
        """Marque la quête comme échouée"""
        self.status = UserQuestStatus.FAILED.value
        self.last_activity = datetime.now(timezone.utc)
    
    def update_progress(self, step_number: int, step_score: float = 0.0):
        """Met à jour la progression"""
        self.current_step = max(self.current_step, step_number)
        self.last_activity = datetime.now(timezone.utc)
        
        if self.quest:
            self.progress_percentage = min(1.0, self.current_step / self.quest.total_steps)
            
            # Mettre à jour le score total
            completed_steps = len([sp for sp in self.step_progress if sp.is_completed])
            if completed_steps > 0:
                total_score = sum(sp.score for sp in self.step_progress if sp.is_completed)
                self.score = total_score / completed_steps
    
    def add_time_spent(self, minutes: int):
        """Ajoute du temps passé"""
        self.time_spent += minutes
        self.last_activity = datetime.now(timezone.utc)
    
    def use_hint(self):
        """Enregistre l'utilisation d'un indice"""
        self.hints_used += 1
        self.last_activity = datetime.now(timezone.utc)

class UserStepProgress(Base):
    """Progression dans les étapes"""
    
    __tablename__ = "user_step_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_quest_id = Column(Integer, ForeignKey("user_quests.id"), nullable=False)
    step_id = Column(Integer, ForeignKey("quest_steps.id"), nullable=False)
    
    # Progression
    is_completed = Column(Boolean, default=False)
    is_skipped = Column(Boolean, default=False)
    attempts = Column(Integer, default=0)
    score = Column(Float, default=0.0)
    
    # Temps
    time_spent = Column(Integer, default=0)  # minutes
    
    # Contenu utilisateur
    user_code = Column(Text)  # Code écrit par l'utilisateur
    user_notes = Column(Text)  # Notes personnelles
    
    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    last_attempt_at = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    user_quest = relationship("UserQuest", back_populates="step_progress")
    step = relationship("QuestStep", back_populates="user_step_progress")
    answers = relationship("UserAnswer", back_populates="step_progress", cascade="all, delete-orphan")
    
    # Contraintes
    __table_args__ = (
        Index('idx_user_step_unique', 'user_quest_id', 'step_id', unique=True),
    )
    
    def __repr__(self):
        return f"<UserStepProgress(id={self.id}, user_quest_id={self.user_quest_id}, step_id={self.step_id})>"
    
    def start_step(self):
        """Démarre l'étape"""
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc)
    
    def complete_step(self, score: float):
        """Complète l'étape"""
        self.is_completed = True
        self.completed_at = datetime.now(timezone.utc)
        self.score = score
        self.last_attempt_at = datetime.now(timezone.utc)
    
    def skip_step(self):
        """Ignore l'étape"""
        self.is_skipped = True
        self.completed_at = datetime.now(timezone.utc)
    
    def add_attempt(self, score: float = 0.0):
        """Ajoute une tentative"""
        self.attempts += 1
        self.last_attempt_at = datetime.now(timezone.utc)
        if score > self.score:
            self.score = score

class UserAnswer(Base):
    """Réponses utilisateur aux questions"""
    
    __tablename__ = "user_answers"
    
    id = Column(Integer, primary_key=True, index=True)
    step_progress_id = Column(Integer, ForeignKey("user_step_progress.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    
    # Réponse
    answer_text = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    points_earned = Column(Integer, default=0)
    
    # Métadonnées
    attempt_number = Column(Integer, default=1)
    time_taken = Column(Integer)  # secondes
    hint_used = Column(Boolean, default=False)
    
    # Timestamps
    answered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relations
    step_progress = relationship("UserStepProgress", back_populates="answers")
    question = relationship("Question", back_populates="user_answers")
    
    def __repr__(self):
        return f"<UserAnswer(id={self.id}, question_id={self.question_id}, correct={self.is_correct})>"

# ===== ÉVÉNEMENTS SQLALCHEMY =====

@event.listens_for(Quest, 'before_update')
def update_quest_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification"""
    target.updated_at = datetime.now(timezone.utc)

@event.listens_for(QuestStep, 'before_update')
def update_step_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification"""
    target.updated_at = datetime.now(timezone.utc)

@event.listens_for(UserQuest, 'before_update')
def update_user_quest_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification"""
    target.updated_at = datetime.now(timezone.utc)

@event.listens_for(UserStepProgress, 'before_update')
def update_step_progress_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification"""
    target.updated_at = datetime.now(timezone.utc)