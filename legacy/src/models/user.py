# src/models/user.py
"""
Modèle utilisateur pour l'assistant pédagogique IA
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
import json

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy import event
import uuid

from pydantic import BaseModel, EmailStr, validator, Field
from passlib.context import CryptContext
import bcrypt

# Base SQLAlchemy
Base = declarative_base()

# Configuration du hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserLevel(str, Enum):
    """Niveaux d'utilisateur"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class UserRole(str, Enum):
    """Rôles d'utilisateur"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class AccountStatus(str, Enum):
    """Statuts de compte"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class LearningStyle(str, Enum):
    """Styles d'apprentissage"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING = "reading"

# ===== MODÈLES SQLALCHEMY =====

class User(Base):
    """Modèle utilisateur principal"""
    
    __tablename__ = "users"
    
    # Identifiants
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    
    # Authentification
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(String(20), default=UserRole.STUDENT.value)
    
    # Informations personnelles
    first_name = Column(String(100))
    last_name = Column(String(100))
    display_name = Column(String(150))
    avatar_url = Column(String(500))
    bio = Column(Text)
    
    # Préférences d'apprentissage
    level = Column(String(20), default=UserLevel.BEGINNER.value)
    learning_style = Column(String(20), default=LearningStyle.VISUAL.value)
    preferred_languages = Column(Text)  # JSON string
    interests = Column(Text)  # JSON string
    
    # Métadonnées
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime)
    last_activity = Column(DateTime)
    
    # Statistiques
    total_sessions = Column(Integer, default=0)
    total_quests_completed = Column(Integer, default=0)
    total_study_time = Column(Integer, default=0)  # en minutes
    current_streak = Column(Integer, default=0)  # jours consécutifs
    longest_streak = Column(Integer, default=0)
    
    # Configuration
    settings = Column(Text)  # JSON string pour les paramètres utilisateur
    
    # Relations
    profiles = relationship("UserProfile", back_populates="user", cascade="all, delete-orphan")
    progress_records = relationship("UserProgress", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    achievements = relationship("UserAchievement", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
    
    def set_password(self, password: str):
        """Définit le mot de passe haché"""
        self.hashed_password = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Vérifie le mot de passe"""
        return pwd_context.verify(password, self.hashed_password)
    
    def get_full_name(self) -> str:
        """Retourne le nom complet"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.username
    
    def get_preferred_languages(self) -> List[str]:
        """Retourne la liste des langages préférés"""
        if self.preferred_languages:
            try:
                return json.loads(self.preferred_languages)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_preferred_languages(self, languages: List[str]):
        """Définit les langages préférés"""
        self.preferred_languages = json.dumps(languages)
    
    def get_interests(self) -> List[str]:
        """Retourne la liste des centres d'intérêt"""
        if self.interests:
            try:
                return json.loads(self.interests)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_interests(self, interests: List[str]):
        """Définit les centres d'intérêt"""
        self.interests = json.dumps(interests)
    
    def get_settings(self) -> Dict[str, Any]:
        """Retourne les paramètres utilisateur"""
        if self.settings:
            try:
                return json.loads(self.settings)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_settings(self, settings: Dict[str, Any]):
        """Définit les paramètres utilisateur"""
        self.settings = json.dumps(settings)
    
    def update_activity(self):
        """Met à jour l'activité de l'utilisateur"""
        self.last_activity = datetime.now(timezone.utc)
    
    def increment_session_count(self):
        """Incrémente le nombre de sessions"""
        self.total_sessions += 1
    
    def add_study_time(self, minutes: int):
        """Ajoute du temps d'étude"""
        self.total_study_time += minutes
    
    def complete_quest(self):
        """Marque une quête comme complétée"""
        self.total_quests_completed += 1
    
    def update_streak(self, days: int):
        """Met à jour la série de jours consécutifs"""
        self.current_streak = days
        if days > self.longest_streak:
            self.longest_streak = days

class UserProfile(Base):
    """Profil détaillé de l'utilisateur"""
    
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Informations démographiques
    age = Column(Integer)
    country = Column(String(2))  # Code pays ISO
    timezone = Column(String(50))
    language = Column(String(10), default="fr")
    
    # Objectifs d'apprentissage
    learning_goals = Column(Text)  # JSON string
    target_skills = Column(Text)  # JSON string
    available_time_per_week = Column(Integer)  # minutes par semaine
    
    # Préférences pédagogiques
    difficulty_preference = Column(String(20))
    feedback_frequency = Column(String(20))  # immediate, periodic, final
    reminder_enabled = Column(Boolean, default=True)
    reminder_time = Column(String(5))  # Format HH:MM
    
    # Accessibilité
    accessibility_needs = Column(Text)  # JSON string
    font_size_preference = Column(String(10), default="medium")
    high_contrast_mode = Column(Boolean, default=False)
    screen_reader_support = Column(Boolean, default=False)
    
    # Métadonnées
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    user = relationship("User", back_populates="profiles")
    
    def __repr__(self):
        return f"<UserProfile(id={self.id}, user_id={self.user_id})>"
    
    def get_learning_goals(self) -> List[str]:
        """Retourne les objectifs d'apprentissage"""
        if self.learning_goals:
            try:
                return json.loads(self.learning_goals)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_learning_goals(self, goals: List[str]):
        """Définit les objectifs d'apprentissage"""
        self.learning_goals = json.dumps(goals)
    
    def get_target_skills(self) -> List[str]:
        """Retourne les compétences ciblées"""
        if self.target_skills:
            try:
                return json.loads(self.target_skills)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_target_skills(self, skills: List[str]):
        """Définit les compétences ciblées"""
        self.target_skills = json.dumps(skills)
    
    def get_accessibility_needs(self) -> Dict[str, Any]:
        """Retourne les besoins d'accessibilité"""
        if self.accessibility_needs:
            try:
                return json.loads(self.accessibility_needs)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_accessibility_needs(self, needs: Dict[str, Any]):
        """Définit les besoins d'accessibilité"""
        self.accessibility_needs = json.dumps(needs)

class UserSession(Base):
    """Session utilisateur"""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Informations de session
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    device_info = Column(Text)  # JSON string
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_activity = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime)
    
    # Statut
    is_active = Column(Boolean, default=True)
    
    # Relations
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"
    
    def is_expired(self) -> bool:
        """Vérifie si la session est expirée"""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at.replace(tzinfo=timezone.utc)
        return False
    
    def extend_session(self, hours: int = 24):
        """Prolonge la session"""
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Retourne les informations de device"""
        if self.device_info:
            try:
                return json.loads(self.device_info)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_device_info(self, info: Dict[str, Any]):
        """Définit les informations de device"""
        self.device_info = json.dumps(info)

class UserAchievement(Base):
    """Achievements/badges utilisateur"""
    
    __tablename__ = "user_achievements"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Informations du badge
    achievement_id = Column(String(100), nullable=False)  # ID unique du badge
    title = Column(String(200), nullable=False)
    description = Column(Text)
    icon_url = Column(String(500))
    category = Column(String(50))  # learning, streak, milestone, etc.
    
    # Progression
    progress = Column(Float, default=0.0)  # Pourcentage de completion (0.0 à 1.0)
    target_value = Column(Integer)  # Valeur cible pour débloquer
    current_value = Column(Integer, default=0)  # Valeur actuelle
    
    # Statut
    is_unlocked = Column(Boolean, default=False)
    unlocked_at = Column(DateTime)
    
    # Métadonnées
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    user = relationship("User", back_populates="achievements")
    
    def __repr__(self):
        return f"<UserAchievement(id={self.id}, user_id={self.user_id}, title='{self.title}')>"
    
    def update_progress(self, new_value: int):
        """Met à jour la progression"""
        self.current_value = new_value
        if self.target_value:
            self.progress = min(1.0, new_value / self.target_value)
            
            if self.progress >= 1.0 and not self.is_unlocked:
                self.unlock()
    
    def unlock(self):
        """Débloque l'achievement"""
        self.is_unlocked = True
        self.unlocked_at = datetime.now(timezone.utc)
        self.progress = 1.0

# ===== SCHÉMAS PYDANTIC =====

class UserBase(BaseModel):
    """Schéma de base pour l'utilisateur"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=1000)
    level: UserLevel = UserLevel.BEGINNER
    learning_style: LearningStyle = LearningStyle.VISUAL
    preferred_languages: Optional[List[str]] = []
    interests: Optional[List[str]] = []
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum() and '_' not in v and '-' not in v:
            raise ValueError('Le nom d\'utilisateur doit contenir uniquement des lettres, chiffres, _ ou -')
        return v.lower()
    
    @validator('preferred_languages')
    def validate_languages(cls, v):
        if v:
            valid_languages = ['python', 'javascript', 'java', 'cpp', 'sql', 'html', 'css']
            for lang in v:
                if lang.lower() not in valid_languages:
                    raise ValueError(f'Langage non supporté: {lang}')
        return v

class UserCreate(UserBase):
    """Schéma pour la création d'utilisateur"""
    password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Les mots de passe ne correspondent pas')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Le mot de passe doit contenir au moins 8 caractères')
        if not any(c.isupper() for c in v):
            raise ValueError('Le mot de passe doit contenir au moins une majuscule')
        if not any(c.islower() for c in v):
            raise ValueError('Le mot de passe doit contenir au moins une minuscule')
        if not any(c.isdigit() for c in v):
            raise ValueError('Le mot de passe doit contenir au moins un chiffre')
        return v

class UserUpdate(BaseModel):
    """Schéma pour la mise à jour d'utilisateur"""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=1000)
    level: Optional[UserLevel] = None
    learning_style: Optional[LearningStyle] = None
    preferred_languages: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    avatar_url: Optional[str] = None

class UserProfileUpdate(BaseModel):
    """Schéma pour la mise à jour du profil"""
    age: Optional[int] = Field(None, ge=13, le=120)
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    timezone: Optional[str] = None
    language: Optional[str] = Field(None, min_length=2, max_length=10)
    learning_goals: Optional[List[str]] = None
    target_skills: Optional[List[str]] = None
    available_time_per_week: Optional[int] = Field(None, ge=0)
    difficulty_preference: Optional[str] = None
    feedback_frequency: Optional[str] = None
    reminder_enabled: Optional[bool] = None
    reminder_time: Optional[str] = None
    font_size_preference: Optional[str] = None
    high_contrast_mode: Optional[bool] = None
    screen_reader_support: Optional[bool] = None

class UserResponse(UserBase):
    """Schéma de réponse pour l'utilisateur"""
    id: int
    uuid: str
    is_active: bool
    is_verified: bool
    role: UserRole
    avatar_url: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    total_sessions: int
    total_quests_completed: int
    total_study_time: int
    current_streak: int
    longest_streak: int
    
    class Config:
        from_attributes = True

class UserStats(BaseModel):
    """Statistiques utilisateur"""
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
    
class UserDashboard(BaseModel):
    """Données du tableau de bord utilisateur"""
    user: UserResponse
    stats: UserStats
    recent_achievements: List[Dict[str, Any]]
    current_quests: List[Dict[str, Any]]
    study_calendar: Dict[str, int]  # Date -> minutes d'étude
    recommendations: List[Dict[str, Any]]

class PasswordReset(BaseModel):
    """Schéma pour la réinitialisation de mot de passe"""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Schéma pour confirmer la réinitialisation"""
    token: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Les mots de passe ne correspondent pas')
        return v

class LoginRequest(BaseModel):
    """Schéma pour la connexion"""
    username_or_email: str
    password: str
    remember_me: bool = False

class LoginResponse(BaseModel):
    """Schéma de réponse pour la connexion"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

# ===== ÉVÉNEMENTS SQLALCHEMY =====

@event.listens_for(User, 'before_update')
def update_user_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification"""
    target.updated_at = datetime.now(timezone.utc)

@event.listens_for(UserProfile, 'before_update')
def update_profile_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification du profil"""
    target.updated_at = datetime.now(timezone.utc)

# ===== FONCTIONS UTILITAIRES =====

def create_user_with_profile(
    user_data: UserCreate,
    profile_data: Optional[UserProfileUpdate] = None
) -> User:
    """
    Crée un utilisateur avec son profil
    
    Args:
        user_data: Données de l'utilisateur
        profile_data: Données du profil (optionnel)
        
    Returns:
        Utilisateur créé
    """
    # Créer l'utilisateur
    user = User(
        username=user_data.username,
        email=user_data.email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        display_name=user_data.display_name,
        bio=user_data.bio,
        level=user_data.level.value,
        learning_style=user_data.learning_style.value
    )
    
    # Définir le mot de passe
    user.set_password(user_data.password)
    
    # Définir les préférences
    if user_data.preferred_languages:
        user.set_preferred_languages(user_data.preferred_languages)
    
    if user_data.interests:
        user.set_interests(user_data.interests)
    
    # Créer le profil si des données sont fournies
    if profile_data:
        profile = UserProfile(
            user=user,
            age=profile_data.age,
            country=profile_data.country,
            timezone=profile_data.timezone,
            language=profile_data.language,
            available_time_per_week=profile_data.available_time_per_week,
            difficulty_preference=profile_data.difficulty_preference,
            feedback_frequency=profile_data.feedback_frequency,
            reminder_enabled=profile_data.reminder_enabled,
            reminder_time=profile_data.reminder_time,
            font_size_preference=profile_data.font_size_preference,
            high_contrast_mode=profile_data.high_contrast_mode,
            screen_reader_support=profile_data.screen_reader_support
        )
        
        if profile_data.learning_goals:
            profile.set_learning_goals(profile_data.learning_goals)
        
        if profile_data.target_skills:
            profile.set_target_skills(profile_data.target_skills)
        
        user.profiles.append(profile)
    
    return user

def get_user_achievements_summary(user: User) -> Dict[str, Any]:
    """
    Retourne un résumé des achievements de l'utilisateur
    
    Args:
        user: Utilisateur
        
    Returns:
        Résumé des achievements
    """
    achievements = user.achievements
    
    unlocked = [a for a in achievements if a.is_unlocked]
    in_progress = [a for a in achievements if not a.is_unlocked and a.progress > 0]
    
    return {
        "total_achievements": len(achievements),
        "unlocked_count": len(unlocked),
        "in_progress_count": len(in_progress),
        "completion_rate": len(unlocked) / len(achievements) if achievements else 0,
        "recent_unlocked": sorted(unlocked, key=lambda x: x.unlocked_at, reverse=True)[:5],
        "next_to_unlock": sorted(in_progress, key=lambda x: x.progress, reverse=True)[:3]
    }