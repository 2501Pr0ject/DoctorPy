# src/models/progress.py
"""
Modèles pour le suivi de progression et l'analytics
"""

from datetime import datetime, timezone, timedelta, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import json
import uuid

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, Date, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import event, CheckConstraint, func
from sqlalchemy.dialects.postgresql import UUID

from pydantic import BaseModel, validator, Field

from .user import Base  # Utiliser la même base

class ProgressType(str, Enum):
    """Types de progression"""
    QUEST_COMPLETION = "quest_completion"
    SKILL_IMPROVEMENT = "skill_improvement"
    STREAK_MILESTONE = "streak_milestone"
    LEVEL_UP = "level_up"
    ACHIEVEMENT_UNLOCK = "achievement_unlock"
    SESSION_COMPLETE = "session_complete"
    CODE_EXECUTION = "code_execution"
    QUIZ_COMPLETION = "quiz_completion"

class SkillCategory(str, Enum):
    """Catégories de compétences"""
    SYNTAX = "syntax"
    LOGIC = "logic"
    PROBLEM_SOLVING = "problem_solving"
    DEBUGGING = "debugging"
    BEST_PRACTICES = "best_practices"
    ALGORITHMS = "algorithms"
    DATA_STRUCTURES = "data_structures"
    OOP = "object_oriented_programming"
    FUNCTIONAL = "functional_programming"
    TESTING = "testing"
    PERFORMANCE = "performance"

class LearningMetricType(str, Enum):
    """Types de métriques d'apprentissage"""
    TIME_SPENT = "time_spent"
    ACCURACY_RATE = "accuracy_rate"
    COMPLETION_RATE = "completion_rate"
    RETRY_COUNT = "retry_count"
    HINT_USAGE = "hint_usage"
    CODE_QUALITY = "code_quality"
    SPEED_IMPROVEMENT = "speed_improvement"

# ===== MODÈLES SQLALCHEMY =====

class UserProgress(Base):
    """Progression générale de l'utilisateur"""
    
    __tablename__ = "user_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Progression globale
    overall_level = Column(String(20), default="beginner")
    xp_points = Column(Integer, default=0)
    level_progress = Column(Float, default=0.0)  # Pourcentage vers le niveau suivant
    
    # Compétences par catégorie (scores de 0 à 100)
    syntax_score = Column(Float, default=0.0)
    logic_score = Column(Float, default=0.0)
    problem_solving_score = Column(Float, default=0.0)
    debugging_score = Column(Float, default=0.0)
    best_practices_score = Column(Float, default=0.0)
    algorithms_score = Column(Float, default=0.0)
    data_structures_score = Column(Float, default=0.0)
    oop_score = Column(Float, default=0.0)
    functional_score = Column(Float, default=0.0)
    testing_score = Column(Float, default=0.0)
    performance_score = Column(Float, default=0.0)
    
    # Statistiques générales
    total_quests_attempted = Column(Integer, default=0)
    total_quests_completed = Column(Integer, default=0)
    total_study_time = Column(Integer, default=0)  # minutes
    total_code_executions = Column(Integer, default=0)
    successful_code_executions = Column(Integer, default=0)
    
    # Streaks et constance
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    days_active = Column(Integer, default=0)
    last_activity_date = Column(Date)
    
    # Préférences d'apprentissage adaptées
    optimal_session_duration = Column(Integer, default=30)  # minutes
    preferred_difficulty = Column(String(20))
    learning_pace = Column(String(20), default="normal")  # slow, normal, fast
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    user = relationship("User", back_populates="progress_records")
    daily_progress = relationship("DailyProgress", back_populates="user_progress", cascade="all, delete-orphan")
    skill_assessments = relationship("SkillAssessment", back_populates="user_progress", cascade="all, delete-orphan")
    learning_sessions = relationship("LearningSession", back_populates="user_progress", cascade="all, delete-orphan")
    progress_milestones = relationship("ProgressMilestone", back_populates="user_progress", cascade="all, delete-orphan")
    
    # Contraintes
    __table_args__ = (
        CheckConstraint('xp_points >= 0', name='positive_xp'),
        CheckConstraint('level_progress >= 0.0 AND level_progress <= 1.0', name='valid_level_progress'),
        CheckConstraint('current_streak >= 0', name='positive_streak'),
        Index('idx_user_progress_user', 'user_id', unique=True),
    )
    
    def __repr__(self):
        return f"<UserProgress(id={self.id}, user_id={self.user_id}, level='{self.overall_level}', xp={self.xp_points})>"
    
    def add_xp(self, points: int):
        """Ajoute des points d'expérience"""
        self.xp_points += points
        self._update_level()
    
    def _update_level(self):
        """Met à jour le niveau basé sur l'XP"""
        # Système de progression : 1000 XP pour beginner->intermediate, 2500 pour intermediate->advanced
        if self.xp_points < 1000:
            self.overall_level = "beginner"
            self.level_progress = self.xp_points / 1000.0
        elif self.xp_points < 3500:  # 1000 + 2500
            self.overall_level = "intermediate"
            self.level_progress = (self.xp_points - 1000) / 2500.0
        else:
            self.overall_level = "advanced"
            self.level_progress = min(1.0, (self.xp_points - 3500) / 5000.0)  # Progression continue
    
    def update_skill_score(self, skill: SkillCategory, score: float):
        """Met à jour le score d'une compétence"""
        skill_attr = f"{skill.value}_score"
        if hasattr(self, skill_attr):
            setattr(self, skill_attr, max(0.0, min(100.0, score)))
    
    def get_skill_scores(self) -> Dict[str, float]:
        """Retourne tous les scores de compétences"""
        return {
            skill.value: getattr(self, f"{skill.value}_score", 0.0)
            for skill in SkillCategory
        }
    
    def calculate_overall_score(self) -> float:
        """Calcule le score global"""
        scores = list(self.get_skill_scores().values())
        return sum(scores) / len(scores) if scores else 0.0
    
    def update_daily_activity(self, study_minutes: int = 0):
        """Met à jour l'activité quotidienne"""
        today = date.today()
        
        if self.last_activity_date != today:
            # Nouveau jour
            if self.last_activity_date == today - timedelta(days=1):
                # Jour consécutif
                self.current_streak += 1
            else:
                # Rupture de streak
                self.current_streak = 1
            
            self.days_active += 1
            self.last_activity_date = today
            
            if self.current_streak > self.longest_streak:
                self.longest_streak = self.current_streak
        
        if study_minutes > 0:
            self.total_study_time += study_minutes
    
    def get_completion_rate(self) -> float:
        """Calcule le taux de completion des quêtes"""
        if self.total_quests_attempted > 0:
            return self.total_quests_completed / self.total_quests_attempted
        return 0.0
    
    def get_code_success_rate(self) -> float:
        """Calcule le taux de succès du code"""
        if self.total_code_executions > 0:
            return self.successful_code_executions / self.total_code_executions
        return 0.0

class DailyProgress(Base):
    """Progression quotidienne détaillée"""
    
    __tablename__ = "daily_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_progress_id = Column(Integer, ForeignKey("user_progress.id"), nullable=False)
    date = Column(Date, nullable=False)
    
    # Activité du jour
    study_time_minutes = Column(Integer, default=0)
    quests_started = Column(Integer, default=0)
    quests_completed = Column(Integer, default=0)
    questions_answered = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    
    # Code et pratique
    code_executions = Column(Integer, default=0)
    successful_executions = Column(Integer, default=0)
    lines_of_code = Column(Integer, default=0)
    
    # XP et progression
    xp_earned = Column(Integer, default=0)
    achievements_unlocked = Column(Integer, default=0)
    skills_improved = Column(Text)  # JSON array des compétences améliorées
    
    # Engagement
    sessions_count = Column(Integer, default=0)
    average_session_duration = Column(Float, default=0.0)  # minutes
    hints_used = Column(Integer, default=0)
    
    # Métadonnées
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relations
    user_progress = relationship("UserProgress", back_populates="daily_progress")
    
    # Contraintes
    __table_args__ = (
        Index('idx_daily_progress_user_date', 'user_progress_id', 'date', unique=True),
        Index('idx_daily_progress_date', 'date'),
    )
    
    def __repr__(self):
        return f"<DailyProgress(user_progress_id={self.user_progress_id}, date={self.date})>"
    
    def get_accuracy_rate(self) -> float:
        """Calcule le taux de précision du jour"""
        if self.questions_answered > 0:
            return self.correct_answers / self.questions_answered
        return 0.0
    
    def get_completion_rate(self) -> float:
        """Calcule le taux de completion du jour"""
        if self.quests_started > 0:
            return self.quests_completed / self.quests_started
        return 0.0
    
    def get_skills_improved(self) -> List[str]:
        """Retourne les compétences améliorées"""
        if self.skills_improved:
            try:
                return json.loads(self.skills_improved)
            except json.JSONDecodeError:
                return []
        return []
    
    def add_skill_improved(self, skill: str):
        """Ajoute une compétence améliorée"""
        skills = self.get_skills_improved()
        if skill not in skills:
            skills.append(skill)
            self.skills_improved = json.dumps(skills)

class SkillAssessment(Base):
    """Évaluations de compétences spécifiques"""
    
    __tablename__ = "skill_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_progress_id = Column(Integer, ForeignKey("user_progress.id"), nullable=False)
    
    # Compétence évaluée
    skill_category = Column(String(50), nullable=False)
    skill_name = Column(String(100), nullable=False)
    
    # Évaluation
    assessment_type = Column(String(50))  # quiz, coding_challenge, peer_review
    score = Column(Float, nullable=False)  # 0.0 à 100.0
    max_score = Column(Float, default=100.0)
    
    # Contexte
    quest_id = Column(Integer, ForeignKey("quests.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    
    # Détails de performance
    response_time = Column(Integer)  # secondes
    attempts_count = Column(Integer, default=1)
    hints_used = Column(Integer, default=0)
    difficulty_level = Column(String(20))
    
    # Feedback automatique
    strengths = Column(Text)  # JSON array
    weaknesses = Column(Text)  # JSON array
    recommendations = Column(Text)  # JSON array
    
    # Timestamps
    assessed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relations
    user_progress = relationship("UserProgress", back_populates="skill_assessments")
    quest = relationship("Quest", foreign_keys=[quest_id])
    question = relationship("Question", foreign_keys=[question_id])
    
    # Contraintes
    __table_args__ = (
        CheckConstraint('score >= 0.0 AND score <= max_score', name='valid_score'),
        Index('idx_skill_assessment_user_skill', 'user_progress_id', 'skill_category'),
    )
    
    def __repr__(self):
        return f"<SkillAssessment(user_progress_id={self.user_progress_id}, skill='{self.skill_name}', score={self.score})>"
    
    def get_percentage_score(self) -> float:
        """Retourne le score en pourcentage"""
        return (self.score / self.max_score) * 100.0 if self.max_score > 0 else 0.0
    
    def get_strengths(self) -> List[str]:
        """Retourne les points forts"""
        if self.strengths:
            try:
                return json.loads(self.strengths)
            except json.JSONDecodeError:
                return []
        return []
    
    def get_weaknesses(self) -> List[str]:
        """Retourne les points faibles"""
        if self.weaknesses:
            try:
                return json.loads(self.weaknesses)
            except json.JSONDecodeError:
                return []
        return []
    
    def get_recommendations(self) -> List[str]:
        """Retourne les recommandations"""
        if self.recommendations:
            try:
                return json.loads(self.recommendations)
            except json.JSONDecodeError:
                return []
        return []

class LearningSession(Base):
    """Sessions d'apprentissage détaillées"""
    
    __tablename__ = "learning_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_progress_id = Column(Integer, ForeignKey("user_progress.id"), nullable=False)
    session_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    
    # Informations de session
    session_type = Column(String(50))  # quest, practice, review, assessment
    focus_area = Column(String(100))  # Zone de focus principal
    difficulty_level = Column(String(20))
    
    # Durée et timing
    duration_minutes = Column(Integer, nullable=False)
    effective_study_time = Column(Integer)  # Temps réellement productif
    break_time = Column(Integer, default=0)
    
    # Performance
    tasks_completed = Column(Integer, default=0)
    tasks_attempted = Column(Integer, default=0)
    average_accuracy = Column(Float, default=0.0)
    xp_earned = Column(Integer, default=0)
    
    # Engagement et flow
    engagement_score = Column(Float)  # 0.0 à 1.0
    flow_state_duration = Column(Integer, default=0)  # minutes en état de flow
    distraction_count = Column(Integer, default=0)
    
    # Résultats et apprentissage
    concepts_learned = Column(Text)  # JSON array
    skills_practiced = Column(Text)  # JSON array
    mistakes_made = Column(Text)  # JSON array des erreurs communes
    insights_gained = Column(Text)  # JSON array des insights
    
    # Feedback utilisateur
    self_assessment = Column(Integer)  # 1-5 rating
    difficulty_rating = Column(Integer)  # 1-5 rating
    enjoyment_rating = Column(Integer)  # 1-5 rating
    notes = Column(Text)
    
    # Contexte technique
    device_type = Column(String(50))
    browser_info = Column(String(200))
    
    # Timestamps
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relations
    user_progress = relationship("UserProgress", back_populates="learning_sessions")
    session_events = relationship("SessionEvent", back_populates="session", cascade="all, delete-orphan")
    
    # Contraintes
    __table_args__ = (
        CheckConstraint('duration_minutes > 0', name='positive_duration'),
        CheckConstraint('ended_at > started_at', name='valid_session_time'),
        Index('idx_learning_session_user_date', 'user_progress_id', 'started_at'),
    )
    
    def __repr__(self):
        return f"<LearningSession(id={self.id}, user_progress_id={self.user_progress_id}, duration={self.duration_minutes}min)>"
    
    def calculate_completion_rate(self) -> float:
        """Calcule le taux de completion de la session"""
        if self.tasks_attempted > 0:
            return self.tasks_completed / self.tasks_attempted
        return 0.0
    
    def calculate_efficiency(self) -> float:
        """Calcule l'efficacité de la session"""
        if self.duration_minutes > 0:
            return (self.effective_study_time or self.duration_minutes) / self.duration_minutes
        return 0.0
    
    def get_concepts_learned(self) -> List[str]:
        """Retourne les concepts appris"""
        if self.concepts_learned:
            try:
                return json.loads(self.concepts_learned)
            except json.JSONDecodeError:
                return []
        return []
    
    def get_skills_practiced(self) -> List[str]:
        """Retourne les compétences pratiquées"""
        if self.skills_practiced:
            try:
                return json.loads(self.skills_practiced)
            except json.JSONDecodeError:
                return []
        return []

class SessionEvent(Base):
    """Événements détaillés dans une session"""
    
    __tablename__ = "session_events"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("learning_sessions.id"), nullable=False)
    
    # Type d'événement
    event_type = Column(String(50), nullable=False)  # question_answered, hint_used, break_taken, etc.
    event_category = Column(String(50))  # learning, navigation, engagement, error
    
    # Détails de l'événement
    event_data = Column(Text)  # JSON avec détails spécifiques
    context = Column(Text)  # Contexte additionnel
    
    # Métadonnées
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    duration = Column(Integer)  # durée en secondes si applicable
    
    # Relations
    session = relationship("LearningSession", back_populates="session_events")
    
    def __repr__(self):
        return f"<SessionEvent(id={self.id}, session_id={self.session_id}, type='{self.event_type}')>"
    
    def get_event_data(self) -> Dict[str, Any]:
        """Retourne les données de l'événement"""
        if self.event_data:
            try:
                return json.loads(self.event_data)
            except json.JSONDecodeError:
                return {}
        return {}

class ProgressMilestone(Base):
    """Jalons de progression importants"""
    
    __tablename__ = "progress_milestones"
    
    id = Column(Integer, primary_key=True, index=True)
    user_progress_id = Column(Integer, ForeignKey("user_progress.id"), nullable=False)
    
    # Type de jalon
    milestone_type = Column(String(50), nullable=False)
    milestone_name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Valeurs du jalon
    target_value = Column(Float, nullable=False)
    achieved_value = Column(Float, nullable=False)
    unit = Column(String(20))  # points, minutes, quests, etc.
    
    # Importance et catégorie
    importance_level = Column(String(20), default="normal")  # low, normal, high, critical
    category = Column(String(50))  # learning, engagement, skill, achievement
    
    # Récompenses
    xp_reward = Column(Integer, default=0)
    badge_unlocked = Column(String(100))
    special_unlock = Column(Text)  # JSON pour déblocages spéciaux
    
    # Timestamps
    achieved_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relations
    user_progress = relationship("UserProgress", back_populates="progress_milestones")
    
    # Contraintes
    __table_args__ = (
        Index('idx_milestone_user_type', 'user_progress_id', 'milestone_type'),
        Index('idx_milestone_achieved', 'achieved_at'),
    )
    
    def __repr__(self):
        return f"<ProgressMilestone(user_progress_id={self.user_progress_id}, name='{self.milestone_name}')>"
    
    def get_special_unlock(self) -> Dict[str, Any]:
        """Retourne les déblocages spéciaux"""
        if self.special_unlock:
            try:
                return json.loads(self.special_unlock)
            except json.JSONDecodeError:
                return {}
        return {}

class LearningAnalytics(Base):
    """Analytics agrégées pour l'apprentissage"""
    
    __tablename__ = "learning_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Période d'analyse
    period_type = Column(String(20), nullable=False)  # weekly, monthly, quarterly
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    
    # Métriques d'engagement
    total_study_time = Column(Integer, default=0)  # minutes
    average_session_duration = Column(Float, default=0.0)
    sessions_count = Column(Integer, default=0)
    active_days = Column(Integer, default=0)
    consistency_score = Column(Float, default=0.0)  # 0.0 à 1.0
    
    # Métriques de performance
    quests_completed = Column(Integer, default=0)
    average_quest_score = Column(Float, default=0.0)
    questions_answered = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=0.0)
    improvement_rate = Column(Float, default=0.0)  # Pourcentage d'amélioration
    
    # Métriques de progression
    xp_gained = Column(Integer, default=0)
    levels_gained = Column(Integer, default=0)
    skills_improved = Column(Integer, default=0)
    achievements_unlocked = Column(Integer, default=0)
    
    # Patterns d'apprentissage
    preferred_study_times = Column(Text)  # JSON array des créneaux préférés
    learning_velocity = Column(Float, default=0.0)  # Vitesse d'apprentissage
    retention_rate = Column(Float, default=0.0)  # Taux de rétention
    
    # Prédictions et recommandations
    predicted_next_level_date = Column(Date)
    recommended_study_time = Column(Integer)  # minutes par jour
    focus_areas = Column(Text)  # JSON array des zones à améliorer
    
    # Timestamps
    calculated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relations
    user = relationship("User")
    
    # Contraintes
    __table_args__ = (
        Index('idx_analytics_user_period', 'user_id', 'period_type', 'period_start'),
        CheckConstraint('period_end >= period_start', name='valid_period'),
    )
    
    def __repr__(self):
        return f"<LearningAnalytics(user_id={self.user_id}, period='{self.period_type}')>"
    
    def get_preferred_study_times(self) -> List[str]:
        """Retourne les créneaux préférés"""
        if self.preferred_study_times:
            try:
                return json.loads(self.preferred_study_times)
            except json.JSONDecodeError:
                return []
        return []
    
    def get_focus_areas(self) -> List[str]:
        """Retourne les zones de focus recommandées"""
        if self.focus_areas:
            try:
                return json.loads(self.focus_areas)
            except json.JSONDecodeError:
                return []
        return []

# ===== ÉVÉNEMENTS SQLALCHEMY =====

@event.listens_for(UserProgress, 'before_update')
def update_progress_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification"""
    target.updated_at = datetime.now(timezone.utc)

@event.listens_for(DailyProgress, 'before_update')
def update_daily_progress_timestamp(mapper, connection, target):
    """Met à jour le timestamp de modification"""
    target.updated_at = datetime.now(timezone.utc)

# ===== FONCTIONS UTILITAIRES =====

def calculate_learning_score(user_progress: UserProgress) -> Dict[str, float]:
    """
    Calcule un score d'apprentissage global
    
    Args:
        user_progress: Progression de l'utilisateur
        
    Returns:
        Dictionnaire avec différents scores
    """
    # Score de compétences (moyenne pondérée)
    skill_weights = {
        'syntax_score': 0.15,
        'logic_score': 0.20,
        'problem_solving_score': 0.25,
        'debugging_score': 0.15,
        'best_practices_score': 0.10,
        'algorithms_score': 0.15
    }
    
    weighted_skill_score = sum(
        getattr(user_progress, skill) * weight 
        for skill, weight in skill_weights.items()
    )
    
    # Score d'engagement
    engagement_score = min(100.0, (
        (user_progress.current_streak * 5) +
        (user_progress.days_active * 0.5) +
        (user_progress.total_study_time / 60 * 2)  # 2 points par heure
    ))
    
    # Score de progression
    completion_rate = user_progress.get_completion_rate() * 100
    code_success_rate = user_progress.get_code_success_rate() * 100
    
    progression_score = (completion_rate + code_success_rate) / 2
    
    # Score global
    overall_score = (
        weighted_skill_score * 0.4 +
        engagement_score * 0.3 +
        progression_score * 0.3
    )
    
    return {
        'overall_score': overall_score,
        'skill_score': weighted_skill_score,
        'engagement_score': engagement_score,
        'progression_score': progression_score,
        'completion_rate': completion_rate,
        'code_success_rate': code_success_rate
    }

def generate_learning_insights(user_progress: UserProgress, 
                             recent_sessions: List[LearningSession]) -> Dict[str, Any]:
    """
    Génère des insights sur l'apprentissage
    
    Args:
        user_progress: Progression de l'utilisateur
        recent_sessions: Sessions récentes
        
    Returns:
        Insights et recommandations
    """
    insights = {
        'strengths': [],
        'improvement_areas': [],
        'recommendations': [],
        'patterns': {},
        'predictions': {}
    }
    
    # Analyser les forces
    skill_scores = user_progress.get_skill_scores()
    strong_skills = [skill for skill, score in skill_scores.items() if score >= 80.0]
    weak_skills = [skill for skill, score in skill_scores.items() if score < 60.0]
    
    insights['strengths'] = strong_skills
    insights['improvement_areas'] = weak_skills
    
    # Patterns d'apprentissage
    if recent_sessions:
        total_duration = sum(s.duration_minutes for s in recent_sessions)
        avg_duration = total_duration / len(recent_sessions)
        
        insights['patterns'] = {
            'average_session_duration': avg_duration,
            'preferred_session_length': 'short' if avg_duration < 30 else 'medium' if avg_duration < 60 else 'long',
            'consistency': len(recent_sessions) / 7,  # Sessions par semaine
            'engagement_trend': 'improving' if len(recent_sessions) > 3 else 'stable'
        }
    
    # Recommandations
    if weak_skills:
        insights['recommendations'].append(f"Concentrez-vous sur {', '.join(weak_skills[:2])}")
    
    if user_progress.current_streak == 0:
        insights['recommendations'].append("Établissez une routine d'étude quotidienne")
    elif user_progress.current_streak > 7:
        insights['recommendations'].append("Excellent streak ! Maintenez cette constance")
    
    return insights