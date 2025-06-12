"""
Suivi de progression des utilisateurs dans les quêtes.
Analyse des performances, identification des difficultés, recommandations d'amélioration.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from src.core.database import get_session
from src.core.logger import get_logger
from src.core.config import get_settings
from src.models.quest import UserQuest, UserAnswer, UserStepProgress
from src.models.user import User, UserProgress
from src.models.progress import DailyProgress, LearningSession, SessionEvent, SkillAssessment

logger = get_logger(__name__)
settings = get_settings()


class ProgressType(Enum):
    """Types de progression."""
    QUEST_COMPLETION = "quest_completion"
    SKILL_IMPROVEMENT = "skill_improvement"
    LEARNING_STREAK = "learning_streak"
    DIFFICULTY_PROGRESSION = "difficulty_progression"
    TIME_EFFICIENCY = "time_efficiency"


class SkillLevel(Enum):
    """Niveaux de compétence."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ProgressMetrics:
    """Métriques de progression."""
    total_quests_started: int
    total_quests_completed: int
    completion_rate: float
    average_score: float
    total_time_spent: int
    average_time_per_quest: float
    current_streak: int
    longest_streak: int
    skill_improvements: Dict[str, float]
    difficulty_distribution: Dict[str, int]


@dataclass
class LearningPattern:
    """Pattern d'apprentissage détecté."""
    pattern_type: str
    confidence: float
    description: str
    recommendations: List[str]
    supporting_data: Dict[str, Any]


@dataclass
class ProgressInsight:
    """Insight sur la progression."""
    insight_type: str
    title: str
    description: str
    impact_level: str  # low, medium, high
    actionable_steps: List[str]
    data_points: Dict[str, Any]


class ProgressAnalyzer:
    """Analyseur de progression utilisateur."""
    
    def __init__(self):
        self.skill_categories = [
            'syntax', 'logic', 'debugging', 'algorithms', 'data_structures',
            'functions', 'classes', 'modules', 'testing', 'optimization'
        ]
    
    def get_user_progress_metrics(self, user_id: int, days: int = 30) -> ProgressMetrics:
        """
        Calcule les métriques de progression d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days: Période d'analyse en jours
            
        Returns:
            Métriques de progression
        """
        try:
            with get_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Quêtes démarrées et complétées
                total_started = db.query(UserQuest).filter(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.started_at >= cutoff_date
                    )
                ).count()
                
                total_completed = db.query(UserQuest).filter(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.status == 'completed',
                        UserQuest.completed_at >= cutoff_date
                    )
                ).count()
                
                completion_rate = total_completed / max(1, total_started)
                
                # Score moyen
                avg_score_result = db.query(func.avg(UserQuest.score)).filter(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.last_activity >= cutoff_date
                    )
                ).scalar()
                average_score = avg_score_result or 0.0
                
                # Temps total passé
                total_time_result = db.query(func.sum(UserQuest.time_spent)).filter(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.last_activity >= cutoff_date
                    )
                ).scalar()
                total_time_spent = total_time_result or 0
                
                # Temps moyen par quête
                avg_time_per_quest = total_time_spent / max(1, total_started)
                
                # Streaks
                current_streak = self._calculate_current_streak(db, user_id)
                longest_streak = self._calculate_longest_streak(db, user_id, days)
                
                # Améliorations des compétences
                skill_improvements = self._calculate_skill_improvements(db, user_id, days)
                
                # Distribution des difficultés
                difficulty_dist = self._get_difficulty_distribution(db, user_id, cutoff_date)
                
                return ProgressMetrics(
                    total_quests_started=total_started,
                    total_quests_completed=total_completed,
                    completion_rate=completion_rate,
                    average_score=average_score,
                    total_time_spent=total_time_spent,
                    average_time_per_quest=avg_time_per_quest,
                    current_streak=current_streak,
                    longest_streak=longest_streak,
                    skill_improvements=skill_improvements,
                    difficulty_distribution=difficulty_dist
                )
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques: {e}")
            return ProgressMetrics(0, 0, 0.0, 0.0, 0, 0.0, 0, 0, {}, {})
    
    def _calculate_current_streak(self, db: Session, user_id: int) -> int:
        """Calcule le streak actuel de l'utilisateur."""
        try:
            # Récupérer les dernières activités
            recent_progress = db.query(DailyProgress).filter(
                DailyProgress.user_id == user_id
            ).order_by(desc(DailyProgress.date)).limit(30).all()
            
            if not recent_progress:
                return 0
            
            # Calculer le streak
            streak = 0
            last_date = datetime.utcnow().date()
            
            for progress in recent_progress:
                if progress.date == last_date or progress.date == last_date - timedelta(days=1):
                    if progress.quests_completed > 0 or progress.exercises_completed > 0:
                        streak += 1
                        last_date = progress.date - timedelta(days=1)
                    else:
                        break
                else:
                    break
            
            return streak
            
        except Exception as e:
            logger.error(f"Erreur calcul streak: {e}")
            return 0
    
    def _calculate_longest_streak(self, db: Session, user_id: int, days: int) -> int:
        """Calcule le plus long streak sur la période."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            daily_progress = db.query(DailyProgress).filter(
                and_(
                    DailyProgress.user_id == user_id,
                    DailyProgress.date >= cutoff_date.date()
                )
            ).order_by(DailyProgress.date).all()
            
            if not daily_progress:
                return 0
            
            max_streak = 0
            current_streak = 0
            
            for progress in daily_progress:
                if progress.quests_completed > 0 or progress.exercises_completed > 0:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            
            return max_streak
            
        except Exception as e:
            logger.error(f"Erreur calcul longest streak: {e}")
            return 0
    
    def _calculate_skill_improvements(self, db: Session, user_id: int, days: int) -> Dict[str, float]:
        """Calcule l'amélioration des compétences."""
        improvements = {}
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Récupérer les évaluations de compétences
            assessments = db.query(SkillAssessment).filter(
                and_(
                    SkillAssessment.user_id == user_id,
                    SkillAssessment.assessed_at >= cutoff_date
                )
            ).order_by(SkillAssessment.assessed_at).all()
            
            # Grouper par compétence
            skills_data = {}
            for assessment in assessments:
                if assessment.skill_name not in skills_data:
                    skills_data[assessment.skill_name] = []
                skills_data[assessment.skill_name].append(assessment.skill_level)
            
            # Calculer l'amélioration
            for skill, levels in skills_data.items():
                if len(levels) >= 2:
                    improvement = levels[-1] - levels[0]
                    improvements[skill] = improvement
            
        except Exception as e:
            logger.error(f"Erreur calcul améliorations: {e}")
        
        return improvements
    
    def _get_difficulty_distribution(self, db: Session, user_id: int, cutoff_date: datetime) -> Dict[str, int]:
        """Obtient la distribution des difficultés des quêtes complétées."""
        try:
            from src.models.quest import Quest
            
            result = db.query(Quest.difficulty, func.count(UserQuest.id)).join(
                UserQuest, Quest.id == UserQuest.quest_id
            ).filter(
                and_(
                    UserQuest.user_id == user_id,
                    UserQuest.last_activity >= cutoff_date
                )
            ).group_by(Quest.difficulty).all()
            
            return {difficulty: count for difficulty, count in result}
            
        except Exception as e:
            logger.error(f"Erreur distribution difficultés: {e}")
            return {}


class LearningPatternDetector:
    """Détecteur de patterns d'apprentissage."""
    
    def detect_patterns(self, user_id: int, days: int = 30) -> List[LearningPattern]:
        """Détecte les patterns d'apprentissage d'un utilisateur."""
        patterns = []
        
        try:
            with get_session() as db:
                # Pattern 1: Heures d'activité préférées
                time_pattern = self._detect_time_preferences(db, user_id, days)
                if time_pattern:
                    patterns.append(time_pattern)
                
                # Pattern 2: Difficultés avec certains types d'exercices
                difficulty_pattern = self._detect_difficulty_patterns(db, user_id, days)
                if difficulty_pattern:
                    patterns.append(difficulty_pattern)
                
                # Pattern 3: Progression dans les difficultés
                progression_pattern = self._detect_progression_pattern(db, user_id, days)
                if progression_pattern:
                    patterns.append(progression_pattern)
                
                # Pattern 4: Tendance au décrochage
                dropout_pattern = self._detect_dropout_risk(db, user_id, days)
                if dropout_pattern:
                    patterns.append(dropout_pattern)
                
                # Pattern 5: Performance selon la durée des sessions
                session_pattern = self._detect_session_length_pattern(db, user_id, days)
                if session_pattern:
                    patterns.append(session_pattern)
        
        except Exception as e:
            logger.error(f"Erreur détection patterns: {e}")
        
        return patterns
    
    def _detect_time_preferences(self, db: Session, user_id: int, days: int) -> Optional[LearningPattern]:
        """Détecte les préférences horaires."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Récupérer les sessions d'apprentissage
            sessions = db.query(LearningSession).filter(
                and_(
                    LearningSession.user_id == user_id,
                    LearningSession.started_at >= cutoff_date
                )
            ).all()
            
            if len(sessions) < 5:  # Pas assez de données
                return None
            
            # Analyser les heures d'activité
            hours = [session.started_at.hour for session in sessions]
            hour_counts = {}
            for hour in hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            # Trouver les heures préférées
            if hour_counts:
                preferred_hour = max(hour_counts, key=hour_counts.get)
                total_sessions = len(sessions)
                preference_ratio = hour_counts[preferred_hour] / total_sessions
                
                if preference_ratio > 0.3:  # Au moins 30% des sessions
                    time_range = self._get_time_range_description(preferred_hour)
                    
                    return LearningPattern(
                        pattern_type="time_preference",
                        confidence=preference_ratio,
                        description=f"Préfère apprendre {time_range}",
                        recommendations=[
                            f"Planifiez vos sessions d'apprentissage {time_range}",
                            "Créez une routine d'apprentissage régulière"
                        ],
                        supporting_data={
                            "preferred_hour": preferred_hour,
                            "preference_ratio": preference_ratio,
                            "hour_distribution": hour_counts
                        }
                    )
        
        except Exception as e:
            logger.error(f"Erreur détection préférences horaires: {e}")
        
        return None
    
    def _detect_difficulty_patterns(self, db: Session, user_id: int, days: int) -> Optional[LearningPattern]:
        """Détecte les patterns de difficulté."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Analyser les réponses par type d'étape
            answers = db.query(UserAnswer).filter(
                and_(
                    UserAnswer.user_id == user_id,
                    UserAnswer.submitted_at >= cutoff_date
                )
            ).all()
            
            if len(answers) < 10:
                return None
            
            # Grouper par type de contenu (basé sur le feedback ou métadonnées)
            problem_areas = {}
            for answer in answers:
                # Analyser le type d'erreur dans le feedback
                if not answer.is_correct and answer.feedback:
                    error_type = self._categorize_error(answer.feedback)
                    if error_type:
                        if error_type not in problem_areas:
                            problem_areas[error_type] = {'total': 0, 'errors': 0}
                        problem_areas[error_type]['total'] += 1
                        problem_areas[error_type]['errors'] += 1
                    else:
                        # Réponse correcte
                        if 'general' not in problem_areas:
                            problem_areas['general'] = {'total': 0, 'errors': 0}
                        problem_areas['general']['total'] += 1
            
            # Identifier les domaines problématiques
            problematic_areas = []
            for area, stats in problem_areas.items():
                if stats['total'] >= 3:  # Au moins 3 tentatives
                    error_rate = stats['errors'] / stats['total']
                    if error_rate > 0.6:  # Plus de 60% d'erreurs
                        problematic_areas.append((area, error_rate))
            
            if problematic_areas:
                worst_area, error_rate = max(problematic_areas, key=lambda x: x[1])
                
                return LearningPattern(
                    pattern_type="difficulty_area",
                    confidence=error_rate,
                    description=f"Difficultés récurrentes en {worst_area}",
                    recommendations=[
                        f"Revisez les concepts de base en {worst_area}",
                        f"Pratiquez davantage les exercices de {worst_area}",
                        "Demandez de l'aide sur ces concepts spécifiques"
                    ],
                    supporting_data={
                        "problematic_areas": problematic_areas,
                        "worst_area": worst_area,
                        "error_rate": error_rate
                    }
                )
        
        except Exception as e:
            logger.error(f"Erreur détection patterns de difficulté: {e}")
        
        return None
    
    def _detect_progression_pattern(self, db: Session, user_id: int, days: int) -> Optional[LearningPattern]:
        """Détecte le pattern de progression dans les difficultés."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            from src.models.quest import Quest
            
            # Récupérer les quêtes complétées par ordre chronologique
            completed_quests = db.query(UserQuest, Quest.difficulty).join(
                Quest, UserQuest.quest_id == Quest.id
            ).filter(
                and_(
                    UserQuest.user_id == user_id,
                    UserQuest.status == 'completed',
                    UserQuest.completed_at >= cutoff_date
                )
            ).order_by(UserQuest.completed_at).all()
            
            if len(completed_quests) < 3:
                return None
            
            # Mapper les difficultés à des valeurs numériques
            difficulty_values = {
                'beginner': 1,
                'intermediate': 2,
                'advanced': 3,
                'expert': 4
            }
            
            progression = [difficulty_values.get(quest.difficulty, 1) for _, quest in completed_quests]
            
            # Analyser la tendance
            if len(progression) >= 3:
                # Calculer la corrélation avec le temps
                x = list(range(len(progression)))
                correlation = self._calculate_correlation(x, progression)
                
                if correlation > 0.5:
                    return LearningPattern(
                        pattern_type="positive_progression",
                        confidence=correlation,
                        description="Progression constante vers des difficultés plus élevées",
                        recommendations=[
                            "Continuez à vous challenger avec des quêtes plus difficiles",
                            "Maintenez cette excellente progression",
                            "Explorez des sujets avancés"
                        ],
                        supporting_data={
                            "correlation": correlation,
                            "progression": progression,
                            "latest_difficulty": progression[-1]
                        }
                    )
                elif correlation < -0.3:
                    return LearningPattern(
                        pattern_type="regression",
                        confidence=abs(correlation),
                        description="Tendance à revenir vers des difficultés plus simples",
                        recommendations=[
                            "Consolidez vos acquis avant de progresser",
                            "Revisez les concepts intermédiaires",
                            "Prenez le temps nécessaire pour chaque niveau"
                        ],
                        supporting_data={
                            "correlation": correlation,
                            "progression": progression
                        }
                    )
        
        except Exception as e:
            logger.error(f"Erreur détection progression: {e}")
        
        return None
    
    def _detect_dropout_risk(self, db: Session, user_id: int, days: int) -> Optional[LearningPattern]:
        """Détecte le risque de décrochage."""
        try:
            # Analyser l'activité récente
            recent_activity = db.query(DailyProgress).filter(
                and_(
                    DailyProgress.user_id == user_id,
                    DailyProgress.date >= (datetime.utcnow() - timedelta(days=7)).date()
                )
            ).all()
            
            # Analyser les sessions abandonnées
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            sessions = db.query(LearningSession).filter(
                and_(
                    LearningSession.user_id == user_id,
                    LearningSession.started_at >= cutoff_date
                )
            ).all()
            
            if not sessions:
                return None
            
            # Calculer les métriques de risque
            abandoned_sessions = [s for s in sessions if s.ended_at and s.engagement_score < 0.3]
            abandonment_rate = len(abandoned_sessions) / len(sessions)
            
            # Activité décroissante
            days_without_activity = 0
            if recent_activity:
                last_activity = max(recent_activity, key=lambda x: x.date).date
                days_without_activity = (datetime.utcnow().date() - last_activity).days
            else:
                days_without_activity = 7
            
            # Évaluer le risque
            risk_score = (abandonment_rate * 0.6) + (min(days_without_activity, 7) / 7 * 0.4)
            
            if risk_score > 0.5:
                return LearningPattern(
                    pattern_type="dropout_risk",
                    confidence=risk_score,
                    description="Risque de décrochage détecté",
                    recommendations=[
                        "Réduisez la durée des sessions d'apprentissage",
                        "Choisissez des quêtes plus courtes et engageantes",
                        "Fixez-vous des objectifs plus petits et atteignables",
                        "Trouvez un partenaire d'apprentissage ou rejoignez une communauté"
                    ],
                    supporting_data={
                        "risk_score": risk_score,
                        "abandonment_rate": abandonment_rate,
                        "days_without_activity": days_without_activity,
                        "abandoned_sessions": len(abandoned_sessions)
                    }
                )
        
        except Exception as e:
            logger.error(f"Erreur détection risque décrochage: {e}")
        
        return None
    
    def _detect_session_length_pattern(self, db: Session, user_id: int, days: int) -> Optional[LearningPattern]:
        """Détecte le pattern optimal de durée de session."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            sessions = db.query(LearningSession).filter(
                and_(
                    LearningSession.user_id == user_id,
                    LearningSession.started_at >= cutoff_date,
                    LearningSession.ended_at.isnot(None)
                )
            ).all()
            
            if len(sessions) < 5:
                return None
            
            # Analyser la relation durée/performance
            session_data = []
            for session in sessions:
                duration = (session.ended_at - session.started_at).total_seconds() / 60  # en minutes
                engagement = session.engagement_score or 0.5
                session_data.append((duration, engagement))
            
            if len(session_data) >= 5:
                durations, engagements = zip(*session_data)
                
                # Trouver la durée optimale
                optimal_range = self._find_optimal_duration_range(list(durations), list(engagements))
                
                if optimal_range:
                    min_duration, max_duration, avg_engagement = optimal_range
                    
                    return LearningPattern(
                        pattern_type="optimal_session_length",
                        confidence=0.7,  # Confiance modérée
                        description=f"Performance optimale avec des sessions de {min_duration:.0f}-{max_duration:.0f} minutes",
                        recommendations=[
                            f"Planifiez des sessions de {min_duration:.0f} à {max_duration:.0f} minutes",
                            "Évitez les sessions trop courtes ou trop longues",
                            "Prenez des pauses régulières pendant les longues sessions"
                        ],
                        supporting_data={
                            "optimal_min": min_duration,
                            "optimal_max": max_duration,
                            "average_engagement": avg_engagement,
                            "total_sessions": len(sessions)
                        }
                    )
        
        except Exception as e:
            logger.error(f"Erreur détection pattern de session: {e}")
        
        return None
    
    def _get_time_range_description(self, hour: int) -> str:
        """Convertit une heure en description textuelle."""
        if 6 <= hour < 12:
            return "le matin"
        elif 12 <= hour < 18:
            return "l'après-midi"
        elif 18 <= hour < 22:
            return "en soirée"
        else:
            return "tard le soir ou tôt le matin"
    
    def _categorize_error(self, feedback: str) -> Optional[str]:
        """Catégorise le type d'erreur basé sur le feedback."""
        feedback_lower = feedback.lower()
        
        if any(word in feedback_lower for word in ['syntaxe', 'syntax', 'indentation']):
            return 'syntaxe'
        elif any(word in feedback_lower for word in ['logique', 'logic', 'algorithme']):
            return 'logique'
        elif any(word in feedback_lower for word in ['fonction', 'function', 'paramètre']):
            return 'fonctions'
        elif any(word in feedback_lower for word in ['boucle', 'loop', 'iteration']):
            return 'boucles'
        elif any(word in feedback_lower for word in ['variable', 'type', 'donnée']):
            return 'variables'
        elif any(word in feedback_lower for word in ['liste', 'list', 'dictionnaire', 'dict']):
            return 'structures_donnees'
        
        return None
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calcule la corrélation entre deux listes."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _find_optimal_duration_range(self, durations: List[float], engagements: List[float]) -> Optional[Tuple[float, float, float]]:
        """Trouve la plage de durée optimale."""
        if len(durations) < 3:
            return None
        
        # Grouper par tranches de durée
        duration_groups = {}
        for duration, engagement in zip(durations, engagements):
            group = int(duration // 15) * 15  # Groupes de 15 minutes
            if group not in duration_groups:
                duration_groups[group] = []
            duration_groups[group].append(engagement)
        
        # Trouver le groupe avec le meilleur engagement moyen
        best_group = None
        best_engagement = 0
        
        for group, group_engagements in duration_groups.items():
            if len(group_engagements) >= 2:  # Au moins 2 sessions
                avg_engagement = statistics.mean(group_engagements)
                if avg_engagement > best_engagement:
                    best_engagement = avg_engagement
                    best_group = group
        
        if best_group is not None:
            return (best_group, best_group + 15, best_engagement)
        
        return None


class ProgressInsightGenerator:
    """Générateur d'insights sur la progression."""
    
    def __init__(self):
        self.analyzer = ProgressAnalyzer()
        self.pattern_detector = LearningPatternDetector()
    
    def generate_insights(self, user_id: int, days: int = 30) -> List[ProgressInsight]:
        """Génère des insights personnalisés pour un utilisateur."""
        insights = []
        
        try:
            # Récupérer les métriques et patterns
            metrics = self.analyzer.get_user_progress_metrics(user_id, days)
            patterns = self.pattern_detector.detect_patterns(user_id, days)
            
            # Insight sur le taux de completion
            if metrics.total_quests_started > 0:
                completion_insight = self._generate_completion_insight(metrics)
                if completion_insight:
                    insights.append(completion_insight)
            
            # Insight sur les streaks
            streak_insight = self._generate_streak_insight(metrics)
            if streak_insight:
                insights.append(streak_insight)
            
            # Insights basés sur les patterns
            for pattern in patterns:
                pattern_insight = self._generate_pattern_insight(pattern)
                if pattern_insight:
                    insights.append(pattern_insight)
            
            # Insight sur l'amélioration des compétences
            skills_insight = self._generate_skills_insight(metrics)
            if skills_insight:
                insights.append(skills_insight)
            
            # Insight sur la progression en difficulté
            difficulty_insight = self._generate_difficulty_insight(metrics)
            if difficulty_insight:
                insights.append(difficulty_insight)
        
        except Exception as e:
            logger.error(f"Erreur génération insights: {e}")
        
        return insights
    
    def _generate_completion_insight(self, metrics: ProgressMetrics) -> Optional[ProgressInsight]:
        """Génère un insight sur le taux de completion."""
        if metrics.completion_rate > 0.8:
            return ProgressInsight(
                insight_type="completion_success",
                title="Excellent taux de completion !",
                description=f"Vous complétez {metrics.completion_rate:.1%} de vos quêtes, c'est fantastique !",
                impact_level="high",
                actionable_steps=[
                    "Continuez sur cette lancée",
                    "Essayez des défis plus difficiles",
                    "Partagez vos stratégies avec d'autres apprenants"
                ],
                data_points={
                    "completion_rate": metrics.completion_rate,
                    "completed": metrics.total_quests_completed,
                    "started": metrics.total_quests_started
                }
            )
        elif metrics.completion_rate < 0.4:
            return ProgressInsight(
                insight_type="completion_struggle",
                title="Difficultés à terminer les quêtes",
                description=f"Vous ne terminez que {metrics.completion_rate:.1%} de vos quêtes. Explorons comment améliorer cela.",
                impact_level="high",
                actionable_steps=[
                    "Choisissez des quêtes plus courtes pour commencer",
                    "Décomposez les grandes quêtes en petites étapes",
                    "Fixez-vous un objectif de completion par semaine",
                    "Demandez de l'aide quand vous êtes bloqué"
                ],
                data_points={
                    "completion_rate": metrics.completion_rate,
                    "abandoned": metrics.total_quests_started - metrics.total_quests_completed
                }
            )
        
        return None
    
    def _generate_streak_insight(self, metrics: ProgressMetrics) -> Optional[ProgressInsight]:
        """Génère un insight sur les streaks."""
        if metrics.current_streak >= 7:
            return ProgressInsight(
                insight_type="streak_excellent",
                title="Streak impressionnant !",
                description=f"Vous maintenez un streak de {metrics.current_streak} jours. Votre constance est remarquable !",
                impact_level="high",
                actionable_steps=[
                    "Maintenez cette routine quotidienne",
                    "Célébrez cette réussite",
                    "Inspirez d'autres apprenants"
                ],
                data_points={
                    "current_streak": metrics.current_streak,
                    "longest_streak": metrics.longest_streak
                }
            )
        elif metrics.current_streak == 0 and metrics.longest_streak > 0:
            return ProgressInsight(
                insight_type="streak_broken",
                title="Temps de redémarrer votre streak",
                description=f"Votre plus long streak était de {metrics.longest_streak} jours. Vous pouvez le battre !",
                impact_level="medium",
                actionable_steps=[
                    "Commencez par une session courte aujourd'hui",
                    "Planifiez 15 minutes d'apprentissage quotidien",
                    "Utilisez des rappels pour maintenir la constance"
                ],
                data_points={
                    "longest_streak": metrics.longest_streak,
                    "current_streak": metrics.current_streak
                }
            )
        
        return None
    
    def _generate_pattern_insight(self, pattern: LearningPattern) -> Optional[ProgressInsight]:
        """Génère un insight basé sur un pattern détecté."""
        impact_level = "high" if pattern.confidence > 0.7 else "medium" if pattern.confidence > 0.4 else "low"
        
        if pattern.pattern_type == "time_preference":
            return ProgressInsight(
                insight_type="optimal_timing",
                title="Votre créneau optimal identifié",
                description=pattern.description,
                impact_level=impact_level,
                actionable_steps=pattern.recommendations,
                data_points=pattern.supporting_data
            )
        elif pattern.pattern_type == "difficulty_area":
            return ProgressInsight(
                insight_type="improvement_area",
                title="Zone d'amélioration identifiée",
                description=pattern.description,
                impact_level="high",
                actionable_steps=pattern.recommendations,
                data_points=pattern.supporting_data
            )
        elif pattern.pattern_type == "dropout_risk":
            return ProgressInsight(
                insight_type="motivation_support",
                title="Restons motivés ensemble !",
                description="Il semble que vous rencontriez quelques difficultés récemment. C'est normal !",
                impact_level="high",
                actionable_steps=pattern.recommendations,
                data_points=pattern.supporting_data
            )
        elif pattern.pattern_type == "positive_progression":
            return ProgressInsight(
                insight_type="progression_success",
                title="Progression excellente !",
                description=pattern.description,
                impact_level="high",
                actionable_steps=pattern.recommendations,
                data_points=pattern.supporting_data
            )
        
        return None
    
    def _generate_skills_insight(self, metrics: ProgressMetrics) -> Optional[ProgressInsight]:
        """Génère un insight sur l'amélioration des compétences."""
        if metrics.skill_improvements:
            best_improvement = max(metrics.skill_improvements.items(), key=lambda x: x[1])
            skill_name, improvement = best_improvement
            
            if improvement > 10:  # Amélioration significative
                return ProgressInsight(
                    insight_type="skill_improvement",
                    title=f"Progrès remarquables en {skill_name} !",
                    description=f"Vos compétences en {skill_name} se sont améliorées de {improvement:.1f} points.",
                    impact_level="medium",
                    actionable_steps=[
                        f"Continuez à pratiquer {skill_name}",
                        f"Explorez des concepts avancés de {skill_name}",
                        "Appliquez ces compétences dans des projets personnels"
                    ],
                    data_points={
                        "skill": skill_name,
                        "improvement": improvement,
                        "all_improvements": metrics.skill_improvements
                    }
                )
        
        return None
    
    def _generate_difficulty_insight(self, metrics: ProgressMetrics) -> Optional[ProgressInsight]:
        """Génère un insight sur la progression en difficulté."""
        if metrics.difficulty_distribution:
            total_quests = sum(metrics.difficulty_distribution.values())
            
            if total_quests > 0:
                advanced_ratio = (
                    metrics.difficulty_distribution.get('advanced', 0) + 
                    metrics.difficulty_distribution.get('expert', 0)
                ) / total_quests
                
                if advanced_ratio > 0.3:
                    return ProgressInsight(
                        insight_type="difficulty_mastery",
                        title="Vous maîtrisez les niveaux avancés !",
                        description=f"{advanced_ratio:.1%} de vos quêtes sont de niveau avancé ou expert.",
                        impact_level="high",
                        actionable_steps=[
                            "Explorez des domaines spécialisés",
                            "Créez vos propres projets complexes",
                            "Mentionnez d'autres apprenants",
                            "Contribuez à des projets open source"
                        ],
                        data_points={
                            "advanced_ratio": advanced_ratio,
                            "distribution": metrics.difficulty_distribution
                        }
                    )
                elif metrics.difficulty_distribution.get('beginner', 0) / total_quests > 0.8:
                    return ProgressInsight(
                        insight_type="difficulty_progression",
                        title="Prêt pour le niveau supérieur ?",
                        description="Vous excellez au niveau débutant. Il est temps de vous challenger !",
                        impact_level="medium",
                        actionable_steps=[
                            "Essayez quelques quêtes de niveau intermédiaire",
                            "Augmentez progressivement la difficulté",
                            "N'ayez pas peur de l'échec - c'est ainsi qu'on apprend"
                        ],
                        data_points={
                            "beginner_ratio": metrics.difficulty_distribution.get('beginner', 0) / total_quests,
                            "distribution": metrics.difficulty_distribution
                        }
                    )
        
        return None


class ProgressTracker:
    """Suivi de progression principal."""
    
    def __init__(self):
        self.analyzer = ProgressAnalyzer()
        self.pattern_detector = LearningPatternDetector()
        self.insight_generator = ProgressInsightGenerator()
    
    def track_session_start(self, user_id: int, quest_id: Optional[str] = None) -> str:
        """Démarre le suivi d'une session d'apprentissage."""
        try:
            with get_session() as db:
                session = LearningSession(
                    user_id=user_id,
                    quest_id=quest_id,
                    started_at=datetime.utcnow(),
                    activity_count=0,
                    engagement_score=0.5
                )
                
                db.add(session)
                db.commit()
                
                logger.info(f"Session démarrée: {session.id} pour utilisateur {user_id}")
                return str(session.id)
                
        except Exception as e:
            logger.error(f"Erreur démarrage session: {e}")
            return ""
    
    def track_session_event(
        self,
        session_id: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None
    ):
        """Enregistre un événement dans une session."""
        try:
            with get_session() as db:
                event = SessionEvent(
                    session_id=session_id,
                    event_type=event_type,
                    event_data=json.dumps(event_data) if event_data else None,
                    timestamp=datetime.utcnow()
                )
                
                db.add(event)
                
                # Mettre à jour les métriques de la session
                session = db.query(LearningSession).filter(
                    LearningSession.id == session_id
                ).first()
                
                if session:
                    session.activity_count += 1
                    session.last_activity = datetime.utcnow()
                    
                    # Calculer l'engagement basé sur le type d'événement
                    engagement_boost = self._calculate_engagement_boost(event_type)
                    session.engagement_score = min(1.0, session.engagement_score + engagement_boost)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Erreur enregistrement événement: {e}")
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """Termine une session et calcule les métriques finales."""
        try:
            with get_session() as db:
                session = db.query(LearningSession).filter(
                    LearningSession.id == session_id
                ).first()
                
                if not session:
                    return {'error': 'Session introuvable'}
                
                session.ended_at = datetime.utcnow()
                
                # Calculer la durée
                duration = (session.ended_at - session.started_at).total_seconds() / 60
                session.duration_minutes = int(duration)
                
                # Calculer les métriques finales
                events_count = db.query(SessionEvent).filter(
                    SessionEvent.session_id == session_id
                ).count()
                
                session.activity_count = events_count
                
                # Mettre à jour les progrès quotidiens
                self._update_daily_progress(db, session.user_id, session)
                
                db.commit()
                
                return {
                    'session_id': session_id,
                    'duration_minutes': session.duration_minutes,
                    'activity_count': session.activity_count,
                    'engagement_score': session.engagement_score,
                    'events': events_count
                }
                
        except Exception as e:
            logger.error(f"Erreur fin de session: {e}")
            return {'error': 'Erreur lors de la fermeture de session'}
    
    def _calculate_engagement_boost(self, event_type: str) -> float:
        """Calcule le boost d'engagement selon le type d'événement."""
        engagement_boosts = {
            'quest_started': 0.1,
            'step_completed': 0.15,
            'answer_correct': 0.2,
            'answer_incorrect': 0.05,
            'hint_used': 0.02,
            'quest_completed': 0.3,
            'idle_time': -0.1,
            'session_abandoned': -0.2
        }
        
        return engagement_boosts.get(event_type, 0.01)
    
    def _update_daily_progress(self, db: Session, user_id: int, session: LearningSession):
        """Met à jour les progrès quotidiens."""
        today = datetime.utcnow().date()
        
        # Récupérer ou créer l'entrée de progrès quotidien
        daily_progress = db.query(DailyProgress).filter(
            and_(
                DailyProgress.user_id == user_id,
                DailyProgress.date == today
            )
        ).first()
        
        if not daily_progress:
            daily_progress = DailyProgress(
                user_id=user_id,
                date=today,
                time_spent_minutes=0,
                quests_completed=0,
                exercises_completed=0,
                score_gained=0,
                streak_maintained=True
            )
            db.add(daily_progress)
        
        # Mettre à jour les métriques
        daily_progress.time_spent_minutes += session.duration_minutes or 0
        
        # Compter les quêtes et exercices complétés aujourd'hui
        if session.quest_id:
            quest_completed_today = db.query(UserQuest).filter(
                and_(
                    UserQuest.user_id == user_id,
                    UserQuest.quest_id == session.quest_id,
                    UserQuest.status == 'completed',
                    func.date(UserQuest.completed_at) == today
                )
            ).first()
            
            if quest_completed_today:
                daily_progress.quests_completed += 1
        
        # Compter les exercices (réponses correctes) aujourd'hui
        exercises_today = db.query(UserAnswer).filter(
            and_(
                UserAnswer.user_id == user_id,
                UserAnswer.is_correct == True,
                func.date(UserAnswer.submitted_at) == today
            )
        ).count()
        
        daily_progress.exercises_completed = exercises_today
        
        db.commit()
    
    def get_progress_dashboard(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Génère un tableau de bord de progression complet."""
        try:
            # Récupérer toutes les données nécessaires
            metrics = self.analyzer.get_user_progress_metrics(user_id, days)
            patterns = self.pattern_detector.detect_patterns(user_id, days)
            insights = self.insight_generator.generate_insights(user_id, days)
            
            # Données additionnelles
            with get_session() as db:
                # Activité quotidienne récente
                recent_activity = db.query(DailyProgress).filter(
                    and_(
                        DailyProgress.user_id == user_id,
                        DailyProgress.date >= (datetime.utcnow() - timedelta(days=days)).date()
                    )
                ).order_by(DailyProgress.date).all()
                
                # Sessions récentes
                recent_sessions = db.query(LearningSession).filter(
                    and_(
                        LearningSession.user_id == user_id,
                        LearningSession.started_at >= datetime.utcnow() - timedelta(days=7)
                    )
                ).order_by(desc(LearningSession.started_at)).limit(10).all()
                
                # Évaluations de compétences récentes
                recent_skills = db.query(SkillAssessment).filter(
                    and_(
                        SkillAssessment.user_id == user_id,
                        SkillAssessment.assessed_at >= datetime.utcnow() - timedelta(days=days)
                    )
                ).order_by(desc(SkillAssessment.assessed_at)).limit(20).all()
            
            return {
                'user_id': user_id,
                'period_days': days,
                'generated_at': datetime.utcnow().isoformat(),
                
                # Métriques principales
                'metrics': {
                    'quests_started': metrics.total_quests_started,
                    'quests_completed': metrics.total_quests_completed,
                    'completion_rate': metrics.completion_rate,
                    'average_score': metrics.average_score,
                    'total_time_spent': metrics.total_time_spent,
                    'average_time_per_quest': metrics.average_time_per_quest,
                    'current_streak': metrics.current_streak,
                    'longest_streak': metrics.longest_streak
                },
                
                # Progression des compétences
                'skills': {
                    'improvements': metrics.skill_improvements,
                    'recent_assessments': [
                        {
                            'skill': skill.skill_name,
                            'level': skill.skill_level,
                            'date': skill.assessed_at.isoformat()
                        }
                        for skill in recent_skills
                    ]
                },
                
                # Distribution des difficultés
                'difficulty_distribution': metrics.difficulty_distribution,
                
                # Activité quotidienne
                'daily_activity': [
                    {
                        'date': activity.date.isoformat(),
                        'time_spent': activity.time_spent_minutes,
                        'quests_completed': activity.quests_completed,
                        'exercises_completed': activity.exercises_completed,
                        'score_gained': activity.score_gained
                    }
                    for activity in recent_activity
                ],
                
                # Sessions récentes
                'recent_sessions': [
                    {
                        'id': str(session.id),
                        'started_at': session.started_at.isoformat(),
                        'duration_minutes': session.duration_minutes,
                        'engagement_score': session.engagement_score,
                        'activity_count': session.activity_count
                    }
                    for session in recent_sessions
                ],
                
                # Patterns détectés
                'patterns': [
                    {
                        'type': pattern.pattern_type,
                        'confidence': pattern.confidence,
                        'description': pattern.description,
                        'recommendations': pattern.recommendations
                    }
                    for pattern in patterns
                ],
                
                # Insights personnalisés
                'insights': [
                    {
                        'type': insight.insight_type,
                        'title': insight.title,
                        'description': insight.description,
                        'impact_level': insight.impact_level,
                        'actionable_steps': insight.actionable_steps
                    }
                    for insight in insights
                ],
                
                # Recommandations d'action
                'recommendations': self._generate_action_recommendations(metrics, patterns, insights)
            }
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard: {e}")
            return {'error': 'Erreur lors de la génération du tableau de bord'}
    
    def _generate_action_recommendations(
        self,
        metrics: ProgressMetrics,
        patterns: List[LearningPattern],
        insights: List[ProgressInsight]
    ) -> List[str]:
        """Génère des recommandations d'action personnalisées."""
        recommendations = []
        
        # Recommandations basées sur les métriques
        if metrics.completion_rate < 0.5:
            recommendations.append("Choisissez des quêtes plus courtes pour améliorer votre taux de completion")
        
        if metrics.current_streak == 0:
            recommendations.append("Commencez une nouvelle session aujourd'hui pour relancer votre streak")
        
        if metrics.average_time_per_quest > 60:
            recommendations.append("Essayez de décomposer les grandes quêtes en sessions plus courtes")
        
        # Recommandations basées sur les patterns
        for pattern in patterns:
            if pattern.pattern_type == "dropout_risk" and pattern.confidence > 0.6:
                recommendations.append("Prenez une pause si nécessaire - l'apprentissage doit rester un plaisir")
        
        # Recommandations basées sur les insights haute priorité
        high_impact_insights = [i for i in insights if i.impact_level == "high"]
        for insight in high_impact_insights[:2]:  # Limiter à 2 pour éviter la surcharge
            if insight.actionable_steps:
                recommendations.append(insight.actionable_steps[0])
        
        # Recommandation générale de motivation
        if not recommendations:
            recommendations.append("Continuez votre excellent travail - chaque étape compte !")
        
        return recommendations[:5]  # Limiter à 5 recommandations


# Instances globales
progress_tracker = ProgressTracker()


# Fonctions utilitaires
def start_learning_session(user_id: int, quest_id: Optional[str] = None) -> str:
    """Démarre une session d'apprentissage."""
    return progress_tracker.track_session_start(user_id, quest_id)


def track_event(session_id: str, event_type: str, event_data: Optional[Dict[str, Any]] = None):
    """Enregistre un événement dans une session."""
    progress_tracker.track_session_event(session_id, event_type, event_data)


def end_learning_session(session_id: str) -> Dict[str, Any]:
    """Termine une session d'apprentissage."""
    return progress_tracker.end_session(session_id)


def get_user_dashboard(user_id: int, days: int = 30) -> Dict[str, Any]:
    """Récupère le tableau de bord de progression d'un utilisateur."""
    return progress_tracker.get_progress_dashboard(user_id, days)


def get_learning_insights(user_id: int, days: int = 30) -> List[ProgressInsight]:
    """Récupère les insights d'apprentissage d'un utilisateur."""
    insight_generator = ProgressInsightGenerator()
    return insight_generator.generate_insights(user_id, days)


if __name__ == "__main__":
    # Test du suivi de progression
    print("=== Test du Suivi de Progression ===")
    
    # Test d'analyse de métriques
    analyzer = ProgressAnalyzer()
    print("Analyseur de progression initialisé")
    
    # Test de détection de patterns
    detector = LearningPatternDetector()
    print("Détecteur de patterns initialisé")
    
    # Test de génération d'insights
    generator = ProgressInsightGenerator()
    print("Générateur d'insights initialisé")
    
    print("Tous les composants de suivi de progression sont opérationnels !")
