"""
Ajusteur de difficulté adaptatif pour les quêtes.
Analyse les performances utilisateur et ajuste automatiquement la difficulté des quêtes.
"""

import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from src.core.database import get_session
from src.core.logger import get_logger
from src.core.config import get_settings
from src.models.quest import Quest, QuestStep, UserQuest, UserAnswer
from src.models.user import User, UserProgress
from src.models.progress import SkillAssessment

logger = get_logger(__name__)
settings = get_settings()


class DifficultyLevel(Enum):
    """Niveaux de difficulté."""
    VERY_EASY = 1
    EASY = 2
    NORMAL = 3
    HARD = 4
    VERY_HARD = 5


class AdjustmentType(Enum):
    """Types d'ajustement."""
    INCREASE_DIFFICULTY = "increase"
    DECREASE_DIFFICULTY = "decrease"
    MAINTAIN_DIFFICULTY = "maintain"
    SKIP_PREREQUISITES = "skip_prerequisites"
    ADD_SCAFFOLDING = "add_scaffolding"


@dataclass
class PerformanceMetrics:
    """Métriques de performance utilisateur."""
    success_rate: float
    average_attempts: float
    average_time: float
    frustration_indicators: int
    flow_indicators: int
    skill_level: float
    confidence_level: float


@dataclass
class DifficultyAdjustment:
    """Recommandation d'ajustement de difficulté."""
    adjustment_type: AdjustmentType
    magnitude: float  # -1.0 à 1.0
    confidence: float  # 0.0 à 1.0
    reasoning: str
    specific_actions: List[str]
    estimated_impact: str


@dataclass
class AdaptiveRecommendation:
    """Recommandation adaptative pour l'utilisateur."""
    quest_id: str
    quest_title: str
    original_difficulty: str
    recommended_difficulty: str
    adjustments: List[DifficultyAdjustment]
    reasoning: str
    estimated_success_rate: float


class PerformanceAnalyzer:
    """Analyseur de performance utilisateur."""
    
    def __init__(self):
        self.performance_thresholds = {
            'high_success': 0.85,
            'good_success': 0.70,
            'acceptable_success': 0.50,
            'low_success': 0.30,
            'very_low_success': 0.15
        }
    
    def analyze_user_performance(
        self,
        user_id: int,
        quest_id: Optional[str] = None,
        days: int = 30
    ) -> PerformanceMetrics:
        """
        Analyse les performances d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            quest_id: ID de quête spécifique (optionnel)
            days: Période d'analyse
            
        Returns:
            Métriques de performance
        """
        try:
            with get_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Construire la requête de base
                query = db.query(UserAnswer).filter(
                    and_(
                        UserAnswer.user_id == user_id,
                        UserAnswer.submitted_at >= cutoff_date
                    )
                )
                
                if quest_id:
                    query = query.filter(UserAnswer.quest_id == quest_id)
                
                answers = query.all()
                
                if not answers:
                    return PerformanceMetrics(0.5, 1.0, 0.0, 0, 0, 0.5, 0.5)
                
                # Calculer les métriques
                success_rate = sum(1 for a in answers if a.is_correct) / len(answers)
                average_attempts = statistics.mean(a.attempts for a in answers)
                average_time = statistics.mean(a.execution_time for a in answers if a.execution_time)
                
                # Indicateurs de frustration (plusieurs tentatives, temps long)
                frustration_indicators = sum(
                    1 for a in answers 
                    if a.attempts > 3 or (a.execution_time and a.execution_time > 300)
                )
                
                # Indicateurs de flow (réponses rapides et correctes)
                flow_indicators = sum(
                    1 for a in answers 
                    if a.is_correct and a.attempts <= 2 and 
                    (a.execution_time and a.execution_time < 60)
                )
                
                # Évaluer le niveau de compétence
                skill_level = self._calculate_skill_level(db, user_id, quest_id)
                
                # Évaluer le niveau de confiance
                confidence_level = self._calculate_confidence_level(answers)
                
                return PerformanceMetrics(
                    success_rate=success_rate,
                    average_attempts=average_attempts,
                    average_time=average_time,
                    frustration_indicators=frustration_indicators,
                    flow_indicators=flow_indicators,
                    skill_level=skill_level,
                    confidence_level=confidence_level
                )
                
        except Exception as e:
            logger.error(f"Erreur analyse performance: {e}")
            return PerformanceMetrics(0.5, 1.0, 0.0, 0, 0, 0.5, 0.5)
    
    def _calculate_skill_level(self, db: Session, user_id: int, quest_id: Optional[str]) -> float:
        """Calcule le niveau de compétence estimé."""
        try:
            # Récupérer les évaluations de compétences récentes
            recent_assessments = db.query(SkillAssessment).filter(
                and_(
                    SkillAssessment.user_id == user_id,
                    SkillAssessment.assessed_at >= datetime.utcnow() - timedelta(days=30)
                )
            ).all()
            
            if recent_assessments:
                return statistics.mean(a.skill_level for a in recent_assessments) / 100.0
            
            # Fallback: analyser les performances récentes
            user_progress = db.query(UserProgress).filter(
                UserProgress.user_id == user_id
            ).first()
            
            if user_progress and user_progress.level:
                return min(1.0, user_progress.level / 20.0)  # Normaliser sur 20 niveaux max
            
            return 0.5  # Niveau par défaut
            
        except Exception as e:
            logger.error(f"Erreur calcul skill level: {e}")
            return 0.5
    
    def _calculate_confidence_level(self, answers: List) -> float:
        """Calcule le niveau de confiance basé sur les patterns de réponse."""
        if not answers:
            return 0.5
        
        # Facteurs de confiance
        confidence_factors = []
        
        # Consistance dans les bonnes réponses
        recent_answers = sorted(answers, key=lambda x: x.submitted_at)[-10:]
        if len(recent_answers) >= 3:
            correct_streak = 0
            for answer in reversed(recent_answers):
                if answer.is_correct:
                    correct_streak += 1
                else:
                    break
            confidence_factors.append(min(1.0, correct_streak / 5.0))
        
        # Rapidité de résolution (indicateur de confiance)
        quick_answers = [a for a in answers if a.execution_time and a.execution_time < 30]
        if answers:
            confidence_factors.append(len(quick_answers) / len(answers))
        
        # Peu de tentatives nécessaires
        easy_answers = [a for a in answers if a.attempts <= 2]
        if answers:
            confidence_factors.append(len(easy_answers) / len(answers))
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5


class DifficultyAdjuster:
    """Ajusteur de difficulté adaptatif."""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.adjustment_strategies = {
            'too_easy': self._create_increase_difficulty_strategy,
            'too_hard': self._create_decrease_difficulty_strategy,
            'optimal': self._create_maintain_strategy,
            'needs_scaffolding': self._create_scaffolding_strategy
        }
    
    def suggest_difficulty_adjustment(
        self,
        user_id: int,
        quest_id: str,
        current_performance: Optional[PerformanceMetrics] = None
    ) -> List[DifficultyAdjustment]:
        """
        Suggère des ajustements de difficulté pour une quête.
        
        Args:
            user_id: ID de l'utilisateur
            quest_id: ID de la quête
            current_performance: Métriques de performance actuelles
            
        Returns:
            Liste d'ajustements recommandés
        """
        if current_performance is None:
            current_performance = self.analyzer.analyze_user_performance(user_id, quest_id)
        
        # Déterminer la situation actuelle
        situation = self._assess_difficulty_situation(current_performance)
        
        # Générer les ajustements appropriés
        strategy_func = self.adjustment_strategies.get(situation)
        if strategy_func:
            return strategy_func(current_performance)
        
        return [self._create_maintain_strategy(current_performance)[0]]
    
    def _assess_difficulty_situation(self, metrics: PerformanceMetrics) -> str:
        """Évalue la situation de difficulté actuelle."""
        # Trop facile : très bon taux de succès, peu de tentatives, temps rapide
        if (metrics.success_rate > 0.90 and 
            metrics.average_attempts < 1.5 and 
            metrics.flow_indicators > metrics.frustration_indicators):
            return 'too_easy'
        
        # Trop difficile : faible taux de succès, beaucoup de tentatives
        elif (metrics.success_rate < 0.40 and 
              metrics.average_attempts > 3.0 and 
              metrics.frustration_indicators > metrics.flow_indicators):
            return 'too_hard'
        
        # Besoin d'accompagnement : performance moyenne mais signes de frustration
        elif (0.40 <= metrics.success_rate <= 0.70 and 
              metrics.frustration_indicators > 2 and 
              metrics.confidence_level < 0.5):
            return 'needs_scaffolding'
        
        # Optimal : bon équilibre
        else:
            return 'optimal'
    
    def _create_increase_difficulty_strategy(self, metrics: PerformanceMetrics) -> List[DifficultyAdjustment]:
        """Crée une stratégie d'augmentation de difficulté."""
        adjustments = []
        
        # Ajustement principal : augmenter la difficulté
        magnitude = min(0.8, (metrics.success_rate - 0.7) * 2)  # Plus le succès est élevé, plus l'augmentation
        
        main_adjustment = DifficultyAdjustment(
            adjustment_type=AdjustmentType.INCREASE_DIFFICULTY,
            magnitude=magnitude,
            confidence=0.8 if metrics.success_rate > 0.85 else 0.6,
            reasoning=f"Taux de succès élevé ({metrics.success_rate:.1%}) indique que la difficulté actuelle est trop faible",
            specific_actions=[
                "Ajouter des contraintes supplémentaires aux exercices",
                "Introduire des concepts plus avancés",
                "Réduire les indices disponibles",
                "Augmenter la complexité des cas de test"
            ],
            estimated_impact="Amélioration de l'engagement et de l'apprentissage"
        )
        adjustments.append(main_adjustment)
        
        # Ajustement secondaire si l'utilisateur est très avancé
        if metrics.skill_level > 0.8:
            skip_adjustment = DifficultyAdjustment(
                adjustment_type=AdjustmentType.SKIP_PREREQUISITES,
                magnitude=0.5,
                confidence=0.7,
                reasoning="Niveau de compétence élevé permet de sauter certains prérequis",
                specific_actions=[
                    "Proposer des défis optionnels plus complexes",
                    "Permettre d'accéder à des quêtes de niveau supérieur",
                    "Introduire des variantes créatives des exercices"
                ],
                estimated_impact="Maintien de la motivation et évitement de l'ennui"
            )
            adjustments.append(skip_adjustment)
        
        return adjustments
    
    def _create_decrease_difficulty_strategy(self, metrics: PerformanceMetrics) -> List[DifficultyAdjustment]:
        """Crée une stratégie de diminution de difficulté."""
        adjustments = []
        
        # Calculer l'ampleur de la réduction nécessaire
        magnitude = min(0.8, (0.7 - metrics.success_rate) * 2)
        
        main_adjustment = DifficultyAdjustment(
            adjustment_type=AdjustmentType.DECREASE_DIFFICULTY,
            magnitude=-magnitude,  # Valeur négative pour diminution
            confidence=0.8 if metrics.success_rate < 0.3 else 0.6,
            reasoning=f"Taux de succès faible ({metrics.success_rate:.1%}) indique que la difficulté actuelle est trop élevée",
            specific_actions=[
                "Simplifier les exercices en décomposant les étapes",
                "Ajouter plus d'exemples et d'explications",
                "Fournir des templates de code plus détaillés",
                "Réduire la complexité des cas de test"
            ],
            estimated_impact="Réduction de la frustration et amélioration de la progression"
        )
        adjustments.append(main_adjustment)
        
        # Ajout d'accompagnement si beaucoup de frustration
        if metrics.frustration_indicators > 3:
            scaffolding_adjustment = DifficultyAdjustment(
                adjustment_type=AdjustmentType.ADD_SCAFFOLDING,
                magnitude=0.6,
                confidence=0.7,
                reasoning="Signes de frustration élevés nécessitent un accompagnement supplémentaire",
                specific_actions=[
                    "Ajouter des étapes intermédiaires guidées",
                    "Fournir des indices progressifs automatiques",
                    "Créer des mini-exercices préparatoires",
                    "Proposer des ressources d'aide contextuelle"
                ],
                estimated_impact="Soutien de l'apprentissage et maintien de la motivation"
            )
            adjustments.append(scaffolding_adjustment)
        
        return adjustments
    
    def _create_maintain_strategy(self, metrics: PerformanceMetrics) -> List[DifficultyAdjustment]:
        """Crée une stratégie de maintien de la difficulté."""
        return [DifficultyAdjustment(
            adjustment_type=AdjustmentType.MAINTAIN_DIFFICULTY,
            magnitude=0.0,
            confidence=0.8,
            reasoning=f"Performance équilibrée (succès: {metrics.success_rate:.1%}) indique une difficulté appropriée",
            specific_actions=[
                "Maintenir le niveau de difficulté actuel",
                "Continuer le monitoring des performances",
                "Varier les types d'exercices pour maintenir l'engagement"
            ],
            estimated_impact="Progression optimale maintenue"
        )]
    
    def _create_scaffolding_strategy(self, metrics: PerformanceMetrics) -> List[DifficultyAdjustment]:
        """Crée une stratégie d'accompagnement supplémentaire."""
        return [DifficultyAdjustment(
            adjustment_type=AdjustmentType.ADD_SCAFFOLDING,
            magnitude=0.4,
            confidence=0.7,
            reasoning="Performance modérée avec signes de frustration nécessite un accompagnement",
            specific_actions=[
                "Ajouter des explications interactives",
                "Fournir des exemples similaires résolus",
                "Créer des exercices d'échauffement",
                "Proposer des hints contextuels intelligents"
            ],
            estimated_impact="Amélioration de la compréhension et de la confiance"
        )]


class AdaptiveQuestRecommender:
    """Recommandeur de quêtes adaptatif."""
    
    def __init__(self):
        self.adjuster = DifficultyAdjuster()
        self.analyzer = PerformanceAnalyzer()
    
    def get_adaptive_recommendations(
        self,
        user_id: int,
        limit: int = 5
    ) -> List[AdaptiveRecommendation]:
        """
        Génère des recommandations de quêtes adaptatives.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre maximum de recommandations
            
        Returns:
            Liste de recommandations adaptatives
        """
        recommendations = []
        
        try:
            with get_session() as db:
                # Analyser les performances globales de l'utilisateur
                global_performance = self.analyzer.analyze_user_performance(user_id)
                
                # Récupérer les quêtes disponibles
                from src.models.quest import Quest
                
                # Exclure les quêtes déjà complétées
                completed_quest_ids = db.query(UserQuest.quest_id).filter(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.status == 'completed'
                    )
                ).subquery()
                
                available_quests = db.query(Quest).filter(
                    and_(
                        Quest.is_active == True,
                        ~Quest.id.in_(completed_quest_ids)
                    )
                ).limit(limit * 2).all()  # Récupérer plus pour avoir du choix
                
                for quest in available_quests:
                    # Analyser les performances sur cette quête si l'utilisateur l'a tentée
                    quest_performance = self.analyzer.analyze_user_performance(user_id, quest.id)
                    
                    # Déterminer les ajustements nécessaires
                    adjustments = self.adjuster.suggest_difficulty_adjustment(
                        user_id, quest.id, quest_performance
                    )
                    
                    # Calculer la difficulté recommandée
                    recommended_difficulty = self._calculate_recommended_difficulty(
                        quest.difficulty, adjustments
                    )
                    
                    # Estimer le taux de succès
                    estimated_success = self._estimate_success_rate(
                        global_performance, quest.difficulty, recommended_difficulty
                    )
                    
                    # Créer la recommandation
                    recommendation = AdaptiveRecommendation(
                        quest_id=quest.id,
                        quest_title=quest.title,
                        original_difficulty=quest.difficulty,
                        recommended_difficulty=recommended_difficulty,
                        adjustments=adjustments,
                        reasoning=self._generate_recommendation_reasoning(
                            quest, adjustments, estimated_success
                        ),
                        estimated_success_rate=estimated_success
                    )
                    
                    recommendations.append(recommendation)
                
                # Trier par taux de succès estimé (zone optimale 60-80%)
                recommendations.sort(
                    key=lambda r: abs(r.estimated_success_rate - 0.7)
                )
                
                return recommendations[:limit]
                
        except Exception as e:
            logger.error(f"Erreur génération recommandations adaptatives: {e}")
            return []
    
    def _calculate_recommended_difficulty(self, original_difficulty: str, adjustments: List[DifficultyAdjustment]) -> str:
        """Calcule la difficulté recommandée basée sur les ajustements."""
        # Mapper les difficultés à des valeurs numériques
        difficulty_values = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 4
        }
        
        difficulty_names = ['beginner', 'intermediate', 'advanced', 'expert']
        
        current_value = difficulty_values.get(original_difficulty, 2)
        
        # Appliquer les ajustements
        total_adjustment = 0
        for adjustment in adjustments:
            if adjustment.adjustment_type == AdjustmentType.INCREASE_DIFFICULTY:
                total_adjustment += adjustment.magnitude
            elif adjustment.adjustment_type == AdjustmentType.DECREASE_DIFFICULTY:
                total_adjustment += adjustment.magnitude  # Déjà négatif
        
        # Calculer la nouvelle valeur
        new_value = current_value + (total_adjustment * 2)  # Scaling factor
        new_value = max(1, min(4, round(new_value)))  # Limiter entre 1 et 4
        
        return difficulty_names[int(new_value) - 1]
    
    def _estimate_success_rate(
        self,
        performance: PerformanceMetrics,
        original_difficulty: str,
        recommended_difficulty: str
    ) -> float:
        """Estime le taux de succès pour la difficulté recommandée."""
        difficulty_values = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 4
        }
        
        original_value = difficulty_values.get(original_difficulty, 2)
        recommended_value = difficulty_values.get(recommended_difficulty, 2)
        
        # Base de calcul : performance actuelle
        base_success = performance.success_rate
        
        # Ajustement basé sur le changement de difficulté
        difficulty_change = recommended_value - original_value
        
        # Impact estimé du changement de difficulté
        if difficulty_change > 0:  # Plus difficile
            estimated_success = base_success * (0.8 ** difficulty_change)
        elif difficulty_change < 0:  # Plus facile
            estimated_success = base_success + (1 - base_success) * (0.3 * abs(difficulty_change))
        else:  # Même difficulté
            estimated_success = base_success
        
        # Ajustement basé sur le niveau de compétence
        skill_factor = (performance.skill_level - 0.5) * 0.2
        estimated_success += skill_factor
        
        # Ajustement basé sur la confiance
        confidence_factor = (performance.confidence_level - 0.5) * 0.1
        estimated_success += confidence_factor
        
        return max(0.1, min(0.95, estimated_success))
    
    def _generate_recommendation_reasoning(
        self,
        quest: Quest,
        adjustments: List[DifficultyAdjustment],
        estimated_success: float
    ) -> str:
        """Génère l'explication de la recommandation."""
        reasoning_parts = []
        
        # Contexte de la quête
        reasoning_parts.append(f"Cette quête sur '{quest.title}' est classée {quest.difficulty}.")
        
        # Ajustements suggérés
        if adjustments:
            main_adjustment = adjustments[0]
            if main_adjustment.adjustment_type == AdjustmentType.INCREASE_DIFFICULTY:
                reasoning_parts.append("Votre excellent niveau suggère une version plus challenging.")
            elif main_adjustment.adjustment_type == AdjustmentType.DECREASE_DIFFICULTY:
                reasoning_parts.append("Une approche simplifiée vous permettra de mieux progresser.")
            elif main_adjustment.adjustment_type == AdjustmentType.ADD_SCAFFOLDING:
                reasoning_parts.append("Un accompagnement supplémentaire optimisera votre apprentissage.")
            else:
                reasoning_parts.append("Le niveau actuel semble adapté à votre profil.")
        
        # Estimation de succès
        if estimated_success > 0.8:
            reasoning_parts.append("Très forte probabilité de réussite.")
        elif estimated_success > 0.6:
            reasoning_parts.append("Bon équilibre défi/réussite.")
        elif estimated_success > 0.4:
            reasoning_parts.append("Défi stimulant avec accompagnement.")
        else:
            reasoning_parts.append("Recommandé avec préparation supplémentaire.")
        
        return " ".join(reasoning_parts)


class DifficultyMonitor:
    """Moniteur de difficulté en temps réel."""
    
    def __init__(self):
        self.adjuster = DifficultyAdjuster()
        self.analyzer = PerformanceAnalyzer()
        self.monitoring_thresholds = {
            'success_rate_low': 0.3,
            'success_rate_high': 0.9,
            'attempts_high': 4.0,
            'frustration_high': 3
        }
    
    def monitor_real_time_difficulty(self, user_id: int, quest_id: str, session_data: Dict[str, Any]) -> Optional[DifficultyAdjustment]:
        """
        Monitore la difficulté en temps réel pendant une session.
        
        Args:
            user_id: ID de l'utilisateur
            quest_id: ID de la quête
            session_data: Données de la session en cours
            
        Returns:
            Ajustement urgent si nécessaire
        """
        try:
            # Analyser les données de session
            current_attempts = session_data.get('current_attempts', 1)
            time_spent = session_data.get('time_spent_minutes', 0)
            errors_count = session_data.get('errors_count', 0)
            hints_used = session_data.get('hints_used', 0)
            
            # Détection de frustration aigüe
            if (current_attempts > 5 and 
                time_spent > 20 and 
                errors_count > 3):
                
                return DifficultyAdjustment(
                    adjustment_type=AdjustmentType.ADD_SCAFFOLDING,
                    magnitude=0.8,
                    confidence=0.9,
                    reasoning="Signes de frustration aigüe détectés en temps réel",
                    specific_actions=[
                        "Proposer un hint immédiat",
                        "Décomposer l'exercice en sous-étapes",
                        "Offrir un exemple similaire résolu",
                        "Suggérer une pause courte"
                    ],
                    estimated_impact="Intervention immédiate pour éviter l'abandon"
                )
            
            # Détection de facilité excessive
            elif (current_attempts == 1 and 
                  time_spent < 2 and 
                  hints_used == 0):
                
                return DifficultyAdjustment(
                    adjustment_type=AdjustmentType.INCREASE_DIFFICULTY,
                    magnitude=0.6,
                    confidence=0.7,
                    reasoning="Résolution trop rapide détectée",
                    specific_actions=[
                        "Proposer une variante plus complexe",
                        "Ajouter une contrainte supplémentaire",
                        "Suggérer un défi bonus"
                    ],
                    estimated_impact="Maintien de l'engagement par le défi"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur monitoring temps réel: {e}")
            return None
    
    def generate_session_report(self, user_id: int, quest_id: str, session_id: str) -> Dict[str, Any]:
        """Génère un rapport de session avec recommandations."""
        try:
            with get_session() as db:
                # Récupérer les données de session
                from src.models.progress import LearningSession, SessionEvent
                
                session = db.query(LearningSession).filter(
                    LearningSession.id == session_id
                ).first()
                
                if not session:
                    return {'error': 'Session introuvable'}
                
                # Analyser les événements de la session
                events = db.query(SessionEvent).filter(
                    SessionEvent.session_id == session_id
                ).all()
                
                # Calculer les métriques de session
                session_metrics = self._calculate_session_metrics(session, events)
                
                # Analyser les performances sur cette quête
                quest_performance = self.analyzer.analyze_user_performance(user_id, quest_id)
                
                # Générer les recommandations
                adjustments = self.adjuster.suggest_difficulty_adjustment(
                    user_id, quest_id, quest_performance
                )
                
                return {
                    'session_id': session_id,
                    'duration_minutes': session.duration_minutes,
                    'engagement_score': session.engagement_score,
                    'session_metrics': session_metrics,
                    'performance_analysis': {
                        'success_rate': quest_performance.success_rate,
                        'confidence_level': quest_performance.confidence_level,
                        'skill_level': quest_performance.skill_level
                    },
                    'difficulty_recommendations': [
                        {
                            'type': adj.adjustment_type.value,
                            'magnitude': adj.magnitude,
                            'confidence': adj.confidence,
                            'reasoning': adj.reasoning,
                            'actions': adj.specific_actions
                        }
                        for adj in adjustments
                    ],
                    'next_session_suggestions': self._generate_next_session_suggestions(
                        quest_performance, adjustments
                    )
                }
                
        except Exception as e:
            logger.error(f"Erreur génération rapport session: {e}")
            return {'error': 'Erreur lors de la génération du rapport'}
    
    def _calculate_session_metrics(self, session, events) -> Dict[str, Any]:
        """Calcule les métriques détaillées d'une session."""
        if not events:
            return {}
        
        # Compter les types d'événements
        event_counts = {}
        for event in events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculer des métriques dérivées
        total_events = len(events)
        success_events = event_counts.get('answer_correct', 0)
        error_events = event_counts.get('answer_incorrect', 0)
        hint_events = event_counts.get('hint_used', 0)
        
        return {
            'total_events': total_events,
            'success_events': success_events,
            'error_events': error_events,
            'hint_events': hint_events,
            'success_ratio': success_events / max(1, success_events + error_events),
            'help_seeking_ratio': hint_events / max(1, total_events),
            'activity_intensity': total_events / max(1, session.duration_minutes or 1),
            'event_distribution': event_counts
        }
    
    def _generate_next_session_suggestions(self, performance: PerformanceMetrics, adjustments: List[DifficultyAdjustment]) -> List[str]:
        """Génère des suggestions pour la prochaine session."""
        suggestions = []
        
        # Basé sur la performance
        if performance.success_rate < 0.5:
            suggestions.append("Commencez par réviser les concepts de base")
            suggestions.append("Prenez votre temps - la qualité prime sur la vitesse")
        elif performance.success_rate > 0.8:
            suggestions.append("Prêt pour des défis plus complexes")
            suggestions.append("Explorez les exercices bonus disponibles")
        
        # Basé sur les ajustements
        for adjustment in adjustments:
            if adjustment.adjustment_type == AdjustmentType.ADD_SCAFFOLDING:
                suggestions.append("Utilisez les ressources d'aide disponibles")
            elif adjustment.adjustment_type == AdjustmentType.INCREASE_DIFFICULTY:
                suggestions.append("Tentez la version avancée des exercices")
        
        # Basé sur la confiance
        if performance.confidence_level < 0.5:
            suggestions.append("Pratiquez des exercices similaires pour renforcer la confiance")
        
        return suggestions[:3]  # Limiter à 3 suggestions


# Instances globales
difficulty_adjuster = DifficultyAdjuster()
adaptive_recommender = AdaptiveQuestRecommender()
difficulty_monitor = DifficultyMonitor()


# Fonctions utilitaires
def get_difficulty_adjustments(user_id: int, quest_id: str) -> List[DifficultyAdjustment]:
    """Obtient les ajustements de difficulté recommandés."""
    return difficulty_adjuster.suggest_difficulty_adjustment(user_id, quest_id)


def get_adaptive_quest_recommendations(user_id: int, limit: int = 5) -> List[AdaptiveRecommendation]:
    """Obtient les recommandations de quêtes adaptatives."""
    return adaptive_recommender.get_adaptive_recommendations(user_id, limit)


def monitor_session_difficulty(user_id: int, quest_id: str, session_data: Dict[str, Any]) -> Optional[DifficultyAdjustment]:
    """Monitore la difficulté en temps réel."""
    return difficulty_monitor.monitor_real_time_difficulty(user_id, quest_id, session_data)


def generate_difficulty_report(user_id: int, quest_id: str, session_id: str) -> Dict[str, Any]:
    """Génère un rapport de difficulté pour une session."""
    return difficulty_monitor.generate_session_report(user_id, quest_id, session_id)


if __name__ == "__main__":
    # Test de l'ajusteur de difficulté
    print("=== Test de l'Ajusteur de Difficulté ===")
    
    # Test d'analyse de performance
    analyzer = PerformanceAnalyzer()
    
    # Simuler des métriques de performance
    test_metrics = PerformanceMetrics(
        success_rate=0.3,  # Faible taux de succès
        average_attempts=4.2,  # Beaucoup de tentatives
        average_time=180.0,  # Temps élevé
        frustration_indicators=5,  # Signes de frustration
        flow_indicators=1,  # Peu de flow
        skill_level=0.4,  # Niveau de compétence moyen
        confidence_level=0.3  # Faible confiance
    )
    
    # Test d'ajustement
    adjuster = DifficultyAdjuster()
    adjustments = adjuster.suggest_difficulty_adjustment(1, "test_quest", test_metrics)
    
    print(f"Nombre d'ajustements suggérés: {len(adjustments)}")
    for adj in adjustments:
        print(f"- Type: {adj.adjustment_type.value}")
        print(f"  Ampleur: {adj.magnitude}")
        print(f"  Confiance: {adj.confidence}")
        print(f"  Raison: {adj.reasoning}")
    
    print("Ajusteur de difficulté testé avec succès!")