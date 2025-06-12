"""
Gestionnaire principal des quêtes pédagogiques.
Gère le cycle de vie complet des quêtes : création, attribution, progression, évaluation.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from src.core.database import get_session
from src.core.logger import get_logger
from src.core.config import get_settings
from src.models.quest import Quest, QuestStep, UserQuest, UserStepProgress, UserAnswer
from src.models.user import User, UserProgress
from src.models.schemas import QuestCreate, QuestUpdate, UserQuestResponse
from src.code_execution.executor import execute_code_with_tests, TestCaseBuilder
from src.code_execution.security import is_code_safe

logger = get_logger(__name__)
settings = get_settings()


class QuestStatus(Enum):
    """Statuts des quêtes utilisateur."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class QuestDifficulty(Enum):
    """Niveaux de difficulté des quêtes."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class QuestRecommendation:
    """Recommandation de quête pour un utilisateur."""
    quest_id: str
    quest_title: str
    difficulty: str
    estimated_time: int
    match_score: float
    reasons: List[str]
    prerequisites_met: bool


@dataclass
class QuestProgress:
    """Progression dans une quête."""
    quest_id: str
    user_id: int
    status: QuestStatus
    current_step: int
    total_steps: int
    score: float
    max_score: float
    time_spent: int
    completion_rate: float
    last_activity: datetime


class QuestManager:
    """Gestionnaire principal des quêtes."""
    
    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR) / "quests"
        self.cache = {}
        self._load_quests_cache()
    
    def _load_quests_cache(self):
        """Charge les quêtes en cache depuis la base de données."""
        try:
            with get_session() as db:
                quests = db.query(Quest).filter(Quest.is_active == True).all()
                for quest in quests:
                    self.cache[quest.id] = quest
                logger.info(f"Chargé {len(quests)} quêtes en cache")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des quêtes: {e}")
    
    def get_quest(self, quest_id: str) -> Optional[Quest]:
        """Récupère une quête par son ID."""
        if quest_id in self.cache:
            return self.cache[quest_id]
        
        try:
            with get_session() as db:
                quest = db.query(Quest).filter(Quest.id == quest_id).first()
                if quest:
                    self.cache[quest_id] = quest
                return quest
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la quête {quest_id}: {e}")
            return None
    
    def create_quest(self, quest_data: QuestCreate, creator_id: int) -> Optional[Quest]:
        """
        Crée une nouvelle quête.
        
        Args:
            quest_data: Données de la quête
            creator_id: ID du créateur
            
        Returns:
            Quest créée ou None si erreur
        """
        try:
            with get_session() as db:
                # Générer un ID unique
                quest_id = str(uuid.uuid4())
                
                quest = Quest(
                    id=quest_id,
                    title=quest_data.title,
                    description=quest_data.description,
                    difficulty=quest_data.difficulty,
                    estimated_time=quest_data.estimated_time,
                    category=quest_data.category,
                    tags=quest_data.tags,
                    learning_objectives=quest_data.learning_objectives,
                    prerequisites=quest_data.prerequisites,
                    creator_id=creator_id,
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                
                db.add(quest)
                db.flush()  # Pour obtenir l'ID
                
                # Créer les étapes
                for i, step_data in enumerate(quest_data.steps):
                    step = QuestStep(
                        quest_id=quest.id,
                        step_number=i + 1,
                        title=step_data.title,
                        content=step_data.content,
                        step_type=step_data.step_type,
                        code_template=step_data.code_template,
                        expected_output=step_data.expected_output,
                        hints=step_data.hints,
                        resources=step_data.resources,
                        max_score=step_data.max_score
                    )
                    db.add(step)
                
                db.commit()
                
                # Mettre à jour le cache
                self.cache[quest.id] = quest
                
                logger.info(f"Quête créée: {quest.id} - {quest.title}")
                return quest
                
        except Exception as e:
            logger.error(f"Erreur lors de la création de la quête: {e}")
            return None
    
    def get_user_quests(
        self,
        user_id: int,
        status: Optional[QuestStatus] = None,
        difficulty: Optional[QuestDifficulty] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Récupère les quêtes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            status: Filtrer par statut
            difficulty: Filtrer par difficulté
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des quêtes avec progression
        """
        try:
            with get_session() as db:
                query = db.query(UserQuest, Quest).join(Quest)
                query = query.filter(UserQuest.user_id == user_id)
                
                if status:
                    query = query.filter(UserQuest.status == status.value)
                
                if difficulty:
                    query = query.filter(Quest.difficulty == difficulty.value)
                
                query = query.order_by(desc(UserQuest.last_activity))
                results = query.limit(limit).all()
                
                quests = []
                for user_quest, quest in results:
                    quest_data = {
                        'quest': {
                            'id': quest.id,
                            'title': quest.title,
                            'description': quest.description,
                            'difficulty': quest.difficulty,
                            'estimated_time': quest.estimated_time,
                            'category': quest.category,
                            'tags': quest.tags
                        },
                        'progress': {
                            'status': user_quest.status,
                            'current_step': user_quest.current_step,
                            'score': user_quest.score,
                            'max_score': user_quest.max_score,
                            'completion_rate': user_quest.completion_rate,
                            'time_spent': user_quest.time_spent,
                            'started_at': user_quest.started_at,
                            'completed_at': user_quest.completed_at,
                            'last_activity': user_quest.last_activity
                        }
                    }
                    quests.append(quest_data)
                
                return quests
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des quêtes utilisateur: {e}")
            return []
    
    def start_quest(self, user_id: int, quest_id: str) -> Optional[UserQuest]:
        """
        Démarre une quête pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            quest_id: ID de la quête
            
        Returns:
            UserQuest créée ou None si erreur
        """
        try:
            with get_session() as db:
                # Vérifier que la quête existe
                quest = self.get_quest(quest_id)
                if not quest:
                    logger.warning(f"Tentative de démarrage d'une quête inexistante: {quest_id}")
                    return None
                
                # Vérifier si l'utilisateur a déjà cette quête
                existing = db.query(UserQuest).filter(
                    and_(UserQuest.user_id == user_id, UserQuest.quest_id == quest_id)
                ).first()
                
                if existing:
                    if existing.status in ['completed']:
                        logger.info(f"Quête déjà complétée par l'utilisateur {user_id}: {quest_id}")
                        return existing
                    else:
                        # Reprendre la quête existante
                        existing.status = QuestStatus.IN_PROGRESS.value
                        existing.last_activity = datetime.utcnow()
                        db.commit()
                        return existing
                
                # Créer une nouvelle UserQuest
                user_quest = UserQuest(
                    user_id=user_id,
                    quest_id=quest_id,
                    status=QuestStatus.IN_PROGRESS.value,
                    current_step=1,
                    score=0.0,
                    max_score=quest.max_score or 100.0,
                    completion_rate=0.0,
                    time_spent=0,
                    started_at=datetime.utcnow(),
                    last_activity=datetime.utcnow()
                )
                
                db.add(user_quest)
                db.commit()
                
                logger.info(f"Quête démarrée: {quest_id} pour l'utilisateur {user_id}")
                return user_quest
                
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de la quête: {e}")
            return None
    
    def submit_step_answer(
        self,
        user_id: int,
        quest_id: str,
        step_number: int,
        answer_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Soumet une réponse pour une étape de quête.
        
        Args:
            user_id: ID de l'utilisateur
            quest_id: ID de la quête
            step_number: Numéro de l'étape
            answer_data: Données de la réponse
            
        Returns:
            Résultats de l'évaluation
        """
        try:
            with get_session() as db:
                # Récupérer la quête et l'étape
                quest = self.get_quest(quest_id)
                if not quest:
                    return {'error': 'Quête introuvable'}
                
                step = db.query(QuestStep).filter(
                    and_(QuestStep.quest_id == quest_id, QuestStep.step_number == step_number)
                ).first()
                
                if not step:
                    return {'error': 'Étape introuvable'}
                
                # Récupérer ou créer UserQuest
                user_quest = db.query(UserQuest).filter(
                    and_(UserQuest.user_id == user_id, UserQuest.quest_id == quest_id)
                ).first()
                
                if not user_quest:
                    user_quest = self.start_quest(user_id, quest_id)
                    if not user_quest:
                        return {'error': 'Impossible de démarrer la quête'}
                
                # Vérifier que l'utilisateur peut soumettre cette étape
                if step_number > user_quest.current_step + 1:
                    return {'error': 'Vous devez compléter les étapes précédentes'}
                
                # Évaluer la réponse
                evaluation_result = self._evaluate_step_answer(step, answer_data)
                
                # Enregistrer la réponse
                answer = UserAnswer(
                    user_id=user_id,
                    quest_id=quest_id,
                    step_number=step_number,
                    answer_type=answer_data.get('type', 'code'),
                    answer_content=answer_data.get('content', ''),
                    is_correct=evaluation_result['is_correct'],
                    score=evaluation_result['score'],
                    max_score=step.max_score,
                    feedback=evaluation_result['feedback'],
                    execution_time=evaluation_result.get('execution_time', 0),
                    attempts=1,
                    submitted_at=datetime.utcnow()
                )
                
                # Vérifier si c'est une nouvelle tentative
                existing_answer = db.query(UserAnswer).filter(
                    and_(
                        UserAnswer.user_id == user_id,
                        UserAnswer.quest_id == quest_id,
                        UserAnswer.step_number == step_number
                    )
                ).first()
                
                if existing_answer:
                    answer.attempts = existing_answer.attempts + 1
                    # Garder le meilleur score
                    if answer.score > existing_answer.score:
                        existing_answer.score = answer.score
                        existing_answer.answer_content = answer.answer_content
                        existing_answer.is_correct = answer.is_correct
                        existing_answer.feedback = answer.feedback
                        existing_answer.attempts = answer.attempts
                        existing_answer.submitted_at = answer.submitted_at
                    answer = existing_answer
                else:
                    db.add(answer)
                
                # Mettre à jour la progression
                self._update_user_progress(db, user_quest, step_number, evaluation_result)
                
                db.commit()
                
                return {
                    'success': True,
                    'is_correct': evaluation_result['is_correct'],
                    'score': evaluation_result['score'],
                    'max_score': step.max_score,
                    'feedback': evaluation_result['feedback'],
                    'step_completed': evaluation_result['is_correct'],
                    'quest_progress': {
                        'current_step': user_quest.current_step,
                        'completion_rate': user_quest.completion_rate,
                        'total_score': user_quest.score
                    },
                    'hints': step.hints if not evaluation_result['is_correct'] else [],
                    'attempts': answer.attempts
                }
                
        except Exception as e:
            logger.error(f"Erreur lors de la soumission de réponse: {e}")
            return {'error': 'Erreur interne lors de l\'évaluation'}
    
    def _evaluate_step_answer(self, step: QuestStep, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue la réponse d'une étape."""
        answer_type = answer_data.get('type', 'code')
        content = answer_data.get('content', '')
        
        if answer_type == 'code':
            return self._evaluate_code_answer(step, content)
        elif answer_type == 'text':
            return self._evaluate_text_answer(step, content)
        elif answer_type == 'multiple_choice':
            return self._evaluate_multiple_choice(step, answer_data)
        else:
            return {
                'is_correct': False,
                'score': 0.0,
                'feedback': 'Type de réponse non supporté'
            }
    
    def _evaluate_code_answer(self, step: QuestStep, code: str) -> Dict[str, Any]:
        """Évalue une réponse de type code."""
        try:
            # Vérification de sécurité
            if not is_code_safe(code):
                return {
                    'is_correct': False,
                    'score': 0.0,
                    'feedback': 'Code non sécurisé détecté. Utilisez seulement les fonctions autorisées.',
                    'execution_time': 0
                }
            
            # Créer des cas de test basés sur l'étape
            test_cases = self._create_test_cases_for_step(step)
            
            if not test_cases:
                # Évaluation simple par sortie attendue
                from src.code_execution.sandbox import execute_code_safely
                result = execute_code_safely(code)
                
                if result.success:
                    expected = step.expected_output.strip() if step.expected_output else ""
                    actual = result.output.strip()
                    
                    is_correct = actual == expected
                    score = step.max_score if is_correct else 0.0
                    
                    feedback = "Excellent travail !" if is_correct else \
                              f"Sortie attendue: '{expected}', obtenue: '{actual}'"
                    
                    return {
                        'is_correct': is_correct,
                        'score': score,
                        'feedback': feedback,
                        'execution_time': result.execution_time
                    }
                else:
                    return {
                        'is_correct': False,
                        'score': 0.0,
                        'feedback': f"Erreur d'exécution: {result.error}",
                        'execution_time': result.execution_time
                    }
            else:
                # Évaluation avec cas de test
                report = execute_code_with_tests(code, test_cases)
                
                is_correct = report.pass_rate >= 0.8  # 80% de réussite minimum
                score = (report.total_score / report.max_score) * step.max_score
                
                if is_correct:
                    feedback = f"Excellent ! {report.total_score}/{report.max_score} points"
                else:
                    failed_tests = [tr for tr in report.test_results if not tr.passed]
                    feedback = f"Tests échoués: {len(failed_tests)}/{len(report.test_results)}"
                    if failed_tests:
                        feedback += f"\nPremier échec: {failed_tests[0].feedback}"
                
                return {
                    'is_correct': is_correct,
                    'score': score,
                    'feedback': feedback,
                    'execution_time': report.execution_time,
                    'test_results': report.test_results
                }
                
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du code: {e}")
            return {
                'is_correct': False,
                'score': 0.0,
                'feedback': 'Erreur lors de l\'évaluation du code',
                'execution_time': 0
            }
    
    def _evaluate_text_answer(self, step: QuestStep, text: str) -> Dict[str, Any]:
        """Évalue une réponse de type texte."""
        # Évaluation simple par mots-clés ou correspondance exacte
        expected = step.expected_output
        if not expected:
            return {
                'is_correct': True,
                'score': step.max_score,
                'feedback': 'Réponse enregistrée'
            }
        
        # Normaliser les textes
        text_normalized = text.strip().lower()
        expected_normalized = expected.strip().lower()
        
        is_correct = text_normalized == expected_normalized
        
        # Évaluation partielle par mots-clés
        if not is_correct:
            keywords = expected_normalized.split()
            found_keywords = sum(1 for keyword in keywords if keyword in text_normalized)
            partial_score = (found_keywords / len(keywords)) * step.max_score
            
            if partial_score >= step.max_score * 0.7:  # 70% des mots-clés
                is_correct = True
                score = step.max_score
            else:
                score = partial_score
        else:
            score = step.max_score
        
        feedback = "Bonne réponse !" if is_correct else \
                  f"Réponse partielle. Mots-clés trouvés: {found_keywords}/{len(keywords)}"
        
        return {
            'is_correct': is_correct,
            'score': score,
            'feedback': feedback
        }
    
    def _evaluate_multiple_choice(self, step: QuestStep, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue une réponse à choix multiple."""
        selected = answer_data.get('selected', [])
        if not isinstance(selected, list):
            selected = [selected]
        
        # Récupérer les bonnes réponses depuis les métadonnées de l'étape
        step_metadata = json.loads(step.metadata) if step.metadata else {}
        correct_answers = step_metadata.get('correct_answers', [])
        
        if not correct_answers:
            return {
                'is_correct': True,
                'score': step.max_score,
                'feedback': 'Réponse enregistrée'
            }
        
        # Vérifier la correspondance
        selected_set = set(selected)
        correct_set = set(correct_answers)
        
        is_correct = selected_set == correct_set
        
        if is_correct:
            score = step.max_score
            feedback = "Excellente réponse !"
        else:
            # Score partiel basé sur les bonnes réponses
            correct_count = len(selected_set & correct_set)
            wrong_count = len(selected_set - correct_set)
            score = max(0, (correct_count - wrong_count) / len(correct_set)) * step.max_score
            
            feedback = f"Réponses correctes: {correct_count}/{len(correct_set)}"
            if wrong_count > 0:
                feedback += f", Erreurs: {wrong_count}"
        
        return {
            'is_correct': is_correct,
            'score': score,
            'feedback': feedback
        }
    
    def _create_test_cases_for_step(self, step: QuestStep) -> List:
        """Crée des cas de test pour une étape."""
        if not step.metadata:
            return []
        
        try:
            metadata = json.loads(step.metadata)
            test_data = metadata.get('tests', [])
            
            test_cases = []
            for test in test_data:
                if test.get('type') == 'function':
                    test_case = TestCaseBuilder.create_function_test(
                        test_id=test.get('id', str(uuid.uuid4())),
                        description=test.get('description', ''),
                        function_name=test.get('function_name'),
                        function_args=test.get('args', []),
                        expected_result=test.get('expected')
                    )
                    test_cases.append(test_case)
                elif test.get('type') == 'output':
                    test_case = TestCaseBuilder.create_output_test(
                        test_id=test.get('id', str(uuid.uuid4())),
                        description=test.get('description', ''),
                        expected_output=test.get('expected'),
                        input_data=test.get('input')
                    )
                    test_cases.append(test_case)
            
            return test_cases
            
        except json.JSONDecodeError:
            logger.warning(f"Métadonnées invalides pour l'étape {step.id}")
            return []
    
    def _update_user_progress(
        self,
        db: Session,
        user_quest: UserQuest,
        step_number: int,
        evaluation_result: Dict[str, Any]
    ):
        """Met à jour la progression utilisateur."""
        # Mettre à jour le score total
        user_quest.score += evaluation_result['score']
        user_quest.last_activity = datetime.utcnow()
        
        # Avancer à l'étape suivante si réussie
        if evaluation_result['is_correct'] and step_number == user_quest.current_step:
            user_quest.current_step += 1
        
        # Calculer le taux de completion
        total_steps = db.query(QuestStep).filter(
            QuestStep.quest_id == user_quest.quest_id
        ).count()
        
        completed_steps = db.query(UserAnswer).filter(
            and_(
                UserAnswer.user_id == user_quest.user_id,
                UserAnswer.quest_id == user_quest.quest_id,
                UserAnswer.is_correct == True
            )
        ).count()
        
        user_quest.completion_rate = completed_steps / max(1, total_steps)
        
        # Marquer comme complétée si toutes les étapes sont terminées
        if user_quest.completion_rate >= 1.0:
            user_quest.status = QuestStatus.COMPLETED.value
            user_quest.completed_at = datetime.utcnow()
        
        db.commit()
        
        # Mettre à jour les compétences utilisateur
        self._update_user_skills(db, user_quest.user_id, evaluation_result)
    
    def _update_user_skills(self, db: Session, user_id: int, evaluation_result: Dict[str, Any]):
        """Met à jour les compétences de l'utilisateur."""
        try:
            user_progress = db.query(UserProgress).filter(
                UserProgress.user_id == user_id
            ).first()
            
            if not user_progress:
                return
            
            # Mettre à jour les compétences basées sur l'évaluation
            if evaluation_result['is_correct']:
                # Améliorer les compétences
                skills = json.loads(user_progress.skills) if user_progress.skills else {}
                
                # Exemple: améliorer les compétences selon le type d'exercice
                if 'syntax' not in skills:
                    skills['syntax'] = 0
                skills['syntax'] = min(100, skills['syntax'] + 5)
                
                if 'logic' not in skills:
                    skills['logic'] = 0
                skills['logic'] = min(100, skills['logic'] + 3)
                
                user_progress.skills = json.dumps(skills)
                user_progress.total_xp += int(evaluation_result['score'] * 10)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des compétences: {e}")
    
    def get_quest_recommendations(self, user_id: int, limit: int = 5) -> List[QuestRecommendation]:
        """
        Génère des recommandations de quêtes pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre maximum de recommandations
            
        Returns:
            Liste de recommandations
        """
        try:
            with get_session() as db:
                # Récupérer le profil utilisateur
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return []
                
                user_progress = db.query(UserProgress).filter(
                    UserProgress.user_id == user_id
                ).first()
                
                # Récupérer les quêtes non commencées
                completed_quest_ids = db.query(UserQuest.quest_id).filter(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.status.in_(['completed', 'in_progress'])
                    )
                ).subquery()
                
                available_quests = db.query(Quest).filter(
                    and_(
                        Quest.is_active == True,
                        ~Quest.id.in_(completed_quest_ids)
                    )
                ).all()
                
                recommendations = []
                
                for quest in available_quests:
                    score = self._calculate_quest_match_score(quest, user, user_progress)
                    
                    if score > 0.3:  # Seuil minimum
                        recommendation = QuestRecommendation(
                            quest_id=quest.id,
                            quest_title=quest.title,
                            difficulty=quest.difficulty,
                            estimated_time=quest.estimated_time,
                            match_score=score,
                            reasons=self._get_recommendation_reasons(quest, user, user_progress),
                            prerequisites_met=self._check_prerequisites(quest, user_progress)
                        )
                        recommendations.append(recommendation)
                
                # Trier par score et limiter
                recommendations.sort(key=lambda x: x.match_score, reverse=True)
                return recommendations[:limit]
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération de recommandations: {e}")
            return []
    
    def _calculate_quest_match_score(
        self,
        quest: Quest,
        user: User,
        user_progress: Optional[UserProgress]
    ) -> float:
        """Calcule le score de correspondance entre une quête et un utilisateur."""
        score = 0.0
        
        if not user_progress:
            return 0.5  # Score par défaut pour nouveaux utilisateurs
        
        # Score basé sur le niveau de difficulté
        user_level = user_progress.level or 1
        difficulty_scores = {
            'beginner': 1.0 if user_level <= 5 else 0.3,
            'intermediate': 1.0 if 3 <= user_level <= 10 else 0.5,
            'advanced': 1.0 if user_level >= 8 else 0.2,
            'expert': 1.0 if user_level >= 15 else 0.1
        }
        score += difficulty_scores.get(quest.difficulty, 0.5) * 0.4
        
        # Score basé sur les compétences
        if user_progress.skills:
            skills = json.loads(user_progress.skills)
            # Associer les tags de quête aux compétences
            for tag in quest.tags:
                skill_level = skills.get(tag.lower(), 0)
                if skill_level < 70:  # L'utilisateur peut encore améliorer cette compétence
                    score += 0.1
        
        # Score basé sur l'activité récente
        if user_progress.last_activity:
            days_inactive = (datetime.utcnow() - user_progress.last_activity).days
            if days_inactive < 7:
                score += 0.2
        
        return min(1.0, score)
    
    def _get_recommendation_reasons(
        self,
        quest: Quest,
        user: User,
        user_progress: Optional[UserProgress]
    ) -> List[str]:
        """Génère les raisons de la recommandation."""
        reasons = []
        
        if not user_progress:
            reasons.append("Parfait pour débuter votre apprentissage")
            return reasons
        
        # Raisons basées sur le niveau
        user_level = user_progress.level or 1
        if quest.difficulty == 'beginner' and user_level <= 3:
            reasons.append("Adapté à votre niveau débutant")
        elif quest.difficulty == 'intermediate' and 3 <= user_level <= 8:
            reasons.append("Progressez vers le niveau intermédiaire")
        elif quest.difficulty == 'advanced' and user_level >= 8:
            reasons.append("Défi avancé pour vos compétences")
        
        # Raisons basées sur les compétences
        if user_progress.skills:
            skills = json.loads(user_progress.skills)
            for tag in quest.tags:
                skill_level = skills.get(tag.lower(), 0)
                if skill_level < 50:
                    reasons.append(f"Améliore vos compétences en {tag}")
        
        # Raisons basées sur la catégorie
        if quest.category:
            reasons.append(f"Explore le domaine: {quest.category}")
        
        if not reasons:
            reasons.append("Recommandé pour votre profil")
        
        return reasons
    
    def _check_prerequisites(self, quest: Quest, user_progress: Optional[UserProgress]) -> bool:
        """Vérifie si les prérequis sont remplis."""
        if not quest.prerequisites or not user_progress:
            return True
        
        # Vérifier les prérequis basiques (niveau, compétences)
        prerequisites = quest.prerequisites
        user_level = user_progress.level or 1
        
        required_level = prerequisites.get('level', 0)
        if user_level < required_level:
            return False
        
        # Vérifier les compétences requises
        if user_progress.skills:
            skills = json.loads(user_progress.skills)
            required_skills = prerequisites.get('skills', {})
            
            for skill, min_level in required_skills.items():
                if skills.get(skill, 0) < min_level:
                    return False
        
        return True
    
    def get_quest_statistics(self, quest_id: str) -> Dict[str, Any]:
        """Récupère les statistiques d'une quête."""
        try:
            with get_session() as db:
                # Statistiques générales
                total_attempts = db.query(UserQuest).filter(
                    UserQuest.quest_id == quest_id
                ).count()
                
                completed = db.query(UserQuest).filter(
                    and_(
                        UserQuest.quest_id == quest_id,
                        UserQuest.status == 'completed'
                    )
                ).count()
                
                # Temps moyen de completion
                avg_time = db.query(func.avg(UserQuest.time_spent)).filter(
                    and_(
                        UserQuest.quest_id == quest_id,
                        UserQuest.status == 'completed'
                    )
                ).scalar() or 0
                
                # Score moyen
                avg_score = db.query(func.avg(UserQuest.score)).filter(
                    UserQuest.quest_id == quest_id
                ).scalar() or 0
                
                # Taux de completion par étape
                steps_stats = []
                steps = db.query(QuestStep).filter(
                    QuestStep.quest_id == quest_id
                ).order_by(QuestStep.step_number).all()
                
                for step in steps:
                    step_attempts = db.query(UserAnswer).filter(
                        and_(
                            UserAnswer.quest_id == quest_id,
                            UserAnswer.step_number == step.step_number
                        )
                    ).count()
                    
                    step_success = db.query(UserAnswer).filter(
                        and_(
                            UserAnswer.quest_id == quest_id,
                            UserAnswer.step_number == step.step_number,
                            UserAnswer.is_correct == True
                        )
                    ).count()
                    
                    steps_stats.append({
                        'step_number': step.step_number,
                        'title': step.title,
                        'attempts': step_attempts,
                        'success': step_success,
                        'success_rate': step_success / max(1, step_attempts)
                    })
                
                return {
                    'quest_id': quest_id,
                    'total_attempts': total_attempts,
                    'completed': completed,
                    'completion_rate': completed / max(1, total_attempts),
                    'average_time': int(avg_time),
                    'average_score': round(avg_score, 2),
                    'steps_statistics': steps_stats
                }
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques: {e}")
            return {}
    
    def refresh_cache(self):
        """Actualise le cache des quêtes."""
        self.cache.clear()
        self._load_quests_cache()
        logger.info("Cache des quêtes actualisé")


# Instance globale
quest_manager = QuestManager()


# Fonctions utilitaires
def get_user_current_quests(user_id: int) -> List[Dict[str, Any]]:
    """Récupère les quêtes en cours d'un utilisateur."""
    return quest_manager.get_user_quests(user_id, status=QuestStatus.IN_PROGRESS)


def start_quest_for_user(user_id: int, quest_id: str) -> bool:
    """Démarre une quête pour un utilisateur."""
    result = quest_manager.start_quest(user_id, quest_id)
    return result is not None


def submit_answer(user_id: int, quest_id: str, step_number: int, answer: Dict[str, Any]) -> Dict[str, Any]:
    """Soumet une réponse pour une étape."""
    return quest_manager.submit_step_answer(user_id, quest_id, step_number, answer)


if __name__ == "__main__":
    # Test du gestionnaire de quêtes
    print("=== Test du Gestionnaire de Quêtes ===")
    
    # Simuler une soumission de réponse
    test_answer = {
        'type': 'code',
        'content': '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
'''
    }
    
    print("Gestionnaire de quêtes initialisé avec succès!")
    print(f"Quêtes en cache: {len(quest_manager.cache)}")