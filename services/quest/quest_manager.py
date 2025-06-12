"""
Gestionnaire principal du service Quest
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from ..shared.cache import CacheManager
from ..shared.events import EventBus, EventType
from ..shared.utils import LoggerFactory

from .models import (
    Quest, QuestProgress, Achievement, UserAchievement, UserStats,
    QuestStatus, QuestDifficulty, QuestCategory, AchievementType,
    StartQuestRequest, SubmitAnswerRequest, CreateQuestRequest,
    QuestSearchRequest, ProgressResponse, SubmitAnswerResponse,
    AnswerFeedback, LeaderboardResponse, LeaderboardEntry,
    QuestServiceConfig, QuestServiceError
)


class QuestManager:
    """Gestionnaire principal des quêtes et gamification"""
    
    def __init__(self, config: QuestServiceConfig, cache: CacheManager, event_bus: EventBus):
        self.config = config
        self.cache = cache
        self.event_bus = event_bus
        self.logger = LoggerFactory.get_logger("quest_manager")
        
        # Stockage en mémoire pour le développement (à remplacer par DB)
        self.quests: Dict[str, Quest] = {}
        self.progress: Dict[str, QuestProgress] = {}
        self.achievements: Dict[str, Achievement] = {}
        self.user_achievements: Dict[str, List[UserAchievement]] = defaultdict(list)
        self.user_stats: Dict[str, UserStats] = {}
        
        # Initialiser les achievements par défaut
        self._init_default_achievements()
        
        # Charger quelques quêtes d'exemple
        self._init_sample_quests()
        
        self.logger.info("✅ Quest Manager initialisé")
    
    def _init_default_achievements(self):
        """Initialise les achievements par défaut"""
        default_achievements = [
            Achievement(
                id="first_quest",
                name="Premier Pas",
                description="Complétez votre première quête",
                icon="🎯",
                achievement_type=AchievementType.QUEST_COMPLETION,
                points_reward=25,
                badge_color="bronze"
            ),
            Achievement(
                id="python_basics_master",
                name="Maître des Bases",
                description="Complétez 5 quêtes de Python Basics",
                icon="🐍",
                achievement_type=AchievementType.SKILL_MASTERY,
                required_category=QuestCategory.PYTHON_BASICS,
                points_reward=100,
                badge_color="silver"
            ),
            Achievement(
                id="speed_demon",
                name="Démon de Vitesse",
                description="Complétez une quête en moins de 50% du temps estimé",
                icon="⚡",
                achievement_type=AchievementType.SPEED_COMPLETION,
                points_reward=50,
                badge_color="gold"
            ),
            Achievement(
                id="perfectionist",
                name="Perfectionniste",
                description="Obtenez un score parfait sur 3 quêtes",
                icon="💎",
                achievement_type=AchievementType.PERFECT_SCORE,
                points_reward=75,
                badge_color="diamond"
            ),
            Achievement(
                id="week_streak",
                name="Habitué",
                description="Maintenez une série de 7 jours",
                icon="🔥",
                achievement_type=AchievementType.STREAK,
                required_streak=7,
                points_reward=150,
                badge_color="red"
            )
        ]
        
        for achievement in default_achievements:
            self.achievements[achievement.id] = achievement
        
        self.logger.info(f"✅ {len(default_achievements)} achievements par défaut initialisés")
    
    def _init_sample_quests(self):
        """Initialise quelques quêtes d'exemple"""
        from .models import Question, QuestionType
        
        # Quête 1: Variables Python
        quest1 = Quest(
            id="python_variables_101",
            title="Les Variables en Python",
            description="Apprenez les bases des variables en Python",
            category=QuestCategory.PYTHON_BASICS,
            difficulty=QuestDifficulty.EASY,
            status=QuestStatus.PUBLISHED,
            questions=[
                Question(
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    question_text="Comment déclarer une variable nommée 'age' avec la valeur 25 en Python ?",
                    options=["age = 25", "var age = 25", "int age = 25", "age := 25"],
                    correct_answer="age = 25",
                    explanation="En Python, on déclare une variable simplement avec le signe égal : nom_variable = valeur",
                    points=10
                ),
                Question(
                    question_type=QuestionType.TRUE_FALSE,
                    question_text="En Python, il faut déclarer le type d'une variable avant de l'utiliser.",
                    correct_answer="False",
                    explanation="Python est un langage à typage dynamique. Le type est inféré automatiquement.",
                    points=10
                )
            ],
            estimated_time_minutes=15,
            tags=["variables", "basics", "débutant"]
        )
        quest1.total_points = sum(q.points for q in quest1.questions)
        
        # Quête 2: Boucles For
        quest2 = Quest(
            id="python_for_loops",
            title="Maîtriser les Boucles For",
            description="Découvrez comment utiliser les boucles for en Python",
            category=QuestCategory.PYTHON_BASICS,
            difficulty=QuestDifficulty.MEDIUM,
            status=QuestStatus.PUBLISHED,
            questions=[
                Question(
                    question_type=QuestionType.CODE_COMPLETION,
                    question_text="Complétez ce code pour afficher les nombres de 1 à 5 :",
                    code_snippet="for i in _____(1, 6):\\n    print(i)",
                    correct_answer="range",
                    explanation="La fonction range(1, 6) génère les nombres de 1 à 5 (6 exclu)",
                    points=15
                ),
                Question(
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    question_text="Que fait ce code : for char in 'Python': print(char)",
                    options=[
                        "Affiche 'Python'",
                        "Affiche chaque lettre sur une ligne",
                        "Provoque une erreur",
                        "Ne fait rien"
                    ],
                    correct_answer="Affiche chaque lettre sur une ligne",
                    explanation="Une boucle for peut itérer sur chaque caractère d'une chaîne",
                    points=15
                )
            ],
            estimated_time_minutes=20,
            tags=["boucles", "for", "itération"],
            prerequisites=["python_variables_101"]
        )
        quest2.total_points = sum(q.points for q in quest2.questions)
        
        # Ajouter les quêtes
        self.quests[quest1.id] = quest1
        self.quests[quest2.id] = quest2
        
        self.logger.info(f"✅ {len(self.quests)} quêtes d'exemple initialisées")
    
    async def get_available_quests(self, user_id: str, search: QuestSearchRequest) -> List[Quest]:
        """Récupère les quêtes disponibles pour un utilisateur"""
        try:
            # Récupérer les stats utilisateur pour vérifier les prérequis
            user_stats = await self.get_user_stats(user_id)
            completed_quests = set()
            
            # Récupérer les quêtes complétées
            for progress_id, progress in self.progress.items():
                if progress.user_id == user_id and progress.status == QuestStatus.COMPLETED:
                    completed_quests.add(progress.quest_id)
            
            # Filtrer les quêtes
            available_quests = []
            for quest in self.quests.values():
                # Vérifier le statut
                if quest.status != QuestStatus.PUBLISHED:
                    continue
                
                # Appliquer les filtres de recherche
                if search.category and quest.category != search.category:
                    continue
                if search.difficulty and quest.difficulty != search.difficulty:
                    continue
                if search.search_term and search.search_term.lower() not in quest.title.lower():
                    continue
                if search.tags and not any(tag in quest.tags for tag in search.tags):
                    continue
                
                # Vérifier les prérequis
                if quest.prerequisites:
                    if not all(prereq in completed_quests for prereq in quest.prerequisites):
                        continue
                
                available_quests.append(quest)
            
            # Pagination
            start = search.offset
            end = start + search.limit
            
            return available_quests[start:end]
            
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération quêtes: {e}")
            raise QuestServiceError(f"Erreur récupération quêtes: {e}", "QUEST_RETRIEVAL_ERROR")
    
    async def start_quest(self, request: StartQuestRequest) -> QuestProgress:
        """Démarre une nouvelle quête pour un utilisateur"""
        try:
            # Vérifier que la quête existe
            if request.quest_id not in self.quests:
                raise QuestServiceError("Quête introuvable", "QUEST_NOT_FOUND")
            
            quest = self.quests[request.quest_id]
            
            # Vérifier que l'utilisateur n'a pas déjà une progression en cours
            for progress in self.progress.values():
                if (progress.user_id == request.user_id and 
                    progress.quest_id == request.quest_id and 
                    progress.status == QuestStatus.IN_PROGRESS):
                    return progress
            
            # Créer une nouvelle progression
            progress = QuestProgress(
                user_id=request.user_id,
                quest_id=request.quest_id,
                status=QuestStatus.IN_PROGRESS,
                max_possible_score=quest.total_points
            )
            
            self.progress[progress.id] = progress
            
            # Mettre en cache
            cache_key = f"quest_progress:{request.user_id}:{request.quest_id}"
            await self.cache.set(cache_key, progress.dict(), ttl=self.config.cache_ttl_seconds)
            
            # Envoyer l'événement
            await self.event_bus.publish(
                EventType.QUEST_STARTED,
                {
                    "user_id": request.user_id,
                    "quest_id": request.quest_id,
                    "progress_id": progress.id,
                    "timestamp": progress.started_at.isoformat()
                }
            )
            
            self.logger.info(f"🎯 Quête démarrée: {quest.title} pour {request.user_id}")
            return progress
            
        except QuestServiceError:
            raise
        except Exception as e:
            self.logger.error(f"❌ Erreur démarrage quête: {e}")
            raise QuestServiceError(f"Erreur démarrage quête: {e}", "QUEST_START_ERROR")
    
    async def submit_answer(self, request: SubmitAnswerRequest) -> SubmitAnswerResponse:
        """Soumet une réponse à une question"""
        try:
            # Récupérer la progression
            if request.progress_id not in self.progress:
                raise QuestServiceError("Progression introuvable", "PROGRESS_NOT_FOUND")
            
            progress = self.progress[request.progress_id]
            quest = self.quests[progress.quest_id]
            
            # Vérifier que la quête est en cours
            if progress.status != QuestStatus.IN_PROGRESS:
                raise QuestServiceError("Cette quête n'est pas en cours", "QUEST_NOT_IN_PROGRESS")
            
            # Trouver la question
            current_question = None
            for question in quest.questions:
                if question.id == request.question_id:
                    current_question = question
                    break
            
            if not current_question:
                raise QuestServiceError("Question introuvable", "QUESTION_NOT_FOUND")
            
            # Évaluer la réponse
            is_correct = self._evaluate_answer(current_question, request.answer)
            points_earned = current_question.points if is_correct else 0
            
            # Mettre à jour la progression
            progress.answers[request.question_id] = request.answer
            progress.score += points_earned
            progress.time_spent_seconds += request.time_spent_seconds
            
            # Créer le feedback
            feedback = AnswerFeedback(
                is_correct=is_correct,
                points_earned=points_earned,
                explanation=current_question.explanation,
                correct_answer=current_question.correct_answer if not is_correct else None,
                hints_available=current_question.hints
            )
            
            # Vérifier si c'est la dernière question
            answered_questions = len(progress.answers)
            total_questions = len(quest.questions)
            quest_completed = answered_questions >= total_questions
            
            achievements_unlocked = []
            
            if quest_completed:
                # Marquer la quête comme complétée
                progress.status = QuestStatus.COMPLETED
                progress.completed_at = datetime.now()
                progress.current_question_index = total_questions
                
                # Vérifier les achievements
                achievements_unlocked = await self._check_achievements(progress.user_id, progress, quest)
                
                # Mettre à jour les stats utilisateur
                await self._update_user_stats(progress.user_id, progress, quest)
                
                # Envoyer l'événement de completion
                await self.event_bus.publish(
                    EventType.QUEST_COMPLETED,
                    {
                        "user_id": progress.user_id,
                        "quest_id": progress.quest_id,
                        "progress_id": progress.id,
                        "score": progress.score,
                        "max_score": progress.max_possible_score,
                        "time_spent": progress.time_spent_seconds,
                        "achievements_unlocked": [ach.id for ach in achievements_unlocked]
                    }
                )
                
                self.logger.info(f"🎉 Quête complétée: {quest.title} par {progress.user_id}")
            else:
                # Passer à la question suivante
                progress.current_question_index += 1
            
            # Sauvegarder la progression
            self.progress[request.progress_id] = progress
            
            return SubmitAnswerResponse(
                feedback=feedback,
                progress=progress,
                quest_completed=quest_completed,
                achievements_unlocked=achievements_unlocked
            )
            
        except QuestServiceError:
            raise
        except Exception as e:
            self.logger.error(f"❌ Erreur soumission réponse: {e}")
            raise QuestServiceError(f"Erreur soumission réponse: {e}", "ANSWER_SUBMISSION_ERROR")
    
    def _evaluate_answer(self, question, user_answer) -> bool:
        """Évalue si une réponse est correcte"""
        correct = question.correct_answer
        
        if question.question_type.value in ["multiple_choice", "true_false", "fill_blank"]:
            return str(user_answer).strip().lower() == str(correct).strip().lower()
        elif question.question_type.value == "code_completion":
            # Pour les questions de code, on peut faire une comparaison plus flexible
            return str(user_answer).strip() == str(correct).strip()
        else:
            # Pour les autres types, comparaison directe
            return user_answer == correct
    
    async def _check_achievements(self, user_id: str, progress: QuestProgress, quest: Quest) -> List[Achievement]:
        """Vérifie et débloque les achievements appropriés"""
        unlocked = []
        user_achievements = {ach.achievement_id for ach in self.user_achievements[user_id]}
        
        for achievement in self.achievements.values():
            # Ignorer si déjà débloqué
            if achievement.id in user_achievements:
                continue
            
            should_unlock = False
            
            if achievement.achievement_type == AchievementType.QUEST_COMPLETION:
                # Premier achievement pour toute completion
                completed_count = len([
                    p for p in self.progress.values() 
                    if p.user_id == user_id and p.status == QuestStatus.COMPLETED
                ])
                should_unlock = completed_count == 1  # Première quête
            
            elif achievement.achievement_type == AchievementType.PERFECT_SCORE:
                # Score parfait
                if progress.score == progress.max_possible_score:
                    perfect_scores = len([
                        p for p in self.progress.values()
                        if (p.user_id == user_id and 
                            p.status == QuestStatus.COMPLETED and 
                            p.score == p.max_possible_score)
                    ])
                    should_unlock = perfect_scores >= 3
            
            elif achievement.achievement_type == AchievementType.SPEED_COMPLETION:
                # Completion rapide
                time_ratio = progress.time_spent_seconds / (quest.estimated_time_minutes * 60)
                should_unlock = time_ratio <= 0.5  # 50% du temps
            
            elif achievement.achievement_type == AchievementType.SKILL_MASTERY:
                # Maîtrise d'une catégorie
                if achievement.required_category == quest.category:
                    category_completions = len([
                        p for p in self.progress.values()
                        if (p.user_id == user_id and 
                            p.status == QuestStatus.COMPLETED and
                            self.quests[p.quest_id].category == quest.category)
                    ])
                    should_unlock = category_completions >= 5
            
            if should_unlock:
                # Débloquer l'achievement
                user_achievement = UserAchievement(
                    user_id=user_id,
                    achievement_id=achievement.id,
                    quest_id=quest.id
                )
                self.user_achievements[user_id].append(user_achievement)
                unlocked.append(achievement)
                
                self.logger.info(f"🏆 Achievement débloqué: {achievement.name} pour {user_id}")
        
        return unlocked
    
    async def _update_user_stats(self, user_id: str, progress: QuestProgress, quest: Quest):
        """Met à jour les statistiques utilisateur"""
        if user_id not in self.user_stats:
            self.user_stats[user_id] = UserStats(user_id=user_id)
        
        stats = self.user_stats[user_id]
        
        # Mise à jour des stats générales
        stats.total_quests_completed += 1
        stats.total_points += progress.score
        stats.total_time_spent_minutes += progress.time_spent_seconds // 60
        
        # Niveau (basé sur les points)
        stats.current_level = max(1, stats.total_points // 100)
        
        # Streak (simplifié)
        if stats.last_active.date() == (datetime.now() - timedelta(days=1)).date():
            stats.current_streak += 1
        else:
            stats.current_streak = 1
        stats.longest_streak = max(stats.longest_streak, stats.current_streak)
        
        # Score moyen
        total_possible = sum(
            p.max_possible_score for p in self.progress.values() 
            if p.user_id == user_id and p.status == QuestStatus.COMPLETED
        )
        total_earned = sum(
            p.score for p in self.progress.values() 
            if p.user_id == user_id and p.status == QuestStatus.COMPLETED
        )
        stats.average_score_percentage = (total_earned / total_possible * 100) if total_possible > 0 else 0
        
        # Perfect scores
        if progress.score == progress.max_possible_score:
            stats.perfect_scores += 1
        
        # Progression par catégorie
        if quest.category not in stats.category_progress:
            stats.category_progress[quest.category] = {"completed": 0, "total_points": 0}
        stats.category_progress[quest.category]["completed"] += 1
        stats.category_progress[quest.category]["total_points"] += progress.score
        
        # Achievements
        stats.total_achievements = len(self.user_achievements[user_id])
        
        # Dernière activité
        stats.last_active = datetime.now()
        
        self.user_stats[user_id] = stats
    
    async def get_user_stats(self, user_id: str) -> UserStats:
        """Récupère les statistiques d'un utilisateur"""
        if user_id not in self.user_stats:
            self.user_stats[user_id] = UserStats(user_id=user_id)
        
        return self.user_stats[user_id]
    
    async def get_leaderboard(self, limit: int = 50) -> LeaderboardResponse:
        """Récupère le classement général"""
        try:
            # Trier les utilisateurs par points
            sorted_users = sorted(
                self.user_stats.items(),
                key=lambda x: x[1].total_points,
                reverse=True
            )
            
            entries = []
            for rank, (user_id, stats) in enumerate(sorted_users[:limit], 1):
                entry = LeaderboardEntry(
                    user_id=user_id,
                    username=f"User{user_id[:8]}",  # Nom simplifié
                    total_points=stats.total_points,
                    quests_completed=stats.total_quests_completed,
                    current_level=stats.current_level,
                    rank=rank
                )
                entries.append(entry)
            
            return LeaderboardResponse(
                entries=entries,
                total_users=len(self.user_stats)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération leaderboard: {e}")
            raise QuestServiceError(f"Erreur récupération leaderboard: {e}", "LEADERBOARD_ERROR")
    
    async def create_quest(self, request: CreateQuestRequest, creator_id: str) -> Quest:
        """Crée une nouvelle quête"""
        try:
            quest = Quest(
                title=request.title,
                description=request.description,
                category=request.category,
                difficulty=request.difficulty,
                questions=request.questions,
                estimated_time_minutes=request.estimated_time_minutes,
                prerequisites=request.prerequisites,
                tags=request.tags,
                created_by=creator_id
            )
            
            # Calculer le total de points
            quest.total_points = sum(q.points for q in quest.questions)
            
            # Sauvegarder
            self.quests[quest.id] = quest
            
            self.logger.info(f"📝 Nouvelle quête créée: {quest.title}")
            return quest
            
        except Exception as e:
            self.logger.error(f"❌ Erreur création quête: {e}")
            raise QuestServiceError(f"Erreur création quête: {e}", "QUEST_CREATION_ERROR")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Retourne le statut de santé du service"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "total_quests": len(self.quests),
            "active_progresses": len([p for p in self.progress.values() if p.status == QuestStatus.IN_PROGRESS]),
            "total_users": len(self.user_stats),
            "total_achievements": len(self.achievements),
            "service_version": "1.0.0"
        }