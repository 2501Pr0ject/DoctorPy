"""
Module des modèles de données pour l'assistant pédagogique IA

Ce module contient tous les modèles SQLAlchemy et schémas Pydantic pour :
- La gestion des utilisateurs et leurs profils
- Les quêtes pédagogiques et leur structure
- Le suivi de progression et l'analytics
- L'API et la validation des données

Architecture :
- Modèles SQLAlchemy pour la persistance en base
- Schémas Pydantic pour la validation et l'API
- Relations et contraintes pour l'intégrité des données
- Événements et triggers pour la maintenance automatique
"""

# Import de la base SQLAlchemy
from .user import Base

# ===== MODÈLES UTILISATEUR =====
from .user import (
    User,
    UserProfile, 
    UserSession,
    UserAchievement,
    UserLevel,
    UserRole,
    AccountStatus,
    LearningStyle,
    create_user_with_profile,
    get_user_achievements_summary
)

# ===== MODÈLES QUÊTE =====
from .quest import (
    Quest,
    QuestStep,
    Question,
    UserQuest,
    UserStepProgress,
    UserAnswer,
    QuestStatus,
    QuestDifficulty,
    QuestCategory,
    QuestionType,
    UserQuestStatus
)

# ===== MODÈLES PROGRESSION =====
from .progress import (
    UserProgress,
    DailyProgress,
    SkillAssessment,
    LearningSession,
    SessionEvent,
    ProgressMilestone,
    LearningAnalytics,
    ProgressType,
    SkillCategory,
    LearningMetricType,
    calculate_learning_score,
    generate_learning_insights
)

# ===== SCHÉMAS PYDANTIC =====
from .schemas import (
    # Schémas de base
    TimestampMixin,
    PaginationParams,
    PaginatedResponse,
    APIResponse,
    
    # Schémas utilisateur
    UserBaseSchema,
    UserCreateSchema,
    UserUpdateSchema,
    UserResponseSchema,
    UserStatsSchema,
    
    # Schémas quête
    QuestBaseSchema,
    QuestCreateSchema,
    QuestUpdateSchema,
    QuestResponseSchema,
    QuestSummarySchema,
    QuestStepBaseSchema,
    QuestStepCreateSchema,
    QuestStepUpdateSchema,
    QuestStepResponseSchema,
    QuestionBaseSchema,
    QuestionCreateSchema,
    QuestionUpdateSchema,
    QuestionResponseSchema,
    
    # Schémas progression
    UserProgressSchema,
    DailyProgressSchema,
    SkillAssessmentSchema,
    LearningSessionSchema,
    
    # Schémas réponse utilisateur
    UserAnswerSchema,
    UserAnswerCreateSchema,
    QuizSubmissionSchema,
    
    # Schémas quest utilisateur
    UserQuestSchema,
    UserQuestStartSchema,
    UserQuestProgressSchema,
    
    # Schémas dashboard
    DashboardStatsSchema,
    UserDashboardSchema,
    LearningInsightsSchema,
    
    # Schémas authentification
    LoginSchema,
    TokenSchema,
    LoginResponseSchema,
    PasswordResetSchema,
    PasswordResetConfirmSchema,
    ChangePasswordSchema,
    
    # Schémas recherche
    QuestFilterSchema,
    QuestSearchSchema,
    UserSearchSchema,
    
    # Schémas analytics
    AnalyticsTimeRange,
    LearningAnalyticsSchema,
    SystemAnalyticsSchema,
    
    # Schémas notifications
    NotificationSchema,
    NotificationCreateSchema,
    
    # Schémas export/import
    ExportRequestSchema,
    ImportQuestSchema,
    
    # Schémas configuration
    SystemConfigSchema,
    UserPreferencesSchema,
    
    # Schémas validation
    ValidationResultSchema,
    CodeValidationSchema,
    CodeExecutionResultSchema,
    
    # Schémas gamification
    AchievementSchema,
    UserAchievementSchema,
    LeaderboardEntrySchema,
    LeaderboardSchema
)

# Version du module
__version__ = "1.0.0"

# Export principal
__all__ = [
    # Base SQLAlchemy
    "Base",
    
    # ===== MODÈLES =====
    # Utilisateur
    "User",
    "UserProfile",
    "UserSession", 
    "UserAchievement",
    
    # Quête
    "Quest",
    "QuestStep",
    "Question",
    "UserQuest",
    "UserStepProgress",
    "UserAnswer",
    
    # Progression
    "UserProgress",
    "DailyProgress",
    "SkillAssessment",
    "LearningSession",
    "SessionEvent",
    "ProgressMilestone",
    "LearningAnalytics",
    
    # ===== ENUMS =====
    # Utilisateur
    "UserLevel",
    "UserRole",
    "AccountStatus",
    "LearningStyle",
    
    # Quête
    "QuestStatus",
    "QuestDifficulty",
    "QuestCategory",
    "QuestionType",
    "UserQuestStatus",
    
    # Progression
    "ProgressType",
    "SkillCategory",
    "LearningMetricType",
    
    # ===== SCHÉMAS DE BASE =====
    "TimestampMixin",
    "PaginationParams",
    "PaginatedResponse",
    "APIResponse",
    
    # ===== SCHÉMAS UTILISATEUR =====
    "UserBaseSchema",
    "UserCreateSchema",
    "UserUpdateSchema",
    "UserResponseSchema",
    "UserStatsSchema",
    
    # ===== SCHÉMAS QUÊTE =====
    "QuestBaseSchema",
    "QuestCreateSchema",
    "QuestUpdateSchema",
    "QuestResponseSchema",
    "QuestSummarySchema",
    "QuestStepBaseSchema",
    "QuestStepCreateSchema",
    "QuestStepUpdateSchema",
    "QuestStepResponseSchema",
    "QuestionBaseSchema",
    "QuestionCreateSchema",
    "QuestionUpdateSchema",
    "QuestionResponseSchema",
    
    # ===== SCHÉMAS PROGRESSION =====
    "UserProgressSchema",
    "DailyProgressSchema",
    "SkillAssessmentSchema",
    "LearningSessionSchema",
    
    # ===== SCHÉMAS INTERACTION =====
    "UserAnswerSchema",
    "UserAnswerCreateSchema",
    "QuizSubmissionSchema",
    "UserQuestSchema",
    "UserQuestStartSchema", 
    "UserQuestProgressSchema",
    
    # ===== SCHÉMAS DASHBOARD =====
    "DashboardStatsSchema",
    "UserDashboardSchema",
    "LearningInsightsSchema",
    
    # ===== SCHÉMAS AUTH =====
    "LoginSchema",
    "TokenSchema",
    "LoginResponseSchema",
    "PasswordResetSchema",
    "PasswordResetConfirmSchema",
    "ChangePasswordSchema",
    
    # ===== SCHÉMAS RECHERCHE =====
    "QuestFilterSchema",
    "QuestSearchSchema",
    "UserSearchSchema",
    
    # ===== SCHÉMAS ANALYTICS =====
    "AnalyticsTimeRange",
    "LearningAnalyticsSchema",
    "SystemAnalyticsSchema",
    
    # ===== SCHÉMAS NOTIFICATIONS =====
    "NotificationSchema",
    "NotificationCreateSchema",
    
    # ===== SCHÉMAS IMPORT/EXPORT =====
    "ExportRequestSchema",
    "ImportQuestSchema",
    
    # ===== SCHÉMAS CONFIG =====
    "SystemConfigSchema",
    "UserPreferencesSchema",
    
    # ===== SCHÉMAS VALIDATION =====
    "ValidationResultSchema",
    "CodeValidationSchema",
    "CodeExecutionResultSchema",
    
    # ===== SCHÉMAS GAMIFICATION =====
    "AchievementSchema",
    "UserAchievementSchema",
    "LeaderboardEntrySchema",
    "LeaderboardSchema",
    
    # ===== FONCTIONS UTILITAIRES =====
    "create_user_with_profile",
    "get_user_achievements_summary",
    "calculate_learning_score",
    "generate_learning_insights"
]

# ===== CONFIGURATION DU MODULE =====

# Mapping des modèles par catégorie
MODEL_CATEGORIES = {
    "user": [User, UserProfile, UserSession, UserAchievement],
    "quest": [Quest, QuestStep, Question, UserQuest, UserStepProgress, UserAnswer],
    "progress": [UserProgress, DailyProgress, SkillAssessment, LearningSession, 
                SessionEvent, ProgressMilestone, LearningAnalytics]
}

# Schémas par catégorie
SCHEMA_CATEGORIES = {
    "user": [
        UserBaseSchema, UserCreateSchema, UserUpdateSchema, UserResponseSchema,
        UserStatsSchema, LoginSchema, TokenSchema, LoginResponseSchema,
        PasswordResetSchema, PasswordResetConfirmSchema, ChangePasswordSchema
    ],
    "quest": [
        QuestBaseSchema, QuestCreateSchema, QuestUpdateSchema, QuestResponseSchema,
        QuestSummarySchema, QuestStepBaseSchema, QuestStepCreateSchema,
        QuestStepUpdateSchema, QuestStepResponseSchema, QuestionBaseSchema,
        QuestionCreateSchema, QuestionUpdateSchema, QuestionResponseSchema
    ],
    "progress": [
        UserProgressSchema, DailyProgressSchema, SkillAssessmentSchema,
        LearningSessionSchema, LearningAnalyticsSchema
    ],
    "interaction": [
        UserAnswerSchema, UserAnswerCreateSchema, QuizSubmissionSchema,
        UserQuestSchema, UserQuestStartSchema, UserQuestProgressSchema
    ],
    "system": [
        DashboardStatsSchema, UserDashboardSchema, LearningInsightsSchema,
        SystemAnalyticsSchema, NotificationSchema, NotificationCreateSchema,
        SystemConfigSchema, UserPreferencesSchema
    ]
}

# Relations principales entre modèles
MAIN_RELATIONSHIPS = {
    "User": {
        "profiles": "UserProfile",
        "progress_records": "UserProgress", 
        "sessions": "UserSession",
        "achievements": "UserAchievement",
        "user_quests": "UserQuest"
    },
    "Quest": {
        "steps": "QuestStep",
        "user_quests": "UserQuest"
    },
    "QuestStep": {
        "questions": "Question",
        "user_step_progress": "UserStepProgress"
    },
    "UserProgress": {
        "daily_progress": "DailyProgress",
        "skill_assessments": "SkillAssessment",
        "learning_sessions": "LearningSession"
    }
}

def get_model_info() -> dict:
    """
    Retourne des informations sur les modèles disponibles
    
    Returns:
        Dictionnaire avec les informations des modèles
    """
    return {
        "version": __version__,
        "total_models": len([model for models in MODEL_CATEGORIES.values() for model in models]),
        "total_schemas": len([schema for schemas in SCHEMA_CATEGORIES.values() for schema in schemas]),
        "categories": list(MODEL_CATEGORIES.keys()),
        "relationships": MAIN_RELATIONSHIPS,
        "enums": {
            "user": ["UserLevel", "UserRole", "AccountStatus", "LearningStyle"],
            "quest": ["QuestStatus", "QuestDifficulty", "QuestCategory", "QuestionType", "UserQuestStatus"],
            "progress": ["ProgressType", "SkillCategory", "LearningMetricType"]
        }
    }

def get_models_by_category(category: str) -> list:
    """
    Retourne les modèles d'une catégorie spécifique
    
    Args:
        category: Nom de la catégorie
        
    Returns:
        Liste des modèles de la catégorie
    """
    return MODEL_CATEGORIES.get(category, [])

def get_schemas_by_category(category: str) -> list:
    """
    Retourne les schémas d'une catégorie spécifique
    
    Args:
        category: Nom de la catégorie
        
    Returns:
        Liste des schémas de la catégorie
    """
    return SCHEMA_CATEGORIES.get(category, [])

def create_all_tables(engine):
    """
    Crée toutes les tables en base de données
    
    Args:
        engine: Moteur SQLAlchemy
    """
    try:
        Base.metadata.create_all(bind=engine)
        print(f"✅ Tables créées avec succès pour {len(Base.metadata.tables)} tables")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la création des tables: {e}")
        return False

def drop_all_tables(engine):
    """
    Supprime toutes les tables (ATTENTION: destructif!)
    
    Args:
        engine: Moteur SQLAlchemy
    """
    try:
        Base.metadata.drop_all(bind=engine)
        print("⚠️  Toutes les tables ont été supprimées")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la suppression des tables: {e}")
        return False

def validate_model_relationships():
    """
    Valide les relations entre modèles
    
    Returns:
        Dictionnaire avec le résultat de la validation
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Vérifications basiques des relations
    try:
        # Vérifier que les clés étrangères correspondent
        for table_name, table in Base.metadata.tables.items():
            for fk in table.foreign_keys:
                target_table = fk.column.table.name
                if target_table not in Base.metadata.tables:
                    validation_results["errors"].append(
                        f"Table {table_name} référence une table inexistante: {target_table}"
                    )
                    validation_results["valid"] = False
        
        # Vérifier les contraintes
        for table_name, table in Base.metadata.tables.items():
            for constraint in table.constraints:
                if hasattr(constraint, 'columns') and not constraint.columns:
                    validation_results["warnings"].append(
                        f"Contrainte vide dans la table {table_name}"
                    )
    
    except Exception as e:
        validation_results["errors"].append(f"Erreur lors de la validation: {str(e)}")
        validation_results["valid"] = False
    
    return validation_results

# ===== UTILITAIRES DE MIGRATION =====

def get_migration_scripts() -> list:
    """
    Retourne la liste des scripts de migration disponibles
    
    Returns:
        Liste des migrations
    """
    return [
        {
            "version": "001",
            "description": "Création des tables utilisateur",
            "models": ["User", "UserProfile", "UserSession"]
        },
        {
            "version": "002", 
            "description": "Création des tables quête",
            "models": ["Quest", "QuestStep", "Question"]
        },
        {
            "version": "003",
            "description": "Création des tables progression",
            "models": ["UserProgress", "DailyProgress", "SkillAssessment"]
        },
        {
            "version": "004",
            "description": "Ajout des relations utilisateur-quête",
            "models": ["UserQuest", "UserStepProgress", "UserAnswer"]
        },
        {
            "version": "005",
            "description": "Analytics et sessions d'apprentissage", 
            "models": ["LearningSession", "SessionEvent", "LearningAnalytics"]
        }
    ]

def check_database_version(engine) -> dict:
    """
    Vérifie la version de la base de données
    
    Args:
        engine: Moteur SQLAlchemy
        
    Returns:
        Informations sur la version de la DB
    """
    from sqlalchemy import text
    
    try:
        with engine.connect() as conn:
            # Vérifier si la table de version existe
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
            ))
            
            if result.fetchone():
                # Récupérer la version actuelle
                version_result = conn.execute(text("SELECT version_num FROM alembic_version"))
                current_version = version_result.fetchone()
                
                return {
                    "has_version_table": True,
                    "current_version": current_version[0] if current_version else None,
                    "needs_migration": False
                }
            else:
                return {
                    "has_version_table": False,
                    "current_version": None,
                    "needs_migration": True
                }
    
    except Exception as e:
        return {
            "error": str(e),
            "has_version_table": False,
            "needs_migration": True
        }

# ===== VALIDATION DES DONNÉES =====

def validate_user_data(user_data: dict) -> ValidationResultSchema:
    """
    Valide les données utilisateur
    
    Args:
        user_data: Données à valider
        
    Returns:
        Résultat de validation
    """
    try:
        UserCreateSchema(**user_data)
        return ValidationResultSchema(is_valid=True)
    except Exception as e:
        return ValidationResultSchema(
            is_valid=False,
            errors=[str(e)]
        )

def validate_quest_data(quest_data: dict) -> ValidationResultSchema:
    """
    Valide les données de quête
    
    Args:
        quest_data: Données à valider
        
    Returns:
        Résultat de validation
    """
    try:
        QuestCreateSchema(**quest_data)
        return ValidationResultSchema(is_valid=True)
    except Exception as e:
        return ValidationResultSchema(
            is_valid=False,
            errors=[str(e)]
        )

# ===== FONCTIONS DE SEED/DEMO =====

def create_demo_data(session):
    """
    Crée des données de démonstration
    
    Args:
        session: Session SQLAlchemy
    """
    from datetime import datetime, timedelta
    import random
    
    try:
        # Créer un utilisateur de démo
        demo_user = User(
            username="demo_user",
            email="demo@example.com",
            first_name="Utilisateur",
            last_name="Démo",
            level=UserLevel.BEGINNER.value
        )
        demo_user.set_password("demo123!")
        session.add(demo_user)
        session.flush()
        
        # Créer une progression de base
        user_progress = UserProgress(
            user_id=demo_user.id,
            xp_points=250,
            syntax_score=75.0,
            logic_score=60.0,
            problem_solving_score=45.0
        )
        session.add(user_progress)
        
        # Créer une quête de démo
        demo_quest = Quest(
            title="Introduction à Python",
            description="Apprenez les bases du langage Python",
            short_description="Concepts de base de Python",
            category=QuestCategory.PYTHON_BASICS.value,
            difficulty=QuestDifficulty.EASY.value,
            level=UserLevel.BEGINNER.value,
            slug="introduction-python",
            created_by=demo_user.id,
            status=QuestStatus.PUBLISHED.value
        )
        demo_quest.set_learning_objectives([
            "Comprendre les variables",
            "Utiliser les types de données de base",
            "Écrire des fonctions simples"
        ])
        session.add(demo_quest)
        session.flush()
        
        # Créer une étape
        demo_step = QuestStep(
            quest_id=demo_quest.id,
            order=1,
            title="Les variables en Python",
            description="Apprenez à déclarer et utiliser des variables",
            content="Les variables permettent de stocker des données...",
            code_template="# Créez une variable\nnom = "
        )
        session.add(demo_step)
        session.flush()
        
        # Créer une question
        demo_question = Question(
            step_id=demo_step.id,
            question_text="Comment déclare-t-on une variable en Python ?",
            question_type=QuestionType.MULTIPLE_CHOICE.value,
            correct_answer="nom = 'valeur'"
        )
        demo_question.set_choices([
            "nom = 'valeur'",
            "var nom = 'valeur'",
            "let nom = 'valeur'",
            "string nom = 'valeur'"
        ])
        session.add(demo_question)
        
        # Créer quelques données de progression quotidienne
        for i in range(7):
            daily_data = DailyProgress(
                user_progress_id=user_progress.id,
                date=datetime.now().date() - timedelta(days=i),
                study_time_minutes=random.randint(15, 60),
                quests_completed=random.randint(0, 2),
                questions_answered=random.randint(5, 20),
                correct_answers=random.randint(3, 18),
                xp_earned=random.randint(20, 100)
            )
            session.add(daily_data)
        
        session.commit()
        print("✅ Données de démo créées avec succès")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Erreur lors de la création des données de démo: {e}")

# ===== INITIALISATION DU MODULE =====

def initialize_models():
    """Initialise le module des modèles"""
    print(f"📊 Module models initialisé (version {__version__})")
    print(f"   - {len([m for models in MODEL_CATEGORIES.values() for m in models])} modèles SQLAlchemy")
    print(f"   - {len([s for schemas in SCHEMA_CATEGORIES.values() for s in schemas])} schémas Pydantic")
    print(f"   - {len(MODEL_CATEGORIES)} catégories de modèles")

# Initialiser automatiquement
initialize_models()