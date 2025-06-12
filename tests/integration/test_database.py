"""Tests d'intégration pour la base de données"""

import pytest
import tempfile
from pathlib import Path
from src.core.database import DatabaseManager
from src.models.user import User
from src.models.quest import Quest


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Tests d'intégration pour les opérations de base de données"""
    
    @pytest.fixture
    def clean_db(self):
        """Base de données propre pour chaque test"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db_manager = DatabaseManager(db_path)
        yield db_manager
        
        # Nettoyer après le test
        try:
            Path(db_path).unlink()
        except FileNotFoundError:
            pass
    
    def test_user_crud_operations(self, clean_db):
        """Test des opérations CRUD pour les utilisateurs"""
        db = clean_db
        
        # CREATE - Créer un utilisateur
        user_id = db.create_user(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        
        assert user_id is not None
        assert isinstance(user_id, int)
        
        # READ - Lire l'utilisateur
        user = db.get_user_by_id(user_id)
        assert user is not None
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        assert user["xp_total"] == 0
        assert user["level"] == 1
        
        # READ by username
        user_by_username = db.get_user_by_username("testuser")
        assert user_by_username["id"] == user_id
        
        # UPDATE - Mettre à jour l'XP
        success = db.update_user_xp(user_id, 250)
        assert success is True
        
        updated_user = db.get_user_by_id(user_id)
        assert updated_user["xp_total"] == 250
        
        # UPDATE - Dernière connexion
        success = db.update_user_login(user_id)
        assert success is True
        
        updated_user = db.get_user_by_id(user_id)
        assert updated_user["last_login"] is not None
    
    def test_quest_operations(self, clean_db, sample_quest_data):
        """Test des opérations de quêtes"""
        db = clean_db
        
        # Insérer une quête
        db.execute_update(
            """INSERT INTO quests (
                id, title, description, difficulty, category, 
                estimated_time, prerequisites, learning_objectives, content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sample_quest_data["id"],
                sample_quest_data["title"],
                sample_quest_data["description"],
                sample_quest_data["difficulty"],
                sample_quest_data["category"],
                sample_quest_data["estimated_time"],
                "[]",  # prerequisites
                "[]",  # learning_objectives
                '{"steps": []}'  # content simplifié
            )
        )
        
        # Récupérer la quête
        quest = db.get_quest_by_id(sample_quest_data["id"])
        assert quest is not None
        assert quest["title"] == sample_quest_data["title"]
        assert quest["difficulty"] == sample_quest_data["difficulty"]
        
        # Récupérer par difficulté
        beginner_quests = db.get_quests_by_difficulty("beginner")
        assert len(beginner_quests) >= 1
        assert any(q["id"] == sample_quest_data["id"] for q in beginner_quests)
        
        # Récupérer par catégorie
        python_quests = db.get_quests_by_category("python")
        assert len(python_quests) >= 1
        assert any(q["id"] == sample_quest_data["id"] for q in python_quests)
    
    def test_user_progress_operations(self, clean_db):
        """Test des opérations de progression utilisateur"""
        db = clean_db
        
        # Créer un utilisateur et une quête
        user_id = db.create_user("testuser", "test@example.com")
        
        db.execute_update(
            """INSERT INTO quests (id, title, difficulty, category, content) 
               VALUES (?, ?, ?, ?, ?)""",
            ("test_quest", "Test Quest", "beginner", "python", '{"steps": []}')
        )
        
        # Créer une progression
        progress_id = db.create_user_progress(user_id, "test_quest")
        assert progress_id is not None
        
        # Mettre à jour la progression
        success = db.update_user_progress(
            user_id, "test_quest", 1, 50.0, {"hints_used": 2}
        )
        assert success is True
        
        # Récupérer la progression
        progress = db.get_user_progress(user_id, "test_quest")
        assert len(progress) == 1
        assert progress[0]["current_step"] == 1
        assert progress[0]["completion_percentage"] == 50.0
        
        # Compléter la quête
        success = db.complete_quest(user_id, "test_quest", 100)
        assert success is True
        
        # Vérifier la complétion
        completed_progress = db.get_user_progress(user_id, "test_quest")
        assert completed_progress[0]["status"] == "completed"
        assert completed_progress[0]["completion_percentage"] == 100.0
        assert completed_progress[0]["xp_earned"] == 100
        
        # Vérifier que l'XP utilisateur a été mis à jour
        user = db.get_user_by_id(user_id)
        assert user["xp_total"] == 100
    
    def test_chat_session_operations(self, clean_db):
        """Test des opérations de sessions de chat"""
        db = clean_db
        
        # Créer un utilisateur
        user_id = db.create_user("chatuser", "chat@example.com")
        
        # Créer une session de chat
        session_id = "test_session_123"
        success = db.create_chat_session(session_id, user_id, "free_chat")
        assert success is True
        
        # Ajouter des messages
        message_id_1 = db.add_message(
            session_id, user_id, "user", "Bonjour!", {"timestamp": "2023-01-01"}
        )
        assert message_id_1 is not None
        
        message_id_2 = db.add_message(
            session_id, user_id, "assistant", "Bonjour! Comment puis-je vous aider?"
        )
        assert message_id_2 is not None
        
        # Récupérer les messages de la session
        messages = db.get_session_messages(session_id, limit=10)
        assert len(messages) == 2
        
        # Les messages devraient être triés par timestamp (le plus récent en premier)
        assert messages[0]["id"] == message_id_2  # Message assistant (plus récent)
        assert messages[1]["id"] == message_id_1  # Message utilisateur
    
    def test_analytics_operations(self, clean_db):
        """Test des opérations d'analytics"""
        db = clean_db
        
        # Créer un utilisateur
        user_id = db.create_user("analyticsuser", "analytics@example.com")
        
        # Enregistrer des événements
        event_id_1 = db.log_event(
            "quest_started", user_id, 
            {"quest_id": "python_basics", "difficulty": "beginner"}
        )
        assert event_id_1 is not None
        
        event_id_2 = db.log_event(
            "quest_completed", user_id,
            {"quest_id": "python_basics", "xp_earned": 100}
        )
        assert event_id_2 is not None
        
        event_id_3 = db.log_event(
            "login", user_id,
            {"login_method": "username", "success": True}
        )
        assert event_id_3 is not None
        
        # Récupérer les analytics de l'utilisateur
        analytics = db.get_user_analytics(user_id, limit=10)
        assert len(analytics) == 3
        
        # Vérifier les données des événements
        quest_started = next(a for a in analytics if a["event_type"] == "quest_started")
        assert "quest_id" in quest_started["event_data"]
        
        quest_completed = next(a for a in analytics if a["event_type"] == "quest_completed")
        assert "xp_earned" in quest_completed["event_data"]
    
    def test_database_statistics(self, clean_db):
        """Test des statistiques de base de données"""
        db = clean_db
        
        # Créer quelques données
        user_id = db.create_user("statsuser", "stats@example.com")
        
        db.execute_update(
            """INSERT INTO quests (id, title, difficulty, category, content) 
               VALUES (?, ?, ?, ?, ?)""",
            ("stats_quest", "Stats Quest", "beginner", "python", '{"steps": []}')
        )
        
        db.create_user_progress(user_id, "stats_quest")
        db.create_chat_session("stats_session", user_id, "quest_mode")
        db.add_message("stats_session", user_id, "user", "Hello")
        db.log_event("test_event", user_id, {"test": True})
        
        # Obtenir les statistiques
        stats = db.get_database_stats()
        
        assert "users" in stats
        assert "quests" in stats
        assert "user_progress" in stats
        assert "chat_sessions" in stats
        assert "messages" in stats
        assert "analytics" in stats
        
        assert stats["users"] >= 1
        assert stats["quests"] >= 1
        assert stats["user_progress"] >= 1
        assert stats["chat_sessions"] >= 1
        assert stats["messages"] >= 1
        assert stats["analytics"] >= 1
    
    def test_foreign_key_constraints(self, clean_db):
        """Test des contraintes de clés étrangères"""
        db = clean_db
        
        # Essayer d'insérer une progression pour un utilisateur inexistant
        with pytest.raises(Exception):  # Should fail due to foreign key constraint
            db.execute_update(
                """INSERT INTO user_progress (user_id, quest_id) 
                   VALUES (?, ?)""",
                (9999, "nonexistent_quest")
            )
        
        # Créer un utilisateur valide
        user_id = db.create_user("fkuser", "fk@example.com")
        
        # Essayer d'insérer une progression pour une quête inexistante
        with pytest.raises(Exception):  # Should fail due to foreign key constraint
            db.execute_update(
                """INSERT INTO user_progress (user_id, quest_id) 
                   VALUES (?, ?)""",
                (user_id, "nonexistent_quest")
            )
    
    def test_unique_constraints(self, clean_db):
        """Test des contraintes d'unicité"""
        db = clean_db
        
        # Créer un utilisateur
        user_id = db.create_user("uniqueuser", "unique@example.com")
        assert user_id is not None
        
        # Essayer de créer un utilisateur avec le même username
        with pytest.raises(Exception):  # Should fail due to unique constraint
            db.create_user("uniqueuser", "different@example.com")
        
        # Essayer de créer un utilisateur avec le même email
        with pytest.raises(Exception):  # Should fail due to unique constraint
            db.create_user("differentuser", "unique@example.com")
    
    def test_cleanup_operations(self, clean_db):
        """Test des opérations de nettoyage"""
        db = clean_db
        
        # Créer des données anciennes
        user_id = db.create_user("cleanupuser", "cleanup@example.com")
        
        # Créer une session inactive
        session_id = "old_session"
        db.create_chat_session(session_id, user_id, "free_chat")
        
        # Marquer la session comme inactive et ancienne
        db.execute_update(
            """UPDATE chat_sessions 
               SET is_active = 0, last_activity = datetime('now', '-35 days') 
               WHERE id = ?""",
            (session_id,)
        )
        
        # Créer des analytics anciennes
        db.log_event("old_event", user_id, {"old": True})
        db.execute_update(
            """UPDATE analytics 
               SET timestamp = datetime('now', '-65 days') 
               WHERE event_type = ?""",
            ("old_event",)
        )
        
        # Effectuer le nettoyage
        cleanup_stats = db.cleanup_old_data(days=30)
        
        assert "old_sessions" in cleanup_stats
        assert "old_analytics" in cleanup_stats
        assert cleanup_stats["old_sessions"] >= 1
        assert cleanup_stats["old_analytics"] >= 1
    
    @pytest.mark.slow
    def test_concurrent_operations(self, clean_db):
        """Test des opérations concurrentes"""
        import threading
        import time
        
        db = clean_db
        user_id = db.create_user("concurrentuser", "concurrent@example.com")
        
        errors = []
        
        def update_xp(amount):
            try:
                time.sleep(0.01)  # Simuler une latence
                db.update_user_xp(user_id, amount)
            except Exception as e:
                errors.append(e)
        
        # Lancer plusieurs threads qui mettent à jour l'XP simultanément
        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_xp, args=(10,))
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        
        # Vérifier qu'il n'y a pas eu d'erreurs
        assert len(errors) == 0
        
        # Vérifier que l'XP total est correct
        user = db.get_user_by_id(user_id)
        assert user["xp_total"] == 100  # 10 threads * 10 XP