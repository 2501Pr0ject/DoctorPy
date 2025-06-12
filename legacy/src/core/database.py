"""Gestionnaire de base de données SQLite pour DoctorPy"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestionnaire centralisé pour les bases de données SQLite"""
    
    def __init__(self, db_path: str = "./data/databases/doctorpy.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialiser la base de données
        self._init_database()
        
    def _init_database(self):
        """Initialise la base de données avec les tables nécessaires"""
        with self.get_connection() as conn:
            # Table des utilisateurs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    profile_data JSON,
                    xp_total INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    streak_days INTEGER DEFAULT 0
                )
            """)
            
            # Table des quêtes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quests (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    difficulty TEXT NOT NULL CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
                    category TEXT NOT NULL,
                    estimated_time INTEGER DEFAULT 0,
                    prerequisites JSON DEFAULT '[]',
                    learning_objectives JSON DEFAULT '[]',
                    content JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    xp_reward INTEGER DEFAULT 100
                )
            """)
            
            # Table de progression des utilisateurs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    quest_id TEXT NOT NULL,
                    status TEXT DEFAULT 'not_started' CHECK (status IN ('not_started', 'in_progress', 'completed', 'failed')),
                    current_step INTEGER DEFAULT 0,
                    completion_percentage REAL DEFAULT 0.0,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    time_spent INTEGER DEFAULT 0,
                    xp_earned INTEGER DEFAULT 0,
                    attempts INTEGER DEFAULT 0,
                    progress_data JSON DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (quest_id) REFERENCES quests (id),
                    UNIQUE(user_id, quest_id)
                )
            """)
            
            # Table des sessions de chat
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    session_data JSON DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Table des messages
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Table des analytics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    event_type TEXT NOT NULL,
                    event_data JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Index pour les performances
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp)")
            
        logger.info(f"Base de données initialisée: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Gestionnaire de contexte pour les connexions à la base de données"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
        conn.execute("PRAGMA foreign_keys = ON")  # Activer les clés étrangères
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur de base de données: {e}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Exécute une requête SELECT et retourne les résultats"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Exécute une requête INSERT/UPDATE/DELETE et retourne le nombre de lignes affectées"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Exécute une requête INSERT et retourne l'ID de la ligne insérée"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid
    
    # ========== MÉTHODES UTILISATEURS ==========
    
    def create_user(self, username: str, email: str, password_hash: Optional[str] = None, 
                   profile_data: Optional[Dict] = None) -> int:
        """Crée un nouvel utilisateur"""
        query = """
            INSERT INTO users (username, email, password_hash, profile_data)
            VALUES (?, ?, ?, ?)
        """
        params = (username, email, password_hash, json.dumps(profile_data or {}))
        return self.execute_insert(query, params)
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Récupère un utilisateur par son ID"""
        query = "SELECT * FROM users WHERE id = ?"
        results = self.execute_query(query, (user_id,))
        return results[0] if results else None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Récupère un utilisateur par son nom d'utilisateur"""
        query = "SELECT * FROM users WHERE username = ?"
        results = self.execute_query(query, (username,))
        return results[0] if results else None
    
    def update_user_xp(self, user_id: int, xp_to_add: int) -> bool:
        """Met à jour l'XP d'un utilisateur"""
        query = "UPDATE users SET xp_total = xp_total + ? WHERE id = ?"
        return self.execute_update(query, (xp_to_add, user_id)) > 0
    
    def update_user_login(self, user_id: int) -> bool:
        """Met à jour la dernière connexion d'un utilisateur"""
        query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?"
        return self.execute_update(query, (user_id,)) > 0
    
    # ========== MÉTHODES QUÊTES ==========
    
    def get_quest_by_id(self, quest_id: str) -> Optional[Dict[str, Any]]:
        """Récupère une quête par son ID"""
        query = "SELECT * FROM quests WHERE id = ? AND is_active = 1"
        results = self.execute_query(query, (quest_id,))
        return results[0] if results else None
    
    def get_quests_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Récupère les quêtes par difficulté"""
        query = "SELECT * FROM quests WHERE difficulty = ? AND is_active = 1 ORDER BY created_at"
        return self.execute_query(query, (difficulty,))
    
    def get_quests_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Récupère les quêtes par catégorie"""
        query = "SELECT * FROM quests WHERE category = ? AND is_active = 1 ORDER BY created_at"
        return self.execute_query(query, (category,))
    
    def get_all_quests(self) -> List[Dict[str, Any]]:
        """Récupère toutes les quêtes actives"""
        query = "SELECT * FROM quests WHERE is_active = 1 ORDER BY difficulty, created_at"
        return self.execute_query(query)
    
    # ========== MÉTHODES PROGRESSION ==========
    
    def create_user_progress(self, user_id: int, quest_id: str) -> int:
        """Crée un nouvel enregistrement de progression"""
        query = """
            INSERT INTO user_progress (user_id, quest_id, status, started_at)
            VALUES (?, ?, 'in_progress', CURRENT_TIMESTAMP)
        """
        return self.execute_insert(query, (user_id, quest_id))
    
    def update_user_progress(self, user_id: int, quest_id: str, 
                           current_step: int, completion_percentage: float,
                           progress_data: Optional[Dict] = None) -> bool:
        """Met à jour la progression d'un utilisateur"""
        query = """
            UPDATE user_progress 
            SET current_step = ?, completion_percentage = ?, progress_data = ?
            WHERE user_id = ? AND quest_id = ?
        """
        params = (current_step, completion_percentage, 
                 json.dumps(progress_data or {}), user_id, quest_id)
        return self.execute_update(query, params) > 0
    
    def complete_quest(self, user_id: int, quest_id: str, xp_earned: int) -> bool:
        """Marque une quête comme terminée"""
        with self.get_connection() as conn:
            # Mettre à jour la progression
            conn.execute("""
                UPDATE user_progress 
                SET status = 'completed', completion_percentage = 100.0, 
                    completed_at = CURRENT_TIMESTAMP, xp_earned = ?
                WHERE user_id = ? AND quest_id = ?
            """, (xp_earned, user_id, quest_id))
            
            # Mettre à jour l'XP utilisateur
            conn.execute("""
                UPDATE users SET xp_total = xp_total + ? WHERE id = ?
            """, (xp_earned, user_id))
            
        return True
    
    def get_user_progress(self, user_id: int, quest_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère la progression d'un utilisateur"""
        if quest_id:
            query = """
                SELECT up.*, q.title, q.difficulty, q.category 
                FROM user_progress up 
                JOIN quests q ON up.quest_id = q.id 
                WHERE up.user_id = ? AND up.quest_id = ?
            """
            return self.execute_query(query, (user_id, quest_id))
        else:
            query = """
                SELECT up.*, q.title, q.difficulty, q.category 
                FROM user_progress up 
                JOIN quests q ON up.quest_id = q.id 
                WHERE up.user_id = ? 
                ORDER BY up.started_at DESC
            """
            return self.execute_query(query, (user_id,))
    
    # ========== MÉTHODES SESSIONS ==========
    
    def create_chat_session(self, session_id: str, user_id: int, mode: str) -> bool:
        """Crée une nouvelle session de chat"""
        query = """
            INSERT INTO chat_sessions (id, user_id, mode)
            VALUES (?, ?, ?)
        """
        return self.execute_insert(query, (session_id, user_id, mode)) is not None
    
    def add_message(self, session_id: str, user_id: int, role: str, 
                   content: str, metadata: Optional[Dict] = None) -> int:
        """Ajoute un message à une session"""
        query = """
            INSERT INTO messages (session_id, user_id, role, content, metadata)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (session_id, user_id, role, content, json.dumps(metadata or {}))
        return self.execute_insert(query, params)
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère les messages d'une session"""
        query = """
            SELECT * FROM messages 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        return self.execute_query(query, (session_id, limit))
    
    # ========== MÉTHODES ANALYTICS ==========
    
    def log_event(self, event_type: str, user_id: Optional[int] = None, 
                  event_data: Optional[Dict] = None, session_id: Optional[str] = None) -> int:
        """Enregistre un événement analytique"""
        query = """
            INSERT INTO analytics (user_id, event_type, event_data, session_id)
            VALUES (?, ?, ?, ?)
        """
        params = (user_id, event_type, json.dumps(event_data or {}), session_id)
        return self.execute_insert(query, params)
    
    def get_user_analytics(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les analytics d'un utilisateur"""
        query = """
            SELECT * FROM analytics 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        return self.execute_query(query, (user_id, limit))
    
    # ========== MÉTHODES UTILITAIRES ==========
    
    def get_database_stats(self) -> Dict[str, int]:
        """Retourne les statistiques de la base de données"""
        stats = {}
        tables = ['users', 'quests', 'user_progress', 'chat_sessions', 'messages', 'analytics']
        
        for table in tables:
            result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = result[0]['count'] if result else 0
        
        return stats
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Nettoie les anciennes données"""
        cleanup_stats = {}
        
        # Supprimer les anciennes sessions inactives
        query = """
            DELETE FROM chat_sessions 
            WHERE is_active = 0 AND last_activity < datetime('now', '-{} days')
        """.format(days)
        cleanup_stats['old_sessions'] = self.execute_update(query)
        
        # Supprimer les anciens analytics
        query = """
            DELETE FROM analytics 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days * 2)  # Garder les analytics plus longtemps
        cleanup_stats['old_analytics'] = self.execute_update(query)
        
        return cleanup_stats


# Instance globale du gestionnaire de base de données
db_manager = DatabaseManager()


# ========== FONCTIONS UTILITAIRES ==========

def init_database(db_path: Optional[str] = None) -> DatabaseManager:
    """Initialise une nouvelle instance de base de données"""
    if db_path:
        return DatabaseManager(db_path)
    return db_manager


def get_or_create_user(username: str, email: str) -> Dict[str, Any]:
    """Récupère un utilisateur existant ou en crée un nouveau"""
    user = db_manager.get_user_by_username(username)
    
    if not user:
        user_id = db_manager.create_user(username, email)
        user = db_manager.get_user_by_id(user_id)
    
    return user


def backup_database(backup_path: str) -> bool:
    """Crée une sauvegarde de la base de données"""
    try:
        import shutil
        shutil.copy2(db_manager.db_path, backup_path)
        logger.info(f"Sauvegarde créée: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        return False