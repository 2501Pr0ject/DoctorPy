#!/usr/bin/env python3

"""Script d'initialisation simple de la base de données"""

import sys
import sqlite3
import json
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.database import DatabaseManager


def create_sample_user(db_manager):
    """Crée un utilisateur d'exemple"""
    try:
        # Vérifier si l'utilisateur existe déjà
        existing_user = db_manager.get_user_by_username("demo_user")
        
        if existing_user:
            print("✅ Utilisateur demo_user existe déjà")
            return existing_user['id']
        
        # Créer l'utilisateur
        user_id = db_manager.create_user(
            username="demo_user",
            email="demo@example.com"
        )
        
        print(f"✅ Utilisateur demo_user créé avec l'ID: {user_id}")
        return user_id
        
    except Exception as e:
        print(f"❌ Erreur lors de la création de l'utilisateur: {e}")
        return None


def create_sample_quests(db_manager):
    """Crée des quêtes d'exemple"""
    try:
        sample_quests = [
            {
                'id': 'python_hello_world',
                'title': 'Premier programme Python',
                'description': 'Créer votre premier programme "Hello, World!" en Python',
                'difficulty': 'beginner',
                'category': 'python',
                'estimated_time': 15,
                'prerequisites': '[]',
                'learning_objectives': '["Comprendre la syntaxe de base", "Utiliser la fonction print"]',
                'content': '{"steps": [{"title": "Votre premier Hello World", "content": "Écrivez un programme qui affiche \'Hello, World!\'", "exercise": "print(\'Hello, World!\')", "solution": "print(\'Hello, World!\')", "tips": ["Utilisez la fonction print()", "N\'oubliez pas les guillemets"]}]}'
            },
            {
                'id': 'python_variables',
                'title': 'Les variables en Python',
                'description': 'Apprendre à créer et utiliser des variables',
                'difficulty': 'beginner',
                'category': 'python',
                'estimated_time': 30,
                'prerequisites': '["python_hello_world"]',
                'learning_objectives': '["Créer des variables", "Comprendre les types de base"]',
                'content': '{"steps": [{"title": "Créer une variable", "content": "Les variables stockent des données", "exercise": "Créez une variable \'nom\' avec votre prénom", "solution": "nom = \'Alice\'", "tips": ["Pas d\'espaces dans les noms de variables", "Utilisez des noms descriptifs"]}]}'
            }
        ]
        
        for quest in sample_quests:
            # Vérifier si la quête existe déjà
            existing = db_manager.execute_query(
                "SELECT id FROM quests WHERE id = ?", (quest['id'],)
            )
            
            if existing:
                print(f"✅ Quête {quest['id']} existe déjà")
                continue
            
            # Insérer la quête
            db_manager.execute_update(
                """INSERT INTO quests (
                    id, title, description, difficulty, category, 
                    estimated_time, prerequisites, learning_objectives, content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    quest['id'], quest['title'], quest['description'],
                    quest['difficulty'], quest['category'], quest['estimated_time'],
                    quest['prerequisites'], quest['learning_objectives'], quest['content']
                )
            )
            
            print(f"✅ Quête {quest['id']} créée")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des quêtes: {e}")


def main():
    """Fonction principale"""
    try:
        print("🗄️  Initialisation de la base de données...")
        
        # Créer le gestionnaire de base de données
        db_manager = DatabaseManager()
        print("✅ Base de données initialisée")
        
        # Créer des données d'exemple
        print("👤 Création de l'utilisateur d'exemple...")
        create_sample_user(db_manager)
        
        print("🎯 Création des quêtes d'exemple...")
        create_sample_quests(db_manager)
        
        # Afficher les statistiques
        stats = db_manager.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"📊 Tables créées: {[stat['name'] for stat in stats]}")
        
        user_count = db_manager.execute_query("SELECT COUNT(*) as count FROM users")[0]['count']
        quest_count = db_manager.execute_query("SELECT COUNT(*) as count FROM quests")[0]['count']
        
        print(f"📈 Statistiques:")
        print(f"   - Utilisateurs: {user_count}")
        print(f"   - Quêtes: {quest_count}")
        
        print("🎉 Base de données configurée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors de la configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()