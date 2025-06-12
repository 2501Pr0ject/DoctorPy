#!/usr/bin/env python3
"""
Démonstration interactive des microservices DoctorPy
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

class DoctorPyDemo:
    """Démonstration des services DoctorPy"""
    
    def __init__(self):
        self.base_urls = {
            "auth": "http://localhost:8001",
            "rag": "http://localhost:8002", 
            "quest": "http://localhost:8004"
        }
        self.processes = []
    
    def start_services(self):
        """Démarre tous les services"""
        print("🚀 Démarrage des services DoctorPy...")
        
        scripts = ["run_auth.py", "run_rag.py", "run_quest.py"]
        for script in scripts:
            try:
                process = subprocess.Popen([
                    sys.executable, script
                ], 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
                )
                self.processes.append(process)
                print(f"   ✅ {script} démarré")
                time.sleep(1)
            except Exception as e:
                print(f"   ❌ Erreur {script}: {e}")
        
        print("⏳ Attente de l'initialisation (8 secondes)...")
        time.sleep(8)
    
    def stop_services(self):
        """Arrête tous les services"""
        print("\n🛑 Arrêt des services...")
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        print("✅ Services arrêtés")
    
    def demo_auth_service(self):
        """Démonstration du service Auth"""
        print("\n" + "="*50)
        print("🔐 DÉMONSTRATION SERVICE AUTH")
        print("="*50)
        
        # Test de santé
        response = requests.get(f"{self.base_urls['auth']}/health")
        print(f"Health check: {response.json()}")
        
        # Login démo
        response = requests.post(f"{self.base_urls['auth']}/auth/login", 
                               json={"username": "demo", "password": "demo"})
        print(f"Login: {response.json()}")
        
        # Profil utilisateur
        response = requests.get(f"{self.base_urls['auth']}/users/profile")
        print(f"Profil: {response.json()}")
    
    def demo_rag_service(self):
        """Démonstration du service RAG"""
        print("\n" + "="*50)
        print("🤖 DÉMONSTRATION SERVICE RAG")
        print("="*50)
        
        # Test différentes requêtes
        queries = [
            {"query": "Comment créer une variable en Python ?", "query_type": "code_help"},
            {"query": "Explique-moi les boucles for", "query_type": "tutorial"},
            {"query": "Qu'est-ce qu'une fonction ?", "query_type": "concept"}
        ]
        
        for query in queries:
            print(f"\n📝 Question: {query['query']}")
            response = requests.post(f"{self.base_urls['rag']}/api/v1/rag/query", json=query)
            result = response.json()
            print(f"💡 Réponse: {result['response'][:100]}...")
            print(f"📚 Sources: {', '.join(result['sources'])}")
    
    def demo_quest_service(self):
        """Démonstration du service Quest"""
        print("\n" + "="*50)
        print("🎮 DÉMONSTRATION SERVICE QUEST")
        print("="*50)
        
        # Lister les quêtes
        response = requests.get(f"{self.base_urls['quest']}/api/v1/quests")
        quests = response.json()['quests']
        print(f"📋 Quêtes disponibles: {len(quests)}")
        for quest in quests[:2]:
            print(f"   • {quest['title']} ({quest['difficulty']}) - {quest['points']} points")
        
        # Démarrer une quête
        quest_data = {"quest_id": "python_variables_101", "user_id": "demo_user"}
        response = requests.post(f"{self.base_urls['quest']}/api/v1/quests/start", json=quest_data)
        quest_progress = response.json()
        print(f"\n🎯 Quête démarrée: {quest_progress['quest_id']}")
        print(f"❓ Première question: {quest_progress['questions'][0]['question']}")
        
        # Soumettre une réponse
        submission = {
            "progress_id": quest_progress['progress_id'],
            "question_id": "q1",
            "answer": "age = 25"
        }
        response = requests.post(f"{self.base_urls['quest']}/api/v1/quests/submit", json=submission)
        result = response.json()
        print(f"✅ Réponse: {'Correcte' if result['is_correct'] else 'Incorrecte'}")
        print(f"💰 Points gagnés: {result['points_earned']}")
        print(f"📝 Feedback: {result['feedback']}")
        
        # Leaderboard
        response = requests.get(f"{self.base_urls['quest']}/api/v1/quests/leaderboard/global")
        leaderboard = response.json()['leaderboard']
        print(f"\n🏆 Leaderboard:")
        for player in leaderboard[:3]:
            print(f"   {player['rank']}. {player['user']} - {player['score']} points ({player['quests_completed']} quêtes)")
    
    def demo_integration(self):
        """Démonstration de l'intégration entre services"""
        print("\n" + "="*50)
        print("🔗 DÉMONSTRATION INTÉGRATION")
        print("="*50)
        
        # Scénario complet : utilisateur qui se connecte, pose une question RAG, puis fait une quête
        print("📖 Scénario: Alice apprend les variables Python")
        
        # 1. Connexion
        auth_response = requests.post(f"{self.base_urls['auth']}/auth/login", 
                                    json={"username": "alice", "password": "demo"})
        print(f"1. ✅ Alice se connecte: {auth_response.json()['message']}")
        
        # 2. Question RAG
        rag_query = {"query": "Comment déclarer une variable en Python?", "query_type": "code_help"}
        rag_response = requests.post(f"{self.base_urls['rag']}/api/v1/rag/query", json=rag_query)
        print(f"2. 🤖 Alice consulte l'aide IA: '{rag_query['query']}'")
        print(f"   💡 Réponse obtenue avec {rag_response.json()['confidence']*100:.0f}% de confiance")
        
        # 3. Quête
        quest_start = {"quest_id": "python_variables_101", "user_id": "alice"}
        quest_response = requests.post(f"{self.base_urls['quest']}/api/v1/quests/start", json=quest_start)
        print(f"3. 🎯 Alice démarre la quête 'Variables Python'")
        
        # 4. Stats
        stats_response = requests.get(f"{self.base_urls['quest']}/stats/public")
        stats = stats_response.json()
        print(f"4. 📊 Système: {stats['total_users']} utilisateurs, {stats['total_completions']} complétions")
        
        print("\n🎉 Intégration réussie! Tous les services communiquent parfaitement.")
    
    def run_demo(self):
        """Lance la démonstration complète"""
        print("🌟 DÉMONSTRATION DOCTORPY MICROSERVICES")
        print("🤖 Assistant IA éducatif pour l'apprentissage Python")
        print("=" * 60)
        
        try:
            self.start_services()
            
            self.demo_auth_service()
            self.demo_rag_service()
            self.demo_quest_service()
            self.demo_integration()
            
            print("\n" + "="*60)
            print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
            print("✅ Tous les microservices sont opérationnels")
            print("🚀 Architecture prête pour développement et déploiement")
            
        except requests.exceptions.ConnectionError:
            print("❌ Erreur: Impossible de se connecter aux services")
            print("   Vérifiez que tous les services sont démarrés")
        except Exception as e:
            print(f"❌ Erreur durant la démonstration: {e}")
        finally:
            self.stop_services()

if __name__ == "__main__":
    demo = DoctorPyDemo()
    demo.run_demo()