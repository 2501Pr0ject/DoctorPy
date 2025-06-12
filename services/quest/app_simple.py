"""
Application FastAPI simplifiée pour le service Quest
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

def create_app() -> FastAPI:
    """Créer l'application FastAPI pour le service Quest"""
    
    app = FastAPI(
        title="DoctorPy Quest Service",
        description="Service de gamification et gestion des quêtes",
        version="1.0.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routes simples
    @app.get("/")
    async def root():
        return {
            "service": "DoctorPy Quest Service",
            "version": "1.0.0",
            "status": "running",
            "mode": "demo",
            "features": [
                "Quest Management",
                "Gamification System", 
                "Progress Tracking",
                "Achievement System",
                "Leaderboard",
                "User Analytics"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Endpoint de vérification de santé"""
        return {
            "status": "healthy",
            "service": "quest",
            "timestamp": "now"
        }
    
    @app.get("/stats/public")
    async def public_stats():
        """Statistiques publiques (mode démo)"""
        return {
            "total_quests": 15,
            "total_categories": 3,
            "total_users": 42,
            "total_completions": 128,
            "categories_available": ["python_basics", "data_structures", "algorithms"],
            "service_uptime": "running"
        }
    
    @app.get("/api/v1/quests")
    async def list_quests(category: str = None):
        """Lister les quêtes disponibles"""
        quests = [
            {
                "id": "python_variables_101",
                "title": "Variables Python - Les bases",
                "description": "Apprenez à créer et utiliser des variables en Python",
                "difficulty": "beginner",
                "category": "python_basics",
                "estimated_time": "15 min",
                "points": 100,
                "questions_count": 5
            },
            {
                "id": "loops_mastery",
                "title": "Maîtrise des boucles",
                "description": "Maîtrisez les boucles for et while en Python",
                "difficulty": "intermediate", 
                "category": "python_basics",
                "estimated_time": "25 min",
                "points": 200,
                "questions_count": 8
            },
            {
                "id": "functions_expert",
                "title": "Expert en fonctions",
                "description": "Créez des fonctions efficaces et réutilisables",
                "difficulty": "intermediate",
                "category": "python_advanced",
                "estimated_time": "30 min",
                "points": 250,
                "questions_count": 10
            },
            {
                "id": "data_structures_ninja",
                "title": "Ninja des structures de données",
                "description": "Maîtrisez les listes, dictionnaires et sets",
                "difficulty": "advanced",
                "category": "data_structures",
                "estimated_time": "45 min",
                "points": 350,
                "questions_count": 12
            }
        ]
        
        if category:
            quests = [q for q in quests if q["category"] == category]
        
        return {"quests": quests}
    
    @app.post("/api/v1/quests/start")
    async def start_quest(quest_data: dict):
        """Démarrer une quête"""
        quest_id = quest_data.get("quest_id")
        user_id = quest_data.get("user_id", "demo_user")
        
        # Questions selon la quête
        questions = {}
        if quest_id == "python_variables_101":
            questions = {
                "questions": [
                    {
                        "id": "q1",
                        "type": "code_completion",
                        "question": "Créez une variable 'age' avec la valeur 25:",
                        "expected_answer": "age = 25",
                        "hint": "Utilisez le signe = pour assigner une valeur"
                    },
                    {
                        "id": "q2", 
                        "type": "multiple_choice",
                        "question": "Quel est le type de la variable: nom = 'Alice'",
                        "choices": ["int", "str", "float", "bool"],
                        "expected_answer": "str"
                    }
                ]
            }
        elif quest_id == "loops_mastery":
            questions = {
                "questions": [
                    {
                        "id": "q1",
                        "type": "code_completion", 
                        "question": "Écrivez une boucle qui affiche les nombres de 0 à 4:",
                        "expected_answer": "for i in range(5):\n    print(i)",
                        "hint": "Utilisez range() avec for"
                    }
                ]
            }
        else:
            questions = {
                "questions": [
                    {
                        "id": "q1",
                        "type": "general",
                        "question": f"Question générale pour la quête {quest_id}",
                        "expected_answer": "Réponse générale"
                    }
                ]
            }
        
        progress_id = f"progress_{user_id}_{quest_id}_{int(time.time())}"
        
        return {
            "progress_id": progress_id,
            "quest_id": quest_id,
            "user_id": user_id,
            "status": "in_progress",
            "current_question": 0,
            **questions
        }
    
    @app.post("/api/v1/quests/submit")
    async def submit_answer(submission: dict):
        """Soumettre une réponse"""
        progress_id = submission.get("progress_id")
        question_id = submission.get("question_id")
        answer = submission.get("answer", "").strip()
        
        # Évaluation simple de la réponse
        is_correct = False
        feedback = ""
        
        if question_id == "q1" and "age" in answer and "25" in answer:
            is_correct = True
            feedback = "Parfait! Vous avez correctement créé une variable."
        elif question_id == "q2" and answer.lower() == "str":
            is_correct = True
            feedback = "Exact! Les chaînes de caractères sont de type str."
        elif "range" in answer and "for" in answer:
            is_correct = True  
            feedback = "Excellente utilisation de la boucle for avec range()!"
        else:
            feedback = "Pas tout à fait. Essayez encore ou consultez l'aide."
        
        points_earned = 50 if is_correct else 10
        
        return {
            "progress_id": progress_id,
            "question_id": question_id,
            "is_correct": is_correct,
            "points_earned": points_earned,
            "feedback": feedback,
            "next_question": question_id != "q2"  # Simplification
        }
    
    @app.get("/api/v1/quests/leaderboard/global")
    async def global_leaderboard():
        """Leaderboard global"""
        return {
            "leaderboard": [
                {"user": "alice", "score": 1250, "rank": 1, "quests_completed": 5},
                {"user": "bob", "score": 980, "rank": 2, "quests_completed": 4},
                {"user": "charlie", "score": 750, "rank": 3, "quests_completed": 3},
                {"user": "demo_user", "score": 100, "rank": 4, "quests_completed": 1}
            ]
        }
    
    return app

# Point d'entrée pour développement
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8004)