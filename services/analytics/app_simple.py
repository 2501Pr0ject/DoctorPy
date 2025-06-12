"""
Application FastAPI simplifiée pour le service Analytics
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List

def create_app() -> FastAPI:
    """Créer l'application FastAPI pour le service Analytics"""
    
    app = FastAPI(
        title="DoctorPy Analytics Service",
        description="Service d'analytics et métriques pour DoctorPy",
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
            "service": "DoctorPy Analytics Service",
            "version": "1.0.0",
            "status": "running",
            "mode": "demo",
            "features": [
                "User Activity Tracking",
                "Learning Progress Analytics", 
                "Performance Metrics",
                "Usage Statistics",
                "Real-time Dashboards",
                "Custom Reports"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Endpoint de vérification de santé"""
        return {
            "status": "healthy",
            "service": "analytics",
            "timestamp": datetime.now().isoformat(),
            "metrics_collected": 1542,
            "active_sessions": 23
        }
    
    @app.get("/api/v1/analytics/overview")
    async def get_analytics_overview():
        """Vue d'ensemble des analytics"""
        return {
            "summary": {
                "total_users": 156,
                "active_users_today": 42,
                "total_sessions": 1847,
                "avg_session_duration": "18m 32s",
                "quest_completion_rate": 0.78,
                "most_popular_topics": ["variables", "loops", "functions"]
            },
            "growth": {
                "new_users_this_week": 12,
                "sessions_growth": "+15%",
                "engagement_score": 8.4
            }
        }
    
    @app.get("/api/v1/analytics/users")
    async def get_user_analytics():
        """Analytics des utilisateurs"""
        # Génération de données démo réalistes
        return {
            "user_metrics": {
                "total_registered": 156,
                "active_last_7_days": 89,
                "active_last_30_days": 134,
                "retention_rate": {
                    "day_1": 0.85,
                    "day_7": 0.62,
                    "day_30": 0.41
                }
            },
            "user_segments": [
                {"segment": "beginners", "count": 67, "percentage": 43},
                {"segment": "intermediate", "count": 54, "percentage": 35},
                {"segment": "advanced", "count": 35, "percentage": 22}
            ],
            "user_activity": [
                {"date": "2025-12-01", "active_users": 38},
                {"date": "2025-12-02", "active_users": 45},
                {"date": "2025-12-03", "active_users": 52},
                {"date": "2025-12-04", "active_users": 41},
                {"date": "2025-12-05", "active_users": 48},
                {"date": "2025-12-06", "active_users": 42}
            ]
        }
    
    @app.get("/api/v1/analytics/quests")
    async def get_quest_analytics():
        """Analytics des quêtes"""
        return {
            "quest_performance": [
                {
                    "quest_id": "python_variables_101",
                    "title": "Variables Python - Les bases",
                    "attempts": 234,
                    "completions": 198,
                    "success_rate": 0.846,
                    "avg_time": "12m 45s",
                    "difficulty_rating": 3.2
                },
                {
                    "quest_id": "loops_mastery",
                    "title": "Maîtrise des boucles",
                    "attempts": 189,
                    "completions": 142,
                    "success_rate": 0.751,
                    "avg_time": "18m 12s",
                    "difficulty_rating": 4.1
                },
                {
                    "quest_id": "functions_expert",
                    "title": "Expert en fonctions",
                    "attempts": 156,
                    "completions": 98,
                    "success_rate": 0.628,
                    "avg_time": "25m 33s",
                    "difficulty_rating": 4.8
                }
            ],
            "popular_categories": [
                {"category": "python_basics", "completions": 340},
                {"category": "data_structures", "completions": 156},
                {"category": "python_advanced", "completions": 98}
            ]
        }
    
    @app.get("/api/v1/analytics/rag")
    async def get_rag_analytics():
        """Analytics du service RAG"""
        return {
            "query_metrics": {
                "total_queries": 2847,
                "queries_today": 156,
                "avg_response_time": "0.8s",
                "satisfaction_rate": 0.91
            },
            "popular_queries": [
                {"query_pattern": "variable", "count": 423, "avg_confidence": 0.94},
                {"query_pattern": "loop", "count": 387, "avg_confidence": 0.89},
                {"query_pattern": "function", "count": 356, "avg_confidence": 0.92},
                {"query_pattern": "list", "count": 298, "avg_confidence": 0.87},
                {"query_pattern": "error", "count": 234, "avg_confidence": 0.85}
            ],
            "query_types": [
                {"type": "code_help", "percentage": 45},
                {"type": "concept_explanation", "percentage": 32},
                {"type": "debugging", "percentage": 15},
                {"type": "best_practices", "percentage": 8}
            ]
        }
    
    @app.get("/api/v1/analytics/performance")
    async def get_performance_metrics():
        """Métriques de performance du système"""
        return {
            "system_health": {
                "uptime": "99.8%",
                "response_times": {
                    "auth_service": "120ms",
                    "rag_service": "340ms", 
                    "quest_service": "95ms",
                    "analytics_service": "80ms"
                },
                "error_rates": {
                    "auth_service": "0.1%",
                    "rag_service": "0.3%",
                    "quest_service": "0.2%",
                    "analytics_service": "0.1%"
                }
            },
            "resource_usage": {
                "cpu_avg": "12%",
                "memory_avg": "68%",
                "disk_usage": "45%",
                "network_io": "moderate"
            },
            "cache_metrics": {
                "hit_rate": "87%",
                "miss_rate": "13%",
                "eviction_rate": "2%"
            }
        }
    
    @app.post("/api/v1/analytics/track")
    async def track_event(event_data: dict):
        """Enregistrer un événement d'analytics"""
        event_type = event_data.get("event_type")
        user_id = event_data.get("user_id")
        metadata = event_data.get("metadata", {})
        
        # Simulation de l'enregistrement
        event_id = f"evt_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return {
            "event_id": event_id,
            "status": "recorded",
            "event_type": event_type,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Événement '{event_type}' enregistré avec succès"
        }
    
    @app.get("/api/v1/analytics/reports/learning-progress")
    async def get_learning_progress_report():
        """Rapport de progression d'apprentissage"""
        return {
            "report_id": "learning_progress_2025_12_06",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_learners": 156,
                "concepts_mastered": 1247,
                "avg_progress": 67.8,
                "improvement_rate": "+12% this month"
            },
            "skill_distribution": [
                {"skill": "Variables", "mastery_rate": 0.89, "avg_score": 8.4},
                {"skill": "Loops", "mastery_rate": 0.76, "avg_score": 7.8},
                {"skill": "Functions", "mastery_rate": 0.65, "avg_score": 7.2},
                {"skill": "Data Structures", "mastery_rate": 0.52, "avg_score": 6.8},
                {"skill": "OOP", "mastery_rate": 0.34, "avg_score": 6.1}
            ],
            "learning_paths": [
                {"path": "Python Fundamentals", "completion_rate": 0.78},
                {"path": "Data Science Basics", "completion_rate": 0.45},
                {"path": "Web Development", "completion_rate": 0.23}
            ]
        }
    
    @app.get("/api/v1/analytics/dashboard")
    async def get_dashboard_data():
        """Données pour le dashboard principal"""
        return {
            "real_time": {
                "active_users": 23,
                "ongoing_quests": 15,
                "rag_queries_per_minute": 3.2,
                "system_load": "normal"
            },
            "today_stats": {
                "new_registrations": 7,
                "quests_completed": 34,
                "rag_queries": 156,
                "avg_session_time": "18m 32s"
            },
            "trends": {
                "user_growth": "+8.5%",
                "engagement": "+12.3%",
                "quest_completion": "+5.7%",
                "rag_satisfaction": "+2.1%"
            },
            "alerts": [
                {
                    "type": "info",
                    "message": "Pic d'activité détecté sur les quêtes de variables",
                    "timestamp": "2025-12-06T19:45:00Z"
                }
            ]
        }
    
    return app

# Point d'entrée pour développement
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8003)