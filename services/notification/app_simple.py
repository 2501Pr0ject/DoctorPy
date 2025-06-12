"""
Application FastAPI simplifiée pour le service Notification
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List

def create_app() -> FastAPI:
    """Créer l'application FastAPI pour le service Notification"""
    
    app = FastAPI(
        title="DoctorPy Notification Service",
        description="Service de notifications multi-canal pour DoctorPy",
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
            "service": "DoctorPy Notification Service",
            "version": "1.0.0",
            "status": "running",
            "mode": "demo",
            "features": [
                "Email Notifications",
                "Push Notifications",
                "In-App Messages",
                "SMS Alerts", 
                "Slack Integration",
                "Webhook Support",
                "Notification Templates",
                "Scheduling & Queuing"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Endpoint de vérification de santé"""
        return {
            "status": "healthy",
            "service": "notification",
            "timestamp": datetime.now().isoformat(),
            "notifications_sent_today": 234,
            "queue_size": 12,
            "channels_active": ["email", "push", "in_app"]
        }
    
    @app.post("/api/v1/notifications/send")
    async def send_notification(notification_data: dict):
        """Envoyer une notification"""
        user_id = notification_data.get("user_id")
        message = notification_data.get("message")
        notification_type = notification_data.get("type", "info")
        channels = notification_data.get("channels", ["in_app"])
        
        notification_id = f"notif_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Simulation d'envoi selon le canal
        results = []
        for channel in channels:
            if channel == "email":
                results.append({
                    "channel": "email",
                    "status": "sent", 
                    "recipient": f"user_{user_id}@doctorpy.com",
                    "sent_at": datetime.now().isoformat()
                })
            elif channel == "push":
                results.append({
                    "channel": "push",
                    "status": "delivered",
                    "device_count": random.randint(1, 3),
                    "sent_at": datetime.now().isoformat()
                })
            elif channel == "in_app":
                results.append({
                    "channel": "in_app",
                    "status": "queued",
                    "will_display": "next_login",
                    "sent_at": datetime.now().isoformat()
                })
            elif channel == "sms":
                results.append({
                    "channel": "sms", 
                    "status": "sent",
                    "phone": f"+33******{random.randint(10, 99)}",
                    "sent_at": datetime.now().isoformat()
                })
        
        return {
            "notification_id": notification_id,
            "status": "processed",
            "message": message,
            "type": notification_type,
            "user_id": user_id,
            "channels_results": results,
            "created_at": datetime.now().isoformat()
        }
    
    @app.get("/api/v1/notifications/user/{user_id}")
    async def get_user_notifications(user_id: str, limit: int = 10):
        """Récupérer les notifications d'un utilisateur"""
        # Génération de notifications démo
        notifications = []
        notification_types = [
            {
                "type": "achievement",
                "message": "🎉 Félicitations ! Vous avez terminé la quête 'Variables Python'",
                "priority": "normal"
            },
            {
                "type": "reminder", 
                "message": "📚 N'oubliez pas de continuer votre apprentissage ! Une nouvelle quête vous attend.",
                "priority": "low"
            },
            {
                "type": "tip",
                "message": "💡 Conseil du jour : Utilisez des noms de variables descriptifs pour un code plus lisible.",
                "priority": "normal"
            },
            {
                "type": "system",
                "message": "🔧 Maintenance programmée demain de 02h00 à 04h00 (temps d'arrêt minimal).",
                "priority": "high"
            },
            {
                "type": "social",
                "message": "👥 Alice a battu votre score sur le leaderboard ! Relevez le défi !",
                "priority": "normal"
            }
        ]
        
        for i in range(min(limit, len(notification_types))):
            notif = notification_types[i]
            notifications.append({
                "id": f"notif_{user_id}_{i}_{int(time.time())}",
                "type": notif["type"],
                "message": notif["message"],
                "priority": notif["priority"],
                "read": random.choice([True, False]),
                "created_at": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
                "channels": ["in_app"]
            })
        
        return {
            "user_id": user_id,
            "notifications": notifications,
            "total_count": len(notifications),
            "unread_count": len([n for n in notifications if not n["read"]])
        }
    
    @app.post("/api/v1/notifications/mark-read")
    async def mark_notifications_read(data: dict):
        """Marquer des notifications comme lues"""
        notification_ids = data.get("notification_ids", [])
        user_id = data.get("user_id")
        
        return {
            "user_id": user_id,
            "marked_read": len(notification_ids),
            "notification_ids": notification_ids,
            "status": "success",
            "updated_at": datetime.now().isoformat()
        }
    
    @app.get("/api/v1/notifications/templates")
    async def get_notification_templates():
        """Récupérer les templates de notifications"""
        return {
            "templates": {
                "quest_completed": {
                    "title": "Quête terminée !",
                    "message": "🎉 Félicitations ! Vous avez terminé la quête '{quest_title}' et gagné {points} points !",
                    "channels": ["in_app", "email"],
                    "priority": "normal"
                },
                "achievement_unlocked": {
                    "title": "Nouveau badge débloqué !",
                    "message": "🏆 Vous avez débloqué le badge '{achievement_name}' ! Continuez comme ça !",
                    "channels": ["in_app", "push"],
                    "priority": "normal"
                },
                "daily_reminder": {
                    "title": "Continuez votre apprentissage",
                    "message": "📚 Bonjour {user_name} ! Il est temps de continuer votre parcours Python.",
                    "channels": ["push", "email"],
                    "priority": "low"
                },
                "system_maintenance": {
                    "title": "Maintenance programmée",
                    "message": "🔧 Maintenance prévue le {date} de {start_time} à {end_time}. Temps d'arrêt minimal.",
                    "channels": ["in_app", "email", "push"],
                    "priority": "high"
                },
                "leaderboard_update": {
                    "title": "Mise à jour du classement",
                    "message": "📊 Votre position dans le leaderboard : #{rank}. {comparison_message}",
                    "channels": ["in_app"],
                    "priority": "low"
                }
            }
        }
    
    @app.post("/api/v1/notifications/broadcast")
    async def broadcast_notification(broadcast_data: dict):
        """Diffuser une notification à plusieurs utilisateurs"""
        message = broadcast_data.get("message")
        user_groups = broadcast_data.get("user_groups", ["all"])
        channels = broadcast_data.get("channels", ["in_app"])
        notification_type = broadcast_data.get("type", "announcement")
        
        # Simulation de diffusion
        broadcast_id = f"broadcast_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Calcul du nombre d'utilisateurs affectés
        affected_users = 0
        if "all" in user_groups:
            affected_users = 156  # Tous les utilisateurs
        elif "active" in user_groups:
            affected_users = 89   # Utilisateurs actifs
        elif "beginners" in user_groups:
            affected_users = 67   # Débutants
        else:
            affected_users = sum([
                {"intermediate": 54, "advanced": 35}.get(group, 10) 
                for group in user_groups
            ])
        
        return {
            "broadcast_id": broadcast_id,
            "status": "scheduled",
            "message": message,
            "type": notification_type,
            "user_groups": user_groups,
            "channels": channels,
            "affected_users": affected_users,
            "estimated_delivery": "2-5 minutes",
            "created_at": datetime.now().isoformat()
        }
    
    @app.get("/api/v1/notifications/stats")
    async def get_notification_stats():
        """Statistiques des notifications"""
        return {
            "delivery_stats": {
                "sent_today": 234,
                "sent_this_week": 1567,
                "sent_this_month": 6234,
                "delivery_rate": 0.978,
                "open_rate": 0.723,
                "click_rate": 0.156
            },
            "channel_performance": [
                {"channel": "in_app", "sent": 1234, "delivered": 1234, "opened": 892},
                {"channel": "email", "sent": 567, "delivered": 551, "opened": 398},
                {"channel": "push", "sent": 456, "delivered": 445, "opened": 289},
                {"channel": "sms", "sent": 123, "delivered": 121, "opened": 95}
            ],
            "notification_types": [
                {"type": "achievement", "count": 345, "engagement": 0.85},
                {"type": "reminder", "count": 234, "engagement": 0.62},
                {"type": "tip", "count": 189, "engagement": 0.71},
                {"type": "system", "count": 78, "engagement": 0.91},
                {"type": "social", "count": 156, "engagement": 0.67}
            ],
            "queue_status": {
                "pending": 12,
                "processing": 3,
                "failed": 2,
                "retry_queue": 1
            }
        }
    
    @app.get("/api/v1/notifications/preferences/{user_id}")
    async def get_user_preferences(user_id: str):
        """Récupérer les préférences de notification d'un utilisateur"""
        return {
            "user_id": user_id,
            "preferences": {
                "channels": {
                    "email": True,
                    "push": True,
                    "in_app": True,
                    "sms": False
                },
                "types": {
                    "achievement": True,
                    "reminder": True,
                    "tip": True,
                    "system": True,
                    "social": True
                },
                "frequency": {
                    "immediate": True,
                    "daily_digest": False,
                    "weekly_summary": True
                },
                "quiet_hours": {
                    "enabled": True,
                    "start": "22:00",
                    "end": "08:00",
                    "timezone": "Europe/Paris"
                }
            },
            "updated_at": datetime.now().isoformat()
        }
    
    @app.put("/api/v1/notifications/preferences/{user_id}")
    async def update_user_preferences(user_id: str, preferences: dict):
        """Mettre à jour les préférences de notification"""
        return {
            "user_id": user_id,
            "status": "updated",
            "preferences": preferences,
            "updated_at": datetime.now().isoformat(),
            "message": "Préférences de notification mises à jour avec succès"
        }
    
    return app

# Point d'entrée pour développement
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8005)