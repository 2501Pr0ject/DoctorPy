"""
Tâches Prefect pour les notifications et alertes
"""

import sys
import smtplib
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from prefect import task, get_run_logger

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@task(
    name="send_notification",
    description="Envoyer une notification de succès",
    retries=1,
    tags=["notification", "success"]
)
async def send_notification(
    title: str,
    message: str,
    channels: List[str] = ["log"],
    priority: str = "normal",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Envoyer une notification de succès ou d'information
    
    Args:
        title: Titre de la notification
        message: Contenu du message
        channels: Canaux de notification ["log", "email", "slack", "webhook"]
        priority: Priorité (low, normal, high)
        metadata: Métadonnées additionnelles
        
    Returns:
        Dict avec le statut d'envoi
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"📢 Envoi de notification: {title}")
        
        notification_data = {
            "title": title,
            "message": message,
            "type": "notification",
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        results = {}
        
        # Canal LOG (toujours actif)
        if "log" in channels:
            logger.info(f"📝 {title}")
            logger.info(f"   {message}")
            results["log"] = {"status": "sent", "method": "logger"}
        
        # Canal EMAIL
        if "email" in channels:
            email_result = await send_email_notification(notification_data)
            results["email"] = email_result
        
        # Canal SLACK
        if "slack" in channels:
            slack_result = await send_slack_notification(notification_data)
            results["slack"] = slack_result
        
        # Canal WEBHOOK
        if "webhook" in channels:
            webhook_result = await send_webhook_notification(notification_data)
            results["webhook"] = webhook_result
        
        # Canal FILE (sauvegarde locale)
        if "file" in channels:
            file_result = await save_notification_to_file(notification_data)
            results["file"] = file_result
        
        # Compter les succès
        successful_channels = sum(1 for result in results.values() if result.get("status") == "sent")
        total_channels = len(channels)
        
        return {
            "status": "success",
            "title": title,
            "channels_attempted": total_channels,
            "channels_successful": successful_channels,
            "results": results,
            "sent_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'envoi de notification: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "title": title
        }


@task(
    name="send_alert",
    description="Envoyer une alerte d'erreur",
    retries=2,
    retry_delay_seconds=[30, 120],
    tags=["alert", "error", "critical"]
)
async def send_alert(
    title: str,
    message: str,
    channels: List[str] = ["log", "email"],
    severity: str = "high",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Envoyer une alerte critique ou d'erreur
    
    Args:
        title: Titre de l'alerte
        message: Contenu du message d'erreur
        channels: Canaux d'alerte ["log", "email", "slack", "webhook"]
        severity: Sévérité (low, medium, high, critical)
        metadata: Métadonnées additionnelles (stack trace, etc.)
        
    Returns:
        Dict avec le statut d'envoi
    """
    logger = get_run_logger()
    
    try:
        logger.error(f"🚨 Envoi d'alerte: {title}")
        
        alert_data = {
            "title": f"🚨 ALERTE: {title}",
            "message": message,
            "type": "alert",
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        results = {}
        
        # Canal LOG (toujours actif avec niveau ERROR)
        if "log" in channels:
            logger.error(f"🚨 {title}")
            logger.error(f"   {message}")
            if metadata:
                logger.error(f"   Métadonnées: {json.dumps(metadata, indent=2)}")
            results["log"] = {"status": "sent", "method": "logger"}
        
        # Canal EMAIL (avec priorité haute)
        if "email" in channels:
            email_result = await send_email_notification(alert_data, is_alert=True)
            results["email"] = email_result
        
        # Canal SLACK (avec formatage d'alerte)
        if "slack" in channels:
            slack_result = await send_slack_notification(alert_data, is_alert=True)
            results["slack"] = slack_result
        
        # Canal WEBHOOK (avec payload alerte)
        if "webhook" in channels:
            webhook_result = await send_webhook_notification(alert_data, is_alert=True)
            results["webhook"] = webhook_result
        
        # Sauvegarde obligatoire pour les alertes
        file_result = await save_notification_to_file(alert_data, is_alert=True)
        results["file"] = file_result
        
        # Compter les succès
        successful_channels = sum(1 for result in results.values() if result.get("status") == "sent")
        total_channels = len(channels) + 1  # +1 pour le file obligatoire
        
        return {
            "status": "success",
            "title": title,
            "severity": severity,
            "channels_attempted": total_channels,
            "channels_successful": successful_channels,
            "results": results,
            "sent_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur critique lors de l'envoi d'alerte: {str(e)}")
        # En cas d'erreur d'alerte, essayer au moins de sauvegarder localement
        try:
            await save_notification_to_file({
                "title": f"ERREUR ALERTE: {title}",
                "message": f"Erreur d'envoi: {str(e)}\nMessage original: {message}",
                "type": "alert_error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }, is_alert=True)
        except:
            pass
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "title": title,
            "severity": severity
        }


async def send_email_notification(
    notification_data: Dict[str, Any],
    is_alert: bool = False
) -> Dict[str, Any]:
    """Envoyer notification par email"""
    try:
        # Configuration email depuis variables d'environnement
        import os
        
        smtp_host = os.getenv("SMTP_HOST", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        from_email = os.getenv("FROM_EMAIL", "doctorpy@localhost")
        to_emails = os.getenv("ALERT_EMAILS", "admin@localhost").split(",")
        
        if not smtp_user or not to_emails[0]:
            return {"status": "skipped", "reason": "email_not_configured"}
        
        # Créer le message
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = notification_data["title"]
        
        # Corps du message
        body = f"""
{notification_data['message']}

Timestamp: {notification_data['timestamp']}
Type: {notification_data['type']}
"""
        
        if is_alert:
            body += f"Sévérité: {notification_data.get('severity', 'unknown')}\n"
        
        if notification_data.get("metadata"):
            body += f"\nMétadonnées:\n{json.dumps(notification_data['metadata'], indent=2)}"
        
        msg.attach(MIMEText(body, "plain"))
        
        # Envoyer l'email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            
            server.send_message(msg)
        
        return {
            "status": "sent",
            "method": "smtp",
            "recipients": len(to_emails),
            "smtp_host": smtp_host
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "method": "smtp"
        }


async def send_slack_notification(
    notification_data: Dict[str, Any],
    is_alert: bool = False
) -> Dict[str, Any]:
    """Envoyer notification vers Slack"""
    try:
        import os
        
        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        if not webhook_url:
            return {"status": "skipped", "reason": "slack_not_configured"}
        
        # Formatage Slack
        color = "#ff0000" if is_alert else "#36a64f"
        icon = "🚨" if is_alert else "ℹ️"
        
        payload = {
            "text": f"{icon} {notification_data['title']}",
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {
                            "title": "Message",
                            "value": notification_data["message"],
                            "short": False
                        },
                        {
                            "title": "Timestamp",
                            "value": notification_data["timestamp"],
                            "short": True
                        },
                        {
                            "title": "Type",
                            "value": notification_data["type"],
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        if is_alert and "severity" in notification_data:
            payload["attachments"][0]["fields"].append({
                "title": "Sévérité",
                "value": notification_data["severity"],
                "short": True
            })
        
        # Envoyer à Slack
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        return {
            "status": "sent",
            "method": "slack_webhook",
            "response_status": response.status_code
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "method": "slack_webhook"
        }


async def send_webhook_notification(
    notification_data: Dict[str, Any],
    is_alert: bool = False
) -> Dict[str, Any]:
    """Envoyer notification vers webhook personnalisé"""
    try:
        import os
        
        webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL", "")
        if not webhook_url:
            return {"status": "skipped", "reason": "webhook_not_configured"}
        
        # Payload pour webhook
        payload = {
            "event": "doctorpy_notification",
            "data": notification_data,
            "is_alert": is_alert,
            "source": "doctorpy_prefect"
        }
        
        # Headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DoctorPy-Prefect/1.0"
        }
        
        # Token d'authentification si disponible
        webhook_token = os.getenv("WEBHOOK_TOKEN", "")
        if webhook_token:
            headers["Authorization"] = f"Bearer {webhook_token}"
        
        # Envoyer la requête
        response = requests.post(
            webhook_url,
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        return {
            "status": "sent",
            "method": "webhook",
            "response_status": response.status_code,
            "webhook_url": webhook_url
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "method": "webhook"
        }


async def save_notification_to_file(
    notification_data: Dict[str, Any],
    is_alert: bool = False
) -> Dict[str, Any]:
    """Sauvegarder notification dans un fichier local"""
    try:
        # Répertoire de sauvegarde
        notifications_dir = Path("data/notifications")
        notifications_dir.mkdir(parents=True, exist_ok=True)
        
        # Nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        file_type = "alerts" if is_alert else "notifications"
        filename = f"{file_type}_{timestamp}.jsonl"
        
        file_path = notifications_dir / filename
        
        # Ajouter au fichier JSONL (une ligne par notification)
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(notification_data, f, ensure_ascii=False)
            f.write("\n")
        
        return {
            "status": "sent",
            "method": "file",
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "method": "file"
        }


@task(
    name="send_report_notification",
    description="Envoyer notification avec rapport",
    retries=1,
    tags=["notification", "report"]
)
async def send_report_notification(
    report_data: Dict[str, Any],
    channels: List[str] = ["log", "email"]
) -> Dict[str, Any]:
    """
    Envoyer une notification avec un rapport de monitoring
    
    Args:
        report_data: Données du rapport généré
        channels: Canaux de notification
        
    Returns:
        Dict avec le statut d'envoi
    """
    logger = get_run_logger()
    
    try:
        if report_data.get("status") != "success":
            return await send_alert(
                title="Échec génération rapport",
                message=f"Erreur lors de la génération du rapport: {report_data.get('error_message', 'Inconnue')}",
                channels=channels
            )
        
        report = report_data.get("report", {})
        
        # Formatage du message
        title = f"Rapport {report.get('report_type', 'monitoring')} - DoctorPy"
        
        message = f"""
📊 Rapport {report.get('report_type', 'monitoring')} généré avec succès

🎯 Score global: {report.get('overall_score', 0)}/100
🏥 Santé système: {report.get('summary', {}).get('overall_health', 'inconnue').upper()}

📈 Activité 24h:
• Utilisateurs actifs: {report.get('application', {}).get('active_users_24h', 0)}
• Nouveaux utilisateurs: {report.get('application', {}).get('new_users_24h', 0)}
• Messages envoyés: {report.get('application', {}).get('messages_24h', 0)}
• Quêtes complétées: {report.get('application', {}).get('quests_completed_24h', 0)}
• Erreurs: {report.get('application', {}).get('errors_24h', 0)}

💻 Système:
• CPU: {report.get('system', {}).get('cpu_usage', 0):.1f}%
• Mémoire: {report.get('system', {}).get('memory_usage', 0):.1f}%
• Disque: {report.get('system', {}).get('disk_usage', 0):.1f}%

💾 Stockage:
• Base de données: {report.get('data_storage', {}).get('database_size_mb', 0):.1f} MB
• Vector store: {report.get('data_storage', {}).get('vector_store_size_mb', 0):.1f} MB
• Total: {report.get('data_storage', {}).get('total_data_size_mb', 0):.1f} MB
"""
        
        # Ajouter les problèmes s'il y en a
        issues = report.get('issues', [])
        if issues:
            message += f"\n⚠️ Problèmes détectés ({len(issues)}):\n"
            for issue in issues[:5]:  # Limiter à 5 problèmes
                message += f"• {issue}\n"
            if len(issues) > 5:
                message += f"• ... et {len(issues) - 5} autre(s)\n"
        
        # Métadonnées pour les canaux avancés
        metadata = {
            "report_file": report_data.get("report_file", ""),
            "report_type": report.get("report_type", ""),
            "overall_score": report.get("overall_score", 0),
            "health_status": report.get("summary", {}).get("overall_health", "unknown"),
            "issues_count": len(issues)
        }
        
        return await send_notification(
            title=title,
            message=message,
            channels=channels,
            priority="normal",
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'envoi du rapport: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }


@task(
    name="send_pipeline_notification",
    description="Envoyer notification de fin de pipeline",
    retries=1,
    tags=["notification", "pipeline"]
)
async def send_pipeline_notification(
    pipeline_result: Dict[str, Any],
    channels: List[str] = ["log"]
) -> Dict[str, Any]:
    """
    Envoyer une notification de fin de pipeline RAG
    
    Args:
        pipeline_result: Résultat du pipeline RAG
        channels: Canaux de notification
        
    Returns:
        Dict avec le statut d'envoi
    """
    logger = get_run_logger()
    
    try:
        pipeline_status = pipeline_result.get("status", "unknown")
        
        if pipeline_status == "success":
            # Notification de succès
            summary = pipeline_result.get("summary", {})
            total_time = pipeline_result.get("total_time_seconds", 0)
            
            title = "Pipeline RAG terminé avec succès"
            message = f"""
✅ Pipeline de mise à jour de la base de connaissances terminé

⏱️ Durée: {total_time:.1f} secondes
📄 Documents scrapés: {summary.get('documents_scraped', 0)}
🧩 Chunks créés: {summary.get('chunks_created', 0)}
🧠 Embeddings générés: {summary.get('embeddings_generated', 0)}
🗂️ Documents indexés: {summary.get('documents_indexed', 0)}
📊 Collection: {summary.get('collection_name', 'doctorpy_docs')}
"""
            
            return await send_notification(
                title=title,
                message=message,
                channels=channels,
                priority="normal",
                metadata=pipeline_result
            )
            
        else:
            # Alerte d'échec
            error_stage = pipeline_result.get("error_stage", "unknown")
            error_message = pipeline_result.get("error_message", "Erreur inconnue")
            total_time = pipeline_result.get("total_time_seconds", 0)
            
            title = f"Échec du pipeline RAG à l'étape: {error_stage}"
            message = f"""
❌ Pipeline de mise à jour de la base de connaissances échoué

🚨 Étape d'échec: {error_stage}
❌ Erreur: {error_message}
⏱️ Temps écoulé: {total_time:.1f} secondes

Les étapes suivantes n'ont pas pu être exécutées.
Intervention manuelle requise.
"""
            
            return await send_alert(
                title=title,
                message=message,
                channels=channels,
                severity="high",
                metadata=pipeline_result
            )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'envoi de notification pipeline: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }