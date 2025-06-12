"""
Workflows Prefect pour la maintenance automatisée du système
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
from prefect import flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .tasks.maintenance import (
    cleanup_sessions,
    cleanup_logs, 
    backup_database,
    optimize_vector_store,
    cleanup_analytics
)
from .tasks.notification import send_notification, send_alert
from .analytics import health_check_flow


@flow(
    name="daily_maintenance",
    description="Maintenance quotidienne automatisée du système",
    version="1.0",
    task_runner=ConcurrentTaskRunner(max_workers=3),
    timeout_seconds=3600,  # 1 heure max
    retries=1
)
async def daily_maintenance(
    notification_channels: List[str] = ["log"],
    skip_backup: bool = False,
    skip_cleanup: bool = False
) -> Dict[str, Any]:
    """
    Workflow de maintenance quotidienne automatisée
    
    Args:
        notification_channels: Canaux de notification
        skip_backup: Ignorer la sauvegarde (pour les tests)
        skip_cleanup: Ignorer le nettoyage (pour les tests)
        
    Returns:
        Dict avec le résumé de la maintenance
    """
    logger = get_run_logger()
    
    logger.info("🧹 Démarrage de la maintenance quotidienne")
    maintenance_start = datetime.now()
    
    try:
        maintenance_results = {
            "started_at": maintenance_start.isoformat(),
            "type": "daily",
            "tasks": {}
        }
        
        # ===== VÉRIFICATION DE SANTÉ PRÉLIMINAIRE =====
        logger.info("🩺 Vérification de santé préliminaire")
        
        health_check = await health_check_flow(
            alert_on_degraded=False,  # Pas d'alerte, juste vérification
            notification_channels=["log"],
            detailed_analysis=False
        )
        
        maintenance_results["tasks"]["health_check"] = health_check
        
        # Si le système est en panne, reporter la maintenance
        if health_check.get("health_check", {}).get("overall_health") == "unhealthy":
            await send_alert(
                title="⚠️ Maintenance quotidienne reportée",
                message="Système en panne détecté. Maintenance automatique reportée pour éviter d'aggraver les problèmes.",
                channels=notification_channels
            )
            
            maintenance_results["status"] = "postponed"
            maintenance_results["reason"] = "system_unhealthy"
            return maintenance_results
        
        # ===== NETTOYAGE DES SESSIONS =====
        if not skip_cleanup:
            logger.info("🗑️ Nettoyage des sessions inactives")
            
            sessions_cleanup = cleanup_sessions(
                inactive_days=7,
                keep_recent_hours=24
            )
            maintenance_results["tasks"]["sessions_cleanup"] = sessions_cleanup
            
            if sessions_cleanup.get("status") == "error":
                logger.warning(f"⚠️ Erreur nettoyage sessions: {sessions_cleanup.get('error_message')}")
        
        # ===== NETTOYAGE DES LOGS =====
        if not skip_cleanup:
            logger.info("📜 Nettoyage et archivage des logs")
            
            logs_cleanup = cleanup_logs(
                keep_days=30,
                archive=True,
                max_size_mb=100
            )
            maintenance_results["tasks"]["logs_cleanup"] = logs_cleanup
            
            if logs_cleanup.get("status") == "error":
                logger.warning(f"⚠️ Erreur nettoyage logs: {logs_cleanup.get('error_message')}")
        
        # ===== SAUVEGARDE DE BASE DE DONNÉES =====
        if not skip_backup:
            logger.info("💾 Sauvegarde de la base de données")
            
            backup_result = backup_database(
                backup_dir="data/backups",
                keep_backups=7,
                compress=True
            )
            maintenance_results["tasks"]["database_backup"] = backup_result
            
            if backup_result.get("status") == "error":
                logger.error(f"❌ Erreur sauvegarde: {backup_result.get('error_message')}")
                await send_alert(
                    title="❌ Échec sauvegarde quotidienne",
                    message=f"La sauvegarde quotidienne a échoué: {backup_result.get('error_message')}",
                    channels=notification_channels
                )
        
        # ===== OPTIMISATION VECTOR STORE =====
        logger.info("⚡ Optimisation du vector store")
        
        vector_optimization = optimize_vector_store(
            collection_name="doctorpy_docs",
            vacuum=True
        )
        maintenance_results["tasks"]["vector_optimization"] = vector_optimization
        
        # ===== NETTOYAGE ANALYTICS =====
        if not skip_cleanup:
            logger.info("📊 Nettoyage des anciennes analytics")
            
            analytics_cleanup = cleanup_analytics(
                keep_days=90,
                archive_old_data=True
            )
            maintenance_results["tasks"]["analytics_cleanup"] = analytics_cleanup
        
        # ===== FINALISATION =====
        maintenance_end = datetime.now()
        maintenance_duration = (maintenance_end - maintenance_start).total_seconds()
        
        # Calculer les statistiques de maintenance
        successful_tasks = sum(1 for task in maintenance_results["tasks"].values() 
                             if isinstance(task, dict) and task.get("status") == "success")
        total_tasks = len(maintenance_results["tasks"])
        
        # Calculer l'espace libéré total
        total_space_freed = 0
        for task_result in maintenance_results["tasks"].values():
            if isinstance(task_result, dict):
                total_space_freed += task_result.get("space_freed_mb", 0)
        
        maintenance_results.update({
            "status": "success",
            "completed_at": maintenance_end.isoformat(),
            "duration_seconds": round(maintenance_duration, 1),
            "summary": {
                "successful_tasks": successful_tasks,
                "total_tasks": total_tasks,
                "success_rate": round(successful_tasks / total_tasks * 100, 1) if total_tasks > 0 else 0,
                "total_space_freed_mb": round(total_space_freed, 1)
            }
        })
        
        # Notification de succès
        success_message = f"""✅ Maintenance quotidienne terminée en {maintenance_duration:.1f}s

📊 Résumé:
• Tâches réussies: {successful_tasks}/{total_tasks} ({maintenance_results['summary']['success_rate']:.1f}%)
• Espace libéré: {total_space_freed:.1f} MB
• Sessions nettoyées: {maintenance_results['tasks'].get('sessions_cleanup', {}).get('sessions_cleaned', 0)}
• Logs archivés: {maintenance_results['tasks'].get('logs_cleanup', {}).get('logs_archived', 0)}
• Sauvegarde: {'✅' if maintenance_results['tasks'].get('database_backup', {}).get('status') == 'success' else '❌'}"""
        
        await send_notification(
            title="✅ Maintenance quotidienne terminée",
            message=success_message,
            channels=notification_channels,
            priority="normal"
        )
        
        logger.info(f"✅ Maintenance quotidienne terminée avec succès")
        logger.info(f"   ⏱️ Durée: {maintenance_duration:.1f}s")
        logger.info(f"   📊 Tâches: {successful_tasks}/{total_tasks}")
        logger.info(f"   💾 Espace libéré: {total_space_freed:.1f} MB")
        
        return maintenance_results
        
    except Exception as e:
        maintenance_end = datetime.now()
        maintenance_duration = (maintenance_end - maintenance_start).total_seconds()
        
        logger.error(f"❌ Erreur lors de la maintenance quotidienne: {str(e)}")
        
        await send_alert(
            title="🚨 Échec maintenance quotidienne",
            message=f"Erreur critique lors de la maintenance: {str(e)}\nDurée avant échec: {maintenance_duration:.1f}s",
            channels=notification_channels,
            severity="high"
        )
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "started_at": maintenance_start.isoformat(),
            "failed_at": maintenance_end.isoformat(),
            "duration_seconds": round(maintenance_duration, 1),
            "tasks": maintenance_results.get("tasks", {}) if 'maintenance_results' in locals() else {}
        }


@flow(
    name="weekly_maintenance",
    description="Maintenance hebdomadaire approfondie du système",
    version="1.0",
    task_runner=ConcurrentTaskRunner(max_workers=2),
    timeout_seconds=7200,  # 2 heures max
    retries=1
)
async def weekly_maintenance(
    notification_channels: List[str] = ["log", "email"],
    include_deep_cleanup: bool = True,
    include_optimization: bool = True
) -> Dict[str, Any]:
    """
    Workflow de maintenance hebdomadaire approfondie
    
    Args:
        notification_channels: Canaux de notification
        include_deep_cleanup: Inclure un nettoyage approfondi
        include_optimization: Inclure les optimisations avancées
        
    Returns:
        Dict avec le résumé de la maintenance
    """
    logger = get_run_logger()
    
    logger.info("🔧 Démarrage de la maintenance hebdomadaire")
    maintenance_start = datetime.now()
    
    try:
        maintenance_results = {
            "started_at": maintenance_start.isoformat(),
            "type": "weekly",
            "tasks": {}
        }
        
        # ===== MAINTENANCE QUOTIDIENNE COMPLÈTE =====
        logger.info("📅 Exécution de la maintenance quotidienne complète")
        
        daily_result = await daily_maintenance(
            notification_channels=["log"],  # Pas de notification pour la partie quotidienne
            skip_backup=False,
            skip_cleanup=False
        )
        maintenance_results["tasks"]["daily_maintenance"] = daily_result
        
        # ===== NETTOYAGE APPROFONDI =====
        if include_deep_cleanup:
            logger.info("🧹 Nettoyage approfondi du système")
            
            # Nettoyage des sessions avec critères plus stricts
            deep_sessions_cleanup = cleanup_sessions(
                inactive_days=3,  # Plus strict pour le nettoyage hebdomadaire
                keep_recent_hours=12
            )
            maintenance_results["tasks"]["deep_sessions_cleanup"] = deep_sessions_cleanup
            
            # Nettoyage des logs avec archivage complet
            deep_logs_cleanup = cleanup_logs(
                keep_days=14,  # Garder moins de logs
                archive=True,
                max_size_mb=50  # Seuil plus bas
            )
            maintenance_results["tasks"]["deep_logs_cleanup"] = deep_logs_cleanup
            
            # Nettoyage analytics plus agressif
            deep_analytics_cleanup = cleanup_analytics(
                keep_days=60,  # Garder moins d'analytics
                archive_old_data=True
            )
            maintenance_results["tasks"]["deep_analytics_cleanup"] = deep_analytics_cleanup
        
        # ===== OPTIMISATIONS AVANCÉES =====
        if include_optimization:
            logger.info("⚡ Optimisations avancées du système")
            
            # Optimisation du vector store avec vacuum complet
            advanced_vector_optimization = optimize_vector_store(
                collection_name="doctorpy_docs",
                vacuum=True
            )
            maintenance_results["tasks"]["advanced_vector_optimization"] = advanced_vector_optimization
            
            # Optimisations supplémentaires
            additional_optimizations = await perform_additional_optimizations()
            maintenance_results["tasks"]["additional_optimizations"] = additional_optimizations
        
        # ===== SAUVEGARDE DE SÉCURITÉ =====
        logger.info("🛡️ Sauvegarde de sécurité hebdomadaire")
        
        security_backup = backup_database(
            backup_dir="data/backups/weekly",
            keep_backups=4,  # Garder 4 sauvegardes hebdomadaires
            compress=True
        )
        maintenance_results["tasks"]["security_backup"] = security_backup
        
        # ===== VÉRIFICATION DE SANTÉ FINALE =====
        logger.info("🩺 Vérification de santé post-maintenance")
        
        final_health_check = await health_check_flow(
            alert_on_degraded=True,
            notification_channels=notification_channels,
            detailed_analysis=True
        )
        maintenance_results["tasks"]["final_health_check"] = final_health_check
        
        # ===== FINALISATION =====
        maintenance_end = datetime.now()
        maintenance_duration = (maintenance_end - maintenance_start).total_seconds()
        
        # Calculer les statistiques
        successful_tasks = sum(1 for task in maintenance_results["tasks"].values() 
                             if isinstance(task, dict) and task.get("status") in ["success", "completed"])
        total_tasks = len(maintenance_results["tasks"])
        
        # Calculer l'espace total libéré
        total_space_freed = 0
        for task_result in maintenance_results["tasks"].values():
            if isinstance(task_result, dict):
                total_space_freed += task_result.get("space_freed_mb", 0)
                if "tasks" in task_result:  # Pour daily_maintenance
                    for subtask in task_result["tasks"].values():
                        if isinstance(subtask, dict):
                            total_space_freed += subtask.get("space_freed_mb", 0)
        
        maintenance_results.update({
            "status": "success",
            "completed_at": maintenance_end.isoformat(),
            "duration_seconds": round(maintenance_duration, 1),
            "summary": {
                "successful_tasks": successful_tasks,
                "total_tasks": total_tasks,
                "success_rate": round(successful_tasks / total_tasks * 100, 1) if total_tasks > 0 else 0,
                "total_space_freed_mb": round(total_space_freed, 1),
                "deep_cleanup_performed": include_deep_cleanup,
                "optimizations_performed": include_optimization
            }
        })
        
        # Rapport détaillé
        final_health = final_health_check.get("health_check", {}).get("overall_health", "unknown")
        
        success_message = f"""✅ Maintenance hebdomadaire terminée en {maintenance_duration/60:.1f} minutes

📊 Résumé complet:
• Tâches réussies: {successful_tasks}/{total_tasks} ({maintenance_results['summary']['success_rate']:.1f}%)
• Espace total libéré: {total_space_freed:.1f} MB
• Nettoyage approfondi: {'✅' if include_deep_cleanup else '⏭️ Ignoré'}
• Optimisations: {'✅' if include_optimization else '⏭️ Ignoré'}
• Santé finale du système: {final_health.upper()}

🛡️ Sauvegarde de sécurité: {'✅' if security_backup.get('status') == 'success' else '❌'}"""
        
        await send_notification(
            title="✅ Maintenance hebdomadaire terminée",
            message=success_message,
            channels=notification_channels,
            priority="normal"
        )
        
        logger.info(f"✅ Maintenance hebdomadaire terminée avec succès")
        logger.info(f"   ⏱️ Durée: {maintenance_duration/60:.1f} minutes")
        logger.info(f"   📊 Tâches: {successful_tasks}/{total_tasks}")
        logger.info(f"   💾 Espace libéré: {total_space_freed:.1f} MB")
        logger.info(f"   🩺 Santé finale: {final_health}")
        
        return maintenance_results
        
    except Exception as e:
        maintenance_end = datetime.now()
        maintenance_duration = (maintenance_end - maintenance_start).total_seconds()
        
        logger.error(f"❌ Erreur lors de la maintenance hebdomadaire: {str(e)}")
        
        await send_alert(
            title="🚨 Échec maintenance hebdomadaire",
            message=f"Erreur critique lors de la maintenance: {str(e)}\nDurée avant échec: {maintenance_duration/60:.1f} minutes",
            channels=notification_channels,
            severity="high"
        )
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "started_at": maintenance_start.isoformat(),
            "failed_at": maintenance_end.isoformat(),
            "duration_seconds": round(maintenance_duration, 1),
            "tasks": maintenance_results.get("tasks", {}) if 'maintenance_results' in locals() else {}
        }


@flow(
    name="emergency_maintenance",
    description="Maintenance d'urgence pour résoudre les problèmes critiques",
    version="1.0",
    timeout_seconds=1800,  # 30 minutes max
    retries=0  # Pas de retry pour la maintenance d'urgence
)
async def emergency_maintenance(
    issue_description: str = "Problème critique détecté",
    notification_channels: List[str] = ["log", "email"],
    force_restart_services: bool = False
) -> Dict[str, Any]:
    """
    Workflow de maintenance d'urgence pour les situations critiques
    
    Args:
        issue_description: Description du problème critique
        notification_channels: Canaux de notification d'urgence
        force_restart_services: Forcer le redémarrage des services
        
    Returns:
        Dict avec le résumé de la maintenance d'urgence
    """
    logger = get_run_logger()
    
    logger.error(f"🚨 MAINTENANCE D'URGENCE: {issue_description}")
    maintenance_start = datetime.now()
    
    try:
        # Notification immédiate
        await send_alert(
            title="🚨 MAINTENANCE D'URGENCE EN COURS",
            message=f"Démarrage de la maintenance d'urgence\nProblème: {issue_description}\nIntervention automatique en cours...",
            channels=notification_channels,
            severity="critical"
        )
        
        emergency_results = {
            "started_at": maintenance_start.isoformat(),
            "type": "emergency",
            "issue_description": issue_description,
            "actions": {}
        }
        
        # ===== VÉRIFICATION DE SANTÉ IMMÉDIATE =====
        logger.info("🩺 Diagnostic immédiat du système")
        
        health_check = await health_check_flow(
            alert_on_degraded=False,
            notification_channels=["log"],
            detailed_analysis=True
        )
        emergency_results["actions"]["health_diagnosis"] = health_check
        
        # ===== ACTIONS D'URGENCE BASÉES SUR LE DIAGNOSTIC =====
        health_status = health_check.get("health_check", {}).get("overall_health", "unknown")
        issues = health_check.get("health_check", {}).get("issues", [])
        
        if "disk" in str(issues).lower() or "space" in str(issues).lower():
            logger.info("💾 Nettoyage d'urgence de l'espace disque")
            
            emergency_cleanup = cleanup_logs(
                keep_days=7,  # Très agressif
                archive=False,  # Pas d'archivage en urgence
                max_size_mb=10
            )
            emergency_results["actions"]["emergency_disk_cleanup"] = emergency_cleanup
            
            # Nettoyage sessions d'urgence
            emergency_sessions = cleanup_sessions(
                inactive_days=1,  # Très agressif
                keep_recent_hours=6
            )
            emergency_results["actions"]["emergency_sessions_cleanup"] = emergency_sessions
        
        if "memory" in str(issues).lower():
            logger.info("💾 Actions d'urgence pour la mémoire")
            
            # Optimisation vector store immédiate
            memory_optimization = optimize_vector_store(
                collection_name="doctorpy_docs",
                vacuum=True
            )
            emergency_results["actions"]["memory_optimization"] = memory_optimization
        
        if "database" in str(issues).lower():
            logger.info("🗄️ Actions d'urgence pour la base de données")
            
            # Sauvegarde d'urgence avant intervention
            emergency_backup = backup_database(
                backup_dir="data/backups/emergency",
                keep_backups=2,
                compress=True
            )
            emergency_results["actions"]["emergency_backup"] = emergency_backup
        
        # ===== REDÉMARRAGE DES SERVICES SI DEMANDÉ =====
        if force_restart_services:
            logger.warning("🔄 Redémarrage forcé des services (simulation)")
            
            # Note: En réalité, ceci nécessiterait des commandes système spécifiques
            restart_result = {
                "status": "simulated",
                "message": "Redémarrage des services simulé",
                "services": ["chromadb", "database_connections", "cache_systems"]
            }
            emergency_results["actions"]["services_restart"] = restart_result
        
        # ===== VÉRIFICATION POST-INTERVENTION =====
        logger.info("🔍 Vérification post-intervention")
        
        post_health_check = await health_check_flow(
            alert_on_degraded=False,
            notification_channels=["log"],
            detailed_analysis=False
        )
        emergency_results["actions"]["post_health_check"] = post_health_check
        
        # ===== FINALISATION =====
        maintenance_end = datetime.now()
        maintenance_duration = (maintenance_end - maintenance_start).total_seconds()
        
        # Évaluer l'amélioration
        initial_health = health_status
        final_health = post_health_check.get("health_check", {}).get("overall_health", "unknown")
        
        improvement = "unknown"
        if initial_health == "unhealthy" and final_health in ["healthy", "degraded"]:
            improvement = "significant"
        elif initial_health == "degraded" and final_health == "healthy":
            improvement = "moderate"
        elif initial_health == final_health:
            improvement = "none"
        else:
            improvement = "partial"
        
        emergency_results.update({
            "status": "completed",
            "completed_at": maintenance_end.isoformat(),
            "duration_seconds": round(maintenance_duration, 1),
            "health_improvement": {
                "initial": initial_health,
                "final": final_health,
                "improvement_level": improvement
            }
        })
        
        # Notification finale
        if improvement in ["significant", "moderate"]:
            await send_notification(
                title="✅ Maintenance d'urgence réussie",
                message=f"""Maintenance d'urgence terminée avec succès en {maintenance_duration:.1f}s

🩺 Amélioration de la santé:
• Avant: {initial_health.upper()}
• Après: {final_health.upper()}
• Amélioration: {improvement.upper()}

Problème traité: {issue_description}""",
                channels=notification_channels,
                priority="high"
            )
        else:
            await send_alert(
                title="⚠️ Maintenance d'urgence partiellement réussie",
                message=f"""Maintenance d'urgence terminée en {maintenance_duration:.1f}s

🩺 État de la santé:
• Avant: {initial_health.upper()}
• Après: {final_health.upper()}
• Amélioration: {improvement.upper()}

Intervention manuelle supplémentaire peut être nécessaire.
Problème initial: {issue_description}""",
                channels=notification_channels,
                severity="medium"
            )
        
        logger.info(f"🚨 Maintenance d'urgence terminée - Amélioration: {improvement}")
        
        return emergency_results
        
    except Exception as e:
        maintenance_end = datetime.now()
        maintenance_duration = (maintenance_end - maintenance_start).total_seconds()
        
        logger.error(f"❌ Erreur critique lors de la maintenance d'urgence: {str(e)}")
        
        await send_alert(
            title="🚨 ÉCHEC MAINTENANCE D'URGENCE",
            message=f"""ERREUR CRITIQUE lors de la maintenance d'urgence:

❌ Erreur: {str(e)}
⏱️ Durée avant échec: {maintenance_duration:.1f}s
🔍 Problème initial: {issue_description}

INTERVENTION MANUELLE URGENTE REQUISE""",
            channels=notification_channels + ["email"] if "email" not in notification_channels else notification_channels,
            severity="critical"
        )
        
        return {
            "status": "failed",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "issue_description": issue_description,
            "started_at": maintenance_start.isoformat(),
            "failed_at": maintenance_end.isoformat(),
            "duration_seconds": round(maintenance_duration, 1),
            "actions": emergency_results.get("actions", {}) if 'emergency_results' in locals() else {}
        }


async def perform_additional_optimizations() -> Dict[str, Any]:
    """
    Effectuer des optimisations supplémentaires du système
    
    Returns:
        Dict avec les résultats des optimisations
    """
    logger = get_run_logger()
    
    try:
        logger.info("⚡ Optimisations supplémentaires")
        
        optimizations = []
        
        # Optimisation 1: Nettoyage des caches temporaires
        temp_dirs = [
            Path("data/temp"),
            Path("data/cache"),
            Path("logs/temp")
        ]
        
        temp_cleaned = 0
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for temp_file in temp_dir.rglob("*"):
                    if temp_file.is_file():
                        temp_file.unlink()
                        temp_cleaned += 1
        
        if temp_cleaned > 0:
            optimizations.append(f"Nettoyé {temp_cleaned} fichiers temporaires")
        
        # Optimisation 2: Vérification de l'intégrité des données
        data_integrity_check = check_data_integrity()
        if data_integrity_check["status"] == "success":
            optimizations.append("Vérification d'intégrité des données réussie")
        
        # Optimisation 3: Mise à jour des métadonnées
        metadata_update = update_system_metadata()
        if metadata_update["status"] == "success":
            optimizations.append("Métadonnées système mises à jour")
        
        return {
            "status": "success",
            "optimizations_performed": optimizations,
            "optimizations_count": len(optimizations),
            "temp_files_cleaned": temp_cleaned
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors des optimisations supplémentaires: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e)
        }


def check_data_integrity() -> Dict[str, Any]:
    """Vérifier l'intégrité des données critiques"""
    try:
        # Vérification basique des fichiers critiques
        critical_files = [
            "data/databases/doctorpy.db",
            "src/core/database.py",
            "ui/streamlit_app.py"
        ]
        
        missing_files = []
        for file_path in critical_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            return {
                "status": "warning",
                "missing_files": missing_files
            }
        
        return {"status": "success"}
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


def update_system_metadata() -> Dict[str, Any]:
    """Mettre à jour les métadonnées système"""
    try:
        # Créer ou mettre à jour un fichier de métadonnées
        metadata = {
            "last_maintenance": datetime.now().isoformat(),
            "system_version": "1.0",
            "maintenance_count": get_maintenance_count() + 1
        }
        
        metadata_file = Path("data/system_metadata.json")
        
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return {"status": "success", "metadata_file": str(metadata_file)}
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


def get_maintenance_count() -> int:
    """Obtenir le nombre de maintenances effectuées"""
    try:
        metadata_file = Path("data/system_metadata.json")
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                return metadata.get("maintenance_count", 0)
        return 0
    except:
        return 0