"""
Workflows Prefect pour l'analytique et le monitoring avancé
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from prefect import flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .tasks.monitoring import (
    collect_system_metrics, 
    collect_application_metrics, 
    check_health, 
    generate_report
)
from .tasks.notification import (
    send_notification, 
    send_alert, 
    send_report_notification
)


@flow(
    name="generate_analytics",
    description="Générer un rapport analytique complet",
    version="1.0",
    task_runner=ConcurrentTaskRunner(max_workers=3),
    timeout_seconds=1800,  # 30 minutes max
    retries=1
)
async def generate_analytics(
    report_type: str = "daily",
    notification_channels: List[str] = ["log", "email"],
    include_health_check: bool = True
) -> Dict[str, Any]:
    """
    Générer un rapport analytique complet du système
    
    Args:
        report_type: Type de rapport (daily, weekly, monthly)
        notification_channels: Canaux de notification pour le rapport
        include_health_check: Inclure une vérification de santé
        
    Returns:
        Dict avec le rapport analytique complet
    """
    logger = get_run_logger()
    
    logger.info(f"📊 Génération du rapport analytique {report_type}")
    
    try:
        # Collecte des métriques en parallèle
        logger.info("📈 Collecte des métriques système et application")
        
        system_metrics, app_metrics = await asyncio.gather(
            collect_system_metrics(),
            collect_application_metrics(),
            return_exceptions=True
        )
        
        # Vérification de santé si demandée
        health_check = None
        if include_health_check:
            logger.info("🩺 Vérification de santé du système")
            health_check = check_health()
        
        # Génération du rapport
        logger.info("📋 Compilation du rapport analytique")
        
        report_result = generate_report(
            system_metrics=system_metrics if not isinstance(system_metrics, Exception) else {"status": "error", "error": str(system_metrics)},
            app_metrics=app_metrics if not isinstance(app_metrics, Exception) else {"status": "error", "error": str(app_metrics)},
            health_check=health_check if health_check and health_check.get("status") == "success" else {"status": "skipped"},
            report_type=report_type
        )
        
        # Analyse des tendances (si rapport quotidien ou plus)
        trends_analysis = None
        if report_type in ["daily", "weekly", "monthly"]:
            trends_analysis = await analyze_trends(report_type)
        
        # Compilation des résultats
        analytics_result = {
            "status": "success",
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "metrics_collection": {
                "system_metrics": system_metrics.get("status") if not isinstance(system_metrics, Exception) else "error",
                "app_metrics": app_metrics.get("status") if not isinstance(app_metrics, Exception) else "error",
                "health_check": health_check.get("status") if health_check else "skipped"
            },
            "report": report_result,
            "trends": trends_analysis
        }
        
        # Notification du rapport
        if report_result.get("status") == "success":
            await send_report_notification(
                report_data=report_result,
                channels=notification_channels
            )
        else:
            await send_alert(
                title=f"Échec génération rapport {report_type}",
                message=f"Erreur lors de la génération: {report_result.get('error_message', 'Inconnue')}",
                channels=notification_channels
            )
        
        logger.info(f"✅ Rapport analytique {report_type} généré avec succès")
        return analytics_result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération analytique: {str(e)}")
        
        await send_alert(
            title=f"Erreur critique - Génération analytique {report_type}",
            message=f"Impossible de générer le rapport: {str(e)}",
            channels=notification_channels
        )
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "report_type": report_type
        }


@flow(
    name="health_check_flow",
    description="Workflow de vérification de santé avec alertes",
    version="1.0",
    timeout_seconds=600,  # 10 minutes max
    retries=2
)
async def health_check_flow(
    alert_on_degraded: bool = True,
    notification_channels: List[str] = ["log"],
    detailed_analysis: bool = False
) -> Dict[str, Any]:
    """
    Workflow complet de vérification de santé avec alertes conditionnelles
    
    Args:
        alert_on_degraded: Envoyer une alerte si le système est dégradé
        notification_channels: Canaux de notification
        detailed_analysis: Inclure une analyse détaillée des problèmes
        
    Returns:
        Dict avec les résultats de la vérification
    """
    logger = get_run_logger()
    
    logger.info("🩺 Démarrage de la vérification de santé complète")
    
    try:
        # Vérification de santé principale
        health_result = check_health()
        
        overall_health = health_result.get("overall_health", "unknown")
        issues = health_result.get("issues", [])
        issues_count = len(issues)
        
        # Analyse détaillée si demandée
        detailed_results = None
        if detailed_analysis and health_result.get("status") == "success":
            detailed_results = await perform_detailed_health_analysis(health_result)
        
        # Déterminer les actions selon le statut
        if overall_health == "unhealthy":
            # Système en panne - alerte critique
            await send_alert(
                title="🚨 SYSTÈME EN PANNE - Intervention urgente requise",
                message=f"Santé système: {overall_health.upper()}\n{issues_count} problème(s) critique(s) détecté(s):\n" + 
                       "\n".join([f"• {issue}" for issue in issues[:10]]),
                channels=notification_channels + ["email"] if "email" not in notification_channels else notification_channels,
                severity="critical"
            )
            
        elif overall_health == "degraded" and alert_on_degraded:
            # Système dégradé - alerte d'avertissement
            await send_alert(
                title="⚠️ Système dégradé - Surveillance renforcée",
                message=f"Santé système: {overall_health.upper()}\n{issues_count} problème(s) détecté(s):\n" + 
                       "\n".join([f"• {issue}" for issue in issues[:5]]),
                channels=notification_channels,
                severity="medium"
            )
            
        elif overall_health == "healthy":
            # Système sain - notification de succès
            await send_notification(
                title="✅ Système en bonne santé",
                message=f"Vérification terminée: aucun problème détecté\nStatut: {overall_health.upper()}",
                channels=notification_channels,
                priority="low"
            )
        
        # Compilation des résultats
        workflow_result = {
            "status": "success",
            "health_check": health_result,
            "detailed_analysis": detailed_results,
            "actions_taken": {
                "alert_sent": overall_health in ["unhealthy", "degraded"] if alert_on_degraded else overall_health == "unhealthy",
                "notification_sent": overall_health == "healthy",
                "detailed_analysis_performed": detailed_analysis and detailed_results is not None
            },
            "checked_at": datetime.now().isoformat()
        }
        
        logger.info(f"🩺 Vérification terminée - Santé: {overall_health.upper()}")
        if issues_count > 0:
            logger.warning(f"⚠️ {issues_count} problème(s) détecté(s)")
        
        return workflow_result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification de santé: {str(e)}")
        
        await send_alert(
            title="🚨 Erreur critique - Vérification de santé",
            message=f"Impossible d'effectuer la vérification de santé: {str(e)}\nIntervention manuelle requise.",
            channels=notification_channels + ["email"] if "email" not in notification_channels else notification_channels,
            severity="critical"
        )
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "checked_at": datetime.now().isoformat()
        }


async def analyze_trends(report_type: str) -> Dict[str, Any]:
    """
    Analyser les tendances des métriques sur une période donnée
    
    Args:
        report_type: Type de rapport pour déterminer la période d'analyse
        
    Returns:
        Dict avec l'analyse des tendances
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"📈 Analyse des tendances {report_type}")
        
        # Déterminer la période d'analyse
        if report_type == "daily":
            analysis_days = 7  # Tendance sur 7 jours
        elif report_type == "weekly":
            analysis_days = 30  # Tendance sur 30 jours
        elif report_type == "monthly":
            analysis_days = 90  # Tendance sur 90 jours
        else:
            analysis_days = 7  # Par défaut
        
        # Rechercher les rapports précédents
        reports_dir = Path("data/reports")
        if not reports_dir.exists():
            return {"status": "skipped", "reason": "no_historical_data"}
        
        # Collecter les rapports récents
        recent_reports = []
        cutoff_date = datetime.now() - timedelta(days=analysis_days)
        
        for report_file in reports_dir.glob(f"{report_type}_report_*.json"):
            try:
                file_date = datetime.fromtimestamp(report_file.stat().st_mtime)
                if file_date >= cutoff_date:
                    import json
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        report_data['file_date'] = file_date.isoformat()
                        recent_reports.append(report_data)
            except Exception:
                continue
        
        if len(recent_reports) < 2:
            return {"status": "insufficient_data", "reports_found": len(recent_reports)}
        
        # Trier par date
        recent_reports.sort(key=lambda x: x['file_date'])
        
        # Analyser les tendances
        trends = {
            "cpu_usage": analyze_metric_trend([r.get("system", {}).get("cpu_usage", 0) for r in recent_reports]),
            "memory_usage": analyze_metric_trend([r.get("system", {}).get("memory_usage", 0) for r in recent_reports]),
            "active_users": analyze_metric_trend([r.get("application", {}).get("active_users_24h", 0) for r in recent_reports]),
            "error_rate": analyze_metric_trend([r.get("application", {}).get("errors_24h", 0) for r in recent_reports]),
            "overall_score": analyze_metric_trend([r.get("overall_score", 0) for r in recent_reports])
        }
        
        # Détecter les anomalies
        anomalies = []
        for metric, trend in trends.items():
            if trend["trend"] == "concerning":
                anomalies.append(f"{metric}: {trend['change_description']}")
        
        return {
            "status": "success",
            "analysis_period_days": analysis_days,
            "reports_analyzed": len(recent_reports),
            "trends": trends,
            "anomalies": anomalies,
            "overall_trend": "improving" if trends["overall_score"]["trend"] == "improving" else 
                            "declining" if trends["overall_score"]["trend"] == "concerning" else "stable"
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse des tendances: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e)
        }


def analyze_metric_trend(values: List[float]) -> Dict[str, Any]:
    """
    Analyser la tendance d'une métrique
    
    Args:
        values: Liste des valeurs de la métrique
        
    Returns:
        Dict avec l'analyse de tendance
    """
    if len(values) < 2:
        return {"trend": "unknown", "reason": "insufficient_data"}
    
    # Calculer la tendance
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]
    
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    
    change_percent = ((avg_second - avg_first) / avg_first * 100) if avg_first > 0 else 0
    
    # Déterminer la direction de la tendance
    if abs(change_percent) < 5:
        trend = "stable"
        change_description = f"Stable ({change_percent:+.1f}%)"
    elif change_percent > 15:
        trend = "concerning" if avg_second > 80 else "improving"  # Dépend du contexte
        change_description = f"Forte hausse ({change_percent:+.1f}%)"
    elif change_percent < -15:
        trend = "improving" if avg_second < 80 else "concerning"
        change_description = f"Forte baisse ({change_percent:+.1f}%)"
    elif change_percent > 0:
        trend = "mild_increase"
        change_description = f"Légère hausse ({change_percent:+.1f}%)"
    else:
        trend = "mild_decrease"
        change_description = f"Légère baisse ({change_percent:+.1f}%)"
    
    return {
        "trend": trend,
        "change_percent": round(change_percent, 1),
        "change_description": change_description,
        "avg_first_period": round(avg_first, 1),
        "avg_second_period": round(avg_second, 1),
        "data_points": len(values)
    }


async def perform_detailed_health_analysis(health_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Effectuer une analyse détaillée des problèmes de santé
    
    Args:
        health_result: Résultat de la vérification de santé
        
    Returns:
        Dict avec l'analyse détaillée
    """
    logger = get_run_logger()
    
    try:
        logger.info("🔍 Analyse détaillée des problèmes de santé")
        
        checks = health_result.get("checks", {})
        issues = health_result.get("issues", [])
        
        # Analyser chaque composant
        component_analysis = {}
        
        for component, result in checks.items():
            status = result.get("status", "unknown")
            
            if status in ["unhealthy", "critical", "warning"]:
                # Analyse spécifique par composant
                if component == "memory":
                    component_analysis[component] = analyze_memory_issues(result)
                elif component == "disk_space":
                    component_analysis[component] = analyze_disk_issues(result)
                elif component == "database":
                    component_analysis[component] = analyze_database_issues(result)
                elif component == "vector_store":
                    component_analysis[component] = analyze_vector_store_issues(result)
                else:
                    component_analysis[component] = {
                        "severity": status,
                        "recommendations": [f"Vérifier manuellement le composant {component}"]
                    }
        
        # Recommandations globales
        global_recommendations = generate_global_recommendations(health_result, component_analysis)
        
        return {
            "status": "success",
            "components_analyzed": len(component_analysis),
            "component_analysis": component_analysis,
            "global_recommendations": global_recommendations,
            "priority_actions": [rec for rec in global_recommendations if "urgent" in rec.lower() or "critique" in rec.lower()]
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse détaillée: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e)
        }


def analyze_memory_issues(memory_result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyser les problèmes de mémoire"""
    usage_percent = memory_result.get("usage_percent", 0)
    available_gb = memory_result.get("available_gb", 0)
    
    recommendations = []
    if usage_percent > 95:
        recommendations.extend([
            "URGENT: Redémarrer les services non-critiques",
            "Identifier les processus consommant le plus de mémoire",
            "Envisager l'ajout de mémoire RAM"
        ])
        severity = "critical"
    elif usage_percent > 85:
        recommendations.extend([
            "Surveiller l'évolution de l'utilisation mémoire",
            "Optimiser les requêtes de base de données",
            "Nettoyer les caches temporaires"
        ])
        severity = "warning"
    else:
        severity = "normal"
    
    return {
        "severity": severity,
        "usage_percent": usage_percent,
        "available_gb": available_gb,
        "recommendations": recommendations
    }


def analyze_disk_issues(disk_result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyser les problèmes d'espace disque"""
    free_percent = disk_result.get("free_percent", 100)
    free_gb = disk_result.get("free_gb", 0)
    
    recommendations = []
    if free_percent < 5:
        recommendations.extend([
            "URGENT: Libérer de l'espace disque immédiatement",
            "Supprimer les anciens logs et sauvegardes",
            "Déplacer les données vers un stockage externe"
        ])
        severity = "critical"
    elif free_percent < 15:
        recommendations.extend([
            "Planifier un nettoyage d'espace disque",
            "Archiver les anciennes données",
            "Surveiller la croissance des données"
        ])
        severity = "warning"
    else:
        severity = "normal"
    
    return {
        "severity": severity,
        "free_percent": free_percent,
        "free_gb": free_gb,
        "recommendations": recommendations
    }


def analyze_database_issues(db_result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyser les problèmes de base de données"""
    recommendations = []
    error = db_result.get("error", "")
    
    if "connection" in error.lower():
        recommendations.extend([
            "Vérifier la connectivité à la base de données",
            "Redémarrer le service de base de données",
            "Vérifier les permissions d'accès"
        ])
        severity = "critical"
    elif "timeout" in error.lower():
        recommendations.extend([
            "Optimiser les requêtes lentes",
            "Analyser les index de base de données",
            "Augmenter les timeout de connexion"
        ])
        severity = "warning"
    else:
        recommendations.append("Analyser les logs de base de données pour plus de détails")
        severity = "warning"
    
    return {
        "severity": severity,
        "error": error,
        "recommendations": recommendations
    }


def analyze_vector_store_issues(vs_result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyser les problèmes du vector store"""
    recommendations = []
    error = vs_result.get("error", "")
    
    if "collection" in error.lower():
        recommendations.extend([
            "Vérifier l'intégrité des collections ChromaDB",
            "Réindexer les documents si nécessaire",
            "Vérifier l'espace disque du vector store"
        ])
        severity = "warning"
    else:
        recommendations.extend([
            "Redémarrer ChromaDB",
            "Vérifier la configuration du vector store",
            "Analyser les logs ChromaDB"
        ])
        severity = "critical"
    
    return {
        "severity": severity,
        "error": error,
        "recommendations": recommendations
    }


def generate_global_recommendations(health_result: Dict[str, Any], component_analysis: Dict[str, Any]) -> List[str]:
    """Générer des recommandations globales"""
    recommendations = []
    overall_health = health_result.get("overall_health", "unknown")
    
    critical_components = [comp for comp, analysis in component_analysis.items() 
                          if analysis.get("severity") == "critical"]
    
    if critical_components:
        recommendations.append(f"URGENT: Résoudre les problèmes critiques sur: {', '.join(critical_components)}")
    
    if overall_health == "unhealthy":
        recommendations.extend([
            "Effectuer une maintenance d'urgence du système",
            "Notifier l'équipe d'administration système",
            "Préparer un plan de récupération"
        ])
    elif overall_health == "degraded":
        recommendations.extend([
            "Planifier une maintenance préventive",
            "Surveiller l'évolution des métriques",
            "Optimiser les composants en avertissement"
        ])
    
    return recommendations


# Imports nécessaires
import asyncio