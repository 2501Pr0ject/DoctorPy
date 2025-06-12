"""
Tâches Prefect pour le monitoring et la surveillance
"""

import sys
import psutil
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
from prefect import task, get_run_logger

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.database import DatabaseManager


@task(
    name="collect_system_metrics",
    description="Collecter les métriques système",
    retries=1,
    tags=["monitoring", "metrics", "system"]
)
def collect_system_metrics() -> Dict[str, Any]:
    """
    Collecter les métriques système (CPU, mémoire, disque, etc.)
    
    Returns:
        Dict avec les métriques système
    """
    logger = get_run_logger()
    
    try:
        logger.info("📊 Collecte des métriques système")
        
        # Métriques CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Métriques mémoire
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Métriques disque
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Métriques réseau
        network_io = psutil.net_io_counters()
        
        # Processus système
        processes = len(psutil.pids())
        
        # Uptime du système
        boot_time = psutil.boot_time()
        uptime_seconds = datetime.now().timestamp() - boot_time
        
        # Charge système (Unix uniquement)
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            load_avg = [0, 0, 0]  # Windows n'a pas getloadavg
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2]
            },
            "memory": {
                "total_gb": round(memory.total / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "used_gb": round(memory.used / 1024**3, 2),
                "usage_percent": memory.percent,
                "swap_total_gb": round(swap.total / 1024**3, 2),
                "swap_used_gb": round(swap.used / 1024**3, 2),
                "swap_percent": swap.percent
            },
            "disk": {
                "total_gb": round(disk_usage.total / 1024**3, 2),
                "used_gb": round(disk_usage.used / 1024**3, 2),
                "free_gb": round(disk_usage.free / 1024**3, 2),
                "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 1),
                "read_mb": round(disk_io.read_bytes / 1024**2, 2) if disk_io else 0,
                "write_mb": round(disk_io.write_bytes / 1024**2, 2) if disk_io else 0
            },
            "network": {
                "bytes_sent_mb": round(network_io.bytes_sent / 1024**2, 2),
                "bytes_recv_mb": round(network_io.bytes_recv / 1024**2, 2),
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            },
            "system": {
                "processes_count": processes,
                "uptime_hours": round(uptime_seconds / 3600, 1),
                "boot_time": datetime.fromtimestamp(boot_time).isoformat()
            }
        }
        
        logger.info(f"📊 Métriques collectées:")
        logger.info(f"   🖥️ CPU: {cpu_percent}% ({cpu_count} cores)")
        logger.info(f"   💾 RAM: {memory.percent}% ({metrics['memory']['used_gb']}/{metrics['memory']['total_gb']} GB)")
        logger.info(f"   💿 Disque: {metrics['disk']['usage_percent']}% ({metrics['disk']['used_gb']}/{metrics['disk']['total_gb']} GB)")
        logger.info(f"   🔄 Processus: {processes}")
        
        return {
            "status": "success",
            "metrics": metrics,
            "collected_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la collecte des métriques: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }


@task(
    name="collect_application_metrics",
    description="Collecter les métriques de l'application",
    retries=1,
    tags=["monitoring", "metrics", "application"]
)
def collect_application_metrics() -> Dict[str, Any]:
    """
    Collecter les métriques spécifiques à l'application DoctorPy
    
    Returns:
        Dict avec les métriques application
    """
    logger = get_run_logger()
    
    try:
        logger.info("🎯 Collecte des métriques application")
        
        db_manager = DatabaseManager()
        
        # Statistiques de base de données
        db_stats = db_manager.get_database_stats()
        
        # Métriques utilisateurs
        active_users_24h = db_manager.execute_query("""
            SELECT COUNT(DISTINCT user_id) as count 
            FROM chat_sessions 
            WHERE last_activity > datetime('now', '-24 hours')
        """)
        
        new_users_24h = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM users 
            WHERE created_at > datetime('now', '-24 hours')
        """)
        
        # Métriques de sessions
        active_sessions = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM chat_sessions 
            WHERE is_active = 1
        """)
        
        avg_session_duration = db_manager.execute_query("""
            SELECT AVG((julianday(last_activity) - julianday(created_at)) * 24 * 60) as avg_minutes
            FROM chat_sessions 
            WHERE last_activity > datetime('now', '-7 days')
        """)
        
        # Métriques de quêtes
        quests_completed_24h = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM user_progress 
            WHERE status = 'completed' AND completed_at > datetime('now', '-24 hours')
        """)
        
        avg_quest_completion = db_manager.execute_query("""
            SELECT AVG(completion_percentage) as avg_completion
            FROM user_progress 
            WHERE status IN ('in_progress', 'completed')
        """)
        
        # Métriques de messages
        messages_24h = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM messages 
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        
        # Métriques d'erreurs (depuis analytics)
        errors_24h = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM analytics 
            WHERE event_type = 'error' AND timestamp > datetime('now', '-24 hours')
        """)
        
        # Taille des fichiers de données
        data_sizes = {}
        data_paths = {
            "database": Path("data/databases/doctorpy.db"),
            "vector_store": Path("data/vector_store"),
            "embeddings": Path("data/embeddings"),
            "processed": Path("data/processed"),
            "raw": Path("data/raw")
        }
        
        for name, path in data_paths.items():
            if path.exists():
                if path.is_file():
                    size_mb = path.stat().st_size / 1024**2
                else:
                    size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2
                data_sizes[name] = round(size_mb, 2)
            else:
                data_sizes[name] = 0
        
        # Statut des services (vérification basique)
        services_status = {}
        
        # Vérifier ChromaDB
        try:
            import chromadb
            chroma_client = chromadb.PersistentClient(path="./data/vector_store")
            collections = chroma_client.list_collections()
            services_status["chromadb"] = {
                "status": "healthy",
                "collections_count": len(collections)
            }
        except Exception as e:
            services_status["chromadb"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Vérifier Ollama (si disponible)
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                services_status["ollama"] = {
                    "status": "healthy",
                    "models_count": len(models)
                }
            else:
                services_status["ollama"] = {
                    "status": "unhealthy",
                    "http_status": response.status_code
                }
        except Exception as e:
            services_status["ollama"] = {
                "status": "offline",
                "error": str(e)
            }
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "database": db_stats,
            "users": {
                "total": db_stats.get("users", 0),
                "active_24h": active_users_24h[0]["count"] if active_users_24h else 0,
                "new_24h": new_users_24h[0]["count"] if new_users_24h else 0
            },
            "sessions": {
                "active": active_sessions[0]["count"] if active_sessions else 0,
                "avg_duration_minutes": round(avg_session_duration[0]["avg_minutes"] or 0, 1) if avg_session_duration else 0
            },
            "quests": {
                "total": db_stats.get("quests", 0),
                "completed_24h": quests_completed_24h[0]["count"] if quests_completed_24h else 0,
                "avg_completion_percent": round(avg_quest_completion[0]["avg_completion"] or 0, 1) if avg_quest_completion else 0
            },
            "messages": {
                "total": db_stats.get("messages", 0),
                "sent_24h": messages_24h[0]["count"] if messages_24h else 0
            },
            "errors": {
                "count_24h": errors_24h[0]["count"] if errors_24h else 0
            },
            "data_sizes_mb": data_sizes,
            "services": services_status
        }
        
        logger.info(f"🎯 Métriques application collectées:")
        logger.info(f"   👥 Utilisateurs actifs 24h: {metrics['users']['active_24h']}")
        logger.info(f"   💬 Messages 24h: {metrics['messages']['sent_24h']}")
        logger.info(f"   🎯 Quêtes complétées 24h: {metrics['quests']['completed_24h']}")
        logger.info(f"   ❌ Erreurs 24h: {metrics['errors']['count_24h']}")
        
        return {
            "status": "success",
            "metrics": metrics,
            "collected_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la collecte des métriques app: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }


@task(
    name="check_health",
    description="Vérifier la santé globale du système",
    retries=1,
    tags=["monitoring", "health", "check"]
)
def check_health() -> Dict[str, Any]:
    """
    Effectuer un check de santé complet du système
    
    Returns:
        Dict avec le statut de santé global
    """
    logger = get_run_logger()
    
    try:
        logger.info("🩺 Vérification de santé du système")
        
        health_checks = {}
        overall_status = "healthy"
        issues = []
        
        # Check 1: Base de données
        try:
            db_manager = DatabaseManager()
            db_stats = db_manager.get_database_stats()
            
            # Vérifier la connectivité
            test_query = db_manager.execute_query("SELECT 1 as test")
            
            if test_query and test_query[0]["test"] == 1:
                health_checks["database"] = {
                    "status": "healthy",
                    "tables": len([k for k, v in db_stats.items() if v > 0]),
                    "total_records": sum(db_stats.values())
                }
            else:
                health_checks["database"] = {"status": "unhealthy", "issue": "Query failed"}
                overall_status = "degraded"
                issues.append("Database query test failed")
                
        except Exception as e:
            health_checks["database"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"
            issues.append(f"Database error: {str(e)}")
        
        # Check 2: ChromaDB Vector Store
        try:
            import chromadb
            chroma_client = chromadb.PersistentClient(path="./data/vector_store")
            collections = chroma_client.list_collections()
            
            if collections:
                # Test d'une collection
                collection = collections[0]
                count = collection.count()
                
                # Test de requête simple
                test_result = collection.query(query_texts=["test"], n_results=1)
                
                health_checks["vector_store"] = {
                    "status": "healthy",
                    "collections": len(collections),
                    "documents": count,
                    "query_test": len(test_result.get("documents", [[]])[0]) > 0
                }
            else:
                health_checks["vector_store"] = {"status": "degraded", "issue": "No collections found"}
                if overall_status == "healthy":
                    overall_status = "degraded"
                issues.append("ChromaDB has no collections")
                
        except Exception as e:
            health_checks["vector_store"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"
            issues.append(f"ChromaDB error: {str(e)}")
        
        # Check 3: Espace disque
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            if free_percent < 5:
                disk_status = "critical"
                overall_status = "unhealthy"
                issues.append(f"Disk space critical: {free_percent:.1f}% free")
            elif free_percent < 15:
                disk_status = "warning"
                if overall_status == "healthy":
                    overall_status = "degraded"
                issues.append(f"Disk space low: {free_percent:.1f}% free")
            else:
                disk_status = "healthy"
            
            health_checks["disk_space"] = {
                "status": disk_status,
                "free_percent": round(free_percent, 1),
                "free_gb": round(free / 1024**3, 1),
                "total_gb": round(total / 1024**3, 1)
            }
            
        except Exception as e:
            health_checks["disk_space"] = {"status": "unknown", "error": str(e)}
        
        # Check 4: Mémoire
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                memory_status = "critical"
                overall_status = "unhealthy"
                issues.append(f"Memory critical: {memory.percent}% used")
            elif memory.percent > 85:
                memory_status = "warning"
                if overall_status == "healthy":
                    overall_status = "degraded"
                issues.append(f"Memory high: {memory.percent}% used")
            else:
                memory_status = "healthy"
            
            health_checks["memory"] = {
                "status": memory_status,
                "usage_percent": memory.percent,
                "available_gb": round(memory.available / 1024**3, 1)
            }
            
        except Exception as e:
            health_checks["memory"] = {"status": "unknown", "error": str(e)}
        
        # Check 5: Services externes
        # Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                health_checks["ollama"] = {"status": "healthy", "models": len(response.json().get("models", []))}
            else:
                health_checks["ollama"] = {"status": "degraded", "http_status": response.status_code}
                if overall_status == "healthy":
                    overall_status = "degraded"
                issues.append(f"Ollama responding with status {response.status_code}")
        except Exception as e:
            health_checks["ollama"] = {"status": "offline", "error": str(e)}
            # Ollama offline n'est pas critique
        
        # Check 6: Fichiers critiques
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
            health_checks["critical_files"] = {
                "status": "unhealthy",
                "missing_files": missing_files
            }
            overall_status = "unhealthy"
            issues.append(f"Missing critical files: {', '.join(missing_files)}")
        else:
            health_checks["critical_files"] = {"status": "healthy", "all_present": True}
        
        # Résumé final
        result = {
            "status": "success",
            "overall_health": overall_status,
            "checks": health_checks,
            "issues": issues,
            "issues_count": len(issues),
            "checked_at": datetime.now().isoformat(),
            "uptime_status": "operational" if overall_status in ["healthy", "degraded"] else "down"
        }
        
        # Log du résumé
        status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "unknown": "❓"}
        logger.info(f"🩺 Vérification terminée: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        
        if issues:
            logger.warning(f"⚠️ {len(issues)} problème(s) détecté(s):")
            for issue in issues:
                logger.warning(f"   • {issue}")
        else:
            logger.info("✅ Aucun problème détecté")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification de santé: {str(e)}")
        return {
            "status": "error",
            "overall_health": "unknown",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "checked_at": datetime.now().isoformat()
        }


@task(
    name="generate_report",
    description="Générer un rapport de monitoring",
    retries=1,
    tags=["monitoring", "report"]
)
def generate_report(
    system_metrics: Dict[str, Any],
    app_metrics: Dict[str, Any],
    health_check: Dict[str, Any],
    report_type: str = "daily"
) -> Dict[str, Any]:
    """
    Générer un rapport de monitoring consolidé
    
    Args:
        system_metrics: Métriques système
        app_metrics: Métriques application
        health_check: Résultats du check de santé
        report_type: Type de rapport (daily, weekly, monthly)
        
    Returns:
        Dict avec le rapport généré
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"📋 Génération du rapport {report_type}")
        
        # Extraire les données des métriques
        timestamp = datetime.now()
        
        # Métriques système
        sys_data = system_metrics.get("metrics", {}) if system_metrics.get("status") == "success" else {}
        
        # Métriques application
        app_data = app_metrics.get("metrics", {}) if app_metrics.get("status") == "success" else {}
        
        # Santé du système
        health_data = health_check if health_check.get("status") == "success" else {}
        
        # Compiler le rapport
        report = {
            "report_type": report_type,
            "generated_at": timestamp.isoformat(),
            "period": {
                "start": (timestamp - timedelta(days=1)).isoformat(),
                "end": timestamp.isoformat()
            },
            "summary": {
                "overall_health": health_data.get("overall_health", "unknown"),
                "issues_count": health_data.get("issues_count", 0),
                "uptime_status": health_data.get("uptime_status", "unknown")
            },
            "system": {
                "cpu_usage": sys_data.get("cpu", {}).get("usage_percent", 0),
                "memory_usage": sys_data.get("memory", {}).get("usage_percent", 0),
                "disk_usage": sys_data.get("disk", {}).get("usage_percent", 0),
                "uptime_hours": sys_data.get("system", {}).get("uptime_hours", 0)
            },
            "application": {
                "active_users_24h": app_data.get("users", {}).get("active_24h", 0),
                "new_users_24h": app_data.get("users", {}).get("new_24h", 0),
                "messages_24h": app_data.get("messages", {}).get("sent_24h", 0),
                "quests_completed_24h": app_data.get("quests", {}).get("completed_24h", 0),
                "errors_24h": app_data.get("errors", {}).get("count_24h", 0),
                "avg_session_duration": app_data.get("sessions", {}).get("avg_duration_minutes", 0)
            },
            "services": app_data.get("services", {}),
            "data_storage": {
                "database_size_mb": app_data.get("data_sizes_mb", {}).get("database", 0),
                "vector_store_size_mb": app_data.get("data_sizes_mb", {}).get("vector_store", 0),
                "total_data_size_mb": sum(app_data.get("data_sizes_mb", {}).values())
            },
            "issues": health_data.get("issues", [])
        }
        
        # Calculer des scores
        report["scores"] = {
            "performance_score": max(0, 100 - report["system"]["cpu_usage"] - report["system"]["memory_usage"]) / 2,
            "availability_score": 100 if report["summary"]["overall_health"] == "healthy" else 
                                 80 if report["summary"]["overall_health"] == "degraded" else 0,
            "user_activity_score": min(100, report["application"]["active_users_24h"] * 10)
        }
        
        # Score global
        scores = report["scores"]
        report["overall_score"] = round(
            (scores["performance_score"] * 0.3 + 
             scores["availability_score"] * 0.5 + 
             scores["user_activity_score"] * 0.2), 1
        )
        
        # Sauvegarder le rapport
        reports_dir = Path("data/reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_filename = f"{report_type}_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_file = reports_dir / report_filename
        
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        result = {
            "status": "success",
            "report": report,
            "report_file": str(report_file),
            "report_size_kb": round(report_file.stat().st_size / 1024, 1),
            "generated_at": timestamp.isoformat()
        }
        
        logger.info(f"📋 Rapport {report_type} généré:")
        logger.info(f"   📊 Score global: {report['overall_score']}/100")
        logger.info(f"   🎯 Santé: {report['summary']['overall_health']}")
        logger.info(f"   👥 Utilisateurs actifs: {report['application']['active_users_24h']}")
        logger.info(f"   💾 Fichier: {report_filename}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération du rapport: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }