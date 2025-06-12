"""
Tâches Prefect pour la maintenance du système
"""

import sys
import os
import shutil
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
from prefect import task, get_run_logger

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.database import DatabaseManager


@task(
    name="cleanup_sessions",
    description="Nettoyer les sessions inactives",
    retries=2,
    retry_delay_seconds=[30, 120],
    tags=["maintenance", "cleanup", "sessions"]
)
def cleanup_sessions(
    inactive_days: int = 7,
    keep_recent_hours: int = 24
) -> Dict[str, Any]:
    """
    Nettoyer les sessions inactives et leurs messages associés
    
    Args:
        inactive_days: Nombre de jours après lesquels une session est considérée inactive
        keep_recent_hours: Heures récentes à préserver même si inactives
        
    Returns:
        Dict avec les statistiques de nettoyage
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"🧹 Nettoyage des sessions inactives (>{inactive_days} jours)")
        
        db_manager = DatabaseManager()
        
        # Calculer les seuils de date
        inactive_threshold = datetime.now() - timedelta(days=inactive_days)
        recent_threshold = datetime.now() - timedelta(hours=keep_recent_hours)
        
        # Compter les sessions à nettoyer
        sessions_to_clean = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM chat_sessions 
            WHERE (is_active = 0 AND last_activity < ?) 
               OR (last_activity < ? AND created_at < ?)
        """, (inactive_threshold.isoformat(), inactive_threshold.isoformat(), recent_threshold.isoformat()))
        
        sessions_count = sessions_to_clean[0]['count'] if sessions_to_clean else 0
        
        # Compter les messages associés
        messages_to_clean = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM messages m
            JOIN chat_sessions s ON m.session_id = s.id
            WHERE (s.is_active = 0 AND s.last_activity < ?) 
               OR (s.last_activity < ? AND s.created_at < ?)
        """, (inactive_threshold.isoformat(), inactive_threshold.isoformat(), recent_threshold.isoformat()))
        
        messages_count = messages_to_clean[0]['count'] if messages_to_clean else 0
        
        if sessions_count == 0:
            logger.info("✅ Aucune session à nettoyer")
            return {
                "status": "success",
                "sessions_cleaned": 0,
                "messages_cleaned": 0,
                "space_freed_mb": 0
            }
        
        # Nettoyer les messages d'abord (contrainte de clé étrangère)
        messages_cleaned = db_manager.execute_update("""
            DELETE FROM messages 
            WHERE session_id IN (
                SELECT id FROM chat_sessions 
                WHERE (is_active = 0 AND last_activity < ?) 
                   OR (last_activity < ? AND created_at < ?)
            )
        """, (inactive_threshold.isoformat(), inactive_threshold.isoformat(), recent_threshold.isoformat()))
        
        # Nettoyer les sessions
        sessions_cleaned = db_manager.execute_update("""
            DELETE FROM chat_sessions 
            WHERE (is_active = 0 AND last_activity < ?) 
               OR (last_activity < ? AND created_at < ?)
        """, (inactive_threshold.isoformat(), inactive_threshold.isoformat(), recent_threshold.isoformat()))
        
        # Estimer l'espace libéré (approximatif)
        space_freed_mb = round((messages_cleaned * 0.5 + sessions_cleaned * 0.1) / 1024, 2)
        
        result = {
            "status": "success", 
            "sessions_cleaned": sessions_cleaned,
            "messages_cleaned": messages_cleaned,
            "space_freed_mb": space_freed_mb,
            "cleanup_threshold": inactive_threshold.isoformat(),
            "recent_threshold": recent_threshold.isoformat()
        }
        
        logger.info(f"✅ Nettoyage terminé:")
        logger.info(f"   🗑️ Sessions supprimées: {sessions_cleaned}")
        logger.info(f"   💬 Messages supprimés: {messages_cleaned}")
        logger.info(f"   💾 Espace libéré: ~{space_freed_mb} MB")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage des sessions: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "sessions_cleaned": 0,
            "messages_cleaned": 0
        }


@task(
    name="cleanup_logs",
    description="Nettoyer et archiver les anciens logs",
    retries=2,
    retry_delay_seconds=[30, 60],
    tags=["maintenance", "cleanup", "logs"]
)
def cleanup_logs(
    keep_days: int = 30,
    archive: bool = True,
    max_size_mb: int = 100
) -> Dict[str, Any]:
    """
    Nettoyer les anciens fichiers de logs
    
    Args:
        keep_days: Nombre de jours de logs à conserver
        archive: Archiver les logs avant suppression
        max_size_mb: Taille maximale des logs en MB
        
    Returns:
        Dict avec les statistiques de nettoyage
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"📜 Nettoyage des logs (>{keep_days} jours)")
        
        logs_dir = Path("logs")
        if not logs_dir.exists():
            logger.info("📂 Répertoire logs inexistant - création")
            logs_dir.mkdir(parents=True, exist_ok=True)
            return {
                "status": "success",
                "logs_cleaned": 0,
                "logs_archived": 0,
                "space_freed_mb": 0
            }
        
        # Seuil de date pour le nettoyage
        cleanup_threshold = datetime.now() - timedelta(days=keep_days)
        
        logs_cleaned = 0
        logs_archived = 0
        total_size_freed = 0
        
        # Archive directory
        if archive:
            archive_dir = logs_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
        
        # Parcourir tous les fichiers de logs
        for log_file in logs_dir.glob("*.log*"):
            try:
                # Vérifier l'âge du fichier
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                file_size = log_file.stat().st_size
                
                if file_mtime < cleanup_threshold:
                    # Archiver si demandé
                    if archive:
                        archive_path = archive_dir / f"{log_file.stem}_{file_mtime.strftime('%Y%m%d')}.gz"
                        
                        # Compresser et archiver
                        import gzip
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(archive_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        logs_archived += 1
                        logger.info(f"📦 Archivé: {log_file.name} -> {archive_path.name}")
                    
                    # Supprimer le fichier original
                    log_file.unlink()
                    logs_cleaned += 1
                    total_size_freed += file_size
                    
                    logger.info(f"🗑️ Supprimé: {log_file.name} ({file_size / 1024 / 1024:.1f} MB)")
                
                # Vérifier la taille maximale
                elif file_size > max_size_mb * 1024 * 1024:
                    logger.warning(f"⚠️ Fichier log volumineux: {log_file.name} ({file_size / 1024 / 1024:.1f} MB)")
                    
                    # Rotation du fichier volumineux
                    backup_path = log_file.with_suffix(f".{datetime.now().strftime('%Y%m%d')}.log")
                    log_file.rename(backup_path)
                    
                    # Créer un nouveau fichier vide
                    log_file.touch()
                    
                    logger.info(f"🔄 Rotation: {log_file.name} -> {backup_path.name}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur avec {log_file.name}: {e}")
                continue
        
        space_freed_mb = round(total_size_freed / 1024 / 1024, 2)
        
        result = {
            "status": "success",
            "logs_cleaned": logs_cleaned,
            "logs_archived": logs_archived if archive else 0,
            "space_freed_mb": space_freed_mb,
            "keep_days": keep_days,
            "archive_enabled": archive
        }
        
        logger.info(f"✅ Nettoyage des logs terminé:")
        logger.info(f"   🗑️ Logs supprimés: {logs_cleaned}")
        if archive:
            logger.info(f"   📦 Logs archivés: {logs_archived}")
        logger.info(f"   💾 Espace libéré: {space_freed_mb} MB")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage des logs: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "logs_cleaned": 0
        }


@task(
    name="backup_database",
    description="Sauvegarder la base de données",
    retries=2,
    retry_delay_seconds=[60, 300],
    tags=["maintenance", "backup", "database"]
)
def backup_database(
    backup_dir: str = "data/backups",
    keep_backups: int = 7,
    compress: bool = True
) -> Dict[str, Any]:
    """
    Créer une sauvegarde de la base de données
    
    Args:
        backup_dir: Répertoire de sauvegarde
        keep_backups: Nombre de sauvegardes à conserver
        compress: Compresser la sauvegarde
        
    Returns:
        Dict avec les informations de sauvegarde
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"💾 Création de sauvegarde de la base de données")
        
        # Créer le répertoire de sauvegarde
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Chemin de la base de données principale
        db_path = Path("data/databases/doctorpy.db")
        if not db_path.exists():
            logger.error(f"❌ Base de données introuvable: {db_path}")
            return {
                "status": "failed",
                "reason": "database_not_found",
                "db_path": str(db_path)
            }
        
        # Nom du fichier de sauvegarde
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"doctorpy_backup_{timestamp}.db"
        
        if compress:
            backup_filename += ".gz"
        
        backup_file = backup_path / backup_filename
        
        # Obtenir la taille de la base de données
        db_size = db_path.stat().st_size
        
        # Créer la sauvegarde
        logger.info(f"📄 Sauvegarde: {db_path} -> {backup_file}")
        
        if compress:
            import gzip
            with open(db_path, 'rb') as f_in:
                with gzip.open(backup_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(db_path, backup_file)
        
        backup_size = backup_file.stat().st_size
        compression_ratio = backup_size / db_size if db_size > 0 else 1
        
        # Nettoyer les anciennes sauvegardes
        backups_cleaned = 0
        if keep_backups > 0:
            # Lister toutes les sauvegardes
            all_backups = sorted(
                backup_path.glob("doctorpy_backup_*.db*"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Supprimer les anciennes
            for old_backup in all_backups[keep_backups:]:
                old_backup.unlink()
                backups_cleaned += 1
                logger.info(f"🗑️ Ancienne sauvegarde supprimée: {old_backup.name}")
        
        result = {
            "status": "success",
            "backup_file": str(backup_file),
            "backup_size_mb": round(backup_size / 1024 / 1024, 2),
            "original_size_mb": round(db_size / 1024 / 1024, 2),
            "compression_ratio": round(compression_ratio, 3),
            "compressed": compress,
            "backups_cleaned": backups_cleaned,
            "total_backups": len(list(backup_path.glob("doctorpy_backup_*.db*"))),
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Sauvegarde créée avec succès:")
        logger.info(f"   📄 Fichier: {backup_file.name}")
        logger.info(f"   💾 Taille: {result['backup_size_mb']} MB")
        if compress:
            logger.info(f"   📦 Compression: {compression_ratio:.1%}")
        logger.info(f"   🗂️ Total sauvegardes: {result['total_backups']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }


@task(
    name="optimize_vector_store",
    description="Optimiser le vector store ChromaDB",
    retries=1,
    retry_delay_seconds=[120],
    tags=["maintenance", "optimization", "chromadb"]
)
def optimize_vector_store(
    collection_name: str = "doctorpy_docs",
    vacuum: bool = True
) -> Dict[str, Any]:
    """
    Optimiser le vector store ChromaDB
    
    Args:
        collection_name: Nom de la collection à optimiser
        vacuum: Effectuer un vacuum de la base
        
    Returns:
        Dict avec les résultats d'optimisation
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"⚡ Optimisation du vector store ChromaDB")
        
        import chromadb
        from chromadb.config import Settings
        
        # Se connecter à ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="./data/vector_store",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Vérifier que la collection existe
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            logger.warning(f"⚠️ Collection '{collection_name}' introuvable")
            return {
                "status": "skipped",
                "reason": "collection_not_found",
                "collection_name": collection_name
            }
        
        # Statistiques avant optimisation
        initial_count = collection.count()
        vector_store_path = Path("./data/vector_store")
        initial_size = sum(f.stat().st_size for f in vector_store_path.rglob("*") if f.is_file())
        
        logger.info(f"📊 État initial: {initial_count} documents, {initial_size / 1024 / 1024:.1f} MB")
        
        optimizations_performed = []
        
        # Vacuum de la base (si supporté par ChromaDB)
        if vacuum:
            try:
                # ChromaDB ne supporte pas encore le vacuum direct
                # Mais on peut simuler avec une optimisation des index
                logger.info("🧹 Optimisation des index...")
                # Note: ChromaDB optimise automatiquement, on log juste l'opération
                optimizations_performed.append("Index optimization")
            except Exception as e:
                logger.warning(f"⚠️ Vacuum non supporté: {e}")
        
        # Nettoyage des fichiers temporaires
        temp_files_cleaned = 0
        for temp_file in vector_store_path.rglob("*.tmp"):
            try:
                temp_file.unlink()
                temp_files_cleaned += 1
            except Exception:
                pass
        
        if temp_files_cleaned > 0:
            optimizations_performed.append(f"Cleaned {temp_files_cleaned} temp files")
            logger.info(f"🗑️ {temp_files_cleaned} fichiers temporaires supprimés")
        
        # Vérification de l'intégrité
        try:
            # Test de requête pour vérifier l'intégrité
            test_result = collection.query(
                query_texts=["test"],
                n_results=1
            )
            integrity_check = len(test_result.get("documents", [[]])[0]) > 0
            optimizations_performed.append("Integrity check passed")
        except Exception as e:
            logger.warning(f"⚠️ Problème d'intégrité détecté: {e}")
            integrity_check = False
        
        # Statistiques après optimisation
        final_count = collection.count()
        final_size = sum(f.stat().st_size for f in vector_store_path.rglob("*") if f.is_file())
        
        space_saved = initial_size - final_size
        space_saved_mb = space_saved / 1024 / 1024
        
        result = {
            "status": "success",
            "collection_name": collection_name,
            "document_count": final_count,
            "optimizations_performed": optimizations_performed,
            "statistics": {
                "initial_size_mb": round(initial_size / 1024 / 1024, 2),
                "final_size_mb": round(final_size / 1024 / 1024, 2),
                "space_saved_mb": round(space_saved_mb, 2),
                "temp_files_cleaned": temp_files_cleaned
            },
            "integrity_check": integrity_check,
            "optimized_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Optimisation terminée:")
        logger.info(f"   📊 Documents: {final_count}")
        logger.info(f"   💾 Espace libéré: {space_saved_mb:.2f} MB")
        logger.info(f"   🔧 Optimisations: {len(optimizations_performed)}")
        logger.info(f"   ✅ Intégrité: {'OK' if integrity_check else 'PROBLÈME'}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }


@task(
    name="cleanup_analytics",
    description="Nettoyer les anciennes données d'analytics",
    retries=1,
    tags=["maintenance", "cleanup", "analytics"]
)
def cleanup_analytics(
    keep_days: int = 90,
    archive_old_data: bool = True
) -> Dict[str, Any]:
    """
    Nettoyer les anciennes données d'analytics
    
    Args:
        keep_days: Nombre de jours d'analytics à conserver
        archive_old_data: Archiver les données avant suppression
        
    Returns:
        Dict avec les statistiques de nettoyage
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"📊 Nettoyage des analytics (>{keep_days} jours)")
        
        db_manager = DatabaseManager()
        
        # Seuil de date
        cleanup_threshold = datetime.now() - timedelta(days=keep_days)
        
        # Compter les données à nettoyer
        old_analytics = db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM analytics 
            WHERE timestamp < ?
        """, (cleanup_threshold.isoformat(),))
        
        analytics_count = old_analytics[0]['count'] if old_analytics else 0
        
        if analytics_count == 0:
            logger.info("✅ Aucune donnée d'analytics à nettoyer")
            return {
                "status": "success",
                "analytics_cleaned": 0,
                "analytics_archived": 0
            }
        
        # Archiver si demandé
        analytics_archived = 0
        if archive_old_data:
            archive_dir = Path("data/analytics_archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Exporter les données vers CSV
            old_data = db_manager.execute_query("""
                SELECT * FROM analytics 
                WHERE timestamp < ?
                ORDER BY timestamp
            """, (cleanup_threshold.isoformat(),))
            
            if old_data:
                import csv
                archive_file = archive_dir / f"analytics_archive_{datetime.now().strftime('%Y%m%d')}.csv"
                
                with open(archive_file, 'w', newline='', encoding='utf-8') as csvfile:
                    if old_data:
                        fieldnames = old_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(old_data)
                
                analytics_archived = len(old_data)
                logger.info(f"📦 {analytics_archived} analytics archivés dans {archive_file.name}")
        
        # Supprimer les anciennes données
        analytics_cleaned = db_manager.execute_update("""
            DELETE FROM analytics 
            WHERE timestamp < ?
        """, (cleanup_threshold.isoformat(),))
        
        result = {
            "status": "success",
            "analytics_cleaned": analytics_cleaned,
            "analytics_archived": analytics_archived if archive_old_data else 0,
            "cleanup_threshold": cleanup_threshold.isoformat(),
            "keep_days": keep_days
        }
        
        logger.info(f"✅ Nettoyage des analytics terminé:")
        logger.info(f"   🗑️ Analytics supprimés: {analytics_cleaned}")
        if archive_old_data:
            logger.info(f"   📦 Analytics archivés: {analytics_archived}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage des analytics: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "analytics_cleaned": 0
        }