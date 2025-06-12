"""
Tâches Prefect pour la génération d'embeddings
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from prefect import task, get_run_logger

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.data_processing.create_embeddings import EmbeddingGenerator


@task(
    name="create_embeddings",
    description="Générer les embeddings pour les chunks",
    retries=2,
    retry_delay_seconds=[60, 300],
    timeout_seconds=1800,  # 30 minutes max
    tags=["data", "embeddings", "ml"]
)
async def create_embeddings(
    processing_result: Dict[str, Any],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Générer les embeddings pour tous les chunks
    
    Args:
        processing_result: Résultat du traitement précédent
        model_name: Nom du modèle sentence-transformers
        batch_size: Taille des batches pour le traitement
        force_regenerate: Force la régénération même si les embeddings existent
        
    Returns:
        Dict avec les métadonnées de génération
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"🧠 Démarrage de la génération d'embeddings")
        logger.info(f"Modèle: {model_name}, Batch size: {batch_size}")
        
        # Vérifier le statut du traitement
        if processing_result.get("status") != "success":
            logger.error("❌ Traitement en échec - impossible de générer les embeddings")
            return {
                "status": "failed",
                "reason": "processing_failed",
                "embedding_count": 0
            }
        
        # Initialiser le générateur d'embeddings
        generator = EmbeddingGenerator(
            model_name=model_name,
            batch_size=batch_size
        )
        
        # Obtenir le répertoire des chunks
        chunks_dir = Path(processing_result.get("output_dir", "data/processed"))
        if not chunks_dir.exists():
            logger.error(f"❌ Répertoire de chunks introuvable: {chunks_dir}")
            return {
                "status": "failed",
                "reason": "chunks_directory_missing",
                "chunks_dir": str(chunks_dir)
            }
        
        # Force regenerate si demandé
        if force_regenerate:
            logger.info("🔄 Mode force regenerate : suppression du cache existant")
            generator.clear_cache()
        
        # Génération des embeddings
        logger.info(f"📂 Génération depuis: {chunks_dir}")
        result_data = await generator.generate_all_embeddings(chunks_dir)
        
        # Préparer le résultat
        result = {
            "status": "success",
            "embedding_count": result_data.get("total_embeddings", 0),
            "chunk_count": result_data.get("total_chunks", 0),
            "model_name": model_name,
            "embedding_dimension": result_data.get("embedding_dimension", 384),
            "batch_size": batch_size,
            "generated_at": result_data.get("generation_date"),
            "output_dir": str(result_data.get("output_dir", "data/embeddings")),
            "processing_time": result_data.get("processing_time_seconds", 0),
            "batches_processed": result_data.get("batches_processed", 0),
            "avg_batch_time": result_data.get("avg_batch_time", 0),
            "sections": result_data.get("sections", {})
        }
        
        logger.info(f"✅ Génération terminée avec succès:")
        logger.info(f"   🧩 Chunks traités: {result['chunk_count']}")
        logger.info(f"   🧠 Embeddings créés: {result['embedding_count']}")
        logger.info(f"   📐 Dimension: {result['embedding_dimension']}D")
        logger.info(f"   ⏱️ Temps de traitement: {result['processing_time']:.1f}s")
        logger.info(f"   📁 Sauvegardé dans: {result['output_dir']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération d'embeddings: {str(e)}")
        logger.error(f"Type d'erreur: {type(e).__name__}")
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "embedding_count": 0
        }


@task(
    name="validate_embeddings",
    description="Valider la qualité des embeddings générés",
    retries=1,
    tags=["validation", "ml"]
)
def validate_embeddings(embedding_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valider la qualité des embeddings générés
    
    Args:
        embedding_result: Résultat de la génération précédente
        
    Returns:
        Dict avec les résultats de validation
    """
    logger = get_run_logger()
    
    try:
        logger.info("🔍 Validation des embeddings générés")
        
        # Vérifier le statut de la génération
        if embedding_result.get("status") != "success":
            logger.error(f"Génération en échec: {embedding_result.get('error_message')}")
            return {
                "status": "failed",
                "reason": "generation_failed",
                "generation_error": embedding_result.get("error_message")
            }
        
        embedding_count = embedding_result.get("embedding_count", 0)
        chunk_count = embedding_result.get("chunk_count", 0)
        embedding_dimension = embedding_result.get("embedding_dimension", 0)
        
        # Critères de validation
        min_embeddings = 100
        expected_dimension = 384  # Pour all-MiniLM-L6-v2
        
        # Validation du nombre d'embeddings
        if embedding_count < min_embeddings:
            logger.warning(f"⚠️ Nombre d'embeddings insuffisant: {embedding_count} < {min_embeddings}")
            return {
                "status": "warning",
                "reason": "insufficient_embeddings",
                "embedding_count": embedding_count,
                "min_expected": min_embeddings
            }
        
        # Validation de la correspondance chunks/embeddings
        if embedding_count != chunk_count:
            logger.warning(f"⚠️ Mismatch chunks/embeddings: {chunk_count} chunks, {embedding_count} embeddings")
            return {
                "status": "warning",
                "reason": "chunk_embedding_mismatch",
                "chunk_count": chunk_count,
                "embedding_count": embedding_count
            }
        
        # Validation de la dimension
        if embedding_dimension != expected_dimension:
            logger.warning(f"⚠️ Dimension inattendue: {embedding_dimension} != {expected_dimension}")
            return {
                "status": "warning",
                "reason": "unexpected_dimension",
                "actual_dimension": embedding_dimension,
                "expected_dimension": expected_dimension
            }
        
        # Validation du répertoire de sortie
        output_dir = Path(embedding_result.get("output_dir", ""))
        if not output_dir.exists():
            logger.error(f"❌ Répertoire de sortie introuvable: {output_dir}")
            return {
                "status": "failed",
                "reason": "output_directory_missing",
                "output_dir": str(output_dir)
            }
        
        # Compter les fichiers créés
        npy_files = list(output_dir.glob("*.npy"))
        json_files = list(output_dir.glob("*.json"))
        
        if not npy_files:
            logger.error("❌ Aucun fichier d'embeddings (.npy) trouvé")
            return {
                "status": "failed",
                "reason": "no_embedding_files_found",
                "output_dir": str(output_dir)
            }
        
        # Vérifier la taille des fichiers embeddings
        import numpy as np
        total_embeddings_in_files = 0
        
        for npy_file in npy_files[:5]:  # Vérifier les 5 premiers fichiers
            try:
                embeddings = np.load(npy_file)
                total_embeddings_in_files += len(embeddings)
                
                # Vérifier la dimension
                if len(embeddings.shape) != 2 or embeddings.shape[1] != embedding_dimension:
                    logger.warning(f"⚠️ Dimension incorrecte dans {npy_file.name}: {embeddings.shape}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors de la lecture de {npy_file.name}: {e}")
        
        # Validation de la qualité des données
        sections = embedding_result.get("sections", {})
        processing_time = embedding_result.get("processing_time", 0)
        avg_batch_time = embedding_result.get("avg_batch_time", 0)
        
        # Validation réussie
        result = {
            "status": "success",
            "embedding_count": embedding_count,
            "chunk_count": chunk_count,
            "embedding_dimension": embedding_dimension,
            "file_count": {
                "embeddings": len(npy_files),
                "metadata": len(json_files)
            },
            "sample_files_checked": min(5, len(npy_files)),
            "performance": {
                "total_time": processing_time,
                "avg_batch_time": avg_batch_time,
                "embeddings_per_second": round(embedding_count / processing_time, 2) if processing_time > 0 else 0
            },
            "sections": list(sections.keys()),
            "output_dir": str(output_dir),
            "validation_passed": True
        }
        
        logger.info(f"✅ Validation réussie:")
        logger.info(f"   🧠 {embedding_count} embeddings valides")
        logger.info(f"   📐 Dimension: {embedding_dimension}D")
        logger.info(f"   📁 {len(npy_files)} fichiers créés")
        logger.info(f"   ⚡ Performance: {result['performance']['embeddings_per_second']} embeddings/sec")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la validation: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "validation_passed": False
        }


@task(
    name="test_embedding_similarity",
    description="Tester la qualité des embeddings avec des similarités",
    tags=["testing", "similarity"]
)
def test_embedding_similarity(embedding_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tester la qualité des embeddings en calculant des similarités
    
    Args:
        embedding_result: Résultat de la génération précédente
        
    Returns:
        Dict avec les résultats de test
    """
    logger = get_run_logger()
    
    try:
        logger.info("🧪 Test de qualité des embeddings par similarité")
        
        if embedding_result.get("status") != "success":
            logger.warning("⚠️ Génération en échec - test ignoré")
            return {
                "status": "skipped",
                "reason": "generation_failed"
            }
        
        import numpy as np
        import json
        from sklearn.metrics.pairwise import cosine_similarity
        
        output_dir = Path(embedding_result.get("output_dir", "data/embeddings"))
        
        # Charger un échantillon d'embeddings pour test
        npy_files = list(output_dir.glob("*.npy"))
        if not npy_files:
            logger.warning("⚠️ Aucun fichier d'embeddings trouvé pour le test")
            return {
                "status": "skipped",
                "reason": "no_embedding_files"
            }
        
        # Prendre le premier fichier comme échantillon
        sample_file = npy_files[0]
        embeddings = np.load(sample_file)
        
        if len(embeddings) < 10:
            logger.warning("⚠️ Échantillon trop petit pour le test")
            return {
                "status": "skipped",
                "reason": "sample_too_small"
            }
        
        # Prendre un échantillon aléatoire de 10 embeddings
        sample_indices = np.random.choice(len(embeddings), min(10, len(embeddings)), replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        # Calculer les similarités cosinus
        similarities = cosine_similarity(sample_embeddings)
        
        # Analyser les résultats
        # Similarités avec soi-même (diagonale) doivent être 1.0
        self_similarities = np.diag(similarities)
        
        # Similarités avec les autres (hors diagonale)
        mask = np.ones_like(similarities, dtype=bool)
        np.fill_diagonal(mask, False)
        other_similarities = similarities[mask]
        
        # Calculer les statistiques
        stats = {
            "self_similarity": {
                "mean": float(np.mean(self_similarities)),
                "std": float(np.std(self_similarities)),
                "min": float(np.min(self_similarities)),
                "max": float(np.max(self_similarities))
            },
            "other_similarity": {
                "mean": float(np.mean(other_similarities)),
                "std": float(np.std(other_similarities)),
                "min": float(np.min(other_similarities)),
                "max": float(np.max(other_similarities))
            }
        }
        
        # Évaluation de la qualité
        quality_issues = []
        quality_score = 100
        
        # Les auto-similarités doivent être proches de 1.0
        if stats["self_similarity"]["mean"] < 0.99:
            quality_score -= 20
            quality_issues.append("Auto-similarités faibles (< 0.99)")
        
        # Les similarités croisées doivent être variées (pas toutes identiques)
        if stats["other_similarity"]["std"] < 0.1:
            quality_score -= 15
            quality_issues.append("Embeddings peu discriminants (std < 0.1)")
        
        # Les similarités croisées ne doivent pas être trop élevées (pas de sur-clustering)
        if stats["other_similarity"]["mean"] > 0.8:
            quality_score -= 10
            quality_issues.append("Embeddings trop similaires (mean > 0.8)")
        
        result = {
            "status": "success",
            "quality_score": quality_score,
            "quality_issues": quality_issues,
            "sample_size": len(sample_embeddings),
            "embedding_dimension": embeddings.shape[1],
            "similarity_stats": stats,
            "tested_file": sample_file.name,
            "test_passed": quality_score >= 70
        }
        
        logger.info(f"🧪 Test terminé:")
        logger.info(f"   🎯 Score de qualité: {quality_score}/100")
        logger.info(f"   🔍 Auto-similarité moyenne: {stats['self_similarity']['mean']:.3f}")
        logger.info(f"   📊 Similarité croisée: {stats['other_similarity']['mean']:.3f} ± {stats['other_similarity']['std']:.3f}")
        
        if quality_issues:
            logger.warning(f"⚠️ Problèmes détectés: {', '.join(quality_issues)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "test_passed": False
        }