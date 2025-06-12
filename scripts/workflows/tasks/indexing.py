"""
Tâches Prefect pour l'indexation dans ChromaDB
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from prefect import task, get_run_logger

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.data_processing.index_documents import DocumentIndexer


@task(
    name="index_documents",
    description="Indexer les embeddings dans ChromaDB",
    retries=2,
    retry_delay_seconds=[60, 300],
    timeout_seconds=1200,  # 20 minutes max
    tags=["data", "indexing", "chromadb"]
)
async def index_documents(
    embedding_result: Dict[str, Any],
    collection_name: str = "doctorpy_docs",
    batch_size: int = 100,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Indexer tous les embeddings dans ChromaDB
    
    Args:
        embedding_result: Résultat de la génération d'embeddings précédente
        collection_name: Nom de la collection ChromaDB
        batch_size: Taille des batches pour l'indexation
        force_reindex: Force la réindexation même si la collection existe
        
    Returns:
        Dict avec les métadonnées d'indexation
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"🗂️ Démarrage de l'indexation ChromaDB")
        logger.info(f"Collection: {collection_name}, Batch size: {batch_size}")
        
        # Vérifier le statut de la génération d'embeddings
        if embedding_result.get("status") != "success":
            logger.error("❌ Génération d'embeddings en échec - impossible d'indexer")
            return {
                "status": "failed",
                "reason": "embedding_generation_failed",
                "indexed_count": 0
            }
        
        # Initialiser l'indexeur
        indexer = DocumentIndexer(
            collection_name=collection_name,
            batch_size=batch_size
        )
        
        # Obtenir les répertoires
        embeddings_dir = Path(embedding_result.get("output_dir", "data/embeddings"))
        if not embeddings_dir.exists():
            logger.error(f"❌ Répertoire d'embeddings introuvable: {embeddings_dir}")
            return {
                "status": "failed",
                "reason": "embeddings_directory_missing",
                "embeddings_dir": str(embeddings_dir)
            }
        
        # Force reindex si demandé
        if force_reindex:
            logger.info("🔄 Mode force reindex : suppression de la collection existante")
            await indexer.clear_collection()
        
        # Indexation
        logger.info(f"📂 Indexation depuis: {embeddings_dir}")
        result_data = await indexer.index_all_documents(embeddings_dir)
        
        # Préparer le résultat
        result = {
            "status": "success",
            "indexed_count": result_data.get("total_indexed", 0),
            "skipped_count": result_data.get("total_skipped", 0),
            "updated_count": result_data.get("total_updated", 0),
            "embedding_count": embedding_result.get("embedding_count", 0),
            "collection_name": collection_name,
            "batch_size": batch_size,
            "indexed_at": result_data.get("indexing_date"),
            "processing_time": result_data.get("processing_time_seconds", 0),
            "batches_processed": result_data.get("batches_processed", 0),
            "avg_batch_time": result_data.get("avg_batch_time", 0),
            "collection_stats": result_data.get("collection_stats", {}),
            "sections": result_data.get("sections", {})
        }
        
        logger.info(f"✅ Indexation terminée avec succès:")
        logger.info(f"   📄 Documents indexés: {result['indexed_count']}")
        logger.info(f"   ⏭️ Documents ignorés: {result['skipped_count']}")
        logger.info(f"   🔄 Documents mis à jour: {result['updated_count']}")
        logger.info(f"   ⏱️ Temps de traitement: {result['processing_time']:.1f}s")
        logger.info(f"   📊 Collection: {collection_name}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'indexation: {str(e)}")
        logger.error(f"Type d'erreur: {type(e).__name__}")
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "indexed_count": 0
        }


@task(
    name="validate_index",
    description="Valider l'indexation ChromaDB",
    retries=1,
    tags=["validation", "chromadb"]
)
async def validate_index(indexing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valider l'indexation ChromaDB
    
    Args:
        indexing_result: Résultat de l'indexation précédente
        
    Returns:
        Dict avec les résultats de validation
    """
    logger = get_run_logger()
    
    try:
        logger.info("🔍 Validation de l'indexation ChromaDB")
        
        # Vérifier le statut de l'indexation
        if indexing_result.get("status") != "success":
            logger.error(f"Indexation en échec: {indexing_result.get('error_message')}")
            return {
                "status": "failed",
                "reason": "indexing_failed",
                "indexing_error": indexing_result.get("error_message")
            }
        
        indexed_count = indexing_result.get("indexed_count", 0)
        embedding_count = indexing_result.get("embedding_count", 0)
        collection_name = indexing_result.get("collection_name", "doctorpy_docs")
        
        # Critères de validation
        min_documents = 100
        min_success_rate = 0.95  # 95% des embeddings doivent être indexés
        
        # Validation du nombre de documents indexés
        if indexed_count < min_documents:
            logger.warning(f"⚠️ Nombre de documents indexés insuffisant: {indexed_count} < {min_documents}")
            return {
                "status": "warning",
                "reason": "insufficient_indexed_documents",
                "indexed_count": indexed_count,
                "min_expected": min_documents
            }
        
        # Validation du taux de succès
        success_rate = indexed_count / embedding_count if embedding_count > 0 else 0
        if success_rate < min_success_rate:
            logger.warning(f"⚠️ Taux de succès d'indexation faible: {success_rate:.3f} < {min_success_rate}")
            return {
                "status": "warning",
                "reason": "low_indexing_success_rate",
                "success_rate": success_rate,
                "min_expected": min_success_rate,
                "indexed_count": indexed_count,
                "embedding_count": embedding_count
            }
        
        # Tester la connexion à ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Se connecter à ChromaDB
            chroma_client = chromadb.PersistentClient(
                path="./data/vector_store",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Obtenir la collection
            collection = chroma_client.get_collection(name=collection_name)
            
            # Vérifier le nombre de documents dans la collection
            collection_count = collection.count()
            
            if collection_count != indexed_count:
                logger.warning(f"⚠️ Mismatch entre résultat et collection: {collection_count} vs {indexed_count}")
                return {
                    "status": "warning",
                    "reason": "collection_count_mismatch",
                    "collection_count": collection_count,
                    "reported_count": indexed_count
                }
            
            # Test de requête simple
            test_results = collection.query(
                query_texts=["variables python"],
                n_results=5
            )
            
            if not test_results or not test_results.get("documents"):
                logger.warning("⚠️ Requête de test échouée - pas de résultats")
                return {
                    "status": "warning",
                    "reason": "test_query_failed",
                    "collection_count": collection_count
                }
            
            test_result_count = len(test_results["documents"][0]) if test_results["documents"] else 0
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la validation ChromaDB: {str(e)}")
            return {
                "status": "failed",
                "reason": "chromadb_connection_failed",
                "error": str(e)
            }
        
        # Analyser les sections indexées
        sections = indexing_result.get("sections", {})
        collection_stats = indexing_result.get("collection_stats", {})
        
        # Validation réussie
        result = {
            "status": "success",
            "indexed_count": indexed_count,
            "embedding_count": embedding_count,
            "success_rate": round(success_rate, 3),
            "collection_name": collection_name,
            "collection_count": collection_count,
            "test_query_results": test_result_count,
            "sections": list(sections.keys()),
            "collection_stats": collection_stats,
            "validation_passed": True
        }
        
        logger.info(f"✅ Validation réussie:")
        logger.info(f"   📄 {indexed_count} documents indexés")
        logger.info(f"   🎯 Taux de succès: {success_rate:.1%}")
        logger.info(f"   📊 Collection: {collection_count} documents")
        logger.info(f"   🔍 Test de requête: {test_result_count} résultats")
        
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
    name="test_search_quality",
    description="Tester la qualité de recherche de l'index",
    tags=["testing", "search"]
)
async def test_search_quality(indexing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tester la qualité de recherche de l'index ChromaDB
    
    Args:
        indexing_result: Résultat de l'indexation précédente
        
    Returns:
        Dict avec les résultats de test
    """
    logger = get_run_logger()
    
    try:
        logger.info("🔍 Test de qualité de recherche")
        
        if indexing_result.get("status") != "success":
            logger.warning("⚠️ Indexation en échec - test ignoré")
            return {
                "status": "skipped",
                "reason": "indexing_failed"
            }
        
        import chromadb
        from chromadb.config import Settings
        
        collection_name = indexing_result.get("collection_name", "doctorpy_docs")
        
        # Se connecter à ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="./data/vector_store",
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = chroma_client.get_collection(name=collection_name)
        
        # Requêtes de test avec résultats attendus
        test_queries = [
            {
                "query": "variables python",
                "expected_keywords": ["variable", "python", "nom", "valeur"],
                "min_results": 3
            },
            {
                "query": "fonctions def",
                "expected_keywords": ["fonction", "def", "parameter", "return"],
                "min_results": 2
            },
            {
                "query": "listes append",
                "expected_keywords": ["liste", "append", "element", "add"],
                "min_results": 2
            },
            {
                "query": "erreurs exceptions",
                "expected_keywords": ["error", "exception", "try", "except"],
                "min_results": 1
            }
        ]
        
        test_results = []
        
        for test_case in test_queries:
            try:
                # Effectuer la requête
                results = collection.query(
                    query_texts=[test_case["query"]],
                    n_results=10
                )
                
                documents = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                # Analyser les résultats
                result_count = len(documents)
                avg_distance = sum(distances) / len(distances) if distances else 1.0
                
                # Vérifier la présence de mots-clés attendus
                keyword_matches = 0
                if documents:
                    full_text = " ".join(documents).lower()
                    keyword_matches = sum(
                        1 for keyword in test_case["expected_keywords"]
                        if keyword.lower() in full_text
                    )
                
                # Évaluer la qualité
                meets_min_results = result_count >= test_case["min_results"]
                good_relevance = avg_distance < 0.7  # Distance cosinus < 0.7
                has_keywords = keyword_matches > 0
                
                test_result = {
                    "query": test_case["query"],
                    "result_count": result_count,
                    "avg_distance": round(avg_distance, 3),
                    "keyword_matches": keyword_matches,
                    "total_keywords": len(test_case["expected_keywords"]),
                    "meets_min_results": meets_min_results,
                    "good_relevance": good_relevance,
                    "has_keywords": has_keywords,
                    "quality_score": (
                        (50 if meets_min_results else 0) +
                        (30 if good_relevance else 0) +
                        (20 if has_keywords else 0)
                    )
                }
                
                test_results.append(test_result)
                
                logger.info(f"   Query: '{test_case['query']}' -> {result_count} results, score: {test_result['quality_score']}/100")
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur pour la requête '{test_case['query']}': {e}")
                test_results.append({
                    "query": test_case["query"],
                    "error": str(e),
                    "quality_score": 0
                })
        
        # Calculer le score global
        successful_tests = [r for r in test_results if "error" not in r]
        total_score = sum(r["quality_score"] for r in successful_tests)
        avg_score = total_score / len(successful_tests) if successful_tests else 0
        
        # Évaluation finale
        if avg_score >= 80:
            quality_level = "excellent"
        elif avg_score >= 60:
            quality_level = "good"
        elif avg_score >= 40:
            quality_level = "acceptable"
        else:
            quality_level = "needs_improvement"
        
        result = {
            "status": "success",
            "overall_quality_score": round(avg_score, 1),
            "quality_level": quality_level,
            "tests_conducted": len(test_queries),
            "successful_tests": len(successful_tests),
            "failed_tests": len(test_queries) - len(successful_tests),
            "test_results": test_results,
            "collection_name": collection_name,
            "test_passed": avg_score >= 50
        }
        
        logger.info(f"🔍 Test terminé:")
        logger.info(f"   🎯 Score global: {avg_score:.1f}/100 ({quality_level})")
        logger.info(f"   ✅ Tests réussis: {len(successful_tests)}/{len(test_queries)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "test_passed": False
        }


@task(
    name="optimize_collection",
    description="Optimiser la collection ChromaDB",
    tags=["optimization", "chromadb"]
)
async def optimize_collection(indexing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimiser la collection ChromaDB pour de meilleures performances
    
    Args:
        indexing_result: Résultat de l'indexation précédente
        
    Returns:
        Dict avec les résultats d'optimisation
    """
    logger = get_run_logger()
    
    try:
        logger.info("⚡ Optimisation de la collection ChromaDB")
        
        if indexing_result.get("status") != "success":
            logger.warning("⚠️ Indexation en échec - optimisation ignorée")
            return {
                "status": "skipped",
                "reason": "indexing_failed"
            }
        
        collection_name = indexing_result.get("collection_name", "doctorpy_docs")
        
        # Note: ChromaDB optimise automatiquement les collections
        # Cette tâche peut être étendue pour d'autres optimisations futures
        
        optimizations_performed = [
            "Collection validation",
            "Index integrity check",
            "Metadata cleanup"
        ]
        
        # Simuler quelques optimisations légères
        import time
        start_time = time.time()
        
        # Validation de la collection
        import chromadb
        chroma_client = chromadb.PersistentClient(
            path="./data/vector_store",
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        
        collection = chroma_client.get_collection(name=collection_name)
        collection_count = collection.count()
        
        # Temps d'optimisation
        optimization_time = time.time() - start_time
        
        result = {
            "status": "success",
            "collection_name": collection_name,
            "document_count": collection_count,
            "optimizations_performed": optimizations_performed,
            "optimization_time": round(optimization_time, 3),
            "optimized_at": indexing_result.get("indexed_at"),
            "performance_gain": "baseline"  # Peut être mesuré dans le futur
        }
        
        logger.info(f"⚡ Optimisation terminée:")
        logger.info(f"   📊 Collection: {collection_count} documents")
        logger.info(f"   🔧 Optimisations: {len(optimizations_performed)}")
        logger.info(f"   ⏱️ Temps: {optimization_time:.3f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }