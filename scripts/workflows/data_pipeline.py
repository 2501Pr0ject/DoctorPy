"""
Pipeline de données RAG complet avec Prefect

Ce module définit le workflow principal de mise à jour de la base de connaissances :
1. Scraping de la documentation Python
2. Traitement et chunking des documents
3. Génération d'embeddings
4. Indexation dans ChromaDB
5. Validation et tests de qualité
"""

from typing import Dict, Any, Optional
from prefect import flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from .tasks.scraping import scrape_python_docs, validate_scraped_docs, check_scraping_freshness
from .tasks.processing import process_documents, validate_chunks, analyze_chunk_quality
from .tasks.embedding import create_embeddings, validate_embeddings, test_embedding_similarity
from .tasks.indexing import index_documents, validate_index, test_search_quality, optimize_collection
from .tasks.notification import send_notification, send_alert


@flow(
    name="update_knowledge_base",
    description="Mise à jour complète de la base de connaissances RAG",
    version="1.0",
    task_runner=ConcurrentTaskRunner(max_workers=2),
    timeout_seconds=7200,  # 2 heures max
    retries=1,
    retry_delay_seconds=300
)
async def update_knowledge_base(
    force_refresh: bool = False,
    force_reprocess: bool = False,
    force_regenerate_embeddings: bool = False,
    force_reindex: bool = False,
    skip_validation: bool = False,
    max_documents: Optional[int] = None,
    notification_channels: list = ["log"]
) -> Dict[str, Any]:
    """
    Pipeline complet de mise à jour de la base de connaissances
    
    Args:
        force_refresh: Force le re-scraping même si les documents sont récents
        force_reprocess: Force le retraitement des documents
        force_regenerate_embeddings: Force la régénération des embeddings
        force_reindex: Force la réindexation ChromaDB
        skip_validation: Ignore les validations (mode rapide)
        max_documents: Limite le nombre de documents (pour tests)
        notification_channels: Canaux de notification ["log", "email", "slack"]
        
    Returns:
        Dict avec le résumé complet du pipeline
    """
    logger = get_run_logger()
    
    logger.info("🚀 Démarrage du pipeline de mise à jour de la base de connaissances")
    logger.info(f"Options: force_refresh={force_refresh}, force_reprocess={force_reprocess}")
    logger.info(f"         force_regenerate={force_regenerate_embeddings}, force_reindex={force_reindex}")
    
    pipeline_start_time = time.time()
    pipeline_results = {
        "pipeline_version": "1.0",
        "started_at": datetime.now().isoformat(),
        "configuration": {
            "force_refresh": force_refresh,
            "force_reprocess": force_reprocess,
            "force_regenerate_embeddings": force_regenerate_embeddings,
            "force_reindex": force_reindex,
            "skip_validation": skip_validation,
            "max_documents": max_documents
        },
        "stages": {}
    }
    
    try:
        # ===== ÉTAPE 1: VÉRIFICATION DE FRAÎCHEUR =====
        logger.info("📋 Étape 1: Vérification de la fraîcheur des données")
        
        if not force_refresh:
            freshness_check = check_scraping_freshness(max_age_hours=24)
            
            if not freshness_check.get("needs_scraping", True):
                logger.info("✅ Données récentes détectées - scraping ignoré")
                # Charger les résultats du dernier scraping pour continuer le pipeline
                scraping_result = {
                    "status": "success",
                    "document_count": freshness_check.get("document_count", 0),
                    "output_dir": "data/raw",
                    "from_cache": True
                }
            else:
                logger.info("🔄 Données obsolètes - scraping nécessaire")
                scraping_result = await scrape_python_docs(
                    force_refresh=True,
                    max_documents=max_documents
                )
        else:
            scraping_result = await scrape_python_docs(
                force_refresh=force_refresh,
                max_documents=max_documents
            )
        
        pipeline_results["stages"]["scraping"] = scraping_result
        
        # Validation du scraping (si activée)
        if not skip_validation and scraping_result.get("status") == "success":
            validation_result = validate_scraped_docs(scraping_result)
            pipeline_results["stages"]["scraping_validation"] = validation_result
            
            if validation_result.get("status") == "failed":
                error_msg = f"Validation du scraping échouée: {validation_result.get('reason')}"
                logger.error(f"❌ {error_msg}")
                await send_alert(
                    title="Échec validation scraping",
                    message=error_msg,
                    channels=notification_channels
                )
                pipeline_results["status"] = "failed"
                pipeline_results["error_stage"] = "scraping_validation"
                return pipeline_results
        
        # ===== ÉTAPE 2: TRAITEMENT DES DOCUMENTS =====
        logger.info("📄 Étape 2: Traitement et chunking des documents")
        
        processing_result = await process_documents(
            scraping_result=scraping_result,
            chunk_size=500,
            chunk_overlap=100,
            force_reprocess=force_reprocess
        )
        
        pipeline_results["stages"]["processing"] = processing_result
        
        if processing_result.get("status") != "success":
            error_msg = f"Traitement des documents échoué: {processing_result.get('error_message')}"
            logger.error(f"❌ {error_msg}")
            await send_alert(
                title="Échec traitement documents",
                message=error_msg,
                channels=notification_channels
            )
            pipeline_results["status"] = "failed"
            pipeline_results["error_stage"] = "processing"
            return pipeline_results
        
        # Validation et analyse des chunks (en parallèle si validation activée)
        if not skip_validation:
            chunk_validation, chunk_analysis = await asyncio.gather(
                validate_chunks(processing_result),
                analyze_chunk_quality(processing_result),
                return_exceptions=True
            )
            
            pipeline_results["stages"]["chunk_validation"] = chunk_validation
            pipeline_results["stages"]["chunk_analysis"] = chunk_analysis
        
        # ===== ÉTAPE 3: GÉNÉRATION D'EMBEDDINGS =====
        logger.info("🧠 Étape 3: Génération des embeddings")
        
        embedding_result = await create_embeddings(
            processing_result=processing_result,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            force_regenerate=force_regenerate_embeddings
        )
        
        pipeline_results["stages"]["embeddings"] = embedding_result
        
        if embedding_result.get("status") != "success":
            error_msg = f"Génération d'embeddings échouée: {embedding_result.get('error_message')}"
            logger.error(f"❌ {error_msg}")
            await send_alert(
                title="Échec génération embeddings",
                message=error_msg,
                channels=notification_channels
            )
            pipeline_results["status"] = "failed"
            pipeline_results["error_stage"] = "embeddings"
            return pipeline_results
        
        # Validation et test des embeddings (en parallèle si validation activée)
        if not skip_validation:
            embedding_validation, embedding_test = await asyncio.gather(
                validate_embeddings(embedding_result),
                test_embedding_similarity(embedding_result),
                return_exceptions=True
            )
            
            pipeline_results["stages"]["embedding_validation"] = embedding_validation
            pipeline_results["stages"]["embedding_test"] = embedding_test
        
        # ===== ÉTAPE 4: INDEXATION CHROMADB =====
        logger.info("🗂️ Étape 4: Indexation dans ChromaDB")
        
        indexing_result = await index_documents(
            embedding_result=embedding_result,
            collection_name="doctorpy_docs",
            batch_size=100,
            force_reindex=force_reindex
        )
        
        pipeline_results["stages"]["indexing"] = indexing_result
        
        if indexing_result.get("status") != "success":
            error_msg = f"Indexation ChromaDB échouée: {indexing_result.get('error_message')}"
            logger.error(f"❌ {error_msg}")
            await send_alert(
                title="Échec indexation ChromaDB",
                message=error_msg,
                channels=notification_channels
            )
            pipeline_results["status"] = "failed"
            pipeline_results["error_stage"] = "indexing"
            return pipeline_results
        
        # Validation, test et optimisation de l'index (en parallèle si validation activée)
        if not skip_validation:
            index_validation, search_test, optimization = await asyncio.gather(
                validate_index(indexing_result),
                test_search_quality(indexing_result),
                optimize_collection(indexing_result),
                return_exceptions=True
            )
            
            pipeline_results["stages"]["index_validation"] = index_validation
            pipeline_results["stages"]["search_test"] = search_test
            pipeline_results["stages"]["optimization"] = optimization
        
        # ===== FINALISATION =====
        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        
        pipeline_results.update({
            "status": "success",
            "completed_at": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 2),
            "summary": {
                "documents_scraped": scraping_result.get("document_count", 0),
                "chunks_created": processing_result.get("chunk_count", 0),
                "embeddings_generated": embedding_result.get("embedding_count", 0),
                "documents_indexed": indexing_result.get("indexed_count", 0),
                "collection_name": indexing_result.get("collection_name", "doctorpy_docs")
            }
        })
        
        # Notification de succès
        success_message = (
            f"Pipeline RAG terminé avec succès en {total_time:.1f}s\n"
            f"📄 {pipeline_results['summary']['documents_scraped']} documents scrapés\n"
            f"🧩 {pipeline_results['summary']['chunks_created']} chunks créés\n" 
            f"🧠 {pipeline_results['summary']['embeddings_generated']} embeddings générés\n"
            f"🗂️ {pipeline_results['summary']['documents_indexed']} documents indexés"
        )
        
        logger.info(f"✅ {success_message}")
        
        await send_notification(
            title="Pipeline RAG terminé avec succès",
            message=success_message,
            channels=notification_channels
        )
        
        return pipeline_results
        
    except Exception as e:
        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        
        error_msg = f"Erreur critique dans le pipeline: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        pipeline_results.update({
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "failed_at": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 2)
        })
        
        await send_alert(
            title="Échec critique du pipeline RAG",
            message=f"{error_msg}\nTemps écoulé: {total_time:.1f}s",
            channels=notification_channels
        )
        
        return pipeline_results


@flow(
    name="rag_quick_update",
    description="Mise à jour rapide de la base de connaissances (sans validation)",
    version="1.0",
    timeout_seconds=3600  # 1 heure max
)
async def rag_quick_update(
    force_refresh: bool = False,
    max_documents: Optional[int] = None
) -> Dict[str, Any]:
    """
    Version rapide du pipeline RAG sans validations détaillées
    
    Args:
        force_refresh: Force le re-scraping
        max_documents: Limite le nombre de documents
        
    Returns:
        Dict avec le résumé du pipeline
    """
    logger = get_run_logger()
    
    logger.info("⚡ Démarrage du pipeline RAG rapide")
    
    return await update_knowledge_base(
        force_refresh=force_refresh,
        force_reprocess=False,
        force_regenerate_embeddings=False,
        force_reindex=False,
        skip_validation=True,  # Mode rapide sans validation
        max_documents=max_documents,
        notification_channels=["log"]
    )


@flow(
    name="rag_full_pipeline",
    description="Pipeline RAG complet avec toutes les validations",
    version="1.0",
    timeout_seconds=10800  # 3 heures max
)
async def rag_full_pipeline(
    force_all: bool = False,
    notification_channels: list = ["log", "email"]
) -> Dict[str, Any]:
    """
    Version complète du pipeline RAG avec toutes les validations et optimisations
    
    Args:
        force_all: Force toutes les étapes (scraping, processing, embeddings, indexing)
        notification_channels: Canaux de notification
        
    Returns:
        Dict avec le résumé complet du pipeline
    """
    logger = get_run_logger()
    
    logger.info("🏗️ Démarrage du pipeline RAG complet")
    
    return await update_knowledge_base(
        force_refresh=force_all,
        force_reprocess=force_all,
        force_regenerate_embeddings=force_all,
        force_reindex=force_all,
        skip_validation=False,  # Toutes les validations
        max_documents=None,
        notification_channels=notification_channels
    )


# Imports nécessaires pour les flow
import time
import asyncio
from datetime import datetime