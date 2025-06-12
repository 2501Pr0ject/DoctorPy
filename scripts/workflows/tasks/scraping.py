"""
Tâches Prefect pour le scraping de données
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from prefect import task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.data_processing.scrape_docs import DocumentScraper


@task(
    name="scrape_python_docs",
    description="Scraper la documentation Python officielle",
    retries=3,
    retry_delay_seconds=[60, 180, 300],  # Backoff progressif
    timeout_seconds=1800,  # 30 minutes max
    tags=["data", "scraping", "python-docs"]
)
async def scrape_python_docs(
    force_refresh: bool = False,
    max_documents: int = None
) -> Dict[str, Any]:
    """
    Scraper la documentation Python avec gestion d'erreurs robuste
    
    Args:
        force_refresh: Force le re-scraping même si les documents existent
        max_documents: Limite le nombre de documents (pour tests)
        
    Returns:
        Dict avec les métadonnées du scraping
    """
    logger = get_run_logger()
    
    try:
        logger.info("🕷️ Démarrage du scraping de la documentation Python")
        
        # Initialiser le scraper
        scraper = DocumentScraper()
        
        # Configuration conditionnelle pour les tests
        if max_documents:
            logger.info(f"Mode test : limitation à {max_documents} documents")
            scraper.max_documents = max_documents
        
        # Scraping avec force refresh si demandé
        if force_refresh:
            logger.info("Mode force refresh : suppression du cache existant")
            scraper.clear_cache()
        
        # Exécuter le scraping
        documents = await scraper.scrape_all_documentation()
        
        # Résultats
        result = {
            "status": "success",
            "document_count": len(documents),
            "scraped_at": scraper.get_timestamp(),
            "output_dir": str(scraper.output_dir),
            "force_refresh": force_refresh,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "url": doc.url,
                    "section": doc.section,
                    "word_count": len(doc.content.split()) if doc.content else 0
                }
                for doc in documents[:10]  # Premières 10 pour les logs
            ]
        }
        
        logger.info(f"✅ Scraping terminé avec succès: {len(documents)} documents")
        logger.info(f"📁 Sauvegardé dans: {scraper.output_dir}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du scraping: {str(e)}")
        logger.error(f"Type d'erreur: {type(e).__name__}")
        
        # Retourner les informations d'erreur pour le debugging
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "document_count": 0
        }


@task(
    name="validate_scraped_docs",
    description="Valider les documents scrapés",
    retries=1,
    tags=["validation", "data-quality"]
)
def validate_scraped_docs(scraping_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valider la qualité des documents scrapés
    
    Args:
        scraping_result: Résultat du scraping précédent
        
    Returns:
        Dict avec les résultats de validation
    """
    logger = get_run_logger()
    
    try:
        logger.info("🔍 Validation des documents scrapés")
        
        # Vérifier le statut du scraping
        if scraping_result.get("status") != "success":
            logger.error(f"Scraping en échec: {scraping_result.get('error_message')}")
            return {
                "status": "failed",
                "reason": "scraping_failed",
                "scraping_error": scraping_result.get("error_message")
            }
        
        document_count = scraping_result.get("document_count", 0)
        
        # Critères de validation
        min_documents = 30  # Minimum attendu
        min_avg_words = 100  # Minimum de mots par document
        
        # Validation du nombre de documents
        if document_count < min_documents:
            logger.warning(f"⚠️ Nombre de documents insuffisant: {document_count} < {min_documents}")
            return {
                "status": "warning",
                "reason": "insufficient_documents",
                "document_count": document_count,
                "min_expected": min_documents
            }
        
        # Validation de la qualité des documents
        documents = scraping_result.get("documents", [])
        if documents:
            avg_words = sum(doc.get("word_count", 0) for doc in documents) / len(documents)
            
            if avg_words < min_avg_words:
                logger.warning(f"⚠️ Qualité des documents faible: {avg_words:.1f} mots en moyenne")
                return {
                    "status": "warning", 
                    "reason": "low_quality_documents",
                    "avg_word_count": avg_words,
                    "min_expected": min_avg_words
                }
        
        # Validation du répertoire de sortie
        output_dir = Path(scraping_result.get("output_dir", ""))
        if not output_dir.exists():
            logger.error(f"❌ Répertoire de sortie introuvable: {output_dir}")
            return {
                "status": "failed",
                "reason": "output_directory_missing",
                "output_dir": str(output_dir)
            }
        
        # Compter les fichiers réellement créés
        md_files = list(output_dir.glob("*.md"))
        json_files = list(output_dir.glob("*.json"))
        
        logger.info(f"📄 Fichiers créés: {len(md_files)} .md, {len(json_files)} .json")
        
        # Validation réussie
        result = {
            "status": "success",
            "document_count": document_count,
            "file_count": {
                "markdown": len(md_files),
                "json": len(json_files)
            },
            "output_dir": str(output_dir),
            "validation_passed": True
        }
        
        logger.info(f"✅ Validation réussie: {document_count} documents valides")
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
    name="check_scraping_freshness",
    description="Vérifier si le scraping est nécessaire",
    tags=["optimization", "cache"]
)
def check_scraping_freshness(max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Vérifier si les données scrapées sont récentes
    
    Args:
        max_age_hours: Age maximum des données en heures
        
    Returns:
        Dict avec les informations de fraîcheur
    """
    logger = get_run_logger()
    
    try:
        from datetime import datetime, timedelta
        import json
        
        # Chemin vers les métadonnées du dernier scraping
        metadata_file = Path("data/raw/scraping_metadata.json")
        
        if not metadata_file.exists():
            logger.info("📋 Aucune métadonnée de scraping trouvée - scraping nécessaire")
            return {
                "needs_scraping": True,
                "reason": "no_previous_scraping",
                "last_scraping": None
            }
        
        # Lire les métadonnées
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        last_scraping = datetime.fromisoformat(metadata.get("scraped_at", ""))
        age_hours = (datetime.now() - last_scraping).total_seconds() / 3600
        
        needs_scraping = age_hours > max_age_hours
        
        result = {
            "needs_scraping": needs_scraping,
            "reason": "data_too_old" if needs_scraping else "data_fresh",
            "last_scraping": metadata.get("scraped_at"),
            "age_hours": round(age_hours, 2),
            "max_age_hours": max_age_hours,
            "document_count": metadata.get("document_count", 0)
        }
        
        if needs_scraping:
            logger.info(f"🔄 Scraping nécessaire - données vieilles de {age_hours:.1f}h")
        else:
            logger.info(f"✅ Données récentes - {age_hours:.1f}h (< {max_age_hours}h)")
        
        return result
        
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors de la vérification: {str(e)}")
        # En cas d'erreur, on recommande le scraping par sécurité
        return {
            "needs_scraping": True,
            "reason": "check_failed",
            "error": str(e)
        }