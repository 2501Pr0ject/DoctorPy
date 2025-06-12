"""
Tâches Prefect pour le traitement de documents
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from prefect import task, get_run_logger

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.data_processing.process_documents import DocumentProcessor


@task(
    name="process_documents",
    description="Traiter et chunker les documents scrapés",
    retries=2,
    retry_delay_seconds=[30, 120],
    timeout_seconds=900,  # 15 minutes max
    tags=["data", "processing", "chunking"]
)
async def process_documents(
    scraping_result: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    force_reprocess: bool = False
) -> Dict[str, Any]:
    """
    Traiter les documents scrapés en chunks pour le RAG
    
    Args:
        scraping_result: Résultat du scraping précédent
        chunk_size: Taille des chunks en tokens
        chunk_overlap: Overlap entre chunks en caractères
        force_reprocess: Force le retraitement même si les chunks existent
        
    Returns:
        Dict avec les métadonnées du traitement
    """
    logger = get_run_logger()
    
    try:
        logger.info(f"📄 Démarrage du traitement de documents")
        logger.info(f"Paramètres: chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Vérifier le statut du scraping
        if scraping_result.get("status") != "success":
            logger.error("❌ Scraping en échec - impossible de traiter les documents")
            return {
                "status": "failed",
                "reason": "scraping_failed",
                "chunk_count": 0
            }
        
        # Initialiser le processeur
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Obtenir le répertoire des documents sources
        source_dir = Path(scraping_result.get("output_dir", "data/raw"))
        if not source_dir.exists():
            logger.error(f"❌ Répertoire source introuvable: {source_dir}")
            return {
                "status": "failed",
                "reason": "source_directory_missing",
                "source_dir": str(source_dir)
            }
        
        # Force reprocess si demandé
        if force_reprocess:
            logger.info("🔄 Mode force reprocess : suppression du cache existant")
            processor.clear_cache()
        
        # Traitement des documents
        logger.info(f"📂 Traitement des documents depuis: {source_dir}")
        result_data = await processor.process_all_documents(source_dir)
        
        # Préparer le résultat
        result = {
            "status": "success",
            "chunk_count": result_data.get("total_chunks", 0),
            "document_count": result_data.get("total_documents", 0),
            "skipped_files": result_data.get("skipped_files", 0),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "processed_at": result_data.get("processing_date"),
            "output_dir": str(result_data.get("output_dir", "data/processed")),
            "sections": result_data.get("sections", {}),
            "processing_time": result_data.get("processing_time_seconds", 0)
        }
        
        logger.info(f"✅ Traitement terminé avec succès:")
        logger.info(f"   📄 Documents traités: {result['document_count']}")
        logger.info(f"   🧩 Chunks créés: {result['chunk_count']}")
        logger.info(f"   ⏱️ Temps de traitement: {result['processing_time']:.1f}s")
        logger.info(f"   📁 Sauvegardé dans: {result['output_dir']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement: {str(e)}")
        logger.error(f"Type d'erreur: {type(e).__name__}")
        
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "chunk_count": 0
        }


@task(
    name="validate_chunks",
    description="Valider la qualité des chunks créés",
    retries=1,
    tags=["validation", "data-quality"]
)
def validate_chunks(processing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valider la qualité des chunks créés
    
    Args:
        processing_result: Résultat du traitement précédent
        
    Returns:
        Dict avec les résultats de validation
    """
    logger = get_run_logger()
    
    try:
        logger.info("🔍 Validation des chunks créés")
        
        # Vérifier le statut du traitement
        if processing_result.get("status") != "success":
            logger.error(f"Traitement en échec: {processing_result.get('error_message')}")
            return {
                "status": "failed",
                "reason": "processing_failed",
                "processing_error": processing_result.get("error_message")
            }
        
        chunk_count = processing_result.get("chunk_count", 0)
        document_count = processing_result.get("document_count", 0)
        
        # Critères de validation
        min_chunks = 100  # Minimum de chunks attendus
        min_ratio = 5     # Minimum 5 chunks par document
        max_ratio = 100   # Maximum 100 chunks par document (détection over-chunking)
        
        # Validation du nombre de chunks
        if chunk_count < min_chunks:
            logger.warning(f"⚠️ Nombre de chunks insuffisant: {chunk_count} < {min_chunks}")
            return {
                "status": "warning",
                "reason": "insufficient_chunks",
                "chunk_count": chunk_count,
                "min_expected": min_chunks
            }
        
        # Validation du ratio chunks/documents
        if document_count > 0:
            ratio = chunk_count / document_count
            
            if ratio < min_ratio:
                logger.warning(f"⚠️ Ratio chunks/documents trop faible: {ratio:.1f} < {min_ratio}")
                return {
                    "status": "warning",
                    "reason": "low_chunk_ratio",
                    "chunk_document_ratio": ratio,
                    "min_expected": min_ratio
                }
            
            if ratio > max_ratio:
                logger.warning(f"⚠️ Ratio chunks/documents trop élevé: {ratio:.1f} > {max_ratio}")
                return {
                    "status": "warning",
                    "reason": "high_chunk_ratio", 
                    "chunk_document_ratio": ratio,
                    "max_expected": max_ratio
                }
        
        # Validation du répertoire de sortie
        output_dir = Path(processing_result.get("output_dir", ""))
        if not output_dir.exists():
            logger.error(f"❌ Répertoire de sortie introuvable: {output_dir}")
            return {
                "status": "failed",
                "reason": "output_directory_missing",
                "output_dir": str(output_dir)
            }
        
        # Compter les fichiers créés
        json_files = list(output_dir.glob("*.json"))
        
        if not json_files:
            logger.error("❌ Aucun fichier de chunks trouvé")
            return {
                "status": "failed",
                "reason": "no_chunk_files_found",
                "output_dir": str(output_dir)
            }
        
        # Vérifier la taille des fichiers (détection fichiers vides)
        empty_files = [f for f in json_files if f.stat().st_size < 100]  # < 100 bytes
        if empty_files:
            logger.warning(f"⚠️ {len(empty_files)} fichiers suspects (< 100 bytes)")
        
        # Validation des sections
        sections = processing_result.get("sections", {})
        if not sections:
            logger.warning("⚠️ Aucune section détectée dans les documents")
        
        # Validation réussie
        result = {
            "status": "success",
            "chunk_count": chunk_count,
            "document_count": document_count,
            "chunk_document_ratio": round(chunk_count / document_count, 2) if document_count > 0 else 0,
            "file_count": len(json_files),
            "empty_files": len(empty_files),
            "sections": list(sections.keys()),
            "output_dir": str(output_dir),
            "validation_passed": True
        }
        
        logger.info(f"✅ Validation réussie:")
        logger.info(f"   🧩 {chunk_count} chunks valides")
        logger.info(f"   📄 Ratio: {result['chunk_document_ratio']} chunks/document")
        logger.info(f"   📁 {len(json_files)} fichiers créés")
        
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
    name="analyze_chunk_quality",
    description="Analyser la qualité des chunks en détail",
    tags=["analysis", "quality"]
)
def analyze_chunk_quality(processing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyser la qualité des chunks créés en détail
    
    Args:
        processing_result: Résultat du traitement précédent
        
    Returns:
        Dict avec l'analyse de qualité détaillée
    """
    logger = get_run_logger()
    
    try:
        logger.info("📊 Analyse détaillée de la qualité des chunks")
        
        if processing_result.get("status") != "success":
            logger.warning("⚠️ Traitement en échec - analyse limitée")
            return {
                "status": "skipped",
                "reason": "processing_failed"
            }
        
        import json
        
        output_dir = Path(processing_result.get("output_dir", "data/processed"))
        
        # Analyser les chunks depuis le rapport
        report_file = output_dir / "processing_report.json"
        if not report_file.exists():
            logger.warning("⚠️ Rapport de traitement introuvable")
            return {
                "status": "partial",
                "reason": "no_report_file"
            }
        
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Calculer les statistiques
        sections = report.get("sections", {})
        total_tokens = sum(section.get("total_tokens", 0) for section in sections.values())
        total_chunks = sum(section.get("chunk_count", 0) for section in sections.values())
        
        avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks > 0 else 0
        
        # Analyser la distribution des sections
        section_distribution = {
            name: {
                "chunk_count": section.get("chunk_count", 0),
                "document_count": section.get("document_count", 0),
                "total_tokens": section.get("total_tokens", 0),
                "percentage": round((section.get("chunk_count", 0) / total_chunks) * 100, 1) if total_chunks > 0 else 0
            }
            for name, section in sections.items()
        }
        
        # Évaluer la qualité
        quality_score = 100
        quality_issues = []
        
        # Vérifier l'équilibre des sections
        if len(sections) > 1:
            percentages = [dist["percentage"] for dist in section_distribution.values()]
            max_percentage = max(percentages) if percentages else 0
            
            if max_percentage > 80:
                quality_score -= 20
                quality_issues.append("Une section domine (> 80% des chunks)")
        
        # Vérifier la taille moyenne des chunks
        if avg_tokens_per_chunk < 200:
            quality_score -= 15
            quality_issues.append("Chunks trop petits (< 200 tokens en moyenne)")
        elif avg_tokens_per_chunk > 800:
            quality_score -= 10
            quality_issues.append("Chunks trop grands (> 800 tokens en moyenne)")
        
        # Évaluation finale
        if quality_score >= 90:
            quality_level = "excellent"
        elif quality_score >= 75:
            quality_level = "good"
        elif quality_score >= 60:
            quality_level = "acceptable"
        else:
            quality_level = "needs_improvement"
        
        result = {
            "status": "success",
            "quality_score": quality_score,
            "quality_level": quality_level,
            "quality_issues": quality_issues,
            "statistics": {
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "avg_tokens_per_chunk": round(avg_tokens_per_chunk, 1),
                "section_count": len(sections)
            },
            "section_distribution": section_distribution,
            "analyzed_at": processing_result.get("processed_at")
        }
        
        logger.info(f"📊 Analyse terminée:")
        logger.info(f"   🎯 Score de qualité: {quality_score}/100 ({quality_level})")
        logger.info(f"   📝 {avg_tokens_per_chunk:.1f} tokens/chunk en moyenne")
        logger.info(f"   📂 {len(sections)} sections détectées")
        
        if quality_issues:
            logger.warning(f"⚠️ Problèmes de qualité: {', '.join(quality_issues)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }