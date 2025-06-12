"""Script d'initialisation du vector store"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.rag.indexer import DocumentIndexer
from src.core.logger import logger
from src.core.config import settings


def main():
    """Fonction principale"""
    try:
        logger.info("🔍 Initialisation du vector store...")
        
        # Créer l'indexeur
        indexer = DocumentIndexer()
        
        # Vérifier si l'index existe déjà
        stats = indexer.get_index_stats()
        
        if stats.get('total_documents', 0) > 0:
            logger.info(f"📚 Vector store existant trouvé: {stats['total_documents']} documents")
            
            response = input("Voulez-vous réindexer la documentation ? (y/N): ")
            force_refresh = response.lower() in ['y', 'yes', 'oui']
        else:
            force_refresh = True
        
        if force_refresh:
            logger.info("📖 Indexation de la documentation Python...")
            logger.info("⚠️  Cela peut prendre plusieurs minutes...")
            
            # Indexer la documentation
            indexer.index_python_documentation(force_refresh=force_refresh)
            
            # Afficher les statistiques finales
            final_stats = indexer.get_index_stats()
            logger.info("📊 Statistiques finales:")
            logger.info(f"   - Documents: {final_stats.get('total_documents', 0)}")
            logger.info(f"   - Collection: {final_stats.get('collection_name', 'N/A')}")
            logger.info(f"   - Modèle d'embedding: {final_stats.get('embedding_model', 'N/A')}")
        
        logger.info("🎉 Vector store configuré avec succès!")
        
        # Test rapide
        logger.info("🧪 Test rapide du système RAG...")
        from src.rag.retriever import DocumentRetriever
        
        retriever = DocumentRetriever()
        test_docs = retriever.retrieve_relevant_documents("variables python", max_docs=2)
        
        if test_docs:
            logger.info(f"✅ Test réussi: {len(test_docs)} documents trouvés")
            logger.info(f"   Premier résultat: {test_docs[0].page_content[:100]}...")
        else:
            logger.warning("⚠️  Aucun document trouvé lors du test")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()