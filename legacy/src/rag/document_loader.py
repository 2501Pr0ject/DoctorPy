import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..core.config import settings
from ..core.logger import logger
from ..core.exceptions import DocumentLoadingError


class DocumentLoader:
    """Chargeur de documents pour la documentation Python"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_python_docs(self, base_url: str = "https://docs.python.org/3/") -> List[Document]:
        """Charge la documentation Python officielle"""
        try:
            logger.info(f"Chargement de la documentation Python depuis {base_url}")
            
            # URLs importantes de la documentation Python
            important_urls = [
                "tutorial/index.html",
                "library/index.html",
                "reference/index.html",
                "tutorial/introduction.html",
                "tutorial/controlflow.html",
                "tutorial/datastructures.html",
                "tutorial/modules.html",
                "tutorial/inputoutput.html",
                "tutorial/errors.html",
                "tutorial/classes.html",
                "library/functions.html",
                "library/stdtypes.html",
                "library/exceptions.html",
            ]
            
            documents = []
            for url_path in important_urls:
                full_url = urljoin(base_url, url_path)
                try:
                    doc = self._load_web_page(full_url)
                    if doc:
                        documents.append(doc)
                    time.sleep(1)  # Respecter le serveur
                except Exception as e:
                    logger.warning(f"Impossible de charger {full_url}: {e}")
                    continue
            
            # Découper les documents en chunks
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            
            logger.info(f"Chargé {len(documents)} pages, créé {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la documentation: {e}")
            raise DocumentLoadingError(f"Impossible de charger la documentation: {e}")
    
    def _load_web_page(self, url: str) -> Optional[Document]:
        """Charge une page web et extrait le contenu texte"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Supprimer les éléments indésirables
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Extraire le contenu principal
            main_content = soup.find('div', class_='body') or soup.find('main') or soup.body
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                
                # Nettoyer le texte
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                clean_text = '\n'.join(lines)
                
                return Document(
                    page_content=clean_text,
                    metadata={
                        'source': url,
                        'title': soup.title.string if soup.title else url,
                        'length': len(clean_text)
                    }
                )
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {url}: {e}")
            return None
    
    def load_local_markdown(self, directory: Path) -> List[Document]:
        """Charge des fichiers Markdown locaux"""
        documents = []
        
        for md_file in directory.glob("**/*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': str(md_file),
                        'title': md_file.stem,
                        'type': 'markdown'
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Impossible de charger {md_file}: {e}")
        
        # Découper en chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Chargé {len(documents)} fichiers markdown, créé {len(chunks)} chunks")
        
        return chunks