#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoctorPy - Script de scraping de documentation
Récupère la documentation Python officielle et autres ressources
"""

import asyncio
import aiohttp
import aiofiles
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """Structure d'un document scrapé"""
    url: str
    title: str
    content: str
    section: str
    subsection: Optional[str]
    language: str
    last_modified: str
    hash: str
    file_path: str
    metadata: Dict


class DocumentationScraper:
    """
    Scraper pour la documentation Python officielle et autres ressources
    """
    
    def __init__(self, output_dir: str = "./data/raw/documentation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs de base pour différentes documentations
        self.base_urls = {
            "python_official": "https://docs.python.org/3/",
            "python_tutorial": "https://docs.python.org/3/tutorial/",
            "python_library": "https://docs.python.org/3/library/",
            "pandas": "https://pandas.pydata.org/docs/",
            "numpy": "https://numpy.org/doc/stable/",
            "requests": "https://requests.readthedocs.io/en/latest/",
        }
        
        # Sections spécifiques à scraper pour chaque source
        self.sections_config = {
            "python_official": {
                "tutorial": [
                    "tutorial/index.html",
                    "tutorial/appetite.html",
                    "tutorial/interpreter.html",
                    "tutorial/introduction.html",
                    "tutorial/controlflow.html",
                    "tutorial/datastructures.html",
                    "tutorial/modules.html",
                    "tutorial/inputoutput.html",
                    "tutorial/errors.html",
                    "tutorial/classes.html",
                    "tutorial/stdlib.html",
                    "tutorial/stdlib2.html",
                    "tutorial/venv.html",
                    "tutorial/whatnow.html"
                ],
                "library": [
                    "library/functions.html",
                    "library/constants.html",
                    "library/stdtypes.html",
                    "library/exceptions.html",
                    "library/string.html",
                    "library/re.html",
                    "library/datetime.html",
                    "library/math.html",
                    "library/random.html",
                    "library/itertools.html",
                    "library/functools.html",
                    "library/operator.html",
                    "library/pathlib.html",
                    "library/os.html",
                    "library/sys.html",
                    "library/json.html",
                    "library/urllib.html",
                    "library/http.html"
                ],
                "reference": [
                    "reference/lexical_analysis.html",
                    "reference/datamodel.html",
                    "reference/executionmodel.html",
                    "reference/import.html",
                    "reference/expressions.html",
                    "reference/simple_stmts.html",
                    "reference/compound_stmts.html"
                ]
            }
        }
        
        # Limites et délais
        self.max_concurrent = 5
        self.delay_between_requests = 1.0
        self.timeout = 30
        self.max_retries = 3
        
        # Cache des URLs déjà visitées
        self.visited_urls: Set[str] = set()
        self.scraped_documents: List[ScrapedDocument] = []
        
    async def scrape_all_documentation(self) -> List[ScrapedDocument]:
        """Scrape toute la documentation configurée"""
        logger.info("🚀 Démarrage du scraping de la documentation")
        
        # Créer une session HTTP avec des limites de connexion
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers={'User-Agent': 'DoctorPy Documentation Scraper 1.0'}
        ) as session:
            
            # Scraper la documentation Python officielle
            await self._scrape_python_docs(session)
            
            # Scraper d'autres documentations importantes
            await self._scrape_external_docs(session)
            
        logger.info(f"✅ Scraping terminé. {len(self.scraped_documents)} documents récupérés")
        return self.scraped_documents
    
    async def _scrape_python_docs(self, session: aiohttp.ClientSession):
        """Scrape la documentation Python officielle"""
        logger.info("📚 Scraping de la documentation Python officielle")
        
        base_url = self.base_urls["python_official"]
        sections = self.sections_config["python_official"]
        
        tasks = []
        
        # Scraper chaque section
        for section_name, pages in sections.items():
            for page in pages:
                url = urljoin(base_url, page)
                if url not in self.visited_urls:
                    task = self._scrape_single_page(
                        session, 
                        url, 
                        section="python_official", 
                        subsection=section_name
                    )
                    tasks.append(task)
                    self.visited_urls.add(url)
        
        # Exécuter avec limitation de concurrence
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async def limited_scrape(task):
            async with semaphore:
                await asyncio.sleep(self.delay_between_requests)
                return await task
        
        results = await asyncio.gather(*[limited_scrape(task) for task in tasks], return_exceptions=True)
        
        # Filtrer les résultats valides
        valid_docs = [doc for doc in results if isinstance(doc, ScrapedDocument)]
        self.scraped_documents.extend(valid_docs)
        
        logger.info(f"✅ Documentation Python: {len(valid_docs)} pages récupérées")
    
    async def _scrape_external_docs(self, session: aiohttp.ClientSession):
        """Scrape les documentations externes importantes"""
        logger.info("🌐 Scraping des documentations externes")
        
        # URLs importantes pour l'apprentissage Python
        external_urls = [
            # Python.org guides
            "https://docs.python.org/3/howto/regex.html",
            "https://docs.python.org/3/howto/logging.html",
            "https://docs.python.org/3/howto/argparse.html",
            "https://docs.python.org/3/howto/urllib2.html",
            
            # PEPs importants
            "https://peps.python.org/pep-0008/",  # Style Guide
            "https://peps.python.org/pep-0020/",  # Zen of Python
            "https://peps.python.org/pep-0257/",  # Docstring Conventions
        ]
        
        tasks = []
        for url in external_urls:
            if url not in self.visited_urls:
                task = self._scrape_single_page(
                    session, 
                    url, 
                    section="external", 
                    subsection="guides"
                )
                tasks.append(task)
                self.visited_urls.add(url)
        
        # Exécuter avec limitation
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async def limited_scrape(task):
            async with semaphore:
                await asyncio.sleep(self.delay_between_requests)
                return await task
        
        results = await asyncio.gather(*[limited_scrape(task) for task in tasks], return_exceptions=True)
        
        # Filtrer les résultats valides
        valid_docs = [doc for doc in results if isinstance(doc, ScrapedDocument)]
        self.scraped_documents.extend(valid_docs)
        
        logger.info(f"✅ Documentation externe: {len(valid_docs)} pages récupérées")
    
    async def _scrape_single_page(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        section: str, 
        subsection: Optional[str] = None,
        retry_count: int = 0
    ) -> Optional[ScrapedDocument]:
        """Scrape une seule page"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    return await self._parse_html_content(
                        html_content, url, section, subsection
                    )
                else:
                    logger.warning(f"⚠️ Erreur HTTP {response.status} pour {url}")
                    return None
                    
        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                logger.warning(f"⏱️ Timeout pour {url}, tentative {retry_count + 1}")
                await asyncio.sleep(2 ** retry_count)  # Backoff exponentiel
                return await self._scrape_single_page(
                    session, url, section, subsection, retry_count + 1
                )
            else:
                logger.error(f"❌ Échec définitif pour {url} après {self.max_retries} tentatives")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du scraping de {url}: {e}")
            return None
    
    async def _parse_html_content(
        self, 
        html_content: str, 
        url: str, 
        section: str, 
        subsection: Optional[str]
    ) -> Optional[ScrapedDocument]:
        """Parse le contenu HTML et extrait les informations importantes"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extraire le titre
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else "Sans titre"
            
            # Nettoyer le titre
            title = title.replace(' — Python 3', '').replace(' documentation', '').strip()
            
            # Extraire le contenu principal
            content = self._extract_main_content(soup)
            
            if not content.strip():
                logger.warning(f"⚠️ Contenu vide pour {url}")
                return None
            
            # Créer le hash du contenu
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Créer le chemin de fichier
            parsed_url = urlparse(url)
            file_name = f"{section}_{subsection or 'main'}_{content_hash}.md"
            file_path = self.output_dir / section / file_name
            
            # Créer le document
            doc = ScrapedDocument(
                url=url,
                title=title,
                content=content,
                section=section,
                subsection=subsection,
                language="fr",
                last_modified=datetime.now().isoformat(),
                hash=content_hash,
                file_path=str(file_path),
                metadata={
                    "source": "scraper",
                    "content_length": len(content),
                    "scraping_date": datetime.now().isoformat(),
                    "original_url": url
                }
            )
            
            # Sauvegarder le document
            await self._save_document(doc)
            
            logger.info(f"✅ Scrapé: {title[:50]}...")
            return doc
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du parsing de {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extrait le contenu principal d'une page"""
        # Supprimer les éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Chercher le contenu principal selon différents sélecteurs
        main_selectors = [
            'main',
            '.document',
            '.body',
            '.content',
            '#content',
            '.rst-content',
            'article'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Si aucun sélecteur ne fonctionne, prendre le body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return ""
        
        # Nettoyer et formater le contenu
        text_content = []
        
        # Extraire les titres et le texte de manière structurée
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'code']):
            if element.name.startswith('h'):
                level = '#' * int(element.name[1])
                text_content.append(f"\n{level} {element.get_text().strip()}\n")
            elif element.name == 'pre':
                text_content.append(f"\n```\n{element.get_text().strip()}\n```\n")
            elif element.name == 'code' and element.parent.name != 'pre':
                text_content.append(f"`{element.get_text().strip()}`")
            else:
                text = element.get_text().strip()
                if text:
                    text_content.append(text)
        
        # Joindre et nettoyer
        content = ' '.join(text_content)
        
        # Nettoyer les espaces multiples et caractères indésirables
        import re
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    async def _save_document(self, doc: ScrapedDocument):
        """Sauvegarde un document sur le disque"""
        try:
            file_path = Path(doc.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Contenu Markdown avec métadonnées
            markdown_content = f"""---
title: {doc.title}
url: {doc.url}
section: {doc.section}
subsection: {doc.subsection or ''}
language: {doc.language}
last_modified: {doc.last_modified}
hash: {doc.hash}
---

# {doc.title}

**Source:** {doc.url}

{doc.content}
"""
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(markdown_content)
            
            # Sauvegarder les métadonnées JSON
            metadata_path = file_path.with_suffix('.json')
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(asdict(doc), indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde de {doc.url}: {e}")
    
    async def save_scraping_report(self):
        """Sauvegarde un rapport du scraping"""
        report = {
            "scraping_date": datetime.now().isoformat(),
            "total_documents": len(self.scraped_documents),
            "sections": {},
            "documents": [asdict(doc) for doc in self.scraped_documents]
        }
        
        # Statistiques par section
        for doc in self.scraped_documents:
            section = doc.section
            if section not in report["sections"]:
                report["sections"][section] = {"count": 0, "subsections": {}}
            
            report["sections"][section]["count"] += 1
            
            if doc.subsection:
                subsections = report["sections"][section]["subsections"]
                subsections[doc.subsection] = subsections.get(doc.subsection, 0) + 1
        
        # Sauvegarder le rapport
        report_path = self.output_dir / "scraping_report.json"
        async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report, indent=2, ensure_ascii=False))
        
        logger.info(f"📊 Rapport de scraping sauvegardé: {report_path}")


async def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape la documentation Python")
    parser.add_argument(
        "--output-dir", 
        default="./data/raw/documentation",
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=5,
        help="Nombre max de requêtes simultanées"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.0,
        help="Délai entre les requêtes (secondes)"
    )
    
    args = parser.parse_args()
    
    scraper = DocumentationScraper(output_dir=args.output_dir)
    scraper.max_concurrent = args.max_concurrent
    scraper.delay_between_requests = args.delay
    
    start_time = time.time()
    
    try:
        # Lancer le scraping
        documents = await scraper.scrape_all_documentation()
        
        # Sauvegarder le rapport
        await scraper.save_scraping_report()
        
        # Statistiques finales
        elapsed_time = time.time() - start_time
        logger.info(f"🎉 Scraping terminé en {elapsed_time:.2f}s")
        logger.info(f"📚 {len(documents)} documents récupérés")
        
        # Statistiques par section
        sections_stats = {}
        for doc in documents:
            sections_stats[doc.section] = sections_stats.get(doc.section, 0) + 1
        
        for section, count in sections_stats.items():
            logger.info(f"  • {section}: {count} documents")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Scraping interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur lors du scraping: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())