import json
import time
from typing import Dict, Any, List, Optional, Generator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..core.config import settings
from ..core.logger import logger
from ..core.exceptions import OllamaError, LLMError


class OllamaClient:
    """Client pour interagir avec Ollama"""
    
    def __init__(self, host: str = None, model: str = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model
        self.timeout = settings.ollama_timeout
        
        # Configuration de session avec retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def is_model_available(self) -> bool:
        """Vérifie si le modèle est disponible"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            
            is_available = self.model in available_models
            logger.info(f"Modèle {self.model} {'disponible' if is_available else 'non disponible'}")
            
            if not is_available:
                logger.info(f"Modèles disponibles: {available_models}")
            
            return is_available
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du modèle: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Télécharge le modèle si nécessaire"""
        try:
            logger.info(f"Téléchargement du modèle {self.model}")
            
            response = self.session.post(
                f"{self.host}/api/pull",
                json={"name": self.model},
                timeout=600,  # 10 minutes pour le téléchargement
                stream=True
            )
            response.raise_for_status()
            
            # Suivre le progrès du téléchargement
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'status' in data:
                            logger.info(f"Téléchargement: {data['status']}")
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Modèle {self.model} téléchargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement: {e}")
            return False
    
    def generate(
        self, 
        prompt: str, 
        system: str = None, 
        stream: bool = False,
        **kwargs
    ) -> str:
        """Génère une réponse avec le modèle"""
        try:
            # Vérifier que le modèle est disponible
            if not self.is_model_available():
                logger.info("Tentative de téléchargement du modèle")
                if not self.pull_model():
                    raise OllamaError(f"Impossible de télécharger le modèle {self.model}")
            
            # Préparer les données
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }
            
            if system:
                data["system"] = system
            
            # Faire la requête
            response = self.session.post(
                f"{self.host}/api/generate",
                json=data,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                result = response.json()
                return result.get('response', '')
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de requête Ollama: {e}")
            raise OllamaError(f"Erreur de communication avec Ollama: {e}")
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise LLMError(f"Erreur lors de la génération: {e}")
    
    def _handle_streaming_response(self, response) -> Generator[str, None, None]:
        """Gère la réponse en streaming"""
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Erreur dans le streaming: {e}")
            raise OllamaError(f"Erreur dans le streaming: {e}")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False,
        **kwargs
    ) -> str:
        """Interface de chat avec historique"""
        try:
            data = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.host}/api/chat",
                json=data,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                result = response.json()
                return result.get('message', {}).get('content', '')
                
        except Exception as e:
            logger.error(f"Erreur lors du chat: {e}")
            raise OllamaError(f"Erreur lors du chat: {e}")