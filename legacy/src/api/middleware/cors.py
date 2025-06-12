# src/api/middleware/cors.py
"""
Middleware CORS (Cross-Origin Resource Sharing) avancé
"""

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union, Dict, Any, Callable
import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AdvancedCORSMiddleware(BaseHTTPMiddleware):
    """
    Middleware CORS avancé avec fonctionnalités supplémentaires
    """
    
    def __init__(
        self,
        app,
        allow_origins: Union[List[str], str] = None,
        allow_methods: Union[List[str], str] = None,
        allow_headers: Union[List[str], str] = None,
        allow_credentials: bool = False,
        expose_headers: Union[List[str], str] = None,
        max_age: int = 600,
        allow_origin_regex: Optional[str] = None,
        dynamic_origins: Optional[Callable[[str], bool]] = None,
        environment_based: bool = True,
        debug_mode: bool = False
    ):
        super().__init__(app)
        
        # Configuration par défaut pour le développement
        if allow_origins is None:
            allow_origins = [
                "http://localhost:3000",
                "http://localhost:3001", 
                "http://localhost:8080",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080"
            ]
        
        if allow_methods is None:
            allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        
        if allow_headers is None:
            allow_headers = [
                "Accept",
                "Accept-Language",
                "Content-Language",
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "X-API-Key",
                "X-Client-Version",
                "X-Request-ID"
            ]
        
        if expose_headers is None:
            expose_headers = [
                "X-Total-Count",
                "X-Rate-Limit-Limit",
                "X-Rate-Limit-Remaining", 
                "X-Rate-Limit-Reset",
                "X-Process-Time"
            ]
        
        # Normaliser les configurations
        self.allow_origins = self._normalize_list(allow_origins)
        self.allow_methods = self._normalize_list(allow_methods)
        self.allow_headers = self._normalize_list(allow_headers)
        self.expose_headers = self._normalize_list(expose_headers)
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.allow_origin_regex = re.compile(allow_origin_regex) if allow_origin_regex else None
        self.dynamic_origins = dynamic_origins
        self.environment_based = environment_based
        self.debug_mode = debug_mode
        
        # Configuration basée sur l'environnement
        if environment_based:
            self._configure_for_environment()
        
        logger.info(f"CORS configuré - Origins: {len(self.allow_origins)}, Credentials: {allow_credentials}")
    
    def _normalize_list(self, value: Union[List[str], str]) -> List[str]:
        """Normalise une valeur en liste de chaînes"""
        if isinstance(value, str):
            if value == "*":
                return ["*"]
            return [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            return [str(item) for item in value]
        else:
            return []
    
    def _configure_for_environment(self):
        """Configure CORS selon l'environnement"""
        try:
            from src.core.config import get_settings
            settings = get_settings()
            
            if settings.ENVIRONMENT == "production":
                # Configuration stricte pour la production
                self.allow_credentials = True
                self.max_age = 3600  # 1 heure
                
                # Retirer les origins de développement
                dev_origins = [
                    "http://localhost",
                    "http://127.0.0.1",
                    "http://0.0.0.0"
                ]
                
                self.allow_origins = [
                    origin for origin in self.allow_origins
                    if not any(origin.startswith(dev) for dev in dev_origins)
                ]
                
                logger.info("CORS configuré pour la production")
                
            elif settings.ENVIRONMENT == "development":
                # Configuration permissive pour le développement
                self.debug_mode = True
                self.max_age = 60  # 1 minute
                
                # Ajouter des origins de développement
                dev_origins = [
                    "http://localhost:3000",
                    "http://localhost:3001",
                    "http://localhost:8080",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:8080"
                ]
                
                for origin in dev_origins:
                    if origin not in self.allow_origins:
                        self.allow_origins.append(origin)
                
                logger.info("CORS configuré pour le développement")
                
        except Exception as e:
            logger.warning(f"Impossible de configurer CORS selon l'environnement: {e}")
    
    async def dispatch(self, request: Request, call_next):
        """Traite la requête CORS"""
        
        origin = request.headers.get("origin")
        
        # Gérer les requêtes préliminaires (preflight)
        if request.method == "OPTIONS":
            return await self._handle_preflight(request, origin)
        
        # Traiter la requête normale
        response = await call_next(request)
        
        # Ajouter les headers CORS
        self._add_cors_headers(response, origin, request)
        
        return response
    
    async def _handle_preflight(self, request: Request, origin: Optional[str]) -> Response:
        """Gère les requêtes preflight OPTIONS"""
        
        # Vérifier l'origin
        if not self._is_origin_allowed(origin):
            if self.debug_mode:
                logger.warning(f"Origin non autorisée pour preflight: {origin}")
            return Response(status_code=403)
        
        # Vérifier la méthode demandée
        requested_method = request.headers.get("access-control-request-method")
        if requested_method and not self._is_method_allowed(requested_method):
            if self.debug_mode:
                logger.warning(f"Méthode non autorisée: {requested_method}")
            return Response(status_code=403)
        
        # Vérifier les headers demandés
        requested_headers = request.headers.get("access-control-request-headers")
        if requested_headers and not self._are_headers_allowed(requested_headers):
            if self.debug_mode:
                logger.warning(f"Headers non autorisés: {requested_headers}")
            return Response(status_code=403)
        
        # Créer la réponse preflight
        response = Response(status_code=200)
        self._add_preflight_headers(response, origin, requested_method, requested_headers)
        
        if self.debug_mode:
            logger.debug(f"Preflight autorisé pour {origin}")
        
        return response
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Vérifie si l'origin est autorisée"""
        if not origin:
            return True  # Pas d'origin = requête même domaine
        
        # Vérifier les origins explicites
        if "*" in self.allow_origins:
            return True
        
        if origin in self.allow_origins:
            return True
        
        # Vérifier avec regex
        if self.allow_origin_regex and self.allow_origin_regex.match(origin):
            return True
        
        # Vérifier avec fonction dynamique
        if self.dynamic_origins and self.dynamic_origins(origin):
            return True
        
        # Vérification des sous-domaines pour les origins configurés
        for allowed_origin in self.allow_origins:
            if self._is_subdomain_allowed(origin, allowed_origin):
                return True
        
        return False
    
    def _is_subdomain_allowed(self, origin: str, allowed_origin: str) -> bool:
        """Vérifie si l'origin est un sous-domaine autorisé"""
        try:
            origin_parsed = urlparse(origin)
            allowed_parsed = urlparse(allowed_origin)
            
            # Même schéma et port
            if origin_parsed.scheme != allowed_parsed.scheme:
                return False
            
            if origin_parsed.port != allowed_parsed.port:
                return False
            
            # Vérifier si c'est un sous-domaine
            origin_domain = origin_parsed.hostname
            allowed_domain = allowed_parsed.hostname
            
            if allowed_domain.startswith("*."):
                # Wildcard subdomain
                base_domain = allowed_domain[2:]
                return origin_domain.endswith(f".{base_domain}") or origin_domain == base_domain
            
            return False
            
        except Exception:
            return False
    
    def _is_method_allowed(self, method: str) -> bool:
        """Vérifie si la méthode est autorisée"""
        return method.upper() in [m.upper() for m in self.allow_methods]
    
    def _are_headers_allowed(self, headers: str) -> bool:
        """Vérifie si les headers sont autorisés"""
        requested_headers = [h.strip().lower() for h in headers.split(",")]
        allowed_headers_lower = [h.lower() for h in self.allow_headers]
        
        return all(header in allowed_headers_lower for header in requested_headers)
    
    def _add_cors_headers(self, response: Response, origin: Optional[str], request: Request):
        """Ajoute les headers CORS à la réponse"""
        
        # Access-Control-Allow-Origin
        if self._is_origin_allowed(origin):
            if "*" in self.allow_origins and not self.allow_credentials:
                response.headers["Access-Control-Allow-Origin"] = "*"
            elif origin:
                response.headers["Access-Control-Allow-Origin"] = origin
        
        # Access-Control-Allow-Credentials
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Access-Control-Expose-Headers
        if self.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        
        # Vary header pour le cache
        vary_headers = []
        if "Vary" in response.headers:
            vary_headers = [h.strip() for h in response.headers["Vary"].split(",")]
        
        if "Origin" not in vary_headers:
            vary_headers.append("Origin")
        
        response.headers["Vary"] = ", ".join(vary_headers)
    
    def _add_preflight_headers(self, response: Response, origin: Optional[str], 
                             requested_method: Optional[str], requested_headers: Optional[str]):
        """Ajoute les headers pour les requêtes preflight"""
        
        # Headers de base
        self._add_cors_headers(response, origin, None)
        
        # Access-Control-Allow-Methods
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        
        # Access-Control-Allow-Headers
        if requested_headers:
            # Retourner les headers demandés s'ils sont autorisés
            response.headers["Access-Control-Allow-Headers"] = requested_headers
        else:
            # Retourner tous les headers autorisés
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        
        # Access-Control-Max-Age
        response.headers["Access-Control-Max-Age"] = str(self.max_age)


class DynamicCORSMiddleware(AdvancedCORSMiddleware):
    """
    Middleware CORS avec configuration dynamique basée sur la base de données
    """
    
    def __init__(self, app, **kwargs):
        # Configuration par défaut
        super().__init__(app, **kwargs)
        
        # Cache pour les origins dynamiques
        self.origin_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = 0
    
    async def _load_allowed_origins_from_db(self) -> List[str]:
        """Charge les origins autorisées depuis la base de données"""
        try:
            from src.core.database import get_db_session
            
            async with get_db_session() as session:
                query = """
                    SELECT origin_url FROM allowed_origins 
                    WHERE is_active = true
                """
                result = await session.execute(query)
                origins = [row[0] for row in result.fetchall()]
                
                logger.debug(f"Origins chargées depuis la DB: {len(origins)}")
                return origins
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des origins: {e}")
            return []
    
    async def _is_origin_allowed_dynamic(self, origin: str) -> bool:
        """Vérifie si l'origin est autorisée avec cache"""
        import time
        
        current_time = time.time()
        
        # Mettre à jour le cache si nécessaire
        if current_time - self.last_cache_update > self.cache_ttl:
            db_origins = await self._load_allowed_origins_from_db()
            self.origin_cache = {origin: True for origin in db_origins}
            self.last_cache_update = current_time
        
        return origin in self.origin_cache
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Version étendue qui vérifie aussi la base de données"""
        # Vérifier d'abord avec la méthode parent
        if super()._is_origin_allowed(origin):
            return True
        
        # Vérifier dans le cache des origins dynamiques
        if origin and origin in self.origin_cache:
            return True
        
        return False


class SecurityCORSMiddleware(AdvancedCORSMiddleware):
    """
    Middleware CORS avec fonctionnalités de sécurité supplémentaires
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        
        # Configuration de sécurité
        self.blocked_origins = set()
        self.origin_request_count = {}
        self.max_requests_per_origin = 1000
        self.block_duration = 3600  # 1 heure
        
        # Patterns d'origins suspects
        self.suspicious_patterns = [
            r".*\.suspicious\.com",
            r".*malicious.*",
            r".*\.tk$",  # Domaines .tk souvent suspects
            r".*\.ml$",  # Domaines .ml souvent suspects
        ]
        
        self.suspicious_regex = [re.compile(pattern) for pattern in self.suspicious_patterns]
    
    def _is_origin_suspicious(self, origin: str) -> bool:
        """Vérifie si l'origin semble suspecte"""
        for pattern in self.suspicious_regex:
            if pattern.match(origin):
                return True
        return False
    
    def _track_origin_requests(self, origin: str):
        """Suit le nombre de requêtes par origin"""
        if origin not in self.origin_request_count:
            self.origin_request_count[origin] = {"count": 0, "first_request": None}
        
        self.origin_request_count[origin]["count"] += 1
        
        if self.origin_request_count[origin]["first_request"] is None:
            self.origin_request_count[origin]["first_request"] = datetime.utcnow()
    
    def _should_block_origin(self, origin: str) -> bool:
        """Détermine si une origin doit être bloquée"""
        # Vérifier si déjà bloquée
        if origin in self.blocked_origins:
            return True
        
        # Vérifier si suspecte
        if self._is_origin_suspicious(origin):
            logger.warning(f"Origin suspecte détectée: {origin}")
            self.blocked_origins.add(origin)
            return True
        
        # Vérifier le taux de requêtes
        if origin in self.origin_request_count:
            count_data = self.origin_request_count[origin]
            if count_data["count"] > self.max_requests_per_origin:
                logger.warning(f"Origin bloquée pour trop de requêtes: {origin}")
                self.blocked_origins.add(origin)
                return True
        
        return False
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Version sécurisée de la vérification d'origin"""
        if not origin:
            return True
        
        # Vérifier si bloquée
        if self._should_block_origin(origin):
            return False
        
        # Suivre les requêtes
        self._track_origin_requests(origin)
        
        # Vérification normale
        return super()._is_origin_allowed(origin)


# Fonctions utilitaires pour la configuration CORS
def get_cors_origins_for_environment(environment: str) -> List[str]:
    """Retourne les origins CORS selon l'environnement"""
    
    if environment == "production":
        return [
            "https://your-domain.com",
            "https://www.your-domain.com",
            "https://app.your-domain.com"
        ]
    elif environment == "staging":
        return [
            "https://staging.your-domain.com",
            "https://staging-app.your-domain.com"
        ]
    else:  # development
        return [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://0.0.0.0:3000"
        ]


def create_cors_middleware(environment: str = "development", **kwargs) -> AdvancedCORSMiddleware:
    """
    Factory pour créer un middleware CORS selon l'environnement
    """
    
    # Configuration par défaut selon l'environnement
    default_config = {
        "allow_origins": get_cors_origins_for_environment(environment),
        "allow_credentials": environment == "production",
        "max_age": 3600 if environment == "production" else 60,
        "debug_mode": environment == "development"
    }
    
    # Fusionner avec la configuration personnalisée
    config = {**default_config, **kwargs}
    
    return AdvancedCORSMiddleware(**config)


def validate_cors_config(config: Dict[str, Any]) -> List[str]:
    """
    Valide une configuration CORS et retourne les erreurs
    """
    errors = []
    
    # Vérifier les origins
    if "allow_origins" in config:
        origins = config["allow_origins"]
        if isinstance(origins, list):
            for origin in origins:
                if not isinstance(origin, str):
                    errors.append(f"Origin invalide: {origin}")
                elif origin != "*" and not origin.startswith(("http://", "https://")):
                    errors.append(f"Origin doit commencer par http:// ou https://: {origin}")
    
    # Vérifier les méthodes
    if "allow_methods" in config:
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
        methods = config["allow_methods"]
        if isinstance(methods, list):
            for method in methods:
                if method.upper() not in valid_methods:
                    errors.append(f"Méthode HTTP invalide: {method}")
    
    # Vérifier max_age
    if "max_age" in config:
        max_age = config["max_age"]
        if not isinstance(max_age, int) or max_age < 0:
            errors.append("max_age doit être un entier positif")
    
    return errors


# Export des classes principales
__all__ = [
    "AdvancedCORSMiddleware",
    "DynamicCORSMiddleware", 
    "SecurityCORSMiddleware",
    "create_cors_middleware",
    "get_cors_origins_for_environment",
    "validate_cors_config"
]