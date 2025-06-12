# src/api/middleware/rate_limit.py
"""
Middleware de limitation du taux de requêtes (Rate Limiting)
"""

from fastapi import Request, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Tuple
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import hashlib

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception levée quand la limite de taux est dépassée"""
    def __init__(self, retry_after: int, message: str = "Rate limit exceeded"):
        self.retry_after = retry_after
        self.message = message
        super().__init__(self.message)


class TokenBucket:
    """
    Implémentation de l'algorithme Token Bucket pour le rate limiting
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Nombre maximum de tokens dans le bucket
            refill_rate: Nombre de tokens ajoutés par seconde
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Tente de consommer des tokens
        
        Args:
            tokens: Nombre de tokens à consommer
            
        Returns:
            True si les tokens ont été consommés, False sinon
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _refill(self):
        """Recharge le bucket avec de nouveaux tokens"""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def time_until_token(self) -> float:
        """Retourne le temps en secondes jusqu'au prochain token disponible"""
        self._refill()
        if self.tokens >= 1:
            return 0
        
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowCounter:
    """
    Implémentation de l'algorithme Sliding Window pour le rate limiting
    """
    
    def __init__(self, window_size: int, max_requests: int):
        """
        Args:
            window_size: Taille de la fenêtre en secondes
            max_requests: Nombre maximum de requêtes dans la fenêtre
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
    
    def is_allowed(self) -> bool:
        """
        Vérifie si une nouvelle requête est autorisée
        
        Returns:
            True si la requête est autorisée, False sinon
        """
        now = time.time()
        
        # Supprimer les requêtes hors de la fenêtre
        while self.requests and self.requests[0] <= now - self.window_size:
            self.requests.popleft()
        
        # Vérifier si on peut ajouter une nouvelle requête
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def time_until_reset(self) -> float:
        """Retourne le temps jusqu'à ce qu'une requête soit à nouveau autorisée"""
        if not self.requests:
            return 0
        
        oldest_request = self.requests[0]
        return max(0, (oldest_request + self.window_size) - time.time())


class RateLimitConfig:
    """Configuration pour le rate limiting"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        algorithm: str = "token_bucket"  # "token_bucket" ou "sliding_window"
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.algorithm = algorithm


class RateLimitStorage:
    """
    Stockage en mémoire pour les données de rate limiting
    """
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.counters: Dict[str, SlidingWindowCounter] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def get_token_bucket(self, key: str, capacity: int, refill_rate: float) -> TokenBucket:
        """Récupère ou crée un token bucket pour une clé"""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(capacity, refill_rate)
        return self.buckets[key]
    
    def get_sliding_counter(self, key: str, window_size: int, max_requests: int) -> SlidingWindowCounter:
        """Récupère ou crée un compteur sliding window pour une clé"""
        if key not in self.counters:
            self.counters[key] = SlidingWindowCounter(window_size, max_requests)
        return self.counters[key]
    
    def cleanup(self):
        """Nettoie les entrées expirées"""
        now = time.time()
        
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Nettoyer les buckets inactifs
        inactive_buckets = []
        for key, bucket in self.buckets.items():
            if now - bucket.last_refill > 3600:  # 1 heure d'inactivité
                inactive_buckets.append(key)
        
        for key in inactive_buckets:
            del self.buckets[key]
        
        # Nettoyer les compteurs inactifs
        inactive_counters = []
        for key, counter in self.counters.items():
            if not counter.requests or now - counter.requests[-1] > 3600:
                inactive_counters.append(key)
        
        for key in inactive_counters:
            del self.counters[key]
        
        self.last_cleanup = now
        
        if inactive_buckets or inactive_counters:
            logger.info(f"Rate limit cleanup: {len(inactive_buckets)} buckets, {len(inactive_counters)} counters supprimés")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware de rate limiting avec support de différents algorithmes
    """
    
    def __init__(
        self,
        app,
        default_config: Optional[RateLimitConfig] = None,
        path_configs: Optional[Dict[str, RateLimitConfig]] = None,
        exempt_paths: Optional[List[str]] = None,
        enable_headers: bool = True
    ):
        super().__init__(app)
        
        self.default_config = default_config or RateLimitConfig()
        self.path_configs = path_configs or {}
        self.exempt_paths = exempt_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        self.enable_headers = enable_headers
        self.storage = RateLimitStorage()
        
        # Configurations spécifiques par défaut
        self.path_configs.update({
            "/api/v1/auth/login": RateLimitConfig(
                requests_per_minute=5,
                requests_per_hour=20,
                burst_size=2,
                algorithm="sliding_window"
            ),
            "/api/v1/chat/sessions": RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_size=5
            ),
            "/api/v1/quests/generate": RateLimitConfig(
                requests_per_minute=3,
                requests_per_hour=20,
                burst_size=1
            ),
            "/api/v1/admin": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=500,
                burst_size=10
            )
        })
    
    async def dispatch(self, request: Request, call_next):
        """Traite la requête avec rate limiting"""
        
        # Nettoyage périodique
        self.storage.cleanup()
        
        # Vérifier si le chemin est exempt
        path = request.url.path
        if any(path.startswith(exempt_path) for exempt_path in self.exempt_paths):
            return await call_next(request)
        
        # Déterminer la configuration à utiliser
        config = self._get_config_for_path(path)
        
        # Générer la clé de rate limiting
        limit_key = self._generate_limit_key(request)
        
        try:
            # Appliquer le rate limiting
            retry_after = await self._check_rate_limit(limit_key, config, request)
            
            if retry_after > 0:
                return self._create_rate_limit_response(retry_after, config)
            
            # Traiter la requête
            response = await call_next(request)
            
            # Ajouter les headers de rate limiting
            if self.enable_headers:
                self._add_rate_limit_headers(response, limit_key, config)
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur dans le rate limiting: {e}")
            # En cas d'erreur, laisser passer la requête
            return await call_next(request)
    
    def _get_config_for_path(self, path: str) -> RateLimitConfig:
        """Détermine la configuration de rate limiting pour un chemin"""
        
        # Chercher une configuration spécifique
        for config_path, config in self.path_configs.items():
            if path.startswith(config_path):
                return config
        
        # Utiliser la configuration par défaut
        return self.default_config
    
    def _generate_limit_key(self, request: Request) -> str:
        """Génère une clé unique pour le rate limiting"""
        
        # Utiliser l'IP client comme base
        client_ip = request.client.host if request.client else "unknown"
        
        # Ajouter l'utilisateur s'il est authentifié
        user_info = ""
        if hasattr(request.state, 'user_id'):
            user_info = f":user:{request.state.user_id}"
        
        # Ajouter le chemin pour des limites spécifiques
        path_hash = hashlib.md5(request.url.path.encode()).hexdigest()[:8]
        
        return f"rate_limit:{client_ip}{user_info}:path:{path_hash}"
    
    async def _check_rate_limit(self, key: str, config: RateLimitConfig, request: Request) -> float:
        """
        Vérifie les limites de taux et retourne le temps d'attente si dépassé
        
        Returns:
            0 si autorisé, temps d'attente en secondes si dépassé
        """
        
        if config.algorithm == "token_bucket":
            return await self._check_token_bucket(key, config)
        elif config.algorithm == "sliding_window":
            return await self._check_sliding_window(key, config)
        else:
            logger.warning(f"Algorithme de rate limiting inconnu: {config.algorithm}")
            return 0
    
    async def _check_token_bucket(self, key: str, config: RateLimitConfig) -> float:
        """Vérifie avec l'algorithme Token Bucket"""
        
        # Vérifier la limite par minute
        minute_bucket = self.storage.get_token_bucket(
            f"{key}:minute",
            config.requests_per_minute,
            config.requests_per_minute / 60.0
        )
        
        # Vérifier la limite par heure
        hour_bucket = self.storage.get_token_bucket(
            f"{key}:hour",
            config.requests_per_hour,
            config.requests_per_hour / 3600.0
        )
        
        # Vérifier la limite de burst
        burst_bucket = self.storage.get_token_bucket(
            f"{key}:burst",
            config.burst_size,
            config.burst_size / 10.0  # Recharge rapide pour le burst
        )
        
        # Toutes les limites doivent être respectées
        if not minute_bucket.consume():
            return minute_bucket.time_until_token()
        
        if not hour_bucket.consume():
            return hour_bucket.time_until_token()
        
        if not burst_bucket.consume():
            return burst_bucket.time_until_token()
        
        return 0
    
    async def _check_sliding_window(self, key: str, config: RateLimitConfig) -> float:
        """Vérifie avec l'algorithme Sliding Window"""
        
        # Vérifier la limite par minute
        minute_counter = self.storage.get_sliding_counter(
            f"{key}:minute",
            60,
            config.requests_per_minute
        )
        
        # Vérifier la limite par heure
        hour_counter = self.storage.get_sliding_counter(
            f"{key}:hour",
            3600,
            config.requests_per_hour
        )
        
        if not minute_counter.is_allowed():
            return minute_counter.time_until_reset()
        
        if not hour_counter.is_allowed():
            return hour_counter.time_until_reset()
        
        return 0
    
    def _create_rate_limit_response(self, retry_after: float, config: RateLimitConfig) -> JSONResponse:
        """Crée une réponse d'erreur de rate limiting"""
        
        retry_after_int = int(retry_after) + 1
        
        headers = {
            "Retry-After": str(retry_after_int),
            "X-RateLimit-Limit": str(config.requests_per_minute),
            "X-RateLimit-Reset": str(int(time.time() + retry_after_int))
        }
        
        content = {
            "error": "Rate limit exceeded",
            "message": f"Trop de requêtes. Réessayez dans {retry_after_int} secondes.",
            "retry_after": retry_after_int
        }
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=content,
            headers=headers
        )
    
    def _add_rate_limit_headers(self, response, key: str, config: RateLimitConfig):
        """Ajoute les headers informatifs de rate limiting"""
        
        try:
            # Obtenir les informations sur les limites actuelles
            if config.algorithm == "token_bucket":
                minute_bucket = self.storage.buckets.get(f"{key}:minute")
                if minute_bucket:
                    remaining = int(minute_bucket.tokens)
                    response.headers["X-RateLimit-Limit"] = str(config.requests_per_minute)
                    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
                    response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            
            elif config.algorithm == "sliding_window":
                minute_counter = self.storage.counters.get(f"{key}:minute")
                if minute_counter:
                    remaining = max(0, config.requests_per_minute - len(minute_counter.requests))
                    response.headers["X-RateLimit-Limit"] = str(config.requests_per_minute)
                    response.headers["X-RateLimit-Remaining"] = str(remaining)
                    
                    if minute_counter.requests:
                        reset_time = int(minute_counter.requests[0] + 60)
                        response.headers["X-RateLimit-Reset"] = str(reset_time)
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'ajout des headers de rate limiting: {e}")


class AdvancedRateLimitMiddleware(RateLimitMiddleware):
    """
    Version avancée du middleware avec fonctionnalités supplémentaires
    """
    
    def __init__(
        self,
        app,
        default_config: Optional[RateLimitConfig] = None,
        path_configs: Optional[Dict[str, RateLimitConfig]] = None,
        exempt_paths: Optional[List[str]] = None,
        enable_headers: bool = True,
        enable_user_tiers: bool = True,
        enable_adaptive_limits: bool = False
    ):
        super().__init__(app, default_config, path_configs, exempt_paths, enable_headers)
        
        self.enable_user_tiers = enable_user_tiers
        self.enable_adaptive_limits = enable_adaptive_limits
        
        # Configurations par niveau d'utilisateur
        self.user_tier_configs = {
            "free": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=500,
                burst_size=5
            ),
            "premium": RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=2000,
                burst_size=20
            ),
            "admin": RateLimitConfig(
                requests_per_minute=500,
                requests_per_hour=10000,
                burst_size=50
            )
        }
        
        # Historique des performances pour l'adaptation
        self.performance_history = defaultdict(list)
    
    def _get_config_for_path(self, path: str) -> RateLimitConfig:
        """Version avancée qui prend en compte le tier utilisateur"""
        
        base_config = super()._get_config_for_path(path)
        
        # Si les tiers utilisateur ne sont pas activés, utiliser la config de base
        if not self.enable_user_tiers:
            return base_config
        
        return base_config
    
    def _get_user_tier_config(self, request: Request) -> Optional[RateLimitConfig]:
        """Détermine la configuration basée sur le tier utilisateur"""
        
        if not hasattr(request.state, 'is_admin') and not hasattr(request.state, 'user_id'):
            return self.user_tier_configs["free"]
        
        # Utilisateur admin
        if hasattr(request.state, 'is_admin') and request.state.is_admin:
            return self.user_tier_configs["admin"]
        
        # Vérifier le tier premium en base (simplifié pour l'exemple)
        if hasattr(request.state, 'user_id'):
            # Ici, on pourrait vérifier en base le statut premium
            # Pour l'exemple, on considère tous les utilisateurs connectés comme premium
            return self.user_tier_configs["premium"]
        
        return self.user_tier_configs["free"]
    
    async def _check_rate_limit(self, key: str, config: RateLimitConfig, request: Request) -> float:
        """Version avancée qui prend en compte le tier utilisateur"""
        
        if self.enable_user_tiers:
            user_config = self._get_user_tier_config(request)
            if user_config:
                config = user_config
        
        # Adaptation dynamique basée sur les performances
        if self.enable_adaptive_limits:
            config = self._adapt_config_based_on_performance(key, config)
        
        return await super()._check_rate_limit(key, config, request)
    
    def _adapt_config_based_on_performance(self, key: str, config: RateLimitConfig) -> RateLimitConfig:
        """Adapte la configuration basée sur les performances historiques"""
        
        history = self.performance_history[key]
        
        # Si pas assez d'historique, utiliser la config par défaut
        if len(history) < 10:
            return config
        
        # Calculer la latence moyenne récente
        recent_latencies = history[-10:]
        avg_latency = sum(recent_latencies) / len(recent_latencies)
        
        # Adapter les limites selon la latence
        if avg_latency > 2.0:  # Latence élevée, réduire les limites
            factor = 0.7
        elif avg_latency < 0.5:  # Latence faible, augmenter les limites
            factor = 1.3
        else:
            factor = 1.0
        
        adapted_config = RateLimitConfig(
            requests_per_minute=int(config.requests_per_minute * factor),
            requests_per_hour=int(config.requests_per_hour * factor),
            burst_size=int(config.burst_size * factor),
            algorithm=config.algorithm
        )
        
        return adapted_config
    
    def _record_request_performance(self, key: str, latency: float):
        """Enregistre la performance d'une requête"""
        
        history = self.performance_history[key]
        history.append(latency)
        
        # Garder seulement les 100 dernières mesures
        if len(history) > 100:
            history.pop(0)


class GeographicRateLimitMiddleware(RateLimitMiddleware):
    """
    Middleware de rate limiting avec support géographique
    """
    
    def __init__(
        self,
        app,
        default_config: Optional[RateLimitConfig] = None,
        country_configs: Optional[Dict[str, RateLimitConfig]] = None,
        **kwargs
    ):
        super().__init__(app, default_config, **kwargs)
        
        self.country_configs = country_configs or {
            # Limites plus strictes pour certains pays
            "CN": RateLimitConfig(requests_per_minute=10, requests_per_hour=100),
            "RU": RateLimitConfig(requests_per_minute=15, requests_per_hour=200),
            # Limites plus souples pour les pays de confiance
            "FR": RateLimitConfig(requests_per_minute=100, requests_per_hour=2000),
            "US": RateLimitConfig(requests_per_minute=80, requests_per_hour=1500),
        }
    
    def _get_country_from_ip(self, ip: str) -> Optional[str]:
        """
        Détermine le pays depuis l'IP (implémentation simplifiée)
        Dans un vrai projet, utiliser une base GeoIP
        """
        
        # Implémentation factice pour la démonstration
        if ip.startswith("192.168") or ip.startswith("127.0"):
            return "FR"  # IP locale considérée comme française
        
        # Ici, intégrer une vraie bibliothèque GeoIP
        return None
    
    def _get_config_for_path(self, path: str) -> RateLimitConfig:
        """Version géographique qui prend en compte le pays"""
        
        base_config = super()._get_config_for_path(path)
        return base_config
    
    def _generate_limit_key(self, request: Request) -> str:
        """Génère une clé incluant des informations géographiques"""
        
        client_ip = request.client.host if request.client else "unknown"
        country = self._get_country_from_ip(client_ip)
        
        base_key = super()._generate_limit_key(request)
        
        if country:
            return f"{base_key}:country:{country}"
        
        return base_key


# Décorateurs pour rate limiting spécifique
def rate_limit(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    burst_size: int = 10,
    algorithm: str = "token_bucket"
):
    """
    Décorateur pour appliquer un rate limiting spécifique à une route
    """
    def decorator(func):
        # Ajouter les métadonnées de rate limiting à la fonction
        func._rate_limit_config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            burst_size=burst_size,
            algorithm=algorithm
        )
        return func
    
    return decorator


class RateLimitManager:
    """
    Gestionnaire centralisé pour les informations de rate limiting
    """
    
    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
    
    def get_user_limits_info(self, user_key: str) -> Dict[str, Any]:
        """Récupère les informations de limite pour un utilisateur"""
        
        info = {
            "limits": {},
            "usage": {},
            "reset_times": {}
        }
        
        # Parcourir les buckets de l'utilisateur
        for key, bucket in self.storage.buckets.items():
            if user_key in key:
                limit_type = key.split(":")[-1]  # minute, hour, burst
                info["usage"][limit_type] = bucket.capacity - bucket.tokens
                info["limits"][limit_type] = bucket.capacity
                info["reset_times"][limit_type] = bucket.last_refill + (bucket.capacity / bucket.refill_rate)
        
        return info
    
    def reset_user_limits(self, user_key: str):
        """Remet à zéro les limites d'un utilisateur"""
        
        keys_to_remove = []
        for key in self.storage.buckets.keys():
            if user_key in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.storage.buckets[key]
        
        # Faire de même pour les compteurs
        keys_to_remove = []
        for key in self.storage.counters.keys():
            if user_key in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.storage.counters[key]
        
        logger.info(f"Limites de rate limiting remises à zéro pour: {user_key}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques globales du système"""
        
        return {
            "total_buckets": len(self.storage.buckets),
            "total_counters": len(self.storage.counters),
            "memory_usage": self._estimate_memory_usage(),
            "last_cleanup": self.storage.last_cleanup
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estime l'utilisation mémoire approximative"""
        
        # Estimation approximative en bytes
        bucket_size = 100  # Taille approximative d'un TokenBucket
        counter_size = 50 + 8 * 60  # SlidingWindowCounter + timestamps
        
        total_size = (
            len(self.storage.buckets) * bucket_size +
            len(self.storage.counters) * counter_size
        )
        
        return total_size


# Fonctions utilitaires
async def check_rate_limit_for_user(user_id: int, action: str, storage: RateLimitStorage) -> bool:
    """
    Vérifie manuellement une limite de taux pour un utilisateur et une action
    """
    
    config = RateLimitConfig()  # Configuration par défaut
    key = f"manual_check:user:{user_id}:action:{action}"
    
    if config.algorithm == "token_bucket":
        bucket = storage.get_token_bucket(key, config.requests_per_minute, config.requests_per_minute / 60.0)
        return bucket.consume()
    
    elif config.algorithm == "sliding_window":
        counter = storage.get_sliding_counter(key, 60, config.requests_per_minute)
        return counter.is_allowed()
    
    return True


def create_custom_rate_limiter(
    requests_per_minute: int,
    requests_per_hour: int,
    burst_size: int = None,
    algorithm: str = "token_bucket"
) -> RateLimitConfig:
    """
    Crée une configuration de rate limiting personnalisée
    """
    
    if burst_size is None:
        burst_size = min(requests_per_minute // 4, 10)
    
    return RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        burst_size=burst_size,
        algorithm=algorithm
    )


# Export des classes et fonctions principales
__all__ = [
    "RateLimitMiddleware",
    "AdvancedRateLimitMiddleware", 
    "GeographicRateLimitMiddleware",
    "RateLimitConfig",
    "RateLimitStorage",
    "RateLimitManager",
    "TokenBucket",
    "SlidingWindowCounter",
    "RateLimitExceeded",
    "rate_limit",
    "check_rate_limit_for_user",
    "create_custom_rate_limiter"
]