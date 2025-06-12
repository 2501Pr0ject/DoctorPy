"""
Utilitaires partagés pour tous les microservices DoctorPy
"""

import logging
import logging.config
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sys


class LoggerFactory:
    """Factory pour créer des loggers configurés"""
    
    _configured = False
    
    @classmethod
    def configure(cls, log_level: str = "INFO", log_format: str = "json") -> None:
        """Configurer le système de logging"""
        if cls._configured:
            return
        
        if log_format == "json":
            formatter_config = {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
            }
        else:
            formatter_config = {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": formatter_config
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "default",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "formatter": "default",
                    "filename": "logs/app.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "loggers": {
                "": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            }
        }
        
        # Créer le répertoire de logs
        Path("logs").mkdir(exist_ok=True)
        
        logging.config.dictConfig(config)
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Obtenir un logger configuré"""
        if not cls._configured:
            cls.configure()
        return logging.getLogger(name)


class HealthChecker:
    """Utilitaire pour vérifier la santé des services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = LoggerFactory.get_logger(f"{service_name}.health")
        self.checks: Dict[str, callable] = {}
    
    def add_check(self, name: str, check_func: callable) -> None:
        """Ajouter une vérification de santé"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Exécuter toutes les vérifications de santé"""
        results = {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        overall_healthy = True
        
        for check_name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()
                
                results["checks"][check_name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "details": check_result if isinstance(check_result, dict) else {}
                }
                
                if not check_result:
                    overall_healthy = False
                    
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                results["checks"][check_name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False
        
        results["status"] = "healthy" if overall_healthy else "unhealthy"
        return results
    
    def check_database(self, db_manager) -> bool:
        """Vérification standard de base de données"""
        try:
            # Test de connexion simple
            result = db_manager.execute_query("SELECT 1 as test")
            return result and result[0]["test"] == 1
        except Exception:
            return False
    
    async def check_redis(self, redis_client) -> bool:
        """Vérification standard de Redis"""
        try:
            await redis_client.ping()
            return True
        except Exception:
            return False
    
    def check_disk_space(self, min_free_gb: float = 1.0) -> Dict[str, Any]:
        """Vérifier l'espace disque disponible"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            
            return {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total / (1024**3), 2),
                "used_percent": round((used / total) * 100, 1),
                "healthy": free_gb >= min_free_gb
            }
        except Exception:
            return {"healthy": False, "error": "Cannot check disk space"}
    
    def check_memory(self, max_usage_percent: float = 90.0) -> Dict[str, Any]:
        """Vérifier l'utilisation mémoire"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "used_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "healthy": memory.percent <= max_usage_percent
            }
        except Exception:
            return {"healthy": False, "error": "Cannot check memory"}


class ServiceRegistry:
    """Registre des services pour service discovery"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.services: Dict[str, Dict[str, Any]] = {}
        self.logger = LoggerFactory.get_logger("service_registry")
    
    async def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        health_endpoint: str = "/health",
        metadata: Dict[str, Any] = None
    ) -> None:
        """Enregistrer un service"""
        service_info = {
            "name": service_name,
            "host": host,
            "port": port,
            "health_endpoint": health_endpoint,
            "metadata": metadata or {},
            "registered_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        
        # Stocker localement
        self.services[service_name] = service_info
        
        # Stocker dans Redis si disponible
        if self.redis_client:
            await self._store_in_redis(service_name, service_info)
        
        self.logger.info(f"Service {service_name} registered at {host}:{port}")
    
    async def unregister_service(self, service_name: str) -> None:
        """Désenregistrer un service"""
        if service_name in self.services:
            del self.services[service_name]
        
        if self.redis_client:
            await self.redis_client.delete(f"service:{service_name}")
        
        self.logger.info(f"Service {service_name} unregistered")
    
    async def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Obtenir les informations d'un service"""
        # Vérifier le cache local
        if service_name in self.services:
            return self.services[service_name]
        
        # Vérifier Redis
        if self.redis_client:
            service_data = await self.redis_client.get(f"service:{service_name}")
            if service_data:
                return json.loads(service_data)
        
        return None
    
    async def list_services(self) -> Dict[str, Dict[str, Any]]:
        """Lister tous les services enregistrés"""
        if self.redis_client:
            # Synchroniser avec Redis
            await self._sync_from_redis()
        
        return self.services.copy()
    
    async def health_check_services(self) -> Dict[str, bool]:
        """Vérifier la santé de tous les services"""
        health_results = {}
        
        for service_name, service_info in self.services.items():
            try:
                import aiohttp
                url = f"http://{service_info['host']}:{service_info['port']}{service_info['health_endpoint']}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        health_results[service_name] = response.status == 200
                        
                        if response.status == 200:
                            # Mettre à jour le heartbeat
                            service_info["last_heartbeat"] = datetime.utcnow().isoformat()
                            if self.redis_client:
                                await self._store_in_redis(service_name, service_info)
                            
            except Exception as e:
                self.logger.error(f"Health check failed for {service_name}: {e}")
                health_results[service_name] = False
        
        return health_results
    
    async def _store_in_redis(self, service_name: str, service_info: Dict[str, Any]) -> None:
        """Stocker les informations de service dans Redis"""
        try:
            await self.redis_client.setex(
                f"service:{service_name}",
                300,  # 5 minutes TTL
                json.dumps(service_info)
            )
        except Exception as e:
            self.logger.error(f"Failed to store service {service_name} in Redis: {e}")
    
    async def _sync_from_redis(self) -> None:
        """Synchroniser les services depuis Redis"""
        try:
            keys = await self.redis_client.keys("service:*")
            for key in keys:
                service_data = await self.redis_client.get(key)
                if service_data:
                    service_info = json.loads(service_data)
                    service_name = service_info["name"]
                    self.services[service_name] = service_info
        except Exception as e:
            self.logger.error(f"Failed to sync services from Redis: {e}")


class CircuitBreaker:
    """Circuit breaker pour la résilience des services"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: int = 60,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = LoggerFactory.get_logger("circuit_breaker")
    
    async def call(self, func, *args, **kwargs):
        """Exécuter une fonction avec protection circuit breaker"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Succès - réinitialiser si on était en HALF_OPEN
            if self.state == "HALF_OPEN":
                self._reset()
                self.logger.info("Circuit breaker reset to CLOSED state")
            
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Vérifier si on doit tenter une réinitialisation"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.utcnow().timestamp() - self.last_failure_time) >= self.timeout_duration
    
    def _record_failure(self) -> None:
        """Enregistrer un échec"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
    
    def _reset(self) -> None:
        """Réinitialiser le circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"


class AsyncRetry:
    """Utilitaire pour retry avec backoff exponentiel"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        self.logger = LoggerFactory.get_logger("async_retry")
    
    async def execute(self, func, *args, **kwargs):
        """Exécuter une fonction avec retry"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed",
                        exc_info=True
                    )
                    raise e
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}"
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculer le délai pour une tentative"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Jitter entre 50% et 100%
        
        return delay


def format_file_size(bytes_size: int) -> str:
    """Formater une taille en bytes en format lisible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Tronquer une chaîne si elle est trop longue"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """JSON dumps sécurisé qui gère les types non-sérialisables"""
    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    return json.dumps(obj, default=default_serializer, ensure_ascii=False, **kwargs)