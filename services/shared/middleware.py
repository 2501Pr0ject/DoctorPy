"""
Middleware communs pour tous les microservices DoctorPy
"""

import time
import uuid
import logging
from typing import Callable, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from .config import ServiceConfig
from .utils import LoggerFactory


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour logger toutes les requêtes"""
    
    def __init__(self, app: FastAPI, logger_name: str = "requests"):
        super().__init__(app)
        self.logger = LoggerFactory.get_logger(logger_name)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Générer un ID de requête unique
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Ajouter l'ID à la requête
        request.state.request_id = request_id
        
        # Logger la requête entrante
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        try:
            # Traiter la requête
            response = await call_next(request)
            
            # Calculer le temps de traitement
            process_time = time.time() - start_time
            
            # Logger la réponse
            self.logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4),
                    "response_size": response.headers.get("content-length", 0)
                }
            )
            
            # Ajouter l'ID de requête dans les headers de réponse
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Logger l'erreur
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time": round(process_time, 4)
                },
                exc_info=True
            )
            
            # Retourner une réponse d'erreur standardisée
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"X-Request-ID": request_id}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de limitation de taux"""
    
    def __init__(
        self,
        app: FastAPI,
        calls: int = 100,
        period: int = 60,
        redis_client = None
    ):
        super().__init__(app)
        self.calls = calls  # Nombre d'appels autorisés
        self.period = period  # Période en secondes
        self.redis_client = redis_client
        self.logger = LoggerFactory.get_logger("rate_limit")
        
        # Cache local si Redis n'est pas disponible
        self.local_cache: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Obtenir l'identifiant client (IP + User-Agent)
        client_id = self._get_client_id(request)
        
        # Vérifier la limite
        if await self._is_rate_limited(client_id):
            self.logger.warning(
                f"Rate limit exceeded for client {client_id}",
                extra={
                    "client_id": client_id,
                    "endpoint": str(request.url),
                    "method": request.method
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls} requests per {self.period} seconds",
                    "retry_after": self.period
                },
                headers={"Retry-After": str(self.period)}
            )
        
        # Incrémenter le compteur
        await self._increment_counter(client_id)
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Générer un identifiant unique pour le client"""
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        return f"{client_ip}:{hash(user_agent) % 10000}"
    
    async def _is_rate_limited(self, client_id: str) -> bool:
        """Vérifier si le client a dépassé la limite"""
        if self.redis_client:
            return await self._check_redis_limit(client_id)
        else:
            return self._check_local_limit(client_id)
    
    async def _check_redis_limit(self, client_id: str) -> bool:
        """Vérifier la limite avec Redis"""
        try:
            key = f"rate_limit:{client_id}"
            current_count = await self.redis_client.get(key)
            
            if current_count is None:
                return False
            
            return int(current_count) >= self.calls
            
        except Exception as e:
            self.logger.error(f"Erreur vérification rate limit Redis: {e}")
            return False
    
    def _check_local_limit(self, client_id: str) -> bool:
        """Vérifier la limite avec cache local"""
        now = time.time()
        
        if client_id not in self.local_cache:
            return False
        
        client_data = self.local_cache[client_id]
        
        # Nettoyer les entrées expirées
        client_data["timestamps"] = [
            ts for ts in client_data["timestamps"]
            if now - ts < self.period
        ]
        
        return len(client_data["timestamps"]) >= self.calls
    
    async def _increment_counter(self, client_id: str) -> None:
        """Incrémenter le compteur de requêtes"""
        if self.redis_client:
            await self._increment_redis_counter(client_id)
        else:
            self._increment_local_counter(client_id)
    
    async def _increment_redis_counter(self, client_id: str) -> None:
        """Incrémenter avec Redis"""
        try:
            key = f"rate_limit:{client_id}"
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, self.period)
            
        except Exception as e:
            self.logger.error(f"Erreur incrémentation Redis: {e}")
    
    def _increment_local_counter(self, client_id: str) -> None:
        """Incrémenter avec cache local"""
        now = time.time()
        
        if client_id not in self.local_cache:
            self.local_cache[client_id] = {"timestamps": []}
        
        self.local_cache[client_id]["timestamps"].append(now)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware pour ajouter des headers de sécurité"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Headers de sécurité
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware pour les vérifications de santé"""
    
    def __init__(self, app: FastAPI, health_endpoint: str = "/health"):
        super().__init__(app)
        self.health_endpoint = health_endpoint
        self.start_time = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Traitement spécial pour l'endpoint de santé
        if request.url.path == self.health_endpoint:
            return await self._handle_health_check(request)
        
        return await call_next(request)
    
    async def _handle_health_check(self, request: Request) -> Response:
        """Traiter la vérification de santé"""
        uptime = time.time() - self.start_time
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": round(uptime, 2),
            "service": getattr(request.app.state, "service_name", "unknown"),
            "version": getattr(request.app.state, "service_version", "unknown")
        }
        
        return JSONResponse(content=health_data)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware pour collecter des métriques"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.metrics = {
            "requests_total": 0,
            "requests_duration_sum": 0.0,
            "requests_by_method": {},
            "requests_by_status": {},
            "active_requests": 0
        }
        self.logger = LoggerFactory.get_logger("metrics")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Incrémenter les métriques
        self.metrics["requests_total"] += 1
        self.metrics["active_requests"] += 1
        
        method = request.method
        if method not in self.metrics["requests_by_method"]:
            self.metrics["requests_by_method"][method] = 0
        self.metrics["requests_by_method"][method] += 1
        
        try:
            response = await call_next(request)
            
            # Métriques de réponse
            status_code = response.status_code
            if status_code not in self.metrics["requests_by_status"]:
                self.metrics["requests_by_status"][status_code] = 0
            self.metrics["requests_by_status"][status_code] += 1
            
            return response
            
        finally:
            # Calculer la durée
            duration = time.time() - start_time
            self.metrics["requests_duration_sum"] += duration
            self.metrics["active_requests"] -= 1
            
            # Logger les métriques périodiquement
            if self.metrics["requests_total"] % 100 == 0:
                avg_duration = (
                    self.metrics["requests_duration_sum"] / 
                    self.metrics["requests_total"]
                )
                
                self.logger.info(
                    "Metrics update",
                    extra={
                        "total_requests": self.metrics["requests_total"],
                        "avg_duration": round(avg_duration, 4),
                        "active_requests": self.metrics["active_requests"],
                        "requests_by_method": self.metrics["requests_by_method"],
                        "requests_by_status": self.metrics["requests_by_status"]
                    }
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtenir les métriques actuelles"""
        avg_duration = (
            self.metrics["requests_duration_sum"] / 
            self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "avg_duration": round(avg_duration, 4)
        }


class CommonMiddleware:
    """Classe utilitaire pour configurer tous les middlewares"""
    
    @staticmethod
    def setup_middleware(app: FastAPI, config: ServiceConfig) -> None:
        """Configurer tous les middlewares pour une app FastAPI"""
        
        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if config.is_development() else ["https://yourdomain.com"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression GZIP
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Sessions
        app.add_middleware(
            SessionMiddleware,
            secret_key=config.auth.secret_key,
            max_age=config.auth.access_token_expire_minutes * 60
        )
        
        # Middleware personnalisés
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(RequestLoggingMiddleware, logger_name=config.service_name)
        app.add_middleware(MetricsMiddleware)
        app.add_middleware(HealthCheckMiddleware)
        
        # Rate limiting (optionnel en développement)
        if not config.is_development():
            app.add_middleware(
                RateLimitMiddleware,
                calls=1000,  # 1000 requêtes
                period=60    # par minute
            )
        
        # Stocker la configuration dans l'app
        app.state.service_name = config.service_name
        app.state.service_version = "2.0.0"
        app.state.config = config
        
        # Endpoint de métriques
        @app.get("/metrics")
        async def get_metrics():
            """Endpoint pour récupérer les métriques"""
            metrics_middleware = None
            for middleware in app.user_middleware:
                if isinstance(middleware.cls, type) and issubclass(middleware.cls, MetricsMiddleware):
                    metrics_middleware = middleware
                    break
            
            if metrics_middleware:
                return {"metrics": "Metrics endpoint - implement Prometheus format"}
            else:
                return {"error": "Metrics middleware not found"}
        
        # Endpoint de configuration (en développement seulement)
        if config.is_development():
            @app.get("/debug/config")
            async def get_config():
                """Endpoint pour récupérer la configuration (debug uniquement)"""
                return config.to_dict()