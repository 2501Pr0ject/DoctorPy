"""
Gestionnaire de cache Redis pour DoctorPy
Support pour cache intelligent des réponses IA, sessions, et données temporaires
"""

import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
from redis.asyncio import Redis


class CacheStrategy(Enum):
    """Stratégies de cache disponibles"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_accessed is None:
            self.last_accessed = self.created_at


class CacheManager:
    """Gestionnaire de cache Redis intelligent"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "doctorpy",
        default_ttl: int = 3600,  # 1 heure par défaut
        strategy: CacheStrategy = CacheStrategy.TTL
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.redis_client: Optional[Redis] = None
        self.logger = logging.getLogger(__name__)
        
        # Statistiques de performance
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
    
    async def connect(self) -> None:
        """Établir la connexion Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.logger.info("Cache Redis connecté avec succès")
        except Exception as e:
            self.logger.error(f"Erreur connexion Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Fermer la connexion Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Cache Redis déconnecté")
    
    def _make_key(self, key: str, namespace: str = "default") -> str:
        """Créer une clé Redis avec préfixe et namespace"""
        return f"{self.prefix}:{namespace}:{key}"
    
    def _hash_complex_key(self, data: Union[str, Dict, List]) -> str:
        """Créer un hash pour des clés complexes"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    async def get(
        self,
        key: str,
        namespace: str = "default",
        default: Any = None
    ) -> Any:
        """Récupérer une valeur du cache"""
        if not self.redis_client:
            raise RuntimeError("Cache non connecté")
        
        cache_key = self._make_key(key, namespace)
        
        try:
            # Récupérer la valeur
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data is None:
                self.stats["misses"] += 1
                return default
            
            # Désérialiser
            entry_data = json.loads(cached_data)
            entry = CacheEntry(**entry_data)
            
            # Vérifier l'expiration
            if entry.expires_at and datetime.fromisoformat(entry.expires_at) < datetime.utcnow():
                await self.delete(key, namespace)
                self.stats["misses"] += 1
                return default
            
            # Mettre à jour les statistiques d'accès
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            # Sauvegarder les statistiques mises à jour
            await self._update_entry_stats(cache_key, entry)
            
            self.stats["hits"] += 1
            self.logger.debug(f"Cache hit: {cache_key}")
            
            return entry.value
            
        except Exception as e:
            self.logger.error(f"Erreur récupération cache {cache_key}: {e}")
            self.stats["misses"] += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "default",
        tags: List[str] = None
    ) -> bool:
        """Stocker une valeur dans le cache"""
        if not self.redis_client:
            raise RuntimeError("Cache non connecté")
        
        cache_key = self._make_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        try:
            # Créer l'entrée de cache
            now = datetime.utcnow()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl) if ttl > 0 else None,
                tags=tags or []
            )
            
            # Sérialiser l'entrée
            entry_data = {
                "key": entry.key,
                "value": entry.value,
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                "tags": entry.tags
            }
            
            # Stocker dans Redis
            await self.redis_client.setex(
                cache_key,
                ttl if ttl > 0 else 86400 * 365,  # 1 an si pas d'expiration
                json.dumps(entry_data, ensure_ascii=False)
            )
            
            # Indexer par tags si nécessaire
            if tags:
                await self._index_by_tags(cache_key, tags, namespace)
            
            self.stats["sets"] += 1
            self.logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur stockage cache {cache_key}: {e}")
            return False
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Supprimer une valeur du cache"""
        if not self.redis_client:
            raise RuntimeError("Cache non connecté")
        
        cache_key = self._make_key(key, namespace)
        
        try:
            result = await self.redis_client.delete(cache_key)
            if result:
                self.stats["deletes"] += 1
                self.logger.debug(f"Cache delete: {cache_key}")
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Erreur suppression cache {cache_key}: {e}")
            return False
    
    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Vérifier si une clé existe dans le cache"""
        if not self.redis_client:
            return False
        
        cache_key = self._make_key(key, namespace)
        
        try:
            return bool(await self.redis_client.exists(cache_key))
        except Exception as e:
            self.logger.error(f"Erreur vérification existence {cache_key}: {e}")
            return False
    
    async def clear_namespace(self, namespace: str) -> int:
        """Vider tout un namespace"""
        if not self.redis_client:
            return 0
        
        pattern = self._make_key("*", namespace)
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.stats["deletes"] += deleted
                self.logger.info(f"Namespace {namespace} vidé: {deleted} clés supprimées")
                return deleted
            return 0
            
        except Exception as e:
            self.logger.error(f"Erreur vidage namespace {namespace}: {e}")
            return 0
    
    async def clear_by_tags(self, tags: List[str], namespace: str = "default") -> int:
        """Supprimer les entrées par tags"""
        if not self.redis_client:
            return 0
        
        deleted_count = 0
        
        try:
            for tag in tags:
                tag_key = self._make_key(f"tags:{tag}", namespace)
                tagged_keys = await self.redis_client.smembers(tag_key)
                
                if tagged_keys:
                    # Supprimer les clés taguées
                    keys_to_delete = [key.decode() for key in tagged_keys]
                    deleted = await self.redis_client.delete(*keys_to_delete)
                    deleted_count += deleted
                    
                    # Supprimer l'index de tag
                    await self.redis_client.delete(tag_key)
            
            self.stats["deletes"] += deleted_count
            self.logger.info(f"Suppression par tags {tags}: {deleted_count} clés")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Erreur suppression par tags {tags}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques du cache"""
        if not self.redis_client:
            return self.stats
        
        try:
            # Statistiques Redis
            redis_info = await self.redis_client.info("memory")
            redis_stats = {
                "used_memory": redis_info.get("used_memory", 0),
                "used_memory_human": redis_info.get("used_memory_human", "0B"),
                "maxmemory": redis_info.get("maxmemory", 0)
            }
            
            # Calculer le hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self.stats,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests,
                "redis_stats": redis_stats
            }
            
        except Exception as e:
            self.logger.error(f"Erreur récupération statistiques: {e}")
            return self.stats
    
    async def _update_entry_stats(self, cache_key: str, entry: CacheEntry) -> None:
        """Mettre à jour les statistiques d'une entrée"""
        try:
            entry_data = {
                "key": entry.key,
                "value": entry.value,
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                "tags": entry.tags
            }
            
            ttl = await self.redis_client.ttl(cache_key)
            if ttl > 0:
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(entry_data, ensure_ascii=False)
                )
                
        except Exception as e:
            self.logger.error(f"Erreur mise à jour statistiques entrée: {e}")
    
    async def _index_by_tags(self, cache_key: str, tags: List[str], namespace: str) -> None:
        """Indexer une entrée par ses tags"""
        try:
            for tag in tags:
                tag_key = self._make_key(f"tags:{tag}", namespace)
                await self.redis_client.sadd(tag_key, cache_key)
                
        except Exception as e:
            self.logger.error(f"Erreur indexation par tags: {e}")


class AIResponseCache(CacheManager):
    """Cache spécialisé pour les réponses IA"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__(
            redis_url=redis_url,
            prefix="doctorpy:ai",
            default_ttl=7200,  # 2 heures pour les réponses IA
            strategy=CacheStrategy.LRU
        )
    
    async def get_response(
        self,
        query: str,
        context: Dict[str, Any] = None,
        similarity_threshold: float = 0.95
    ) -> Optional[Dict[str, Any]]:
        """Récupérer une réponse IA mise en cache"""
        # Créer une clé basée sur la requête et le contexte
        cache_data = {
            "query": query.lower().strip(),
            "context": context or {}
        }
        cache_key = self._hash_complex_key(cache_data)
        
        cached_response = await self.get(cache_key, "responses")
        
        if cached_response:
            # Vérifier la similarité de la requête (optionnel)
            # Ici on pourrait ajouter une vérification de similarité sémantique
            return cached_response
        
        return None
    
    async def cache_response(
        self,
        query: str,
        response: Dict[str, Any],
        context: Dict[str, Any] = None,
        ttl: int = None
    ) -> bool:
        """Mettre en cache une réponse IA"""
        cache_data = {
            "query": query.lower().strip(),
            "context": context or {}
        }
        cache_key = self._hash_complex_key(cache_data)
        
        # Enrichir la réponse avec des métadonnées
        enriched_response = {
            **response,
            "original_query": query,
            "cached_at": datetime.utcnow().isoformat(),
            "context": context
        }
        
        return await self.set(
            cache_key,
            enriched_response,
            ttl=ttl,
            namespace="responses",
            tags=["ai_response", "rag"]
        )
    
    async def invalidate_by_knowledge_update(self) -> int:
        """Invalider le cache lors d'une mise à jour de la base de connaissances"""
        return await self.clear_by_tags(["ai_response", "rag"])


class SessionCache(CacheManager):
    """Cache spécialisé pour les sessions utilisateur"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__(
            redis_url=redis_url,
            prefix="doctorpy:session",
            default_ttl=1800,  # 30 minutes pour les sessions
            strategy=CacheStrategy.TTL
        )
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Récupérer une session utilisateur"""
        return await self.get(session_id, "user_sessions")
    
    async def set_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: int = None
    ) -> bool:
        """Stocker une session utilisateur"""
        return await self.set(
            session_id,
            session_data,
            ttl=ttl,
            namespace="user_sessions",
            tags=["session", "user"]
        )
    
    async def extend_session(self, session_id: str, additional_ttl: int = 1800) -> bool:
        """Prolonger une session existante"""
        session_data = await self.get_session(session_id)
        if session_data:
            return await self.set_session(session_id, session_data, additional_ttl)
        return False
    
    async def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalider toutes les sessions d'un utilisateur"""
        # Rechercher toutes les sessions de l'utilisateur
        pattern = self._make_key("*", "user_sessions")
        deleted_count = 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            for key in keys:
                session_data = await self.redis_client.get(key)
                if session_data:
                    session = json.loads(session_data)
                    if session.get("value", {}).get("user_id") == user_id:
                        await self.redis_client.delete(key)
                        deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Erreur invalidation sessions utilisateur {user_id}: {e}")
            return 0


# Factory pour créer les différents types de cache
class CacheFactory:
    """Factory pour créer différents types de cache"""
    
    @staticmethod
    def create_ai_cache(redis_url: str = "redis://localhost:6379") -> AIResponseCache:
        """Créer un cache pour les réponses IA"""
        return AIResponseCache(redis_url)
    
    @staticmethod
    def create_session_cache(redis_url: str = "redis://localhost:6379") -> SessionCache:
        """Créer un cache pour les sessions"""
        return SessionCache(redis_url)
    
    @staticmethod
    def create_general_cache(
        redis_url: str = "redis://localhost:6379",
        prefix: str = "doctorpy",
        default_ttl: int = 3600
    ) -> CacheManager:
        """Créer un cache général"""
        return CacheManager(redis_url, prefix, default_ttl)