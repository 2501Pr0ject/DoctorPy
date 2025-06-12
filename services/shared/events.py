"""
Event-Driven Architecture pour DoctorPy
Support Redis et RabbitMQ pour la communication inter-services
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis
import aio_pika
from aio_pika import connect_robust, ExchangeType


class EventType(Enum):
    """Types d'événements système"""
    # Authentication Events
    USER_AUTHENTICATED = "user.authenticated"
    USER_LOGOUT = "user.logout"
    USER_REGISTERED = "user.registered"
    
    # Quest Events
    QUEST_STARTED = "quest.started"
    QUEST_COMPLETED = "quest.completed"
    QUEST_FAILED = "quest.failed"
    PROGRESS_UPDATED = "quest.progress_updated"
    
    # RAG Events
    RAG_QUERY_PROCESSED = "rag.query_processed"
    RAG_KNOWLEDGE_UPDATED = "rag.knowledge_updated"
    RAG_EMBEDDING_GENERATED = "rag.embedding_generated"
    
    # Analytics Events
    METRIC_UPDATED = "analytics.metric_updated"
    REPORT_GENERATED = "analytics.report_generated"
    ALERT_TRIGGERED = "analytics.alert_triggered"
    
    # System Events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_MAINTENANCE = "system.maintenance"
    SYSTEM_ERROR = "system.error"
    
    # Notification Events
    NOTIFICATION_SENT = "notification.sent"
    NOTIFICATION_FAILED = "notification.failed"


@dataclass
class Event:
    """Structure d'événement standardisée"""
    type: EventType
    data: Dict[str, Any]
    source_service: str
    timestamp: str = None
    correlation_id: str = None
    user_id: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.correlation_id:
            import uuid
            self.correlation_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour sérialisation"""
        result = asdict(self)
        result['type'] = self.type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Créer depuis dictionnaire"""
        data['type'] = EventType(data['type'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Sérialiser en JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Désérialiser depuis JSON"""
        return cls.from_dict(json.loads(json_str))


class EventHandler(ABC):
    """Interface pour les handlers d'événements"""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Traiter un événement"""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """Vérifier si ce handler peut traiter ce type d'événement"""
        pass


class EventBus(ABC):
    """Interface abstraite pour le bus d'événements"""
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publier un événement"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_types: List[EventType], handler: EventHandler) -> None:
        """S'abonner à des types d'événements"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Démarrer le bus d'événements"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Arrêter le bus d'événements"""
        pass


class RedisEventBus(EventBus):
    """Implémentation Redis du bus d'événements"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.pubsub = None
        self.logger = logging.getLogger(__name__)
        self._running = False
    
    async def start(self) -> None:
        """Démarrer la connexion Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            self._running = True
            self.logger.info("Redis EventBus démarré avec succès")
        except Exception as e:
            self.logger.error(f"Erreur démarrage Redis EventBus: {e}")
            raise
    
    async def stop(self) -> None:
        """Arrêter la connexion Redis"""
        self._running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("Redis EventBus arrêté")
    
    async def publish(self, event: Event) -> None:
        """Publier un événement via Redis"""
        if not self.redis_client:
            raise RuntimeError("EventBus non démarré")
        
        channel = f"events:{event.type.value}"
        try:
            await self.redis_client.publish(channel, event.to_json())
            self.logger.debug(f"Événement publié: {event.type.value}")
        except Exception as e:
            self.logger.error(f"Erreur publication événement: {e}")
            raise
    
    async def subscribe(self, event_types: List[EventType], handler: EventHandler) -> None:
        """S'abonner à des types d'événements"""
        for event_type in event_types:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
            # S'abonner au canal Redis
            channel = f"events:{event_type.value}"
            await self.pubsub.subscribe(channel)
        
        self.logger.info(f"Abonnement créé pour {len(event_types)} types d'événements")
        
        # Démarrer l'écoute des messages
        if not hasattr(self, '_listener_task'):
            self._listener_task = asyncio.create_task(self._listen_messages())
    
    async def _listen_messages(self) -> None:
        """Écouter les messages Redis en continu"""
        try:
            while self._running:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message['type'] == 'message':
                    await self._handle_message(message)
        except Exception as e:
            self.logger.error(f"Erreur écoute messages: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Traiter un message reçu"""
        try:
            event = Event.from_json(message['data'])
            event_type = event.type
            
            if event_type in self.handlers:
                # Traiter avec tous les handlers enregistrés
                tasks = []
                for handler in self.handlers[event_type]:
                    if handler.can_handle(event_type):
                        tasks.append(handler.handle(event))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self.logger.debug(f"Événement traité: {event_type.value}")
        
        except Exception as e:
            self.logger.error(f"Erreur traitement message: {e}")


class RabbitMQEventBus(EventBus):
    """Implémentation RabbitMQ du bus d'événements"""
    
    def __init__(self, rabbitmq_url: str = "amqp://localhost:5672"):
        self.rabbitmq_url = rabbitmq_url
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.logger = logging.getLogger(__name__)
        self._running = False
    
    async def start(self) -> None:
        """Démarrer la connexion RabbitMQ"""
        try:
            self.connection = await connect_robust(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            
            # Créer l'exchange pour les événements
            self.exchange = await self.channel.declare_exchange(
                "doctorpy_events",
                ExchangeType.TOPIC,
                durable=True
            )
            
            self._running = True
            self.logger.info("RabbitMQ EventBus démarré avec succès")
        
        except Exception as e:
            self.logger.error(f"Erreur démarrage RabbitMQ EventBus: {e}")
            raise
    
    async def stop(self) -> None:
        """Arrêter la connexion RabbitMQ"""
        self._running = False
        if self.connection:
            await self.connection.close()
        self.logger.info("RabbitMQ EventBus arrêté")
    
    async def publish(self, event: Event) -> None:
        """Publier un événement via RabbitMQ"""
        if not self.exchange:
            raise RuntimeError("EventBus non démarré")
        
        routing_key = event.type.value
        message = aio_pika.Message(
            event.to_json().encode(),
            content_type="application/json",
            headers={
                "source_service": event.source_service,
                "correlation_id": event.correlation_id,
                "timestamp": event.timestamp
            }
        )
        
        try:
            await self.exchange.publish(message, routing_key=routing_key)
            self.logger.debug(f"Événement publié: {event.type.value}")
        except Exception as e:
            self.logger.error(f"Erreur publication événement: {e}")
            raise
    
    async def subscribe(self, event_types: List[EventType], handler: EventHandler) -> None:
        """S'abonner à des types d'événements"""
        if not self.channel:
            raise RuntimeError("EventBus non démarré")
        
        # Créer une queue unique pour ce service
        service_name = handler.__class__.__name__.lower()
        queue_name = f"doctorpy_{service_name}_queue"
        
        queue = await self.channel.declare_queue(queue_name, durable=True)
        
        # Lier la queue aux types d'événements
        for event_type in event_types:
            await queue.bind(self.exchange, routing_key=event_type.value)
            
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
        
        # Commencer à consommer les messages
        await queue.consume(self._handle_message)
        
        self.logger.info(f"Abonnement créé pour {len(event_types)} types d'événements")
    
    async def _handle_message(self, message: aio_pika.IncomingMessage) -> None:
        """Traiter un message reçu"""
        async with message.process():
            try:
                event = Event.from_json(message.body.decode())
                event_type = event.type
                
                if event_type in self.handlers:
                    # Traiter avec tous les handlers enregistrés
                    tasks = []
                    for handler in self.handlers[event_type]:
                        if handler.can_handle(event_type):
                            tasks.append(handler.handle(event))
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                        self.logger.debug(f"Événement traité: {event_type.value}")
            
            except Exception as e:
                self.logger.error(f"Erreur traitement message: {e}")
                # En cas d'erreur, rejeter le message pour retry
                raise


class EventBusFactory:
    """Factory pour créer le bon type de bus d'événements"""
    
    @staticmethod
    def create(bus_type: str = "redis", **kwargs) -> EventBus:
        """Créer un bus d'événements selon le type"""
        if bus_type.lower() == "redis":
            return RedisEventBus(kwargs.get("redis_url", "redis://localhost:6379"))
        elif bus_type.lower() == "rabbitmq":
            return RabbitMQEventBus(kwargs.get("rabbitmq_url", "amqp://localhost:5672"))
        else:
            raise ValueError(f"Type de bus non supporté: {bus_type}")


# Exemples d'handlers pour différents services
class AuthEventHandler(EventHandler):
    """Handler pour les événements d'authentification"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"{service_name}.events")
    
    async def handle(self, event: Event) -> None:
        """Traiter un événement d'authentification"""
        if event.type == EventType.USER_AUTHENTICATED:
            await self._handle_user_authenticated(event)
        elif event.type == EventType.USER_REGISTERED:
            await self._handle_user_registered(event)
    
    def can_handle(self, event_type: EventType) -> bool:
        """Vérifier si ce handler peut traiter ce type d'événement"""
        return event_type in [
            EventType.USER_AUTHENTICATED,
            EventType.USER_REGISTERED,
            EventType.USER_LOGOUT
        ]
    
    async def _handle_user_authenticated(self, event: Event) -> None:
        """Traiter l'authentification d'un utilisateur"""
        user_id = event.data.get("user_id")
        self.logger.info(f"Utilisateur authentifié: {user_id}")
        # Logique métier spécifique au service
    
    async def _handle_user_registered(self, event: Event) -> None:
        """Traiter l'enregistrement d'un nouvel utilisateur"""
        user_data = event.data
        self.logger.info(f"Nouvel utilisateur enregistré: {user_data.get('email')}")
        # Logique métier spécifique au service


class RAGEventHandler(EventHandler):
    """Handler pour les événements RAG"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"{service_name}.events")
    
    async def handle(self, event: Event) -> None:
        """Traiter un événement RAG"""
        if event.type == EventType.RAG_QUERY_PROCESSED:
            await self._handle_query_processed(event)
        elif event.type == EventType.RAG_KNOWLEDGE_UPDATED:
            await self._handle_knowledge_updated(event)
    
    def can_handle(self, event_type: EventType) -> bool:
        """Vérifier si ce handler peut traiter ce type d'événement"""
        return event_type in [
            EventType.RAG_QUERY_PROCESSED,
            EventType.RAG_KNOWLEDGE_UPDATED,
            EventType.RAG_EMBEDDING_GENERATED
        ]
    
    async def _handle_query_processed(self, event: Event) -> None:
        """Traiter une requête RAG"""
        query_data = event.data
        self.logger.info(f"Requête RAG traitée: {query_data.get('query')}")
        # Analytics, cache, etc.
    
    async def _handle_knowledge_updated(self, event: Event) -> None:
        """Traiter la mise à jour de la base de connaissances"""
        update_data = event.data
        self.logger.info(f"Base de connaissances mise à jour: {update_data.get('documents_count')} documents")
        # Invalidation du cache, notifications, etc.