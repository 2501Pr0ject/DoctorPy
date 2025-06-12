"""
Gestionnaire d'authentification et JWT pour DoctorPy
"""

import jwt
import bcrypt
import secrets
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict

from ..shared.config import get_auth_config
from ..shared.utils import LoggerFactory
from ..shared.events import EventBus, Event, EventType

from .models import User, Session, AuthToken, UserRole, SecurityEvent
from .database import AuthDatabase


class JWTHandler:
    """Gestionnaire JWT pour les tokens d'authentification"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.logger = LoggerFactory.get_logger("jwt_handler")
    
    def create_access_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None,
        scopes: list = None
    ) -> str:
        """Créer un token d'accès JWT"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "scopes": scopes or []
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Créer un token de rafraîchissement"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=7)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Vérifier et décoder un token JWT"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expiré")
            return None
        except jwt.JWTError as e:
            self.logger.warning(f"Token invalide: {e}")
            return None
    
    def create_reset_token(self, user_id: str) -> str:
        """Créer un token de réinitialisation de mot de passe"""
        expire = datetime.utcnow() + timedelta(hours=1)  # 1 heure
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "password_reset",
            "jti": secrets.token_urlsafe(32)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token


class PasswordManager:
    """Gestionnaire de mots de passe avec bcrypt"""
    
    def __init__(self, rounds: int = 12):
        self.rounds = rounds
        self.logger = LoggerFactory.get_logger("password_manager")
    
    def hash_password(self, password: str) -> str:
        """Hasher un mot de passe"""
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Vérifier un mot de passe"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            self.logger.error(f"Erreur vérification mot de passe: {e}")
            return False
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Générer un mot de passe sécurisé"""
        import string
        import random
        
        characters = (
            string.ascii_lowercase +
            string.ascii_uppercase +
            string.digits +
            "!@#$%^&*"
        )
        
        password = ''.join(random.choice(characters) for _ in range(length))
        return password


class AuthManager:
    """Gestionnaire principal d'authentification"""
    
    def __init__(
        self,
        database: AuthDatabase,
        cache_manager,
        event_bus: EventBus
    ):
        self.db = database
        self.cache = cache_manager
        self.event_bus = event_bus
        
        config = get_auth_config()
        self.jwt_handler = JWTHandler(
            config.auth.secret_key,
            config.auth.algorithm
        )
        self.password_manager = PasswordManager(config.auth.bcrypt_rounds)
        self.access_token_expire = timedelta(minutes=config.auth.access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=config.auth.refresh_token_expire_days)
        
        # Limites de sécurité
        self.max_login_attempts = config.custom_config.get("max_login_attempts", 5)
        self.lockout_duration = timedelta(
            minutes=config.custom_config.get("lockout_duration_minutes", 15)
        )
        
        self.logger = LoggerFactory.get_logger("auth_manager")
    
    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Créer un nouvel utilisateur"""
        # Vérifier si l'utilisateur existe déjà
        existing_user = await self.db.get_user_by_email(email)
        if existing_user:
            raise ValueError("Un utilisateur avec cet email existe déjà")
        
        existing_username = await self.db.get_user_by_username(username)
        if existing_username:
            raise ValueError("Ce nom d'utilisateur est déjà pris")
        
        # Hasher le mot de passe
        hashed_password = self.password_manager.hash_password(password)
        
        # Créer l'utilisateur
        user = User(
            id=secrets.token_urlsafe(16),
            email=email,
            username=username,
            hashed_password=hashed_password,
            role=role
        )
        
        # Sauvegarder en base
        await self.db.create_user(user)
        
        # Publier événement
        await self.event_bus.publish(
            Event(
                type=EventType.USER_REGISTERED,
                data={
                    "user_id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "role": user.role.value
                },
                source_service="auth_service",
                user_id=user.id
            )
        )
        
        self.logger.info(f"Utilisateur créé: {user.email}")
        return user
    
    async def authenticate(
        self,
        email: str,
        password: str,
        ip_address: str = None,
        user_agent: str = None
    ) -> Tuple[Optional[User], Optional[Session]]:
        """Authentifier un utilisateur"""
        # Récupérer l'utilisateur
        user = await self.db.get_user_by_email(email)
        if not user:
            await self._log_security_event(
                "login_failed",
                None,
                ip_address,
                user_agent,
                {"reason": "user_not_found", "email": email}
            )
            return None, None
        
        # Vérifier si le compte est verrouillé
        if user.is_locked():
            await self._log_security_event(
                "login_blocked",
                user.id,
                ip_address,
                user_agent,
                {"reason": "account_locked", "locked_until": user.locked_until.isoformat()}
            )
            return None, None
        
        # Vérifier le mot de passe
        if not self.password_manager.verify_password(password, user.hashed_password):
            # Incrémenter les tentatives de connexion
            user.login_attempts += 1
            
            # Verrouiller le compte si trop de tentatives
            if user.login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
                await self._log_security_event(
                    "account_locked",
                    user.id,
                    ip_address,
                    user_agent,
                    {"attempts": user.login_attempts, "locked_until": user.locked_until.isoformat()}
                )
            
            await self.db.update_user(user)
            
            await self._log_security_event(
                "login_failed",
                user.id,
                ip_address,
                user_agent,
                {"reason": "invalid_password", "attempts": user.login_attempts}
            )
            return None, None
        
        # Authentification réussie
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        await self.db.update_user(user)
        
        # Créer une session
        session = await self._create_session(user, ip_address, user_agent)
        
        # Publier événement
        await self.event_bus.publish(
            Event(
                type=EventType.USER_AUTHENTICATED,
                data={
                    "user_id": user.id,
                    "session_id": session.session_id,
                    "ip_address": ip_address,
                    "user_agent": user_agent
                },
                source_service="auth_service",
                user_id=user.id
            )
        )
        
        await self._log_security_event(
            "login_success",
            user.id,
            ip_address,
            user_agent,
            {"session_id": session.session_id}
        )
        
        self.logger.info(f"Authentification réussie: {user.email}")
        return user, session
    
    async def refresh_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """Rafraîchir un token d'accès"""
        # Vérifier le refresh token
        payload = self.jwt_handler.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("sub")
        user = await self.db.get_user_by_id(user_id)
        if not user or not user.is_active:
            return None
        
        # Créer de nouveaux tokens
        new_access_token = self.jwt_handler.create_access_token(
            user_id,
            self.access_token_expire
        )
        new_refresh_token = self.jwt_handler.create_refresh_token(
            user_id,
            self.refresh_token_expire
        )
        
        # Mettre à jour la session dans le cache
        session_key = f"session:{user_id}"
        session_data = await self.cache.get(session_key)
        if session_data:
            session_data["access_token"] = new_access_token
            session_data["refresh_token"] = new_refresh_token
            await self.cache.set(session_key, session_data)
        
        return new_access_token, new_refresh_token
    
    async def logout(self, user_id: str, session_id: str = None) -> bool:
        """Déconnecter un utilisateur"""
        try:
            # Invalider la session spécifique ou toutes les sessions
            if session_id:
                await self.cache.delete(f"session:{session_id}")
            else:
                await self.cache.invalidate_user_sessions(user_id)
            
            # Publier événement
            await self.event_bus.publish(
                Event(
                    type=EventType.USER_LOGOUT,
                    data={
                        "user_id": user_id,
                        "session_id": session_id
                    },
                    source_service="auth_service",
                    user_id=user_id
                )
            )
            
            self.logger.info(f"Déconnexion utilisateur: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur déconnexion: {e}")
            return False
    
    async def verify_token(self, token: str) -> Optional[User]:
        """Vérifier un token et retourner l'utilisateur"""
        payload = self.jwt_handler.verify_token(token)
        if not payload or payload.get("type") != "access":
            return None
        
        user_id = payload.get("sub")
        user = await self.db.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            return None
        
        return user
    
    async def create_password_reset_token(self, email: str) -> Optional[str]:
        """Créer un token de réinitialisation de mot de passe"""
        user = await self.db.get_user_by_email(email)
        if not user:
            return None
        
        reset_token = self.jwt_handler.create_reset_token(user.id)
        
        # Stocker le token dans le cache (1 heure)
        await self.cache.set(
            f"reset_token:{user.id}",
            reset_token,
            ttl=3600  # 1 heure
        )
        
        return reset_token
    
    async def reset_password(self, token: str, new_password: str) -> bool:
        """Réinitialiser le mot de passe avec un token"""
        payload = self.jwt_handler.verify_token(token)
        if not payload or payload.get("type") != "password_reset":
            return False
        
        user_id = payload.get("sub")
        
        # Vérifier que le token est dans le cache
        cached_token = await self.cache.get(f"reset_token:{user_id}")
        if cached_token != token:
            return False
        
        # Récupérer l'utilisateur
        user = await self.db.get_user_by_id(user_id)
        if not user:
            return False
        
        # Mettre à jour le mot de passe
        user.hashed_password = self.password_manager.hash_password(new_password)
        user.updated_at = datetime.utcnow()
        await self.db.update_user(user)
        
        # Supprimer le token du cache
        await self.cache.delete(f"reset_token:{user_id}")
        
        # Invalider toutes les sessions de l'utilisateur
        await self.cache.invalidate_user_sessions(user_id)
        
        self.logger.info(f"Mot de passe réinitialisé: {user.email}")
        return True
    
    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """Changer le mot de passe d'un utilisateur"""
        user = await self.db.get_user_by_id(user_id)
        if not user:
            return False
        
        # Vérifier le mot de passe actuel
        if not self.password_manager.verify_password(current_password, user.hashed_password):
            return False
        
        # Mettre à jour le mot de passe
        user.hashed_password = self.password_manager.hash_password(new_password)
        user.updated_at = datetime.utcnow()
        await self.db.update_user(user)
        
        # Invalider toutes les sessions sauf la courante (optionnel)
        # await self.cache.invalidate_user_sessions(user_id)
        
        self.logger.info(f"Mot de passe changé: {user.email}")
        return True
    
    async def _create_session(
        self,
        user: User,
        ip_address: str = None,
        user_agent: str = None
    ) -> Session:
        """Créer une session utilisateur"""
        session_id = secrets.token_urlsafe(32)
        
        # Créer les tokens
        access_token = self.jwt_handler.create_access_token(
            user.id,
            self.access_token_expire
        )
        refresh_token = self.jwt_handler.create_refresh_token(
            user.id,
            self.refresh_token_expire
        )
        
        # Créer la session
        session = Session(
            session_id=session_id,
            user_id=user.id,
            access_token=access_token,
            refresh_token=refresh_token,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.access_token_expire,
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Stocker dans le cache
        await self.cache.set_session(session_id, session.to_dict())
        
        return session
    
    async def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        details: Dict[str, Any]
    ) -> None:
        """Logger un événement de sécurité"""
        security_event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            timestamp=datetime.utcnow().isoformat(),
            severity="warning" if "failed" in event_type else "info"
        )
        
        # Logger l'événement
        if security_event.severity == "warning":
            self.logger.warning(f"Security event: {event_type}", extra=details)
        else:
            self.logger.info(f"Security event: {event_type}", extra=details)
        
        # Stocker dans la base (optionnel)
        # await self.db.log_security_event(security_event)