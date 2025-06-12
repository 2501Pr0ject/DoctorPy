# src/api/middleware/auth.py
"""
Middleware et utilitaires d'authentification
"""

from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import jwt
import logging
from datetime import datetime, timedelta
import hashlib
import secrets
from passlib.context import CryptContext

from src.core.config import get_settings
from src.core.database import get_db_session
from src.core.exceptions import AuthenticationError
from src.models import User
from src.models.schemas import UserBase

logger = logging.getLogger(__name__)

# Configuration
settings = get_settings()
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Modèles d'authentification
class TokenData(BaseModel):
    user_id: int
    username: str
    email: str
    is_admin: bool = False
    exp: datetime
    iat: datetime


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserBase


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# Utilitaires de mot de passe
def hash_password(password: str) -> str:
    """Hash un mot de passe"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe"""
    return pwd_context.verify(plain_password, hashed_password)


# Utilitaires JWT
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Crée un token JWT d'accès
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """
    Crée un token de rafraîchissement
    """
    data = {
        "user_id": user_id,
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
        "jti": secrets.token_urlsafe(32)  # JWT ID unique
    }
    
    return jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_token(token: str) -> Optional[TokenData]:
    """
    Vérifie et décode un token JWT
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Vérifier le type de token
        if payload.get("type") != "access":
            return None
        
        # Extraire les données
        user_id = payload.get("user_id")
        username = payload.get("username")
        email = payload.get("email")
        is_admin = payload.get("is_admin", False)
        exp = datetime.fromtimestamp(payload.get("exp", 0))
        iat = datetime.fromtimestamp(payload.get("iat", 0))
        
        if not all([user_id, username, email]):
            return None
        
        return TokenData(
            user_id=user_id,
            username=username,
            email=email,
            is_admin=is_admin,
            exp=exp,
            iat=iat
        )
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token expiré")
        return None
    except jwt.JWTError as e:
        logger.warning(f"Erreur JWT: {e}")
        return None


async def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authentifie un utilisateur
    """
    try:
        async with get_db_session() as session:
            # Chercher l'utilisateur par nom d'utilisateur ou email
            query = """
                SELECT id, username, email, password_hash, is_active, is_admin, full_name
                FROM users 
                WHERE (username = ? OR email = ?) AND is_active = true
            """
            result = await session.execute(query, [username, username])
            user_data = result.fetchone()
            
            if not user_data:
                return None
            
            # Vérifier le mot de passe
            if not verify_password(password, user_data[3]):
                return None
            
            # Mettre à jour la dernière connexion
            update_query = "UPDATE users SET last_login = ? WHERE id = ?"
            await session.execute(update_query, [datetime.utcnow(), user_data[0]])
            await session.commit()
            
            # Créer l'objet utilisateur
            return User(
                id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                password_hash=user_data[3],
                is_active=user_data[4],
                is_admin=user_data[5],
                full_name=user_data[6],
                last_login=datetime.utcnow()
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de l'authentification: {e}")
        return None


# Dépendances FastAPI
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[UserBase]:
    """
    Récupère l'utilisateur actuel depuis le token JWT (optionnel)
    """
    if not credentials:
        return None
    
    token_data = verify_token(credentials.credentials)
    if not token_data:
        return None
    
    return UserBase(
        id=token_data.user_id,
        username=token_data.username,
        email=token_data.email,
        is_admin=token_data.is_admin
    )


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserBase:
    """
    Dépendance qui exige une authentification valide
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token d'authentification requis",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = verify_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Vérifier que l'utilisateur existe toujours et est actif
    try:
        async with get_db_session() as session:
            query = "SELECT id, is_active FROM users WHERE id = ?"
            result = await session.execute(query, [token_data.user_id])
            user_check = result.fetchone()
            
            if not user_check or not user_check[1]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Utilisateur inactif ou supprimé"
                )
    except Exception as e:
        logger.error(f"Erreur lors de la vérification utilisateur: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur de vérification d'authentification"
        )
    
    return UserBase(
        id=token_data.user_id,
        username=token_data.username,
        email=token_data.email,
        is_admin=token_data.is_admin
    )


async def require_admin(current_user: UserBase = Depends(require_auth)) -> UserBase:
    """
    Dépendance qui exige des droits administrateur
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Droits administrateur requis"
        )
    
    return current_user


async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[UserBase]:
    """
    Récupère l'utilisateur actuel de manière optionnelle (pour les endpoints publics)
    """
    return await get_current_user(credentials)


# Gestionnaire de sessions de sécurité
class SecuritySessionManager:
    """
    Gestionnaire pour les sessions de sécurité (tentatives de connexion, etc.)
    """
    
    def __init__(self):
        self.failed_attempts: Dict[str, list] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self.max_attempts = 5
        self.block_duration = timedelta(minutes=15)
        self.attempt_window = timedelta(minutes=5)
    
    def is_blocked(self, ip: str) -> bool:
        """Vérifie si une IP est bloquée"""
        if ip in self.blocked_ips:
            if datetime.utcnow() - self.blocked_ips[ip] < self.block_duration:
                return True
            else:
                # Le blocage a expiré
                del self.blocked_ips[ip]
                self.failed_attempts.pop(ip, None)
        return False
    
    def record_failed_attempt(self, ip: str):
        """Enregistre une tentative de connexion échouée"""
        now = datetime.utcnow()
        
        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = []
        
        # Nettoyer les anciennes tentatives
        self.failed_attempts[ip] = [
            attempt for attempt in self.failed_attempts[ip]
            if now - attempt < self.attempt_window
        ]
        
        # Ajouter la nouvelle tentative
        self.failed_attempts[ip].append(now)
        
        # Vérifier si on doit bloquer l'IP
        if len(self.failed_attempts[ip]) >= self.max_attempts:
            self.blocked_ips[ip] = now
            logger.warning(f"IP {ip} bloquée pour tentatives de connexion multiples")
    
    def record_successful_login(self, ip: str):
        """Enregistre une connexion réussie (reset les tentatives)"""
        self.failed_attempts.pop(ip, None)
        self.blocked_ips.pop(ip, None)


# Instance globale du gestionnaire de sécurité
security_manager = SecuritySessionManager()


# Middleware d'authentification
class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour gérer l'authentification et les logs de sécurité
    """
    
    def __init__(self, app, exempt_paths: Optional[list] = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Enregistrer l'IP et l'User-Agent
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Ajouter les informations au request state
        request.state.client_ip = client_ip
        request.state.user_agent = user_agent
        request.state.start_time = datetime.utcnow()
        
        # Vérifier si le chemin est exempt d'authentification
        path = request.url.path
        if any(path.startswith(exempt_path) for exempt_path in self.exempt_paths):
            response = await call_next(request)
            return response
        
        # Extraire et valider le token pour les chemins protégés
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            token_data = verify_token(token)
            
            if token_data:
                # Ajouter les informations utilisateur au request state
                request.state.user_id = token_data.user_id
                request.state.username = token_data.username
                request.state.is_admin = token_data.is_admin
                
                # Logger la requête authentifiée
                logger.info(f"Requête authentifiée: {token_data.username} -> {path}")
            else:
                # Token invalide
                logger.warning(f"Token invalide pour {path} depuis {client_ip}")
        
        response = await call_next(request)
        
        # Enregistrer la durée de traitement
        if hasattr(request.state, 'start_time'):
            duration = (datetime.utcnow() - request.state.start_time).total_seconds()
            response.headers["X-Process-Time"] = str(duration)
        
        return response


# Middleware de protection contre le brute force
class BruteForceProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware de protection contre les attaques par force brute
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.protected_paths = ["/api/v1/auth/login"]
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        
        # Vérifier si c'est un chemin protégé
        if any(path.startswith(protected_path) for protected_path in self.protected_paths):
            # Vérifier si l'IP est bloquée
            if security_manager.is_blocked(client_ip):
                logger.warning(f"Tentative d'accès depuis IP bloquée: {client_ip}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Trop de tentatives de connexion. Réessayez plus tard."
                )
        
        response = await call_next(request)
        
        # Enregistrer les tentatives de connexion
        if path.startswith("/api/v1/auth/login") and request.method == "POST":
            if response.status_code == 401:
                security_manager.record_failed_attempt(client_ip)
            elif response.status_code == 200:
                security_manager.record_successful_login(client_ip)
        
        return response


# Utilitaires pour la validation des permissions
class PermissionChecker:
    """
    Vérificateur de permissions pour des actions spécifiques
    """
    
    @staticmethod
    def can_manage_users(user: UserBase) -> bool:
        """Vérifie si l'utilisateur peut gérer d'autres utilisateurs"""
        return user.is_admin
    
    @staticmethod
    def can_manage_quests(user: UserBase) -> bool:
        """Vérifie si l'utilisateur peut gérer les quêtes"""
        return user.is_admin
    
    @staticmethod
    def can_access_analytics(user: UserBase) -> bool:
        """Vérifie si l'utilisateur peut accéder aux analytiques"""
        return user.is_admin
    
    @staticmethod
    def can_modify_quest(user: UserBase, quest_creator_id: int) -> bool:
        """Vérifie si l'utilisateur peut modifier une quête"""
        return user.is_admin or user.id == quest_creator_id
    
    @staticmethod
    def can_access_user_data(user: UserBase, target_user_id: int) -> bool:
        """Vérifie si l'utilisateur peut accéder aux données d'un autre utilisateur"""
        return user.is_admin or user.id == target_user_id


# Dépendances de permission
def require_user_management_permission(current_user: UserBase = Depends(require_auth)) -> UserBase:
    """Dépendance qui exige la permission de gestion des utilisateurs"""
    if not PermissionChecker.can_manage_users(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission de gestion des utilisateurs requise"
        )
    return current_user


def require_quest_management_permission(current_user: UserBase = Depends(require_auth)) -> UserBase:
    """Dépendance qui exige la permission de gestion des quêtes"""
    if not PermissionChecker.can_manage_quests(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission de gestion des quêtes requise"
        )
    return current_user


def require_analytics_permission(current_user: UserBase = Depends(require_auth)) -> UserBase:
    """Dépendance qui exige la permission d'accès aux analytiques"""
    if not PermissionChecker.can_access_analytics(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission d'accès aux analytiques requise"
        )
    return current_user


# Fonction utilitaire pour créer des tokens de réinitialisation de mot de passe
def create_password_reset_token(email: str) -> str:
    """
    Crée un token de réinitialisation de mot de passe
    """
    data = {
        "email": email,
        "type": "password_reset",
        "exp": datetime.utcnow() + timedelta(hours=1),  # Expire dans 1 heure
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Vérifie un token de réinitialisation de mot de passe et retourne l'email
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        if payload.get("type") != "password_reset":
            return None
        
        return payload.get("email")
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token de réinitialisation expiré")
        return None
    except jwt.JWTError as e:
        logger.warning(f"Erreur JWT dans token de réinitialisation: {e}")
        return None


# Fonction pour auditer les actions des utilisateurs
async def audit_user_action(user_id: int, action: str, details: Optional[Dict[str, Any]] = None, ip: str = "unknown"):
    """
    Enregistre une action utilisateur pour l'audit
    """
    try:
        async with get_db_session() as session:
            query = """
                INSERT INTO user_audit_log (user_id, action, details, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """
            await session.execute(query, [
                user_id,
                action,
                str(details) if details else None,
                ip,
                datetime.utcnow()
            ])
            await session.commit()
            
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement d'audit: {e}")


# Export des fonctions principales
__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "authenticate_user",
    "get_current_user",
    "require_auth",
    "require_admin",
    "get_optional_user",
    "AuthMiddleware",
    "BruteForceProtectionMiddleware",
    "PermissionChecker",
    "require_user_management_permission",
    "require_quest_management_permission",
    "require_analytics_permission",
    "create_password_reset_token",
    "verify_password_reset_token",
    "audit_user_action",
    "security_manager"
]