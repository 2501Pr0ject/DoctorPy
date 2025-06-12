"""
Modèles de données pour le service d'authentification
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, EmailStr, validator
import uuid


class UserRole(Enum):
    """Rôles utilisateur disponibles"""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"
    VIEWER = "viewer"


class Permission(Enum):
    """Permissions disponibles"""
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Quest management
    QUEST_CREATE = "quest:create"
    QUEST_READ = "quest:read"
    QUEST_UPDATE = "quest:update"
    QUEST_DELETE = "quest:delete"
    
    # RAG management
    RAG_QUERY = "rag:query"
    RAG_ADMIN = "rag:admin"
    RAG_UPDATE = "rag:update"
    
    # Analytics
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_ADMIN = "analytics:admin"
    
    # System
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"


# Permissions par rôle
ROLE_PERMISSIONS = {
    UserRole.ADMIN: list(Permission),  # Toutes les permissions
    UserRole.MODERATOR: [
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.QUEST_CREATE,
        Permission.QUEST_READ,
        Permission.QUEST_UPDATE,
        Permission.RAG_QUERY,
        Permission.RAG_ADMIN,
        Permission.ANALYTICS_READ,
        Permission.SYSTEM_MONITOR
    ],
    UserRole.USER: [
        Permission.USER_READ,
        Permission.QUEST_READ,
        Permission.RAG_QUERY
    ],
    UserRole.VIEWER: [
        Permission.USER_READ,
        Permission.QUEST_READ,
        Permission.RAG_QUERY,
        Permission.ANALYTICS_READ
    ]
}


@dataclass
class User:
    """Modèle utilisateur"""
    id: str
    email: str
    username: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Vérifier si l'utilisateur a une permission"""
        if not self.is_active:
            return False
        
        role_permissions = ROLE_PERMISSIONS.get(self.role, [])
        return permission in role_permissions
    
    def is_locked(self) -> bool:
        """Vérifier si le compte est verrouillé"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire (sans mot de passe)"""
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata
        }


# Modèles Pydantic pour les API
class UserCreate(BaseModel):
    """Modèle pour créer un utilisateur"""
    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.USER
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if len(v) > 50:
            raise ValueError('Username must be less than 50 characters')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class UserUpdate(BaseModel):
    """Modèle pour mettre à jour un utilisateur"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    """Modèle de réponse utilisateur"""
    id: str
    email: str
    username: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: str
    updated_at: str
    last_login: Optional[str]
    metadata: Dict[str, Any]


class LoginRequest(BaseModel):
    """Modèle pour la connexion"""
    email: str
    password: str
    remember_me: bool = False


class LoginResponse(BaseModel):
    """Modèle de réponse de connexion"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenRefresh(BaseModel):
    """Modèle pour rafraîchir le token"""
    refresh_token: str


class PasswordReset(BaseModel):
    """Modèle pour réinitialisation de mot de passe"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Modèle pour confirmer la réinitialisation"""
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class PasswordChange(BaseModel):
    """Modèle pour changer le mot de passe"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


@dataclass
class Session:
    """Modèle de session utilisateur"""
    session_id: str
    user_id: str
    access_token: str
    refresh_token: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Vérifier si la session a expiré"""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active
        }


@dataclass
class AuthToken:
    """Modèle de token d'authentification"""
    token: str
    token_type: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    scopes: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Vérifier si le token a expiré"""
        return datetime.utcnow() > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """Vérifier si le token a un scope spécifique"""
        return scope in self.scopes


class Role(BaseModel):
    """Modèle de rôle (pour API admin)"""
    name: str
    description: str
    permissions: List[str]
    is_active: bool = True
    created_at: str
    updated_at: str


class PermissionCheck(BaseModel):
    """Modèle pour vérifier une permission"""
    permission: str
    resource_id: Optional[str] = None
    

class UserStats(BaseModel):
    """Statistiques utilisateur"""
    total_users: int
    active_users: int
    verified_users: int
    users_by_role: Dict[str, int]
    recent_registrations: int
    recent_logins: int


class SecurityEvent(BaseModel):
    """Événement de sécurité"""
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any]
    timestamp: str
    severity: str = "info"  # info, warning, error, critical