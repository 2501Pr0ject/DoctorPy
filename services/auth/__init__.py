"""
Service d'Authentification et d'Autorisation pour DoctorPy

Ce service gère :
- Authentification JWT
- Gestion des utilisateurs
- Contrôle d'accès basé sur les rôles (RBAC)
- Sessions utilisateur
"""

from .app import create_auth_app
from .models import User, Role, Permission
from .auth import AuthManager, JWTHandler
from .dependencies import get_current_user, require_permission

__version__ = "2.0.0"
__all__ = [
    "create_auth_app",
    "User", 
    "Role",
    "Permission",
    "AuthManager",
    "JWTHandler",
    "get_current_user",
    "require_permission"
]