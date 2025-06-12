"""
Routes pour le service d'authentification
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Dict, Any

# Création des routeurs
auth_router = APIRouter()
users_router = APIRouter()

security = HTTPBearer()

@auth_router.post("/login")
async def login(credentials: Dict[str, str]):
    """Connexion utilisateur"""
    return {
        "access_token": "fake_token_for_demo",
        "token_type": "bearer",
        "message": "Connexion réussie (mode démo)"
    }

@auth_router.post("/logout")
async def logout():
    """Déconnexion utilisateur"""
    return {"message": "Déconnexion réussie"}

@auth_router.post("/refresh")
async def refresh_token():
    """Rafraîchir le token"""
    return {
        "access_token": "new_fake_token",
        "token_type": "bearer"
    }

@users_router.get("/profile")
async def get_profile():
    """Profil utilisateur"""
    return {
        "user_id": "demo_user",
        "username": "demo",
        "email": "demo@doctorpy.com"
    }

@users_router.post("/register")
async def register(user_data: Dict[str, str]):
    """Inscription utilisateur"""
    return {
        "user_id": "new_demo_user",
        "message": "Inscription réussie (mode démo)"
    }