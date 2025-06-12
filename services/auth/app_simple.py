"""
Application FastAPI simplifiée pour le service d'authentification
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

def create_auth_app() -> FastAPI:
    """Créer l'application FastAPI pour le service d'authentification"""
    
    app = FastAPI(
        title="DoctorPy Auth Service",
        description="Service d'authentification et d'autorisation",
        version="1.0.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routes simples
    @app.get("/")
    async def root():
        return {
            "service": "DoctorPy Auth Service",
            "version": "1.0.0",
            "status": "running",
            "mode": "demo"
        }
    
    @app.get("/health")
    async def health_check():
        """Endpoint de vérification de santé"""
        return {
            "status": "healthy",
            "service": "auth",
            "timestamp": "now"
        }
    
    @app.post("/auth/login")
    async def login():
        """Connexion utilisateur (mode démo)"""
        return {
            "access_token": "demo_token_12345",
            "token_type": "bearer",
            "message": "Connexion réussie en mode démo"
        }
    
    @app.get("/users/profile")
    async def get_profile():
        """Profil utilisateur (mode démo)"""
        return {
            "user_id": "demo_user",
            "username": "demo",
            "email": "demo@doctorpy.com",
            "role": "student"
        }
    
    return app

# Point d'entrée pour développement
if __name__ == "__main__":
    app = create_auth_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)