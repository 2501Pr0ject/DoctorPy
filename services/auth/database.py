"""
Gestionnaire de base de données pour l'authentification
"""

import asyncio
from typing import Optional

class AuthDatabase:
    """Gestionnaire de base de données d'authentification (version démo)"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connected = False
    
    async def connect(self):
        """Connexion à la base de données"""
        # Simulation de connexion
        await asyncio.sleep(0.1)
        self.connected = True
        print("✅ Auth Database connectée (mode démo)")
    
    async def disconnect(self):
        """Déconnexion de la base de données"""
        self.connected = False
        print("✅ Auth Database déconnectée")
    
    async def health_check(self) -> bool:
        """Vérification de santé de la base de données"""
        return self.connected