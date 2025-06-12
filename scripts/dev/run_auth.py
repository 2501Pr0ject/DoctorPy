#!/usr/bin/env python3
"""
Script de lancement direct du service Auth
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    from services.auth.app_simple import create_auth_app
    import uvicorn
    
    print("🔐 Démarrage du service Auth (mode simplifié)...")
    app = create_auth_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)