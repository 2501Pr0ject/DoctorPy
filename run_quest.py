#!/usr/bin/env python3
"""
Script de lancement direct du service Quest
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    from services.quest.app import create_app
    import uvicorn
    
    print("🎮 Démarrage du service Quest...")
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8004)