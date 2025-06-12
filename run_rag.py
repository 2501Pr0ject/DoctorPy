#!/usr/bin/env python3
"""
Script de lancement direct du service RAG
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    from services.rag.app import create_app
    import uvicorn
    
    print("🤖 Démarrage du service RAG...")
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8002)