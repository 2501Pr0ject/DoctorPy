#!/usr/bin/env python3
"""
Test simple pour vérifier si l'architecture fonctionne
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test Service")

@app.get("/")
def root():
    return {"status": "OK", "service": "Test simple"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🧪 Démarrage service de test...")
    uvicorn.run(app, host="0.0.0.0", port=8000)