#!/usr/bin/env python3
"""
Script de déploiement des workflows Prefect pour DoctorPy
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.workflows.deployment import deploy_all_workflows, setup_prefect_environment

if __name__ == "__main__":
    print("🚀 Déploiement des workflows DoctorPy...")
    
    # Configurer l'environnement si nécessaire
    setup_prefect_environment()
    
    # Déployer tous les workflows
    deploy_all_workflows()
    
    print("\n✅ Déploiement terminé!")
    print("\n🔗 Commandes utiles:")
    print("   prefect server start                    # Démarrer le serveur")
    print("   prefect agent start --pool default      # Démarrer l'agent")
    print("   prefect deployment run <nom>             # Exécuter un workflow")
    print("   prefect flow-run list                    # Voir les exécutions")