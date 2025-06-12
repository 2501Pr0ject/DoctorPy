"""
Scripts de déploiement et configuration Prefect pour DoctorPy
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from prefect import flow, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import des workflows
from .data_pipeline import update_knowledge_base, rag_quick_update, rag_full_pipeline
from .maintenance import daily_maintenance, weekly_maintenance, emergency_maintenance
from .analytics import generate_analytics, health_check_flow


def create_deployments():
    """
    Créer tous les déploiements Prefect pour DoctorPy
    """
    deployments = []
    
    # ===== DÉPLOIEMENTS RAG PIPELINE =====
    
    # Pipeline RAG quotidien (mise à jour incrémentale)
    daily_rag_deployment = Deployment.build_from_flow(
        flow=rag_quick_update,
        name="daily-rag-update",
        version="1.0",
        description="Mise à jour quotidienne rapide de la base de connaissances",
        tags=["rag", "daily", "automatic"],
        schedule=CronSchedule(
            cron="0 2 * * *",  # Tous les jours à 2h du matin
            timezone="Europe/Paris"
        ),
        parameters={
            "force_refresh": False,
            "max_documents": None
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(daily_rag_deployment)
    
    # Pipeline RAG hebdomadaire (mise à jour complète)
    weekly_rag_deployment = Deployment.build_from_flow(
        flow=rag_full_pipeline,
        name="weekly-rag-full-update",
        version="1.0",
        description="Mise à jour hebdomadaire complète avec validations",
        tags=["rag", "weekly", "complete", "validation"],
        schedule=CronSchedule(
            cron="0 1 * * 0",  # Dimanche à 1h du matin
            timezone="Europe/Paris"
        ),
        parameters={
            "force_all": True,
            "notification_channels": ["log", "email"]
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(weekly_rag_deployment)
    
    # Pipeline RAG manuel (à la demande)
    manual_rag_deployment = Deployment.build_from_flow(
        flow=update_knowledge_base,
        name="manual-rag-update",
        version="1.0",
        description="Mise à jour manuelle de la base de connaissances",
        tags=["rag", "manual", "on-demand"],
        # Pas de schedule - déclenchement manuel
        parameters={
            "force_refresh": False,
            "force_reprocess": False,
            "force_regenerate_embeddings": False,
            "force_reindex": False,
            "skip_validation": False,
            "max_documents": None,
            "notification_channels": ["log", "email"]
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(manual_rag_deployment)
    
    # ===== DÉPLOIEMENTS MAINTENANCE =====
    
    # Maintenance quotidienne
    daily_maintenance_deployment = Deployment.build_from_flow(
        flow=daily_maintenance,
        name="daily-maintenance",
        version="1.0",
        description="Maintenance quotidienne automatisée",
        tags=["maintenance", "daily", "cleanup"],
        schedule=CronSchedule(
            cron="0 3 * * *",  # Tous les jours à 3h du matin (après le RAG)
            timezone="Europe/Paris"
        ),
        parameters={
            "notification_channels": ["log"],
            "skip_backup": False,
            "skip_cleanup": False
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(daily_maintenance_deployment)
    
    # Maintenance hebdomadaire
    weekly_maintenance_deployment = Deployment.build_from_flow(
        flow=weekly_maintenance,
        name="weekly-maintenance",
        version="1.0",
        description="Maintenance hebdomadaire approfondie",
        tags=["maintenance", "weekly", "deep-cleanup"],
        schedule=CronSchedule(
            cron="0 4 * * 0",  # Dimanche à 4h du matin (après le RAG complet)
            timezone="Europe/Paris"
        ),
        parameters={
            "notification_channels": ["log", "email"],
            "include_deep_cleanup": True,
            "include_optimization": True
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(weekly_maintenance_deployment)
    
    # Maintenance d'urgence (manuel)
    emergency_maintenance_deployment = Deployment.build_from_flow(
        flow=emergency_maintenance,
        name="emergency-maintenance",
        version="1.0",
        description="Maintenance d'urgence pour problèmes critiques",
        tags=["maintenance", "emergency", "critical"],
        # Pas de schedule - déclenchement manuel uniquement
        parameters={
            "issue_description": "Problème critique détecté",
            "notification_channels": ["log", "email"],
            "force_restart_services": False
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(emergency_maintenance_deployment)
    
    # ===== DÉPLOIEMENTS MONITORING =====
    
    # Vérification de santé toutes les heures
    hourly_health_check_deployment = Deployment.build_from_flow(
        flow=health_check_flow,
        name="hourly-health-check",
        version="1.0",
        description="Vérification de santé toutes les heures",
        tags=["monitoring", "health", "hourly"],
        schedule=CronSchedule(
            cron="0 * * * *",  # Toutes les heures
            timezone="Europe/Paris"
        ),
        parameters={
            "alert_on_degraded": True,
            "notification_channels": ["log"],
            "detailed_analysis": False
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(hourly_health_check_deployment)
    
    # Vérification de santé détaillée (4 fois par jour)
    detailed_health_check_deployment = Deployment.build_from_flow(
        flow=health_check_flow,
        name="detailed-health-check",
        version="1.0",
        description="Vérification de santé détaillée 4 fois par jour",
        tags=["monitoring", "health", "detailed"],
        schedule=CronSchedule(
            cron="0 */6 * * *",  # Toutes les 6 heures
            timezone="Europe/Paris"
        ),
        parameters={
            "alert_on_degraded": True,
            "notification_channels": ["log", "email"],
            "detailed_analysis": True
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(detailed_health_check_deployment)
    
    # ===== DÉPLOIEMENTS ANALYTICS =====
    
    # Rapport quotidien
    daily_analytics_deployment = Deployment.build_from_flow(
        flow=generate_analytics,
        name="daily-analytics-report",
        version="1.0",
        description="Génération du rapport analytique quotidien",
        tags=["analytics", "daily", "report"],
        schedule=CronSchedule(
            cron="0 8 * * *",  # Tous les jours à 8h du matin
            timezone="Europe/Paris"
        ),
        parameters={
            "report_type": "daily",
            "notification_channels": ["log", "email"],
            "include_health_check": True
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(daily_analytics_deployment)
    
    # Rapport hebdomadaire
    weekly_analytics_deployment = Deployment.build_from_flow(
        flow=generate_analytics,
        name="weekly-analytics-report",
        version="1.0",
        description="Génération du rapport analytique hebdomadaire",
        tags=["analytics", "weekly", "report"],
        schedule=CronSchedule(
            cron="0 9 * * 1",  # Lundi à 9h du matin
            timezone="Europe/Paris"
        ),
        parameters={
            "report_type": "weekly",
            "notification_channels": ["log", "email"],
            "include_health_check": True
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(weekly_analytics_deployment)
    
    # Rapport mensuel
    monthly_analytics_deployment = Deployment.build_from_flow(
        flow=generate_analytics,
        name="monthly-analytics-report",
        version="1.0",
        description="Génération du rapport analytique mensuel",
        tags=["analytics", "monthly", "report"],
        schedule=CronSchedule(
            cron="0 10 1 * *",  # 1er de chaque mois à 10h
            timezone="Europe/Paris"
        ),
        parameters={
            "report_type": "monthly",
            "notification_channels": ["log", "email"],
            "include_health_check": True
        },
        work_pool_name="default-agent-pool"
    )
    deployments.append(monthly_analytics_deployment)
    
    return deployments


def deploy_all_workflows():
    """
    Déployer tous les workflows Prefect
    """
    print("🚀 Déploiement des workflows Prefect pour DoctorPy...")
    
    deployments = create_deployments()
    
    success_count = 0
    error_count = 0
    
    for deployment in deployments:
        try:
            print(f"📤 Déploiement de {deployment.name}...")
            deployment_id = deployment.apply()
            print(f"✅ {deployment.name} déployé avec succès (ID: {deployment_id})")
            success_count += 1
        except Exception as e:
            print(f"❌ Erreur lors du déploiement de {deployment.name}: {str(e)}")
            error_count += 1
    
    print(f"\n📊 Résumé du déploiement:")
    print(f"   ✅ Succès: {success_count}")
    print(f"   ❌ Erreurs: {error_count}")
    print(f"   📦 Total: {len(deployments)}")
    
    if error_count == 0:
        print("\n🎉 Tous les workflows ont été déployés avec succès!")
        print_deployment_summary()
    else:
        print(f"\n⚠️ {error_count} déploiement(s) ont échoué. Vérifiez la configuration Prefect.")


def print_deployment_summary():
    """
    Afficher un résumé des déploiements créés
    """
    print("\n📋 Résumé des workflows déployés:")
    
    print("\n🔄 RAG Pipeline:")
    print("   • daily-rag-update: Mise à jour quotidienne (2h00)")
    print("   • weekly-rag-full-update: Mise à jour complète (Dimanche 1h00)")
    print("   • manual-rag-update: Déclenchement manuel")
    
    print("\n🧹 Maintenance:")
    print("   • daily-maintenance: Nettoyage quotidien (3h00)")
    print("   • weekly-maintenance: Maintenance approfondie (Dimanche 4h00)")
    print("   • emergency-maintenance: Maintenance d'urgence (manuel)")
    
    print("\n🩺 Monitoring:")
    print("   • hourly-health-check: Vérification toutes les heures")
    print("   • detailed-health-check: Vérification détaillée (toutes les 6h)")
    
    print("\n📊 Analytics:")
    print("   • daily-analytics-report: Rapport quotidien (8h00)")
    print("   • weekly-analytics-report: Rapport hebdomadaire (Lundi 9h00)")
    print("   • monthly-analytics-report: Rapport mensuel (1er du mois 10h00)")


def create_prefect_config():
    """
    Créer la configuration Prefect pour DoctorPy
    """
    config = {
        "prefect": {
            "api": {
                "url": "http://localhost:4200/api"
            },
            "cloud": {
                "api": "https://api.prefect.cloud/api/accounts/[ACCOUNT_ID]/workspaces/[WORKSPACE_ID]"
            },
            "server": {
                "analytics": False
            }
        },
        "logging": {
            "level": "INFO",
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            }
        },
        "flows": {
            "work_pool": "default-agent-pool",
            "storage": "./data/prefect",
            "results_storage": "./data/prefect/results"
        }
    }
    
    config_file = Path("prefect_config.yaml")
    
    import yaml
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📄 Configuration Prefect créée: {config_file}")
    return config_file


def setup_prefect_environment():
    """
    Configurer l'environnement Prefect pour DoctorPy
    """
    print("🔧 Configuration de l'environnement Prefect...")
    
    # Créer les répertoires nécessaires
    directories = [
        "data/prefect",
        "data/prefect/results", 
        "data/prefect/flows",
        "data/prefect/deployments",
        "logs/prefect"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Répertoire créé: {directory}")
    
    # Créer la configuration
    config_file = create_prefect_config()
    
    # Créer un script de démarrage
    startup_script = Path("start_prefect.sh")
    startup_content = """#!/bin/bash

# Script de démarrage Prefect pour DoctorPy

echo "🚀 Démarrage de Prefect pour DoctorPy..."

# Définir les variables d'environnement
export PREFECT_API_URL="http://localhost:4200/api"
export PREFECT_LOGGING_LEVEL="INFO"

# Démarrer le serveur Prefect en arrière-plan
echo "🖥️ Démarrage du serveur Prefect..."
prefect server start --host 0.0.0.0 --port 4200 &
SERVER_PID=$!

# Attendre que le serveur démarre
sleep 10

# Créer l'agent de travail
echo "🤖 Création de l'agent de travail..."
prefect work-pool create default-agent-pool --type process &
POOL_PID=$!

# Attendre que le pool soit créé
sleep 5

# Démarrer l'agent
echo "▶️ Démarrage de l'agent..."
prefect agent start --pool default-agent-pool &
AGENT_PID=$!

echo "✅ Prefect démarré avec succès!"
echo "🌐 Interface web: http://localhost:4200"
echo "📊 Dashboard: http://localhost:4200/dashboard"

# Garder le script en vie
echo "🔄 Prefect en cours d'exécution... (Ctrl+C pour arrêter)"
wait $SERVER_PID $POOL_PID $AGENT_PID
"""
    
    with open(startup_script, 'w', encoding='utf-8') as f:
        f.write(startup_content)
    
    # Rendre le script exécutable
    import stat
    startup_script.chmod(startup_script.stat().st_mode | stat.S_IEXEC)
    
    print(f"📜 Script de démarrage créé: {startup_script}")
    
    # Créer un script de déploiement
    deploy_script = Path("deploy_workflows.py")
    deploy_content = """#!/usr/bin/env python3
\"\"\"
Script de déploiement des workflows Prefect pour DoctorPy
\"\"\"

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
    
    print("\\n✅ Déploiement terminé!")
    print("\\n🔗 Commandes utiles:")
    print("   prefect server start                    # Démarrer le serveur")
    print("   prefect agent start --pool default      # Démarrer l'agent")
    print("   prefect deployment run <nom>             # Exécuter un workflow")
    print("   prefect flow-run list                    # Voir les exécutions")
"""
    
    with open(deploy_script, 'w', encoding='utf-8') as f:
        f.write(deploy_content)
    
    deploy_script.chmod(deploy_script.stat().st_mode | stat.S_IEXEC)
    
    print(f"🎯 Script de déploiement créé: {deploy_script}")
    
    # Créer un fichier README pour Prefect
    readme_file = Path("PREFECT_README.md")
    readme_content = """# Prefect Configuration pour DoctorPy

## Vue d'ensemble

Ce projet utilise Prefect pour orchestrer les workflows de données RAG, maintenance et monitoring.

## Workflows Disponibles

### 🔄 RAG Pipeline
- **daily-rag-update**: Mise à jour quotidienne (2h00)
- **weekly-rag-full-update**: Mise à jour complète (Dimanche 1h00)
- **manual-rag-update**: Déclenchement manuel

### 🧹 Maintenance
- **daily-maintenance**: Nettoyage quotidien (3h00)
- **weekly-maintenance**: Maintenance approfondie (Dimanche 4h00)
- **emergency-maintenance**: Maintenance d'urgence (manuel)

### 🩺 Monitoring
- **hourly-health-check**: Vérification toutes les heures
- **detailed-health-check**: Vérification détaillée (toutes les 6h)

### 📊 Analytics
- **daily-analytics-report**: Rapport quotidien (8h00)
- **weekly-analytics-report**: Rapport hebdomadaire (Lundi 9h00)
- **monthly-analytics-report**: Rapport mensuel (1er du mois 10h00)

## Démarrage Rapide

1. **Installer Prefect**:
   ```bash
   pip install prefect>=2.14.0
   ```

2. **Démarrer l'environnement**:
   ```bash
   ./start_prefect.sh
   ```

3. **Déployer les workflows**:
   ```bash
   python deploy_workflows.py
   ```

4. **Accéder à l'interface web**:
   - URL: http://localhost:4200
   - Dashboard: http://localhost:4200/dashboard

## Commandes Utiles

```bash
# Voir tous les déploiements
prefect deployment list

# Exécuter un workflow manuellement
prefect deployment run "manual-rag-update"

# Voir l'historique des exécutions
prefect flow-run list

# Voir les logs d'une exécution
prefect flow-run logs <flow-run-id>

# Suspendre/reprendre un déploiement
prefect deployment pause <deployment-name>
prefect deployment resume <deployment-name>
```

## Configuration des Notifications

### Email (SMTP)
```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export FROM_EMAIL="doctorpy@yourcompany.com"
export ALERT_EMAILS="admin@yourcompany.com,team@yourcompany.com"
```

### Slack
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

### Webhook Personnalisé
```bash
export NOTIFICATION_WEBHOOK_URL="https://api.yourcompany.com/notifications"
export WEBHOOK_TOKEN="your-api-token"
```

## Surveillance et Maintenance

- Les logs sont stockés dans `logs/prefect/`
- Les données Prefect sont dans `data/prefect/`
- Les rapports sont générés dans `data/reports/`
- Les notifications sont archivées dans `data/notifications/`

## Dépannage

### Problèmes Courants

1. **Erreur de connexion à la base de données**:
   - Vérifier que `data/databases/doctorpy.db` existe
   - Vérifier les permissions de fichier

2. **ChromaDB indisponible**:
   - Vérifier que `data/vector_store/` existe
   - Redémarrer le workflow d'indexation

3. **Notifications non envoyées**:
   - Vérifier les variables d'environnement
   - Tester la connectivité SMTP/Slack

### Maintenance d'Urgence

En cas de problème critique:

```bash
# Exécuter la maintenance d'urgence
prefect deployment run "emergency-maintenance" --param issue_description="Description du problème"

# Suspendre tous les déploiements automatiques
prefect deployment pause --all

# Reprendre après résolution
prefect deployment resume --all
```
"""
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📚 Documentation créée: {readme_file}")
    
    print("\n✅ Environnement Prefect configuré avec succès!")
    print("\n🔗 Étapes suivantes:")
    print("   1. ./start_prefect.sh          # Démarrer Prefect")
    print("   2. python deploy_workflows.py  # Déployer les workflows")
    print("   3. http://localhost:4200       # Accéder à l'interface")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gestion des déploiements Prefect pour DoctorPy")
    parser.add_argument("--setup", action="store_true", help="Configurer l'environnement Prefect")
    parser.add_argument("--deploy", action="store_true", help="Déployer tous les workflows")
    parser.add_argument("--all", action="store_true", help="Setup + Deploy")
    
    args = parser.parse_args()
    
    if args.setup or args.all:
        setup_prefect_environment()
    
    if args.deploy or args.all:
        deploy_all_workflows()
    
    if not any(vars(args).values()):
        print("Utilisez --setup, --deploy ou --all")
        parser.print_help()