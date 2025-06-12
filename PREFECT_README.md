# Prefect Configuration pour DoctorPy

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

## Architecture des Workflows

```
DoctorPy Prefect Workflows
├── RAG Pipeline
│   ├── Scraping (Documentation Python)
│   ├── Processing (Chunking)
│   ├── Embeddings (sentence-transformers)
│   └── Indexing (ChromaDB)
├── Maintenance
│   ├── Sessions cleanup
│   ├── Logs archival
│   ├── Database backup
│   └── Vector store optimization
├── Monitoring
│   ├── System metrics
│   ├── Application metrics
│   ├── Health checks
│   └── Alert management
└── Analytics
    ├── Daily reports
    ├── Trend analysis
    └── Performance metrics
```

## Planification des Workflows

| Workflow | Fréquence | Heure | Description |
|----------|-----------|-------|-------------|
| RAG Update | Quotidien | 02:00 | Mise à jour incrémentale |
| Maintenance | Quotidien | 03:00 | Nettoyage et sauvegarde |
| Health Check | Horaire | xx:00 | Vérification de santé |
| Analytics | Quotidien | 08:00 | Rapport quotidien |
| Full RAG | Hebdomadaire | Dim 01:00 | Mise à jour complète |
| Deep Maintenance | Hebdomadaire | Dim 04:00 | Maintenance approfondie |
| Weekly Report | Hebdomadaire | Lun 09:00 | Rapport hebdomadaire |

## Sécurité et Bonnes Pratiques

- Utilisez des variables d'environnement pour les secrets
- Configurez les sauvegardes automatiques
- Surveillez les métriques de performance
- Configurez les alertes pour les échecs critiques
- Testez régulièrement les workflows manuellement

## Support

Pour toute question ou problème avec les workflows Prefect :
1. Consultez les logs dans `logs/prefect/`
2. Vérifiez le dashboard Prefect
3. Exécutez les workflows manuellement pour diagnostic
4. Consultez la documentation Prefect officielle