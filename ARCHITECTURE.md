# Architecture DoctorPy v2.0 - Microservices Event-Driven

## 🏗️ Vue d'Ensemble Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (Kong/Nginx)                │
├─────────────────────────────────────────────────────────────────┤
│                      Load Balancer                             │
└─────────────────┬───────────────┬───────────────┬───────────────┘
                  │               │               │
    ┌─────────────▼─────────────┐ │ ┌─────────────▼─────────────┐
    │    Auth Service           │ │ │    RAG Service            │
    │  (FastAPI + JWT)          │ │ │  (FastAPI + ChromaDB)     │
    │  Port: 8001               │ │ │  Port: 8002               │
    └─────────────┬─────────────┘ │ └─────────────┬─────────────┘
                  │               │               │
    ┌─────────────▼─────────────┐ │ ┌─────────────▼─────────────┐
    │  Analytics Service        │ │ │   Quest Service           │
    │  (FastAPI + TimeSeries)   │ │ │  (FastAPI + SQLite)       │
    │  Port: 8003               │ │ │  Port: 8004               │
    └─────────────┬─────────────┘ │ └─────────────┬─────────────┘
                  │               │               │
┌─────────────────▼───────────────▼───────────────▼───────────────┐
│                     Event Bus (Redis/RabbitMQ)                 │
│                          Port: 6379/5672                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                  Shared Services Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │    Cache    │ │  Database   │ │   Storage   │ │  Monitoring ││
│  │   (Redis)   │ │ (PostgreSQL)│ │    (S3)     │ │(Prometheus) ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Services Architecture

### 1. **Auth Service** (Port 8001)
- JWT Authentication & Authorization
- User Management
- Role-Based Access Control (RBAC)
- Session Management

### 2. **RAG Service** (Port 8002)
- Knowledge Base Management
- Document Processing & Embeddings
- Semantic Search
- AI Response Generation

### 3. **Quest Service** (Port 8004)
- Quest Management
- Progress Tracking
- Gamification Logic
- Achievement System

### 4. **Analytics Service** (Port 8003)
- User Analytics
- Performance Metrics
- Real-time Monitoring
- Reporting & Dashboards

### 5. **Notification Service** (Port 8005)
- Multi-channel Notifications
- Event-driven Alerts
- Email/Slack Integration
- Real-time WebSocket

## 🔄 Event-Driven Communication

```
Events Flow:
User Action → API Gateway → Service → Event Bus → Other Services
```

### Event Types:
- `user.authenticated`
- `quest.completed`
- `rag.query_processed`
- `analytics.metric_updated`
- `system.health_check`