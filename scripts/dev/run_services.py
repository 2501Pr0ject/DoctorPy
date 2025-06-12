#!/usr/bin/env python3
"""
Script principal pour lancer les microservices DoctorPy
Nouvelle architecture apps/
"""

import subprocess
import sys
import time
from pathlib import Path
import signal
from typing import List
import psutil

# Chemin vers la racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent

class ServiceManager:
    """Gestionnaire des microservices avec nouvelle architecture"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.services = [
            {
                "name": "Auth Service",
                "path": "apps/auth/app_simple.py",
                "port": 8001,
                "description": "Service d'authentification et gestion des utilisateurs"
            },
            {
                "name": "RAG Service", 
                "path": "apps/rag/app_simple.py",
                "port": 8002,
                "description": "Service de récupération et génération assistée (RAG)"
            },
            {
                "name": "Analytics Service",
                "path": "apps/analytics/app_simple.py",
                "port": 8003,
                "description": "Service d'analytics et métriques"
            },
            {
                "name": "Quest Service",
                "path": "apps/quest/app_simple.py", 
                "port": 8004,
                "description": "Service de gamification et gestion des quêtes"
            },
            {
                "name": "Notification Service",
                "path": "apps/notification/app_simple.py",
                "port": 8005,
                "description": "Service de notifications multi-canal"
            }
        ]
    
    def check_port_available(self, port: int) -> bool:
        """Vérifie si un port est disponible"""
        try:
            for conn in psutil.net_connections():
                if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == port:
                    return False
            return True
        except Exception as e:
            print(f"⚠️ Erreur vérification port {port}: {e}")
            return True
    
    def start_service(self, service: dict) -> subprocess.Popen:
        """Démarre un service spécifique"""
        print(f"🚀 Démarrage {service['name']} sur le port {service['port']}...")
        
        if not self.check_port_available(service['port']):
            print(f"⚠️  Port {service['port']} déjà utilisé pour {service['name']}")
            return None
        
        try:
            service_path = PROJECT_ROOT / service['path']
            process = subprocess.Popen([
                sys.executable, str(service_path)
            ], 
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
            )
            
            print(f"   PID: {process.pid} - Attente du démarrage...")
            time.sleep(3)
            
            if process.poll() is None:
                print(f"✅ {service['name']} démarré avec succès (PID: {process.pid})")
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Échec du démarrage de {service['name']}")
                if stderr:
                    print(f"   Erreur: {stderr[:200]}...")
                return None
                
        except Exception as e:
            print(f"❌ Erreur démarrage {service['name']}: {e}")
            return None
    
    def start_all_services(self):
        """Démarre tous les services"""
        print("🌟 Démarrage des microservices DoctorPy")
        print("🏗️ Nouvelle architecture apps/")
        print("=" * 50)
        
        for service in self.services:
            print(f"\n📋 {service['description']}")
            process = self.start_service(service)
            if process:
                self.processes.append(process)
            time.sleep(1)
        
        if self.processes:
            print(f"\n🎉 {len(self.processes)} services démarrés avec succès!")
            self.print_status()
        else:
            print("\n❌ Aucun service n'a pu être démarré")
    
    def print_status(self):
        """Affiche le statut des services"""
        print("\n" + "=" * 50)
        print("📊 STATUT DES SERVICES")
        print("=" * 50)
        
        for i, service in enumerate(self.services):
            if i < len(self.processes) and self.processes[i] and self.processes[i].poll() is None:
                status = "🟢 RUNNING"
                pid = self.processes[i].pid
            else:
                status = "🔴 STOPPED"
                pid = "N/A"
            
            print(f"{service['name']:<20} {status:<12} Port: {service['port']:<6} PID: {pid}")
        
        print("\n🌐 URLs d'accès:")
        for service in self.services:
            print(f"• {service['name']}: http://localhost:{service['port']}/docs")
        
        print(f"\n💡 Pour arrêter tous les services: Ctrl+C")
    
    def stop_all_services(self):
        """Arrête tous les services"""
        print("\n🛑 Arrêt des services...")
        
        for i, process in enumerate(self.processes):
            if process and process.poll() is None:
                service_name = self.services[i]['name'] if i < len(self.services) else f"Service {i}"
                print(f"🔄 Arrêt de {service_name}...")
                
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"✅ {service_name} arrêté")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force l'arrêt de {service_name}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {service_name} forcé à s'arrêter")
                except Exception as e:
                    print(f"❌ Erreur arrêt {service_name}: {e}")
        
        print("✅ Tous les services ont été arrêtés")
    
    def monitor_services(self):
        """Surveille les services en continu"""
        try:
            print("\n🔍 Surveillance des services (Ctrl+C pour arrêter)...")
            while True:
                time.sleep(30)
                
                running_count = 0
                for i, process in enumerate(self.processes):
                    if process and process.poll() is None:
                        running_count += 1
                    else:
                        service_name = self.services[i]['name'] if i < len(self.services) else f"Service {i}"
                        print(f"⚠️  {service_name} s'est arrêté de manière inattendue")
                
                if running_count == 0:
                    print("❌ Tous les services se sont arrêtés")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interruption demandée par l'utilisateur")
        finally:
            self.stop_all_services()

def main():
    """Point d'entrée principal"""
    print("🤖 DoctorPy Microservices Manager")
    print("🏗️ Architecture apps/ v2.0")
    print()
    
    manager = ServiceManager()
    
    def signal_handler(signum, frame):
        print("\n⚠️  Signal d'arrêt reçu")
        manager.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        manager.start_all_services()
        
        if manager.processes:
            manager.monitor_services()
        
    except Exception as e:
        print(f"💥 Erreur critique: {e}")
        manager.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()