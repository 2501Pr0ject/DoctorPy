#!/usr/bin/env python3
"""
Tests d'intégration automatisés pour les microservices DoctorPy
Démarre automatiquement les services, les teste, puis les arrête
"""

import asyncio
import aiohttp
import subprocess
import time
import sys
import signal
from typing import List, Dict, Any
from pathlib import Path

class AutoIntegrationTester:
    """Testeur d'intégration automatisé"""
    
    def __init__(self):
        self.services = [
            {"name": "Auth Service", "script": "run_auth.py", "port": 8001},
            {"name": "RAG Service", "script": "run_rag.py", "port": 8002}, 
            {"name": "Analytics Service", "script": "run_analytics.py", "port": 8003},
            {"name": "Quest Service", "script": "run_quest.py", "port": 8004},
            {"name": "Notification Service", "script": "run_notification.py", "port": 8005}
        ]
        self.processes: List[subprocess.Popen] = []
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def start_services(self):
        """Démarre tous les services"""
        print("🚀 Démarrage automatique des services...")
        
        for service in self.services:
            try:
                process = subprocess.Popen([
                    sys.executable, service['script']
                ], 
                cwd=Path(__file__).parent,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
                )
                self.processes.append(process)
                print(f"   ✅ {service['name']} démarré (PID: {process.pid})")
                time.sleep(1)  # Délai entre les services
            except Exception as e:
                print(f"   ❌ Erreur démarrage {service['name']}: {e}")
        
        print(f"⏳ Attente de l'initialisation des services...")
        time.sleep(8)  # Temps pour que tous les services démarrent
    
    def stop_services(self):
        """Arrête tous les services"""
        print("\n🛑 Arrêt des services...")
        for i, process in enumerate(self.processes):
            if process and process.poll() is None:
                service_name = self.services[i]['name'] if i < len(self.services) else f"Service {i}"
                try:
                    process.terminate()
                    process.wait(timeout=3)
                    print(f"   ✅ {service_name} arrêté")
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    print(f"   ⚡ {service_name} forcé à s'arrêter")
                except Exception as e:
                    print(f"   ⚠️ Erreur arrêt {service_name}: {e}")
    
    async def test_service_health(self, service: dict) -> bool:
        """Test de santé d'un service"""
        try:
            url = f"http://localhost:{service['port']}/health"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ {service['name']} - Healthy")
                    return True
                else:
                    print(f"   ⚠️ {service['name']} - Status {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ {service['name']} - {str(e)[:50]}...")
            return False
    
    async def test_service_endpoints(self, service: dict) -> int:
        """Test des endpoints spécifiques d'un service"""
        tests_passed = 0
        port = service['port']
        name = service['name']
        
        # Test endpoint racine
        try:
            async with self.session.get(f"http://localhost:{port}/") as response:
                if response.status == 200:
                    tests_passed += 1
                    print(f"   ✅ {name} - Endpoint racine OK")
                else:
                    print(f"   ⚠️ {name} - Endpoint racine: {response.status}")
        except Exception as e:
            print(f"   ❌ {name} - Endpoint racine: {str(e)[:30]}...")
        
        # Tests spécifiques par service
        if "Auth" in name:
            try:
                # Test endpoint de profil
                async with self.session.get(f"http://localhost:{port}/users/profile") as response:
                    if response.status == 200:
                        tests_passed += 1
                        print(f"   ✅ {name} - Endpoint profil OK")
            except Exception as e:
                print(f"   ❌ {name} - Endpoint profil: {str(e)[:30]}...")
        
        elif "RAG" in name:
            try:
                # Test endpoint templates
                async with self.session.get(f"http://localhost:{port}/api/v1/rag/templates") as response:
                    if response.status == 200:
                        tests_passed += 1
                        print(f"   ✅ {name} - Endpoint templates OK")
            except Exception as e:
                print(f"   ❌ {name} - Endpoint templates: {str(e)[:30]}...")
        
        elif "Quest" in name:
            try:
                # Test endpoint stats publiques
                async with self.session.get(f"http://localhost:{port}/stats/public") as response:
                    if response.status == 200:
                        tests_passed += 1
                        print(f"   ✅ {name} - Endpoint stats OK")
            except Exception as e:
                print(f"   ❌ {name} - Endpoint stats: {str(e)[:30]}...")
        
        elif "Analytics" in name:
            try:
                # Test endpoint overview
                async with self.session.get(f"http://localhost:{port}/api/v1/analytics/overview") as response:
                    if response.status == 200:
                        tests_passed += 1
                        print(f"   ✅ {name} - Endpoint overview OK")
            except Exception as e:
                print(f"   ❌ {name} - Endpoint overview: {str(e)[:30]}...")
        
        elif "Notification" in name:
            try:
                # Test endpoint stats
                async with self.session.get(f"http://localhost:{port}/api/v1/notifications/stats") as response:
                    if response.status == 200:
                        tests_passed += 1
                        print(f"   ✅ {name} - Endpoint stats OK")
            except Exception as e:
                print(f"   ❌ {name} - Endpoint stats: {str(e)[:30]}...")
        
        return tests_passed
    
    async def run_comprehensive_tests(self):
        """Exécute tous les tests d'intégration"""
        print("🧪 TESTS D'INTÉGRATION AUTOMATISÉS DOCTORPY")
        print("=" * 55)
        
        start_time = time.time()
        
        # Démarrer les services
        self.start_services()
        
        try:
            # Tests de santé
            print("\n🏥 Tests de santé des services")
            health_results = []
            for service in self.services:
                result = await self.test_service_health(service)
                health_results.append(result)
            
            # Tests des endpoints
            print("\n🔍 Tests des endpoints")
            endpoint_tests = 0
            total_endpoint_tests = 0
            for service in self.services:
                print(f"\n📋 Test de {service['name']}:")
                passed = await self.test_service_endpoints(service)
                endpoint_tests += passed
                total_endpoint_tests += 2  # Chaque service a 2 tests
            
            # Résumé
            print("\n" + "=" * 55)
            print("📊 RÉSUMÉ DES TESTS")
            print("=" * 55)
            
            healthy_services = sum(health_results)
            print(f"🏥 Services en santé: {healthy_services}/{len(self.services)}")
            print(f"🔍 Tests d'endpoints réussis: {endpoint_tests}/{total_endpoint_tests}")
            
            # Score global
            total_tests = len(self.services) + total_endpoint_tests
            passed_tests = healthy_services + endpoint_tests
            score = (passed_tests / total_tests) * 100
            
            print(f"\n🎯 Score global: {score:.1f}% ({passed_tests}/{total_tests} tests réussis)")
            
            duration = time.time() - start_time
            print(f"⏱️ Durée totale: {duration:.2f}s")
            
            # Recommandations
            if score >= 90:
                print("\n🎉 Excellent! Architecture microservices opérationnelle")
                print("   ✅ Tous les services sont prêts pour utilisation")
            elif score >= 70:
                print("\n👍 Bon état général, quelques améliorations possibles")
                print("   🔧 La plupart des fonctionnalités sont opérationnelles")
            else:
                print("\n⚠️ Problèmes détectés, vérification nécessaire")
                print("   🛠️ Consulter les logs pour plus de détails")
            
            return score >= 70
            
        finally:
            # Arrêter les services
            self.stop_services()

async def main():
    """Point d'entrée principal"""
    try:
        async with AutoIntegrationTester() as tester:
            success = await tester.run_comprehensive_tests()
            return 0 if success else 1
    except Exception as e:
        print(f"\n💥 Erreur critique: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)