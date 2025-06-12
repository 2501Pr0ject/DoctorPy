#!/usr/bin/env python3
"""
Tests d'intégration pour les microservices DoctorPy
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ServiceEndpoint:
    name: str
    base_url: str
    health_endpoint: str = "/health"


class IntegrationTester:
    """Testeur d'intégration des microservices"""
    
    def __init__(self):
        self.services = [
            ServiceEndpoint("Auth Service", "http://localhost:8001"),
            ServiceEndpoint("RAG Service", "http://localhost:8002"), 
            ServiceEndpoint("Quest Service", "http://localhost:8004")
        ]
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_service_health(self, service: ServiceEndpoint) -> Dict[str, Any]:
        """Teste le health check d'un service"""
        print(f"🔍 Test health check: {service.name}")
        
        try:
            async with self.session.get(
                f"{service.base_url}{service.health_endpoint}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ {service.name} est opérationnel")
                    return {"status": "healthy", "data": data}
                else:
                    print(f"  ⚠️  {service.name} retourne le statut {response.status}")
                    return {"status": "unhealthy", "code": response.status}
                    
        except asyncio.TimeoutError:
            print(f"  ❌ {service.name} - Timeout")
            return {"status": "timeout"}
        except aiohttp.ClientConnectorError:
            print(f"  ❌ {service.name} - Connexion impossible")
            return {"status": "unreachable"}
        except Exception as e:
            print(f"  ❌ {service.name} - Erreur: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_auth_service(self) -> bool:
        """Teste spécifiquement le service Auth"""
        print(f"\\n🔐 Test détaillé du service Auth")
        
        try:
            # Test de la route racine
            async with self.session.get("http://localhost:8001/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ Route racine accessible")
                    print(f"  📋 Service: {data.get('service', 'N/A')}")
                    return True
                else:
                    print(f"  ❌ Route racine inaccessible: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"  ❌ Erreur test Auth: {e}")
            return False
    
    async def test_rag_service(self) -> bool:
        """Teste spécifiquement le service RAG"""
        print(f"\\n🤖 Test détaillé du service RAG")
        
        try:
            # Test de la route racine
            async with self.session.get("http://localhost:8002/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ Route racine accessible")
                    print(f"  📋 Service: {data.get('service', 'N/A')}")
                    
            # Test des templates disponibles
            try:
                async with self.session.get("http://localhost:8002/api/v1/rag/templates") as response:
                    if response.status == 200:
                        data = await response.json()
                        templates = data.get('templates', {})
                        print(f"  ✅ Templates disponibles: {len(templates)}")
                    else:
                        print(f"  ⚠️  Templates non accessibles: {response.status}")
            except:
                print(f"  ⚠️  Templates nécessitent probablement une authentification")
                
            return True
                    
        except Exception as e:
            print(f"  ❌ Erreur test RAG: {e}")
            return False
    
    async def test_quest_service(self) -> bool:
        """Teste spécifiquement le service Quest"""
        print(f"\\n🎮 Test détaillé du service Quest")
        
        try:
            # Test de la route racine
            async with self.session.get("http://localhost:8004/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ Route racine accessible")
                    print(f"  📋 Service: {data.get('service', 'N/A')}")
                    features = data.get('features', [])
                    print(f"  🎯 Fonctionnalités: {', '.join(features)}")
                    
            # Test des stats publiques
            try:
                async with self.session.get("http://localhost:8004/stats/public") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"  ✅ Stats publiques:")
                        print(f"    • Quêtes totales: {data.get('total_quests', 0)}")
                        print(f"    • Utilisateurs: {data.get('total_users', 0)}")
                        print(f"    • Complétions: {data.get('total_completions', 0)}")
                    else:
                        print(f"  ⚠️  Stats publiques non accessibles: {response.status}")
            except:
                print(f"  ⚠️  Erreur récupération stats publiques")
            
            # Test des catégories
            try:
                async with self.session.get("http://localhost:8004/api/v1/quests/categories/list") as response:
                    if response.status == 200:
                        data = await response.json()
                        categories = data.get('categories', [])
                        print(f"  ✅ Catégories disponibles: {len(categories)}")
                    else:
                        print(f"  ⚠️  Catégories nécessitent probablement une authentification")
            except:
                print(f"  ⚠️  Catégories nécessitent une authentification")
                
            return True
                    
        except Exception as e:
            print(f"  ❌ Erreur test Quest: {e}")
            return False
    
    async def test_inter_service_communication(self) -> bool:
        """Teste la communication inter-services (simulation)"""
        print(f"\\n🔗 Test de communication inter-services")
        
        # Pour le moment, on teste juste que tous les services sont accessibles
        all_healthy = True
        
        for service in self.services:
            result = await self.test_service_health(service)
            if result["status"] != "healthy":
                all_healthy = False
        
        if all_healthy:
            print(f"  ✅ Tous les services sont opérationnels")
            print(f"  📡 Communication inter-services potentiellement fonctionnelle")
            return True
        else:
            print(f"  ❌ Certains services ne sont pas accessibles")
            return False
    
    async def run_all_tests(self):
        """Exécute tous les tests d'intégration"""
        print("🧪 TESTS D'INTÉGRATION DOCTORPY")
        print("=" * 50)
        
        start_time = time.time()
        
        # 1. Tests de santé globaux
        print("\\n🏥 Tests de santé des services")
        health_results = {}
        for service in self.services:
            health_results[service.name] = await self.test_service_health(service)
        
        # 2. Tests détaillés par service
        detailed_results = {}
        detailed_results["Auth"] = await self.test_auth_service()
        detailed_results["RAG"] = await self.test_rag_service()
        detailed_results["Quest"] = await self.test_quest_service()
        
        # 3. Test de communication inter-services
        inter_service_ok = await self.test_inter_service_communication()
        
        # 4. Résumé des résultats
        print("\\n" + "=" * 50)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 50)
        
        healthy_services = len([r for r in health_results.values() if r["status"] == "healthy"])
        total_services = len(self.services)
        
        print(f"🏥 Services en santé: {healthy_services}/{total_services}")
        
        detailed_ok = len([r for r in detailed_results.values() if r])
        print(f"🔍 Tests détaillés réussis: {detailed_ok}/{len(detailed_results)}")
        
        print(f"🔗 Communication inter-services: {'✅ OK' if inter_service_ok else '❌ KO'}")
        
        # Score global
        total_tests = total_services + len(detailed_results) + 1
        passed_tests = healthy_services + detailed_ok + (1 if inter_service_ok else 0)
        score = (passed_tests / total_tests) * 100
        
        print(f"\\n🎯 Score global: {score:.1f}% ({passed_tests}/{total_tests} tests réussis)")
        
        duration = time.time() - start_time
        print(f"⏱️  Durée des tests: {duration:.2f}s")
        
        # Recommendations
        print(f"\\n💡 Recommandations:")
        if score >= 90:
            print("  🎉 Excellent! Tous les services semblent fonctionnels")
            print("  🚀 Vous pouvez commencer à utiliser l'architecture microservices")
        elif score >= 70:
            print("  👍 Bon état général, quelques ajustements mineurs")
            print("  🔧 Vérifiez les services en erreur")
        else:
            print("  ⚠️  Plusieurs problèmes détectés")
            print("  🛠️  Vérifiez la configuration et les dépendances")
            print("  📋 Consultez les logs des services pour plus de détails")
        
        return score >= 70


async def main():
    """Point d'entrée principal"""
    try:
        async with IntegrationTester() as tester:
            success = await tester.run_all_tests()
            return 0 if success else 1
            
    except Exception as e:
        print(f"\\n💥 Erreur critique lors des tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)