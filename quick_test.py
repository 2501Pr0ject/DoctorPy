#!/usr/bin/env python3
"""
Test rapide des services simplifiés
"""

import subprocess
import time
import requests
import sys

def test_service(port, name):
    """Test d'un service sur un port donné"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            print(f"✅ {name} (port {port}) : Opérationnel")
            return True
        else:
            print(f"⚠️ {name} (port {port}) : Status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print(f"❌ {name} (port {port}) : Inaccessible")
        return False

def main():
    """Test d'intégration rapide"""
    print("🧪 Test rapide d'intégration DoctorPy")
    print("=" * 45)
    
    # Services à tester
    services = [
        (8001, "Auth Service"),
        (8002, "RAG Service"), 
        (8004, "Quest Service")
    ]
    
    # Démarrer les services
    print("\n🚀 Démarrage des services...")
    processes = []
    
    for port, name in services:
        script = f"run_{name.split()[0].lower()}.py"
        try:
            process = subprocess.Popen([sys.executable, script], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
            processes.append(process)
            print(f"   Lancé {name}")
            time.sleep(2)  # Délai entre les services
        except Exception as e:
            print(f"   ❌ Erreur lancement {name}: {e}")
    
    # Attendre un peu que les services démarrent
    print("\n⏳ Attente du démarrage des services...")
    time.sleep(10)
    
    # Tester les services
    print("\n🔍 Test des services...")
    results = []
    for port, name in services:
        results.append(test_service(port, name))
    
    # Résumé
    print(f"\n📊 Résultats:")
    success_count = sum(results)
    print(f"   Services opérationnels: {success_count}/{len(services)}")
    
    if success_count == len(services):
        print("   🎉 Tous les services fonctionnent!")
        final_status = "SUCCÈS"
    elif success_count > 0:
        print("   ⚠️ Certains services ont des problèmes")
        final_status = "PARTIEL"
    else:
        print("   ❌ Aucun service ne répond")
        final_status = "ÉCHEC"
    
    # Arrêt des services
    print("\n🛑 Arrêt des services...")
    for i, process in enumerate(processes):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
    
    print(f"\n🏁 Test terminé: {final_status}")
    return success_count == len(services)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)