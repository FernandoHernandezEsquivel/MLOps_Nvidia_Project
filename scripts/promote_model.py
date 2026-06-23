# scripts/promote_model.py
"""
Herramienta para promocionar modelos en MLflow.
Permite listar, promocionar y hacer rollback de modelos.
"""
import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import argparse
from datetime import datetime

# Configurar MLflow
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlflow_runs"
mlflow.set_tracking_uri("file:./mlflow_runs")

def list_models():
    """Lista todos los modelos registrados con sus versiones."""
    client = MlflowClient()
    models = client.search_registered_models()
    
    print("\n" + "="*60)
    print("📋 MODELOS REGISTRADOS")
    print("="*60)
    
    if not models:
        print("❌ No hay modelos registrados.")
        return
    
    for model in models:
        print(f"\n📦 {model.name}")
        versions = client.search_model_versions(f"name='{model.name}'")
        for v in versions:
            stage_emoji = {
                'None': '📦',
                'Staging': '🧪',
                'Production': '🚀',
                'Archived': '📁'
            }.get(v.stage, '📦')
            
            # Obtener métricas del run
            run = client.get_run(v.run_id)
            metrics = run.data.metrics
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in list(metrics.items())[:3]])
            
            print(f"   {stage_emoji} Versión {v.version} ({v.stage})")
            print(f"      Run ID: {v.run_id[:8]}...")
            print(f"      Métricas: {metrics_str}")
            print(f"      Creado: {datetime.fromtimestamp(v.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M')}")

def get_model_versions(model_name):
    """Obtiene todas las versiones de un modelo."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return sorted(versions, key=lambda v: int(v.version))
    except:
        return []

def promote_model(model_name, version, stage="Production", archive=True, require_approval=True):
    """
    Promociona un modelo a una etapa específica.
    
    Args:
        model_name: Nombre del modelo
        version: Número de versión
        stage: Etapa destino ('Staging', 'Production', 'Archived')
        archive: Si archivar versiones existentes
        require_approval: Si requiere confirmación manual
    """
    client = MlflowClient()
    
    # Verificar que el modelo existe
    versions = get_model_versions(model_name)
    if not versions:
        print(f"❌ Modelo '{model_name}' no encontrado")
        return False
    
    # Verificar que la versión existe
    version_exists = any(str(v.version) == str(version) for v in versions)
    if not version_exists:
        print(f"❌ Versión {version} no existe")
        print(f"   Versiones disponibles: {[v.version for v in versions]}")
        return False
    
    # Verificar etapa válida
    valid_stages = ['Staging', 'Production', 'Archived']
    if stage not in valid_stages:
        print(f"❌ Etapa inválida. Opciones: {valid_stages}")
        return False
    
    # Solicitar aprobación si está activada
    if require_approval:
        print(f"\n📋 REVISIÓN DE MODELO")
        print(f"   Modelo: {model_name}")
        print(f"   Versión: {version}")
        print(f"   Etapa destino: {stage}")
        print(f"   📊 Revisa métricas en: http://localhost:5000")
        
        response = input("\n¿Aprobar promoción? (s/n): ")
        if response.lower() != 's':
            print("❌ Promoción cancelada")
            return False
    
    # Promocionar
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive
        )
        print(f"✅ Modelo {model_name} versión {version} promocionado a {stage}")
        return True
    except Exception as e:
        print(f"❌ Error al promocionar: {e}")
        return False

def rollback_model(model_name, version):
    """Hace rollback a una versión anterior en producción."""
    client = MlflowClient()
    
    # Obtener versión actual en producción
    versions = get_model_versions(model_name)
    current_prod = [v for v in versions if v.stage == "Production"]
    
    if not current_prod:
        print("❌ No hay versión en producción")
        return False
    
    current_version = current_prod[0].version
    
    # Confirmar rollback
    print(f"\n⚠️ ROLLBACK")
    print(f"   Modelo: {model_name}")
    print(f"   Versión actual en producción: {current_version}")
    print(f"   Versión destino: {version}")
    
    response = input("¿Confirmar rollback? (s/n): ")
    if response.lower() != 's':
        print("❌ Rollback cancelado")
        return False
    
    try:
        # Archivar versión actual
        client.transition_model_version_stage(
            name=model_name,
            version=current_version,
            stage="Archived"
        )
        # Promocionar versión anterior
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        print(f"✅ Rollback completado. Producción ahora en versión {version}")
        return True
    except Exception as e:
        print(f"❌ Error en rollback: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Promocionar modelos en MLflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/promote_model.py --list
  python scripts/promote_model.py --model NVIDIA_Price_Predictor --version 1 --stage Production
  python scripts/promote_model.py --model NVIDIA_Price_Predictor --rollback 1
  python scripts/promote_model.py --auto --stage Production
        """
    )
    parser.add_argument('--list', action='store_true', help='Listar todos los modelos')
    parser.add_argument('--model', type=str, default="NVIDIA_Price_Predictor", help='Nombre del modelo')
    parser.add_argument('--version', type=int, help='Versión a promocionar')
    parser.add_argument('--stage', type=str, default="Production", help='Etapa destino (Staging/Production/Archived)')
    parser.add_argument('--rollback', type=int, help='Hacer rollback a la versión especificada')
    parser.add_argument('--auto', action='store_true', help='Promocionar automáticamente la última versión (sin aprobación)')
    parser.add_argument('--no-approval', action='store_true', help='Saltar aprobación manual')
    
    args = parser.parse_args()
    
    # Caso 1: Listar modelos
    if args.list:
        list_models()
        sys.exit(0)
    
    # Caso 2: Rollback
    if args.rollback:
        rollback_model(args.model, args.rollback)
        sys.exit(0)
    
    # Caso 3: Promoción automática (última versión)
    if args.auto:
        versions = get_model_versions(args.model)
        if versions:
            latest = versions[-1].version
            promote_model(args.model, latest, args.stage, require_approval=not args.no_approval)
        else:
            print(f"❌ No hay versiones para {args.model}")
        sys.exit(0)
    
    # Caso 4: Promoción manual
    if args.version:
        promote_model(args.model, args.version, args.stage, require_approval=not args.no_approval)
    else:
        print("❌ Especifica --version, --auto o --list")
        parser.print_help()

if __name__ == "__main__":
    main()