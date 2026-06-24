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

# ============ CONFIGURACIÓN ============
# Obtener la ruta absoluta del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow_runs")

# Verificar que el directorio existe
if not os.path.exists(MLFLOW_DIR):
    print(f"❌ Error: El directorio {MLFLOW_DIR} no existe")
    print(f"💡 Asegúrate de ejecutar el script desde la raíz del proyecto")
    sys.exit(1)

# Configurar MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"file:{MLFLOW_DIR}"
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")

print(f"🔍 MLflow configurado en: {MLFLOW_DIR}")
print(f"📁 Existe: {os.path.exists(MLFLOW_DIR)}")

# Verificar que MLflow está instalado
try:
    print(f"✅ MLflow {mlflow.__version__} instalado")
except:
    print("❌ Error: MLflow no está instalado")
    print("💡 Ejecuta: pip install mlflow")
    sys.exit(1)

# ============ FUNCIONES ============
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
            
            print(f"   {stage_emoji} Versión {v.version} ({v.stage})")
            print(f"      Run ID: {v.run_id[:8]}...")
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
    """Promociona un modelo a una etapa específica."""
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

# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(
        description='Promocionar modelos en MLflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/promote_model.py --list
  python scripts/promote_model.py --model NVIDIA_Price_Predictor --version 1 --stage Production
  python scripts/promote_model.py --auto --stage Production --no-approval
        """
    )
    parser.add_argument('--list', action='store_true', help='Listar todos los modelos')
    parser.add_argument('--model', type=str, default="NVIDIA_Price_Predictor", help='Nombre del modelo')
    parser.add_argument('--version', type=int, help='Versión a promocionar')
    parser.add_argument('--stage', type=str, default="Production", help='Etapa destino')
    parser.add_argument('--auto', action='store_true', help='Promocionar automáticamente la última versión')
    parser.add_argument('--no-approval', action='store_true', help='Saltar aprobación manual')
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        sys.exit(0)
    
    if args.auto:
        versions = get_model_versions(args.model)
        if versions:
            latest = versions[-1].version
            promote_model(args.model, latest, args.stage, require_approval=not args.no_approval)
        else:
            print(f"❌ No hay versiones para {args.model}")
        sys.exit(0)
    
    if args.version:
        promote_model(args.model, args.version, args.stage, require_approval=not args.no_approval)
    else:
        print("❌ Especifica --version, --auto o --list")
        parser.print_help()

if __name__ == "__main__":
    main()