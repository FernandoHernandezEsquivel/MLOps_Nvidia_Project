# monitoring/alert_system.py
"""
Sistema de alertas para monitoreo MLOps.
"""
from datetime import datetime
import os

def send_alert(subject, message, severity="warning"):
    """
    Envía alerta por consola y archivo de log.
    
    Args:
        subject: Asunto de la alerta
        message: Mensaje detallado
        severity: 'info', 'warning', 'error'
    """
    icons = {
        'info': 'ℹ️',
        'warning': '⚠️',
        'error': '❌'
    }
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Mostrar en consola
    print(f"\n{'='*60}")
    print(f"{icons.get(severity, '📢')} ALERTA: {subject}")
    print(f"{'='*60}")
    print(f"📝 {message}")
    print(f"🕐 {timestamp}")
    print(f"📊 Severidad: {severity.upper()}")
    print(f"{'='*60}\n")
    
    # Guardar en archivo de log
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/alerts_{datetime.now().strftime('%Y%m')}.log"
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] [{severity.upper()}] {subject}: {message}\n")
    
    # Aquí se puede agregar:
    # - Email con smtplib
    # - Slack con webhook
    # - Teams con webhook
    
    return True

def send_drift_alert(drift_share, features_affected):
    """
    Alerta específica para deriva de datos.
    """
    subject = "⚠️ Deriva de Datos Detectada"
    message = f"""
    Se ha detectado deriva de datos en el modelo NVIDIA_Price_Predictor.
    
    📊 Porcentaje de deriva: {drift_share*100:.1f}%
    🔧 Features afectadas: {', '.join(features_affected[:5])}
    
    📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    🔄 Acción recomendada: Reentrenar el modelo.
    """
    
    return send_alert(subject, message, "error")

if __name__ == "__main__":
    # Prueba
    send_alert(
        "🧪 Prueba de Alertas",
        "Este es un mensaje de prueba del sistema de alertas.",
        "info"
    )