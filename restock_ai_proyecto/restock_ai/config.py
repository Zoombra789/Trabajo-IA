"""
====================================================================
CONFIGURACION GENERAL DEL PROYECTO
====================================================================
Centraliza rutas, umbrales de negocio y nombres de modelo para que
sea facil ajustar el comportamiento del sistema sin tener que tocar
el resto del codigo. Todos los demas modulos importan este archivo.
====================================================================
"""

import os

# --------------------------------------------------------------
# Rutas de archivos (se calculan automaticamente, no hace falta
# tocarlas al mover el proyecto de carpeta o de computadora)
# --------------------------------------------------------------
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS_VENTAS = os.path.join(RUTA_BASE, "datos", "ventas_historicas.csv")
RUTA_MODELO_YOLO_CUSTOM = os.path.join(RUTA_BASE, "modelos", "best.pt")

# --------------------------------------------------------------
# Parametros de negocio (ajustables segun la operacion real)
# --------------------------------------------------------------
DIAS_ALERTA_MIN = 3            # dias restantes para considerar "URGENTE"
DIAS_ALERTA_MAX = 5            # dias restantes para considerar "Alerta"
CAPACIDAD_MAXIMA_ESTANTE = 20  # productos esperados en un estante "lleno"

# --------------------------------------------------------------
# Modelos de LLM para el chatbot.
# Actualiza estos nombres si tu proveedor lanza versiones mas nuevas;
# los proveedores suelen renombrar/retirar modelos con el tiempo.
# --------------------------------------------------------------
MODELO_OPENAI = "gpt-4o-mini"
MODELO_GEMINI = "gemini-1.5-flash"
