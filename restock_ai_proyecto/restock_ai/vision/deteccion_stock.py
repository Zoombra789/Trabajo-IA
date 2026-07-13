"""
====================================================================
DETECCION DE STOCK EN ESTANTERIAS CON YOLOv8 (ultralytics)
====================================================================
Usa un modelo YOLOv8 para detectar productos en una foto de un
estante y estimar el porcentaje de "llenado" comparando la cantidad
de objetos detectados contra una capacidad esperada.

MODO DEMO: si todavia no entrenaste un modelo propio, se usa el
modelo generico 'yolov8n.pt' (entrenado con el dataset COCO) solo
para probar que el pipeline funciona de punta a punta. Para produccion
real hay que entrenar un modelo personalizado con fotos de tus
productos y estantes.

Para entrenar tu propio modelo: ver notebook
entrenamiento_colab/entrenar_yolo.ipynb (se ejecuta gratis en Google
Colab, sin instalar nada en tu computadora).
====================================================================
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

import cv2
from ultralytics import YOLO

MODELO_DEMO = "yolov8n.pt"

# Clases reales de COCO (las que yolov8n puede reconocer) agrupadas en
# categorias de negocio, para que la etiqueta se vea mas profesional.
# Todo lo demas que el modelo detecte se muestra como "Producto".
# Importante: esto NO inventa una deteccion -- solo traduce el nombre
# real de la clase que YOLO ya identifico a un rotulo mas legible.
CLASES_BEBIDA = {"bottle", "wine glass", "cup"}
CLASES_PERECIBLE = {"banana", "apple", "orange", "sandwich", "cake", "pizza",
                     "donut", "hot dog", "broccoli", "carrot"}

# Un color distinto (BGR) por categoria, para diferenciarlas de un vistazo
COLORES_ETIQUETA = {
    "Bebida": (0, 140, 255),      # naranja
    "Perecible": (0, 170, 0),     # verde
    "Producto": (255, 120, 0),    # azul
}


def etiqueta_negocio(nombre_clase_real: str) -> str:
    """Traduce el nombre de clase que YOLO detecto de verdad (en ingles,
    del dataset COCO) a una etiqueta de negocio en espanol."""
    if nombre_clase_real in CLASES_BEBIDA:
        return "Bebida"
    if nombre_clase_real in CLASES_PERECIBLE:
        return "Perecible"
    return "Producto"


def cargar_modelo():
    """Carga el modelo custom entrenado si existe; si no, el modelo
    demo generico (ultralytics lo descarga automaticamente la
    primera vez que se usa)."""
    if os.path.exists(config.RUTA_MODELO_YOLO_CUSTOM):
        print(f"Usando modelo entrenado propio: {config.RUTA_MODELO_YOLO_CUSTOM}")
        return YOLO(config.RUTA_MODELO_YOLO_CUSTOM)

    print(f"[MODO DEMO] No se encontro un modelo propio, usando '{MODELO_DEMO}'")
    return YOLO(MODELO_DEMO)


def analizar_estante(ruta_imagen: str, capacidad_maxima: int = None):
    """
    Analiza una imagen de un estante y devuelve:
      - cantidad de productos detectados
      - porcentaje de llenado estimado
      - estado (CRITICO / ALERTA / OK)
      - la imagen anotada con las cajas de deteccion (para mostrarla
        en el dashboard)
    """
    if capacidad_maxima is None:
        capacidad_maxima = config.CAPACIDAD_MAXIMA_ESTANTE

    modelo = cargar_modelo()
    resultados = modelo(ruta_imagen)
    resultado = resultados[0]

    cantidad_detectada = len(resultado.boxes)
    porcentaje_llenado = min(100, round((cantidad_detectada / capacidad_maxima) * 100, 1))

    if porcentaje_llenado < 20:
        estado = "CRITICO - Estante casi vacio, re-stock inmediato"
    elif porcentaje_llenado < 50:
        estado = "ALERTA - Nivel bajo de stock"
    else:
        estado = "OK - Stock suficiente"

    # ------------------------------------------------------------
    # Dibujamos nosotros las cajas (en vez de resultado.plot()) para
    # mostrar una etiqueta de negocio legible ("Producto" / "Bebida")
    # en lugar del nombre crudo de la clase de COCO + score de
    # confianza. La etiqueta se calcula a partir de la clase que el
    # modelo detecto de verdad (ver etiqueta_negocio), no es un texto
    # fijo: si cambia lo que hay en la caja, cambia la etiqueta.
    # ------------------------------------------------------------
    imagen_anotada = resultado.orig_img.copy()
    for caja in resultado.boxes:
        x1, y1, x2, y2 = map(int, caja.xyxy[0])
        nombre_real = modelo.names[int(caja.cls[0])]
        etiqueta = etiqueta_negocio(nombre_real)
        color = COLORES_ETIQUETA[etiqueta]

        cv2.rectangle(imagen_anotada, (x1, y1), (x2, y2), color, 2)
        cv2.putText(imagen_anotada, etiqueta, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return {
        "cantidad_detectada": cantidad_detectada,
        "porcentaje_llenado": porcentaje_llenado,
        "estado": estado,
        "imagen_anotada": imagen_anotada,
    }


# --------------------------------------------------------------
# Prueba rapida: coloca una imagen en datos/foto_estante_ejemplo.jpg
# y ejecuta: python vision/deteccion_stock.py
# --------------------------------------------------------------
if __name__ == "__main__":
    ruta_prueba = os.path.join(config.RUTA_BASE, "datos", "foto_estante_ejemplo.jpg")
    if os.path.exists(ruta_prueba):
        resultado = analizar_estante(ruta_prueba)
        print(resultado["estado"], "-", resultado["porcentaje_llenado"], "%")
    else:
        print("Coloca una imagen de prueba en datos/foto_estante_ejemplo.jpg para probar este modulo.")
