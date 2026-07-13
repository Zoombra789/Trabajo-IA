"""
====================================================================
GENERADOR DE DATOS DE PRUEBA - VENTAS HISTORICAS
====================================================================
Crea un dataset SINTETICO de ventas para poder probar todo el sistema
de analisis predictivo sin necesitar todavia datos reales del negocio.

Genera un CSV con columnas:
    fecha, producto_id, producto_nombre, categoria,
    unidades_vendidas, precio_unitario, stock_actual

Se puede ejecutar de forma independiente:
    python datos/generar_datos_prueba.py
o se llama automaticamente desde el dashboard la primera vez que
se abre (si el CSV todavia no existe).
====================================================================
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------
# 1. CATALOGO DE PRODUCTOS DE EJEMPLO
#    (reemplaza esta lista por tu catalogo real cuando tengas datos)
# --------------------------------------------------------------
PRODUCTOS_DEMO = [
    {"id": "P001", "nombre": "Coca Cola 500ml",     "categoria": "Bebidas",    "precio": 3.5,  "venta_base": 40},
    {"id": "P002", "nombre": "Pan de Molde",         "categoria": "Panaderia", "precio": 5.2,  "venta_base": 25},
    {"id": "P003", "nombre": "Leche Entera 1L",      "categoria": "Lacteos",   "precio": 4.8,  "venta_base": 30},
    {"id": "P004", "nombre": "Arroz 1kg",            "categoria": "Abarrotes", "precio": 6.0,  "venta_base": 15},
    {"id": "P005", "nombre": "Detergente 1L",        "categoria": "Limpieza",  "precio": 12.5, "venta_base": 8},
    {"id": "P006", "nombre": "Papel Higienico x4",   "categoria": "Limpieza",  "precio": 9.0,  "venta_base": 10},
    {"id": "P007", "nombre": "Cerveza Lata",         "categoria": "Bebidas",   "precio": 4.0,  "venta_base": 20},
    {"id": "P008", "nombre": "Galletas Chocolate",   "categoria": "Snacks",    "precio": 3.0,  "venta_base": 18},
]


def generar_datos_ventas(dias_historia: int = 180, semilla: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame de ventas historicas simuladas para varios
    productos, incluyendo:
      - tendencia creciente leve (el negocio va vendiendo un poco mas)
      - estacionalidad semanal (mas ventas viernes/sabado/domingo)
      - ruido aleatorio (para que no sea una linea perfecta)
      - un ciclo de reabastecimiento periodico tipo "diente de sierra":
        cada producto recibe un pedido del proveedor cada cierta
        cantidad de dias (como en un negocio real), y entre pedido y
        pedido el stock baja con las ventas diarias. Esto hace que,
        al ultimo dia de la historia, cada producto quede en un punto
        distinto de su ciclo (algunos con buen stock, otros a punto
        de agotarse), en vez de que todos terminen igual.
    """
    np.random.seed(semilla)
    fecha_inicio = datetime.today() - timedelta(days=dias_historia)
    filas = []

    for prod in PRODUCTOS_DEMO:
        # Cada producto tiene su propio ciclo de reabastecimiento
        intervalo_reposicion = np.random.randint(10, 21)          # cada cuantos dias llega un pedido
        dias_desde_reposicion = np.random.randint(0, intervalo_reposicion)  # desfase inicial aleatorio
        nivel_objetivo = prod["venta_base"] * np.random.uniform(8, 15)      # dias de venta que cubre cada pedido

        stock_actual = nivel_objetivo  # arranca como si acabara de llegar un pedido

        for d in range(dias_historia):
            fecha = fecha_inicio + timedelta(days=d)

            estacionalidad = 1.3 if fecha.weekday() in [4, 5, 6] else 1.0
            tendencia = 1 + (d / dias_historia) * 0.15
            ruido = np.random.normal(1, 0.2)
            demanda_dia = max(0, int(prod["venta_base"] * estacionalidad * tendencia * ruido))

            # No se puede vender mas de lo que hay en el estante
            unidades_vendidas = min(demanda_dia, int(stock_actual))
            stock_actual -= unidades_vendidas

            filas.append({
                "fecha": fecha.strftime("%Y-%m-%d"),
                "producto_id": prod["id"],
                "producto_nombre": prod["nombre"],
                "categoria": prod["categoria"],
                "unidades_vendidas": unidades_vendidas,
                "precio_unitario": prod["precio"],
                "stock_actual": round(stock_actual),
            })

            # Revisar si toca reabastecer (llega el pedido del proveedor)
            dias_desde_reposicion += 1
            if dias_desde_reposicion >= intervalo_reposicion:
                stock_actual = nivel_objetivo * np.random.uniform(0.85, 1.15)
                dias_desde_reposicion = 0

    return pd.DataFrame(filas)


def guardar_datos(ruta_salida: str = None) -> str:
    """Genera los datos y los guarda en un archivo CSV. Devuelve la ruta usada."""
    if ruta_salida is None:
        carpeta = os.path.dirname(os.path.abspath(__file__))
        ruta_salida = os.path.join(carpeta, "ventas_historicas.csv")

    df = generar_datos_ventas()
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    return ruta_salida


# --------------------------------------------------------------
# Permite ejecutar este archivo directamente para generar el CSV
# --------------------------------------------------------------
if __name__ == "__main__":
    ruta = guardar_datos()
    print(f"Datos generados correctamente en: {ruta}")
