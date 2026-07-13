"""
====================================================================
MODELO DE PREDICCION DE DEMANDA Y ALERTAS DE RE-STOCK
====================================================================
Para cada producto entrena un modelo de regresion (Random Forest,
scikit-learn) que usa el historico de ventas para proyectar la
demanda de los proximos dias. Con esa proyeccion calcula en cuantos
dias se agotaria el stock actual y genera una alerta cuando faltan
pocos dias para el quiebre (por defecto: entre 3 y 5 dias).
====================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def _crear_features(df_producto: pd.DataFrame) -> pd.DataFrame:
    """Crea variables predictoras a partir de la fecha: dia de la
    semana, un indice temporal, y las ventas de dias anteriores
    (lags), que suelen ser las que mejor explican el consumo."""
    df_producto = df_producto.sort_values("fecha").reset_index(drop=True)
    df_producto["fecha"] = pd.to_datetime(df_producto["fecha"])
    df_producto["dia_semana"] = df_producto["fecha"].dt.weekday
    df_producto["dia_index"] = np.arange(len(df_producto))

    for lag in [1, 2, 3, 7]:
        df_producto[f"lag_{lag}"] = df_producto["unidades_vendidas"].shift(lag)

    return df_producto.dropna().reset_index(drop=True)


def predecir_demanda_producto(df_producto: pd.DataFrame, dias_a_futuro: int = 7):
    """Entrena un RandomForestRegressor con el historico de un
    producto y proyecta la demanda diaria para los proximos dias."""
    datos = _crear_features(df_producto)

    if len(datos) < 15:
        # Muy pocos datos para un modelo confiable: se usa el promedio simple
        promedio = df_producto["unidades_vendidas"].mean()
        return [max(0, round(promedio))] * dias_a_futuro

    columnas_features = ["dia_semana", "dia_index", "lag_1", "lag_2", "lag_3", "lag_7"]
    X = datos[columnas_features]
    y = datos["unidades_vendidas"]

    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo.fit(X, y)

    # ------------------------------------------------------------
    # Proyeccion iterativa dia a dia: cada prediccion nueva se usa
    # como "lag" para seguir prediciendo el dia siguiente
    # ------------------------------------------------------------
    historial_ventas = list(datos["unidades_vendidas"].values)
    ultimo_dia_index = datos["dia_index"].iloc[-1]
    ultima_fecha = datos["fecha"].iloc[-1]

    predicciones = []
    for i in range(1, dias_a_futuro + 1):
        fecha_futura = ultima_fecha + pd.Timedelta(days=i)
        fila = {
            "dia_semana": fecha_futura.weekday(),
            "dia_index": ultimo_dia_index + i,
            "lag_1": historial_ventas[-1],
            "lag_2": historial_ventas[-2],
            "lag_3": historial_ventas[-3],
            "lag_7": historial_ventas[-7],
        }
        X_pred = pd.DataFrame([fila])[columnas_features]
        pred = max(0, modelo.predict(X_pred)[0])
        predicciones.append(round(pred, 1))
        historial_ventas.append(pred)

    return predicciones


def generar_alertas_restock(df_ventas: pd.DataFrame, df_abc: pd.DataFrame, dias_a_futuro: int = 7) -> pd.DataFrame:
    """Recorre todos los productos, proyecta su demanda futura,
    calcula en cuantos dias se quedarian sin stock segun el stock
    actual, y arma la tabla de alertas ordenada por urgencia."""
    alertas = []

    for producto_id in df_ventas["producto_id"].unique():
        df_prod = df_ventas[df_ventas["producto_id"] == producto_id]
        nombre = df_prod["producto_nombre"].iloc[0]
        stock_actual = df_prod.sort_values("fecha")["stock_actual"].iloc[-1]

        predicciones = predecir_demanda_producto(df_prod, dias_a_futuro)

        # Restar la demanda proyectada dia a dia hasta encontrar el quiebre
        stock_restante = stock_actual
        dia_quiebre = None
        for i, demanda_dia in enumerate(predicciones, start=1):
            stock_restante -= demanda_dia
            if stock_restante <= 0:
                dia_quiebre = i
                break

        categoria_abc = "N/A"
        fila_abc = df_abc[df_abc["producto_id"] == producto_id]
        if not fila_abc.empty:
            categoria_abc = fila_abc["categoria_abc"].iloc[0]

        alertas.append({
            "producto_id": producto_id,
            "producto_nombre": nombre,
            "categoria_abc": categoria_abc,
            "stock_actual": stock_actual,
            "demanda_promedio_diaria": round(np.mean(predicciones), 1),
            "dias_hasta_quiebre": dia_quiebre if dia_quiebre is not None else f"> {dias_a_futuro}",
        })

    df_alertas = pd.DataFrame(alertas)

    # ------------------------------------------------------------
    # Nivel de urgencia segun los dias que faltan para el quiebre
    # ------------------------------------------------------------
    def nivel_urgencia(dias):
        if isinstance(dias, str):
            return "Sin riesgo"
        if dias <= config.DIAS_ALERTA_MIN:
            return "URGENTE"
        elif dias <= config.DIAS_ALERTA_MAX:
            return "Alerta"
        else:
            return "Seguimiento"

    df_alertas["urgencia"] = df_alertas["dias_hasta_quiebre"].apply(nivel_urgencia)

    # ------------------------------------------------------------
    # Ordenar: primero lo mas urgente, y dentro de cada urgencia,
    # primero los productos categoria A (los mas importantes)
    # ------------------------------------------------------------
    orden_urgencia = {"URGENTE": 0, "Alerta": 1, "Seguimiento": 2, "Sin riesgo": 3}
    orden_abc = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "N/A": 5}
    df_alertas["_orden_urgencia"] = df_alertas["urgencia"].map(orden_urgencia)
    df_alertas["_orden_abc"] = df_alertas["categoria_abc"].map(orden_abc)
    df_alertas = (
        df_alertas.sort_values(["_orden_urgencia", "_orden_abc"])
        .drop(columns=["_orden_urgencia", "_orden_abc"])
        .reset_index(drop=True)
    )

    return df_alertas


# --------------------------------------------------------------
# Prueba rapida: python analitica/prediccion_demanda.py
# --------------------------------------------------------------
if __name__ == "__main__":
    from clustering_abc import clasificar_abc

    df_ventas = pd.read_csv(config.RUTA_DATOS_VENTAS)
    df_abc = clasificar_abc(df_ventas)
    df_alertas = generar_alertas_restock(df_ventas, df_abc)
    print(df_alertas)
