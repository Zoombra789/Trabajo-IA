"""
====================================================================
CLASIFICACION ABC DE PRODUCTOS (CLUSTERING CON K-MEANS)
====================================================================
Agrupa los productos en clusters segun su comportamiento de venta
(ingresos totales, unidades vendidas y frecuencia de venta) usando
K-Means de scikit-learn. Despues, los clusters se ordenan por su
ingreso promedio y se etiquetan como:

    A = cluster de mayor ingreso promedio  -> productos criticos
    B = cluster intermedio                 -> productos importantes
    C = cluster de menor ingreso promedio   -> bajo impacto

Esta es la clasica clasificacion ABC de inventarios, pero calculada
mediante clustering no supervisado en vez de un simple corte manual
por porcentaje acumulado (ese porcentaje tambien se calcula, como
dato de referencia adicional para el dashboard).
====================================================================
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _resumir_por_producto(df_ventas: pd.DataFrame) -> pd.DataFrame:
    """Calcula, por cada producto, el ingreso total, las unidades
    totales vendidas y la frecuencia de venta (numero de dias en los
    que se vendio al menos 1 unidad)."""
    df = df_ventas.copy()
    df["ingreso"] = df["unidades_vendidas"] * df["precio_unitario"]

    resumen = (
        df.groupby(["producto_id", "producto_nombre"])
        .agg(
            ingreso_total=("ingreso", "sum"),
            unidades_totales=("unidades_vendidas", "sum"),
            frecuencia_venta=("unidades_vendidas", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    return resumen


def clasificar_abc(df_ventas: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Recibe el historico de ventas y devuelve un resumen por producto
    con su categoria ABC, calculada mediante K-Means sobre ingreso,
    unidades vendidas y frecuencia de venta.
    """
    resumen = _resumir_por_producto(df_ventas)

    # ------------------------------------------------------------
    # 1. Normalizar variables (K-Means es sensible a la escala:
    #    sin esto, el ingreso en soles/dolares dominaria sobre las
    #    otras variables solo por tener numeros mas grandes)
    # ------------------------------------------------------------
    features = resumen[["ingreso_total", "unidades_totales", "frecuencia_venta"]]
    escalador = StandardScaler()
    features_escaladas = escalador.fit_transform(features)

    # ------------------------------------------------------------
    # 2. Entrenar K-Means para agrupar productos con comportamiento similar
    # ------------------------------------------------------------
    n_clusters_real = min(n_clusters, len(resumen))  # por si hay pocos productos
    kmeans = KMeans(n_clusters=n_clusters_real, random_state=42, n_init=10)
    resumen["cluster"] = kmeans.fit_predict(features_escaladas)

    # ------------------------------------------------------------
    # 3. Ordenar los clusters por ingreso promedio y asignarles A/B/C
    # ------------------------------------------------------------
    ingreso_promedio_por_cluster = (
        resumen.groupby("cluster")["ingreso_total"].mean().sort_values(ascending=False)
    )
    etiquetas_disponibles = ["A", "B", "C", "D", "E"][:n_clusters_real]
    mapa_cluster_a_categoria = {
        cluster_id: etiquetas_disponibles[i]
        for i, cluster_id in enumerate(ingreso_promedio_por_cluster.index)
    }
    resumen["categoria_abc"] = resumen["cluster"].map(mapa_cluster_a_categoria)

    # ------------------------------------------------------------
    # 4. Calcular tambien el % acumulado (referencia visual tipo Pareto)
    # ------------------------------------------------------------
    resumen = resumen.sort_values("ingreso_total", ascending=False).reset_index(drop=True)
    resumen["ingreso_acumulado"] = resumen["ingreso_total"].cumsum()
    resumen["porcentaje_acumulado"] = (
        resumen["ingreso_acumulado"] / resumen["ingreso_total"].sum() * 100
    )

    return resumen.drop(columns=["cluster"])


# --------------------------------------------------------------
# Prueba rapida: python analitica/clustering_abc.py
# --------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config

    df = pd.read_csv(config.RUTA_DATOS_VENTAS)
    resultado = clasificar_abc(df)
    print(resultado[["producto_nombre", "ingreso_total", "frecuencia_venta", "categoria_abc"]])
