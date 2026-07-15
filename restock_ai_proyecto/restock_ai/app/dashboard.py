"""
====================================================================
DASHBOARD PRINCIPAL - SISTEMA DE RE-STOCK AUTOMATIZADO CON IA
====================================================================
Punto de entrada de la aplicacion. Para ejecutarlo, desde la carpeta
raiz del proyecto corre en la terminal:

    streamlit run app/dashboard.py

Este dashboard integra los 3 pilares del proyecto:
  1. Analisis predictivo   -> clustering ABC + prediccion de demanda
  2. Vision artificial      -> deteccion de stock (foto o camara en vivo)
  3. Chatbot interactivo    -> consultas en lenguaje natural

Se puede usar con datos de ejemplo (generados automaticamente) o con
un CSV de ventas propio que el usuario sube desde la seccion
"Cargar datos".
====================================================================
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Permite importar los modulos de las carpetas hermanas (analitica, vision, chatbot, datos)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from analitica.clustering_abc import clasificar_abc
from analitica.prediccion_demanda import generar_alertas_restock
from chatbot.asistente_llm import responder_pregunta

# --------------------------------------------------------------
# 1. CONFIGURACION GENERAL DE LA PAGINA
# --------------------------------------------------------------
st.set_page_config(page_title="Re-Stock IA - Prototipo", page_icon="📦", layout="wide")

COLUMNAS_REQUERIDAS = [
    "fecha", "producto_id", "producto_nombre", "categoria",
    "unidades_vendidas", "precio_unitario", "stock_actual",
]


# --------------------------------------------------------------
# 2. DATOS DE EJEMPLO (se generan una sola vez, con cache)
# --------------------------------------------------------------
@st.cache_data
def cargar_datos_demo():
    if not os.path.exists(config.RUTA_DATOS_VENTAS):
        from datos.generar_datos_prueba import guardar_datos
        guardar_datos(config.RUTA_DATOS_VENTAS)
    return pd.read_csv(config.RUTA_DATOS_VENTAS)


@st.cache_data
def procesar_analisis(df_ventas):
    df_abc = clasificar_abc(df_ventas)
    df_alertas = generar_alertas_restock(df_ventas, df_abc)
    return df_abc, df_alertas


# --------------------------------------------------------------
# 3. DATASET ACTIVO EN LA SESION
#    (datos de ejemplo por defecto, o el CSV que el usuario suba)
# --------------------------------------------------------------
if "df_ventas" not in st.session_state:
    st.session_state.df_ventas = cargar_datos_demo()
    st.session_state.fuente_datos = "demo"

df_ventas = st.session_state.df_ventas
df_abc, df_alertas = procesar_analisis(df_ventas)

# --------------------------------------------------------------
# 4. BARRA LATERAL DE NAVEGACION
# --------------------------------------------------------------
st.sidebar.title("📦 Re-Stock IA")
seccion = st.sidebar.radio(
    "Ir a:",
    ["Cargar datos", "Resumen general", "Clasificacion ABC", "Alertas de Re-Stock",
     "Deteccion visual (camara)", "Chatbot"],
)
st.sidebar.markdown("---")
fuente = "tus datos" if st.session_state.fuente_datos == "propio" else "datos de ejemplo"
st.sidebar.caption(f"Fuente de datos actual: **{fuente}**")
st.sidebar.caption("Prototipo - Fase de producto de prueba")

# --------------------------------------------------------------
# 5. SECCION: CARGAR DATOS (CSV propio o datos de ejemplo)
# --------------------------------------------------------------
if seccion == "Cargar datos":
    st.title("Cargar datos de ventas")
    st.write(
        "Sube tu propio historial de ventas en CSV para que el sistema calcule las "
        "alertas de re-stock sobre datos reales. Si no subes nada, se usan datos de "
        "ejemplo generados automaticamente."
    )

    st.markdown(f"**El archivo debe tener estas columnas:** `{'`, `'.join(COLUMNAS_REQUERIDAS)}`")

    archivo_csv = st.file_uploader("Archivo CSV de ventas", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Usar este archivo", disabled=archivo_csv is None):
            try:
                nuevo_df = pd.read_csv(archivo_csv)
                faltantes = set(COLUMNAS_REQUERIDAS) - set(nuevo_df.columns)
                if faltantes:
                    st.error(f"Faltan columnas en el archivo: {', '.join(faltantes)}")
                else:
                    st.session_state.df_ventas = nuevo_df
                    st.session_state.fuente_datos = "propio"
                    st.success("Datos propios cargados. Ve a 'Resumen general' o 'Alertas de Re-Stock' para ver el analisis.")
                    st.rerun()
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

    with col2:
        if st.button("Volver a datos de ejemplo"):
            st.session_state.df_ventas = cargar_datos_demo()
            st.session_state.fuente_datos = "demo"
            st.success("Usando datos de ejemplo otra vez.")
            st.rerun()

    st.subheader("Vista previa de los datos activos")
    st.dataframe(df_ventas.head(20), width="stretch")
    st.caption(f"{len(df_ventas)} filas cargadas.")

# --------------------------------------------------------------
# 6. SECCION: RESUMEN GENERAL
# --------------------------------------------------------------
elif seccion == "Resumen general":
    st.title("Resumen general del inventario")

    col1, col2, col3 = st.columns(3)
    col1.metric("Productos monitoreados", len(df_abc))
    col2.metric("Alertas urgentes", int((df_alertas["urgencia"] == "URGENTE").sum()))
    col3.metric("Productos categoria A", int((df_abc["categoria_abc"] == "A").sum()))

    st.subheader("Ventas historicas (ultimos 30 dias)")
    df_ventas_fecha = df_ventas.copy()
    df_ventas_fecha["fecha"] = pd.to_datetime(df_ventas_fecha["fecha"])
    df_reciente = df_ventas_fecha[
        df_ventas_fecha["fecha"] >= df_ventas_fecha["fecha"].max() - pd.Timedelta(days=30)
    ]
    ventas_por_dia = df_reciente.groupby("fecha")["unidades_vendidas"].sum().reset_index()

    fig = px.line(ventas_por_dia, x="fecha", y="unidades_vendidas", title="Unidades vendidas por dia")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------
# 7. SECCION: CLASIFICACION ABC
# --------------------------------------------------------------
elif seccion == "Clasificacion ABC":
    st.title("Clasificacion ABC de productos (K-Means)")
    st.write(
        "Los productos se agrupan con K-Means segun ingresos, unidades vendidas y "
        "frecuencia de venta. Categoria A = criticos, B = importantes, C = bajo impacto."
    )

    color_map = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e74c3c"}
    fig = px.bar(
        df_abc, x="producto_nombre", y="ingreso_total", color="categoria_abc",
        color_discrete_map=color_map, title="Ingresos por producto y categoria ABC",
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        df_abc[["producto_nombre", "ingreso_total", "frecuencia_venta", "porcentaje_acumulado", "categoria_abc"]],
        width="stretch",
    )

# --------------------------------------------------------------
# 8. SECCION: ALERTAS DE RE-STOCK
# --------------------------------------------------------------
elif seccion == "Alertas de Re-Stock":
    st.title("Alertas de Re-Stock (proyeccion de demanda)")
    st.write("Alerta generada con 3 a 5 dias de antelacion antes de que un producto se quede sin stock.")

    def resaltar_urgencia(fila):
        color = {
            "URGENTE": "background-color: #ffcccc",
            "Alerta": "background-color: #fff3cd",
            "Seguimiento": "background-color: #d4edda",
            "Sin riesgo": "",
        }.get(fila["urgencia"], "")
        return [color] * len(fila)

    st.dataframe(df_alertas.style.apply(resaltar_urgencia, axis=1), width="stretch")

# --------------------------------------------------------------
# 9. SECCION: DETECCION VISUAL (FOTO O CAMARA EN VIVO)
# --------------------------------------------------------------
elif seccion == "Deteccion visual (camara)":
    st.title("Deteccion de stock en estanterias (YOLOv8)")
    st.write(
        "Usa la camara para tomar una foto de un estante (o, para pruebas, de una caja "
        "con productos adentro) y el sistema te dira si esta lleno o casi vacio."
    )

    capacidad = st.number_input(
        "Capacidad maxima esperada (cuantos productos caben si esta 'lleno')",
        min_value=1, max_value=200, value=config.CAPACIDAD_MAXIMA_ESTANTE, step=1,
        help="Para probar con una caja chica en vez de un estante real, baja este numero (ej: 5 o 6).",
    )

    modo_captura = st.radio("¿Como le das la imagen?", ["Usar camara", "Subir foto"], horizontal=True)

    if modo_captura == "Usar camara":
        archivo = st.camera_input("Toma una foto del estante o la caja")
    else:
        archivo = st.file_uploader("Foto del estante", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        ruta_temp = os.path.join(config.RUTA_BASE, "temp_imagen.jpg")
        with open(ruta_temp, "wb") as f:
            f.write(archivo.getbuffer())

        try:
            from vision.deteccion_stock import analizar_estante
            with st.spinner("Analizando imagen con YOLOv8..."):
                resultado = analizar_estante(ruta_temp, capacidad_maxima=capacidad)

            st.image(resultado["imagen_anotada"], caption="Objetos detectados", channels="BGR")
            colA, colB = st.columns(2)
            colA.metric("Productos detectados", resultado["cantidad_detectada"])
            colB.metric("Nivel de llenado estimado", f"{resultado['porcentaje_llenado']}%")

            # Aviso grande y visible, ideal para la demo en vivo
            if resultado["porcentaje_llenado"] < 20:
                st.error(f"🔴 {resultado['estado']}")
            elif resultado["porcentaje_llenado"] < 50:
                st.warning(f"🟡 {resultado['estado']}")
            else:
                st.success(f"🟢 {resultado['estado']}")
        except ImportError:
            st.error("Falta instalar 'ultralytics'. Ejecuta: pip install ultralytics")
        finally:
            if os.path.exists(ruta_temp):
                os.remove(ruta_temp)

    

# --------------------------------------------------------------
# 10. SECCION: CHATBOT
# --------------------------------------------------------------
elif seccion == "Chatbot":
    st.title("Chatbot de inventario")
    st.write("Pregunta sobre el estado del stock, por ejemplo: '¿Que productos son urgentes?'")

    if "historial_chat" not in st.session_state:
        st.session_state.historial_chat = []

    for autor, mensaje in st.session_state.historial_chat:
        with st.chat_message(autor):
            st.write(mensaje)

    pregunta = st.chat_input("Escribe tu pregunta sobre el inventario...")

    if pregunta:
        st.session_state.historial_chat.append(("user", pregunta))
        with st.chat_message("user"):
            st.write(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = responder_pregunta(pregunta, df_alertas)
            st.write(respuesta)
        st.session_state.historial_chat.append(("assistant", respuesta))
