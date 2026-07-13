"""
====================================================================
DASHBOARD PRINCIPAL - SISTEMA DE RE-STOCK AUTOMATIZADO CON IA
====================================================================
Punto de entrada de la aplicacion. Para ejecutarlo, desde la carpeta
raiz del proyecto corre en la terminal:

    streamlit run app/dashboard.py

Este dashboard integra los 3 pilares del proyecto:
  1. Analisis predictivo   -> clustering ABC + prediccion de demanda
  2. Vision artificial      -> deteccion de stock en fotos de estantes
  3. Chatbot interactivo    -> consultas en lenguaje natural
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


# --------------------------------------------------------------
# 2. CARGA / GENERACION DE DATOS
#    (con cache para no recalcular en cada clic del usuario)
# --------------------------------------------------------------
@st.cache_data
def cargar_datos():
    if not os.path.exists(config.RUTA_DATOS_VENTAS):
        # Si todavia no existen datos de ventas, se generan datos de
        # ejemplo automaticamente para que el prototipo funcione al instante
        from datos.generar_datos_prueba import guardar_datos
        guardar_datos(config.RUTA_DATOS_VENTAS)
    return pd.read_csv(config.RUTA_DATOS_VENTAS)


@st.cache_data
def procesar_analisis(df_ventas):
    df_abc = clasificar_abc(df_ventas)
    df_alertas = generar_alertas_restock(df_ventas, df_abc)
    return df_abc, df_alertas


df_ventas = cargar_datos()
df_abc, df_alertas = procesar_analisis(df_ventas)

# --------------------------------------------------------------
# 3. BARRA LATERAL DE NAVEGACION
# --------------------------------------------------------------
st.sidebar.title("📦 Re-Stock IA")
seccion = st.sidebar.radio(
    "Ir a:",
    ["Resumen general", "Clasificacion ABC", "Alertas de Re-Stock", "Deteccion visual (YOLO)", "Chatbot"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Prototipo - Fase de producto de prueba")

# --------------------------------------------------------------
# 4. SECCION: RESUMEN GENERAL
# --------------------------------------------------------------
if seccion == "Resumen general":
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
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------
# 5. SECCION: CLASIFICACION ABC
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
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_abc[["producto_nombre", "ingreso_total", "frecuencia_venta", "porcentaje_acumulado", "categoria_abc"]],
        use_container_width=True,
    )

# --------------------------------------------------------------
# 6. SECCION: ALERTAS DE RE-STOCK
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

    st.dataframe(df_alertas.style.apply(resaltar_urgencia, axis=1), use_container_width=True)

# --------------------------------------------------------------
# 7. SECCION: DETECCION VISUAL CON YOLO
# --------------------------------------------------------------
elif seccion == "Deteccion visual (YOLO)":
    st.title("Deteccion de stock en estanterias (YOLOv8)")
    st.write("Sube una foto de un estante para detectar cuantos productos hay y estimar su nivel de llenado.")

    archivo = st.file_uploader("Foto del estante", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        ruta_temp = os.path.join(config.RUTA_BASE, "temp_imagen.jpg")
        with open(ruta_temp, "wb") as f:
            f.write(archivo.getbuffer())

        try:
            from vision.deteccion_stock import analizar_estante
            with st.spinner("Analizando imagen con YOLOv8..."):
                resultado = analizar_estante(ruta_temp)

            st.image(resultado["imagen_anotada"], caption="Detecciones", channels="BGR")
            colA, colB = st.columns(2)
            colA.metric("Productos detectados", resultado["cantidad_detectada"])
            colB.metric("Nivel de llenado estimado", f"{resultado['porcentaje_llenado']}%")
            st.info(resultado["estado"])
        except ImportError:
            st.error("Falta instalar 'ultralytics'. Ejecuta: pip install ultralytics")
        finally:
            if os.path.exists(ruta_temp):
                os.remove(ruta_temp)

# --------------------------------------------------------------
# 8. SECCION: CHATBOT
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
