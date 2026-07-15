"""
====================================================================
CHATBOT DE RE-STOCK (LangChain + OpenAI / Gemini)
====================================================================
Responde preguntas sobre el estado del stock usando como contexto
las alertas generadas por el modulo de prediccion de demanda.

Funciona con OpenAI o con Gemini segun la clave de API que este
disponible en las variables de entorno (ver archivo .env.example).
Si no hay ninguna clave configurada, funciona en "MODO DEMO" con
respuestas basadas en reglas simples, para que el prototipo siempre
se pueda mostrar sin depender de ningun servicio externo de pago.
====================================================================
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Carga variables desde un archivo .env si existe (opcional, no falla si no esta instalado)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")


def _construir_contexto(df_alertas: pd.DataFrame) -> str:
    """Convierte la tabla de alertas en texto plano, para dárselo
    como contexto al modelo de lenguaje."""
    lineas = []
    for _, fila in df_alertas.iterrows():
        lineas.append(
            f"- {fila['producto_nombre']} (Categoria {fila['categoria_abc']}): "
            f"stock actual {fila['stock_actual']}, "
            f"quiebre estimado en {fila['dias_hasta_quiebre']} dias "
            f"[{fila['urgencia']}]"
        )
    return "\n".join(lineas)


def responder_pregunta(pregunta: str, df_alertas: pd.DataFrame) -> str:
    """Punto de entrada principal del chatbot. Usa OpenAI o Gemini
    (via LangChain) si hay una API key configurada; si no, responde
    en modo demo con reglas simples. Si el modelo real falla por
    cualquier motivo (limite de cuota diario, sin internet, clave
    invalida, etc.), cae automaticamente al modo demo en vez de
    romper la aplicacion -- asi el chatbot siempre responde algo."""
    contexto = _construir_contexto(df_alertas)

    try:
        if OPENAI_KEY:
            return _responder_con_openai(pregunta, contexto)
        elif GEMINI_KEY:
            return _responder_con_gemini(pregunta, contexto)
    except Exception:
        pass  # el modelo real fallo -> seguimos abajo con el modo demo

    return _responder_modo_demo(pregunta, df_alertas)


def _responder_con_openai(pregunta: str, contexto: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    modelo = ChatOpenAI(model=config.MODELO_OPENAI, temperature=0.3)
    mensajes = [
        SystemMessage(content=(
            "Eres un asistente de inventario para una tienda. Responde de forma breve "
            "y concreta usando UNICAMENTE la siguiente informacion de stock:\n" + contexto
        )),
        HumanMessage(content=pregunta),
    ]
    return _extraer_texto(modelo.invoke(mensajes).content)


def _extraer_texto(contenido) -> str:
    """Los modelos mas nuevos (Gemini 3.x y similares) a veces devuelven el
    contenido como una lista de bloques (texto, firma de razonamiento, etc.)
    en vez de un simple string. Aqui nos quedamos solo con el texto."""
    if isinstance(contenido, str):
        return contenido

    if isinstance(contenido, list):
        partes = []
        for bloque in contenido:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(bloque.get("text", ""))
        return "".join(partes) if partes else str(contenido)

    return str(contenido)


def _responder_con_gemini(pregunta: str, contexto: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    modelo = ChatGoogleGenerativeAI(model=config.MODELO_GEMINI, temperature=0.3)
    mensajes = [
        SystemMessage(content=(
            "Eres un asistente de inventario para una tienda. Responde de forma breve "
            "y concreta usando UNICAMENTE la siguiente informacion de stock:\n" + contexto
        )),
        HumanMessage(content=pregunta),
    ]
    return _extraer_texto(modelo.invoke(mensajes).content)


def _responder_modo_demo(pregunta: str, df_alertas: pd.DataFrame) -> str:
    """Respuestas simples basadas en palabras clave, sin necesitar
    ninguna API key. Sirve para probar el dashboard completo sin
    configurar ningun servicio externo."""
    pregunta_lower = pregunta.lower()
    urgentes = df_alertas[df_alertas["urgencia"] == "URGENTE"]

    if "urgente" in pregunta_lower or "critico" in pregunta_lower:
        if urgentes.empty:
            return "No hay productos en estado urgente en este momento."
        nombres = ", ".join(urgentes["producto_nombre"].tolist())
        return f"Productos en estado URGENTE: {nombres}."

    if "categoria a" in pregunta_lower or "producto a" in pregunta_lower:
        productos_a = df_alertas[df_alertas["categoria_abc"] == "A"]["producto_nombre"].tolist()
        return f"Productos categoria A (los mas importantes): {', '.join(productos_a)}."

    if "resumen" in pregunta_lower or "estado general" in pregunta_lower:
        total = len(df_alertas)
        n_urgentes = len(urgentes)
        return (
            f"Hay {total} productos monitoreados. {n_urgentes} requieren re-stock urgente "
            f"(3 dias o menos). Configura OPENAI_API_KEY o GOOGLE_API_KEY para respuestas mas naturales."
        )

    return (
        "[Modo demo sin API key] Prueba preguntando por 'productos urgentes', "
        "'productos categoria A' o pide un 'resumen'. Para respuestas mas inteligentes, "
        "configura una clave de OpenAI o Gemini en el archivo .env"
    )
