# 📦 Sistema de Re-Stock Automatizado con IA (Prototipo)

Prototipo rápido para un negocio en línea con re-stock automatizado por IA.
Combina visión artificial (YOLOv8), analítica predictiva (clustering ABC +
regresión con scikit-learn) y un chatbot con LLM (OpenAI/Gemini vía LangChain),
todo mostrado en un dashboard interactivo con Streamlit.

## Estructura del proyecto

```
restock_ai/
├── config.py                     # rutas y parámetros centrales
├── requirements.txt               # dependencias
├── .env.example                   # plantilla para tus API keys
├── datos/
│   ├── generar_datos_prueba.py    # genera ventas históricas simuladas
│   └── ventas_historicas.csv      # (se crea automáticamente)
├── analitica/
│   ├── clustering_abc.py          # clasificación ABC con K-Means
│   └── prediccion_demanda.py      # regresión (Random Forest) + alertas
├── vision/
│   └── deteccion_stock.py         # detección de stock con YOLOv8
├── chatbot/
│   └── asistente_llm.py           # chatbot con LangChain (OpenAI/Gemini)
├── app/
│   └── dashboard.py                # ⭐ app principal de Streamlit
├── entrenamiento_colab/
│   └── entrenar_yolo.ipynb        # notebook para entrenar YOLO en Google Colab
└── modelos/
    └── best.pt                    # (aquí va tu modelo YOLO ya entrenado)
```

## 1. Instalación (VS Code / Spyder / terminal)

```bash
# 1. Crea un entorno virtual (recomendado)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 2. Instala las dependencias
pip install -r requirements.txt
```

> Si solo quieres probar rápido el dashboard sin la parte de visión (YOLO),
> puedes omitir `ultralytics` y `opencv-python-headless` del `requirements.txt`;
> esa sección mostrará un aviso pero el resto de la app funciona igual.

## 2. Generar datos de ejemplo (opcional, se hace automático)

```bash
python datos/generar_datos_prueba.py
```

## 3. Ejecutar el dashboard localmente

```bash
streamlit run app/dashboard.py
```

Se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## 4. Configurar el chatbot (opcional)

Copia `.env.example` a `.env` y coloca tu clave de OpenAI **o** de Gemini:

```
OPENAI_API_KEY=tu_clave_aqui
```

Si no configuras ninguna clave, el chatbot sigue funcionando en **modo demo**
(respuestas basadas en reglas), para que siempre puedas mostrar el prototipo
completo sin depender de un servicio externo de pago.

## 5. Entrenar tu propio modelo YOLOv8 (sin instalar nada, en Google Colab)

1. Ve a https://colab.research.google.com
2. `Archivo > Subir cuaderno` y selecciona `entrenamiento_colab/entrenar_yolo.ipynb`
3. Sigue las celdas en orden (instalar, preparar dataset, entrenar, descargar)
4. Descarga el archivo `best.pt` generado y colócalo en la carpeta `modelos/`
   de este proyecto. `vision/deteccion_stock.py` lo detectará automáticamente.

Mientras no tengas un modelo propio, el sistema usa el modelo genérico
`yolov8n.pt` en "modo demo" solo para que puedas probar el flujo completo.

## 6. Compartir el proyecto sin que nadie tenga que descargar nada (gratis)

La forma más simple de dar acceso a este dashboard con un link, sin que la
otra persona instale Python ni nada, es **Streamlit Community Cloud**:

1. Crea una cuenta gratuita en https://github.com y sube esta carpeta como
   un repositorio (puedes arrastrar los archivos desde la web de GitHub,
   no necesitas usar la terminal).
2. Ve a https://share.streamlit.io e inicia sesión con tu cuenta de GitHub.
3. Elige "New app", selecciona el repositorio y en "Main file path" escribe:
   `app/dashboard.py`
4. Streamlit Cloud instala automáticamente lo que está en `requirements.txt`
   y te entrega una URL pública (algo como
   `https://tu-usuario-restock-ai.streamlit.app`) que cualquiera puede abrir
   desde el navegador, sin descargar ni instalar nada.

> Nota: `ultralytics`/`torch` son librerías pesadas y pueden tardar varios
> minutos en instalarse la primera vez en Streamlit Cloud. Si solo quieres
> una demo veloz de la parte analítica y el chatbot, puedes comentar esas
> líneas de `requirements.txt` para un despliegue más rápido.

Para el notebook de entrenamiento YOLO, el link a compartir es simplemente
el propio archivo `.ipynb` subido a tu Google Drive (clic derecho > Abrir con
> Google Colaboratory) — la otra persona solo necesita un navegador.

## Decisión técnica: ¿por qué no Java?

Se descarta Java por su mayor complejidad y desarrollo más lento para IA,
y por tener un ecosistema y soporte comunitario menores que Python en
librerías de analítica y machine learning (scikit-learn, ultralytics,
streamlit, langchain no tienen equivalentes tan maduros en Java).
