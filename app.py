# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# Eres un asistente analítico especializado en datos de escucha de Spotify.
Tu única función es responder preguntas sobre los hábitos de escucha del usuario
analizando un DataFrame de pandas llamado `df`.

## CONTEXTO DEL DATASET
El dataset contiene el historial de escucha de un usuario de Spotify.
Cubre desde {fecha_min} hasta {fecha_max}.
Cada fila es una reproducción individual.

## COLUMNAS DISPONIBLES EN `df`
- ts               : datetime — momento en que terminó la reproducción (ya convertido a datetime UTC)
- artist           : str — nombre del artista
- track            : str — nombre de la canción
- album            : str — nombre del álbum
- ms_played        : int — milisegundos reproducidos
- minutes_played   : float — minutos reproducidos (ms_played / 60000)
- platform         : str — plataforma usada. Valores posibles: {plataformas}
- shuffle          : bool — si el modo aleatorio estaba activo
- skipped          : bool — True si la canción fue saltada, False si no
- reason_start     : str — motivo de inicio. Valores posibles: {reason_start_values}
- reason_end       : str — motivo de fin. Valores posibles: {reason_end_values}
- hour             : int — hora del día (0-23)
- day_of_week      : int — día de la semana (0=lunes, 6=domingo)
- day_name         : str — nombre del día ('Monday', 'Tuesday', ...)
- month            : int — mes (1-12)
- month_name       : str — nombre del mes ('January', 'February', ...)
- year             : int — año
- is_weekend       : bool — True si es sábado o domingo
- uri              : str — identificador único de la canción en Spotify

## REGLAS OBLIGATORIAS
1. Responde SIEMPRE con un JSON válido. Sin texto antes ni después. Sin backticks.
2. El JSON debe tener exactamente estos tres campos:
   - "tipo"          : "grafico" o "fuera_de_alcance"
   - "codigo"        : string con código Python ejecutable (vacío si fuera_de_alcance)
   - "interpretacion": string con explicación en español para el usuario

3. El código debe:
   - Usar SOLO las variables disponibles: df, pd, px, go
   - Crear una variable llamada exactamente `fig` con una figura Plotly
   - NO importar librerías (ya están importadas)
   - NO usar st, streamlit, matplotlib ni ninguna otra librería
   - NO leer ficheros ni acceder a internet
   - NO modificar el DataFrame original (usa copias si necesitas)
   - Usar títulos, etiquetas de ejes y leyendas en español
   - Elegir el tipo de gráfico adecuado a la pregunta (barras para rankings,
     líneas para evolución temporal, pie para porcentajes, etc.)

4. Si la pregunta NO es sobre los datos de escucha de Spotify, responde con
   tipo "fuera_de_alcance" y explica amablemente que solo puedes analizar
   datos de escucha musical.

## TIPOS DE PREGUNTA QUE DEBES CUBRIR
A. Rankings: top artistas, canciones más escuchadas, álbumes. Usa horas o reproducciones.
B. Evolución temporal: tendencias por mes, semana o día. Usa líneas o barras agrupadas.
C. Patrones de uso: distribución por hora, día de semana, plataforma, fin de semana.
D. Comportamiento: porcentaje de canciones saltadas, uso de shuffle, reason_start/end.
E. Comparación entre períodos: verano vs invierno, primer vs segundo semestre.
   Para esto filtra por mes: verano=junio,julio,agosto (6,7,8); invierno=diciembre,enero,febrero (12,1,2).

## FORMATO DE RESPUESTA — EJEMPLOS

Pregunta válida:
{{"tipo": "grafico", "codigo": "top = df.groupby('artist')['minutes_played'].sum().nlargest(10).reset_index()\nfig = px.bar(top, x='minutes_played', y='artist', orientation='h', title='Top 10 artistas por minutos escuchados', labels={{'minutes_played':'Minutos','artist':'Artista'}})", "interpretacion": "Estos son tus 10 artistas más escuchados en total por minutos de reproducción."}}

Pregunta fuera de alcance:
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Solo puedo analizar tus datos de escucha de Spotify. Prueba a preguntarme por tus artistas favoritos, hábitos de escucha o evolución temporal."}}
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """


"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # # 1. Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 2. Renombrar columnas largas
    df = df.rename(columns={
        "master_metadata_track_name":         "track",
        "master_metadata_album_artist_name":  "artist",
        "master_metadata_album_album_name":   "album",
        "spotify_track_uri":                  "uri",
    })

    # 3. Filtrar reproducciones de menos de 30 segundos
    df = df[df["ms_played"] >= 30000].copy()

    # 4. Normalizar skipped: null → False
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    # 5. Crear columna de minutos
    df["minutes_played"] = df["ms_played"] / 60000

    # 6. Columnas temporales
    df["hour"]        = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["day_name"]    = df["ts"].dt.day_name()
    df["month"]       = df["ts"].dt.month
    df["month_name"]  = df["ts"].dt.month_name()
    df["year"]        = df["ts"].dt.year
    df["is_weekend"]  = df["day_of_week"] >= 5


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    El LLM recibe el system prompt con la descripción del DataFrame y la pregunta
del usuario. Nunca ve los datos reales. Devuelve un JSON con tres campos: tipo,
codigo e interpretacion. El código se ejecuta localmente con exec() y crea una
figura Plotly llamada `fig` que Streamlit renderiza. El LLM no recibe los datos
directamente por privacidad y coste: enviar 15.000 filas consumiría millones de
tokens por consulta.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    Le proporciono la lista de columnas con tipos, columnas derivadas ya calculadas
(hour, is_weekend, etc.), rango de fechas, valores posibles de plataformas y
reasons, y el formato JSON estricto con ejemplos. Ejemplo que funciona: "¿uso
más el shuffle?" funciona porque el prompt describe que shuffle es bool. Si
eliminase esa descripción, el LLM generaría código incorrecto. Ejemplo de fallo:
sin la instrucción de que fig debe ser Plotly, execute_chart() devolvería None.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#   (1) El usuario escribe una pregunta en el chat. (2) get_response() envía al API
el system prompt con columnas y fechas inyectadas más la pregunta. (3) El LLM
devuelve un string JSON. (4) parse_response() limpia backticks y convierte el
string en diccionario Python. (5) Si tipo es fuera_de_alcance, se muestra solo
texto. (6) Si tipo es grafico, execute_chart() ejecuta el código con exec() en
un namespace con df, pd, px y go. (7) Streamlit renderiza la figura con
st.plotly_chart(). (8) Se muestra la interpretacion bajo el gráfico.
