# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/marialberro/bc5-spotify

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
MODEL = "gpt-4.1-mini"

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
Eres un asistente analítico especializado en datos de escucha de Spotify.
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
{{"tipo": "grafico", "codigo": "top = df.groupby('artist')['minutes_played'].sum().nlargest(10).reset_index()\\nfig = px.bar(top, x='minutes_played', y='artist', orientation='h', title='Top 10 artistas por minutos escuchados', labels={{'minutes_played':'Minutos','artist':'Artista'}})", "interpretacion": "Estos son tus 10 artistas más escuchados en total por minutos de reproducción."}}

Pregunta fuera de alcance:
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Solo puedo analizar tus datos de escucha de Spotify. Prueba a preguntarme por tus artistas favoritos, hábitos de escucha o evolución temporal."}}
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # 1. Convertir timestamp a datetime
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

    return df


def build_prompt(df):
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
def get_response(user_msg, system_prompt):
    import openai
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    response = openai.chat.completions.create(
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
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.set_page_config(page_title="Spotify Analytics", layout="wide")

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

st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

df = load_data()
system_prompt = build_prompt(df)

if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                raw = get_response(prompt, system_prompt)
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    st.write(parsed["interpretacion"])
                else:
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
# 1. ARQUITECTURA TEXT-TO-CODE
#    El LLM recibe el system prompt con la descripción del DataFrame y la pregunta
#    del usuario. Nunca ve los datos reales. Devuelve un JSON con tres campos: tipo,
#    codigo e interpretacion. El código se ejecuta localmente con exec() y crea una
#    figura Plotly llamada `fig` que Streamlit renderiza. El LLM no recibe los datos
#    directamente por privacidad y coste: enviar 15.000 filas consumiría millones de
#    tokens por consulta.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    Le proporciono la lista de columnas con tipos, columnas derivadas ya calculadas
#    (hour, is_weekend, etc.), rango de fechas, valores posibles de plataformas y
#    reasons, y el formato JSON estricto con ejemplos. Ejemplo que funciona: "¿uso
#    más el shuffle?" funciona porque el prompt describe que shuffle es bool. Si
#    eliminase esa descripción, el LLM generaría código incorrecto. Ejemplo de fallo:
#    sin la instrucción de que fig debe ser Plotly, execute_chart() devolvería None.
#
# 3. EL FLUJO COMPLETO
#    (1) El usuario escribe una pregunta en el chat. (2) get_response() envía al API
#    el system prompt con columnas y fechas inyectadas más la pregunta. (3) El LLM
#    devuelve un string JSON. (4) parse_response() limpia backticks y convierte el
#    string en diccionario Python. (5) Si tipo es fuera_de_alcance, se muestra solo
#    texto. (6) Si tipo es grafico, execute_chart() ejecuta el código con exec() en
#    un namespace con df, pd, px y go. (7) Streamlit renderiza la figura con
#    st.plotly_chart(). (8) Se muestra la interpretacion bajo el gráfico.
