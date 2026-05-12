import streamlit as st
import pandas as pd
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# CONFIGURACIÓN DE PÁGINA
# =====================================================

st.set_page_config(
    page_title="Chatbot Universitario",
    page_icon="🎓",
    layout="centered"
)


# =====================================================
# CARGAR DATA
# =====================================================

df = pd.read_csv(
    "base_chatbot_universitario.csv",
    sep=";",
    encoding="latin1"
)


# =====================================================
# FUNCIÓN DE LIMPIEZA DE TEXTO
# =====================================================

def limpiar_texto(texto):

    texto = str(texto).lower()

    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")

    texto = re.sub(r"[^a-zA-Z\s]", "", texto)

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


# =====================================================
# PREPARACIÓN DE DATA
# =====================================================

df["pregunta_limpia"] = df["pregunta"].apply(limpiar_texto)


# =====================================================
# TF-IDF
# =====================================================

vectorizador = TfidfVectorizer()
matriz_tfidf = vectorizador.fit_transform(df["pregunta_limpia"])


# =====================================================
# FUNCIÓN DEL CHATBOT
# =====================================================

def responder_chatbot(pregunta_usuario):

    pregunta_limpia = limpiar_texto(pregunta_usuario)

    saludos = [
        "hola",
        "buenas",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
        "holi"
    ]

    if pregunta_limpia in saludos:
        return "Hola, soy el chatbot universitario. ¿En qué puedo ayudarte?"

    despedidas = [
        "adios",
        "hasta luego",
        "nos vemos",
        "bye",
        "chao"
    ]

    if pregunta_limpia in despedidas:
        return "Hasta luego. Que tengas un excelente día."

    agradecimientos = [
        "gracias",
        "muchas gracias",
        "te agradezco"
    ]

    if pregunta_limpia in agradecimientos:
        return "Con gusto. Estoy para ayudarte."

    vector_pregunta = vectorizador.transform([pregunta_limpia])

    similitudes = cosine_similarity(vector_pregunta, matriz_tfidf)

    indice = similitudes.argmax()

    similitud_maxima = similitudes[0, indice]

    if similitud_maxima < 0.20:
        return "Lo siento, no tengo información suficiente para responder esa consulta."

    respuesta = df.loc[indice, "respuesta"]

    return respuesta


# =====================================================
# INTERFAZ STREAMLIT TIPO CHAT
# =====================================================

st.title("🎓 Chatbot Universitario")

st.write("Realiza una consulta académica. El chatbot responderá automáticamente según la base de preguntas frecuentes.")

# Crear historial de conversación
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mensaje inicial del chatbot
if len(st.session_state.mensajes) == 0:
    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": "Hola, soy el chatbot universitario. Puedes hacerme consultas sobre matrícula, pagos, horarios, trámites, cursos, becas y servicios académicos."
    })

# Mostrar historial
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["contenido"])

# Entrada del usuario
pregunta_usuario = st.chat_input("Escribe tu pregunta aquí...")

if pregunta_usuario:

    st.session_state.mensajes.append({
        "rol": "user",
        "contenido": pregunta_usuario
    })

    respuesta = responder_chatbot(pregunta_usuario)

    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta
    })

    st.rerun()
