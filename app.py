import streamlit as st
import pandas as pd
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# CARGAR DATA
# =====================================================

df = pd.read_csv(
    "base_chatbot_universitario.csv",
    sep=";"
)

# =====================================================
# LIMPIEZA
# =====================================================

def limpiar_texto(texto):

    texto = texto.lower()

    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")

    texto = re.sub(r"[^a-zA-Z\s]", "", texto)

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


# =====================================================
# LIMPIAR PREGUNTAS
# =====================================================

df["pregunta_limpia"] = df["pregunta"].apply(limpiar_texto)


# =====================================================
# TF-IDF
# =====================================================

vectorizador = TfidfVectorizer()

matriz_tfidf = vectorizador.fit_transform(df["pregunta_limpia"])


# =====================================================
# FUNCIÓN CHATBOT
# =====================================================

def responder_chatbot(pregunta_usuario):

    pregunta_limpia = limpiar_texto(pregunta_usuario)

    # SALUDOS
    saludos = [
        "hola",
        "buenas",
        "buenos dias",
        "buenas tardes",
        "buenas noches"
    ]

    if pregunta_limpia in saludos:
        return "Hola, soy el chatbot universitario. ¿En qué puedo ayudarte?"


    # DESPEDIDAS
    despedidas = [
        "adios",
        "hasta luego",
        "bye"
    ]

    if pregunta_limpia in despedidas:
        return "Hasta luego. Que tengas un excelente día."


    # AGRADECIMIENTO
    agradecimientos = [
        "gracias",
        "muchas gracias"
    ]

    if pregunta_limpia in agradecimientos:
        return "Con gusto. Estoy para ayudarte."


    # TF-IDF
    vector_pregunta = vectorizador.transform([pregunta_limpia])

    similitudes = cosine_similarity(vector_pregunta, matriz_tfidf)

    indice = similitudes.argmax()

    similitud_maxima = similitudes[0, indice]

    if similitud_maxima < 0.20:
        return "Lo siento, no tengo información suficiente para responder esa consulta."

    respuesta = df.loc[indice, "respuesta"]

    return respuesta


# =====================================================
# STREAMLIT
# =====================================================

st.set_page_config(
    page_title="Chatbot Universitario",
    page_icon="🎓"
)

st.title("🎓 Chatbot Universitario")

st.write("Realiza una consulta académica.")

pregunta_usuario = st.text_input(
    "Escribe tu pregunta:"
)

if st.button("Consultar"):

    if pregunta_usuario.strip() == "":
        st.warning("Por favor ingresa una pregunta.")

    else:

        respuesta = responder_chatbot(pregunta_usuario)

        st.success(respuesta)
