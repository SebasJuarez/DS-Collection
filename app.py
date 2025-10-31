import os
import re
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PALETTE = ["#FFB200", "#EB5B00", "#D91656", "#640D5F"]
PRIMARY, SECONDARY, ACCENT, DARK = PALETTE

st.set_page_config(page_title="Desastres Naturales - Dashboard", layout="wide")

st.sidebar.title("Controles")

@st.cache_data(show_spinner=True)
def load_data(path: str):
    df = pd.read_csv(path)
    if "target" in df.columns:
        df["target"] = df["target"].astype(int)
    for c in ["keyword", "location", "text"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")
    if "text" in df.columns:
        df["text_len"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().apply(len)
    return df

default_paths = ["train.csv", "data.csv", "disasters.csv"]
df = None
for p in default_paths:
    if os.path.exists(p):
        df = load_data(p)
        break

if df is None:
    st.sidebar.info("Sube el dataset (CSV) con columnas 'text', 'keyword', 'location' y 'target'.")
    uploaded = st.sidebar.file_uploader("Cargar CSV", type=["csv"])
    if uploaded:
        df = load_data(uploaded)
if df is None:
    st.stop()

if "target" in df.columns:
    clase = st.sidebar.multiselect("Clase (target)", options=sorted(df["target"].dropna().unique().tolist()), default=sorted(df["target"].dropna().unique().tolist()), help="1=desastre, 0=no desastre")
else:
    clase = None

has_keyword_only = st.sidebar.checkbox("Solo filas con 'keyword'", value=False)
topn_kw = st.sidebar.slider("Top N keywords", 5, 40, 15, 1)
bins = st.sidebar.slider("Bins histograma longitud", 5, 100, 30)

STOPWORDS = set("a an the of and or in on at to for with from by about as into like through after over between out against during without before under around among i you he she it we they me him her them my your his its our their this that these those is are was were be been being have has had do does did not no nor only just rt http https amp via & : ; , . ! ? ( ) [ ] { } ' \" \\n \\t 0 1 2 3 4 5 6 7 8 9".split())
token_pattern = re.compile(r"[\\w']+")

def tokenize(text):
    tokens = [t.lower() for t in token_pattern.findall(text)]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

work = df.copy()
if has_keyword_only and "keyword" in work.columns:
    work = work[work["keyword"].str.len() > 0]
if clase is not None and "target" in work.columns:
    work = work[work["target"].isin(clase)]

st.markdown("# Dashboard Interactivo — Desastres Naturales")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1) Distribución de clases")
    if "target" in work.columns:
        counts = work["target"].value_counts().rename({0:"No desastre", 1:"Desastre"})
        fig = px.bar(counts, x=counts.index, y=counts.values, labels=dict(x="Clase", y="Cantidad"),
                     color=counts.index, color_discrete_sequence=[PRIMARY, ACCENT])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No existe columna 'target' en el dataset.")

with col2:
    st.subheader("2) Top palabras clave (keyword)")
    if "keyword" in work.columns and work["keyword"].str.len().sum() > 0:
        top_kw = work["keyword"].value_counts().head(topn_kw).sort_values(ascending=True)
        fig2 = px.bar(top_kw, x=top_kw.values, y=top_kw.index, orientation="h",
                      labels=dict(x="Frecuencia", y="keyword"),
                      color=top_kw.values, color_continuous_scale=[PRIMARY, SECONDARY, ACCENT, DARK])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No hay columna 'keyword' o está vacía.")

col3, col4 = st.columns(2)
with col3:
    st.subheader("3) Longitud del texto")
    if "text_len" in work.columns and "target" in work.columns:
        fig3 = px.histogram(work, x="text_len", nbins=bins, color="target",
                            color_discrete_map={0: PRIMARY, 1: ACCENT},
                            labels=dict(text_len="Caracteres", target="Clase"))
        st.plotly_chart(fig3, use_container_width=True)
    elif "text_len" in work.columns:
        fig3 = px.histogram(work, x="text_len", nbins=bins, labels=dict(text_len="Caracteres"))
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("4) Frecuencia de palabras")
    if "text" in work.columns:
        tokens = []
        sample = work["text"]
        if len(sample) > 4000:
            sample = sample.sample(4000, random_state=42)
        for t in sample:
            tokens.extend(tokenize(t))
        if tokens:
            s = pd.Series(tokens).value_counts().head(25).sort_values(ascending=True)
            fig4 = px.bar(s, x=s.values, y=s.index, orientation="h",
                          labels=dict(x="Frecuencia", y="Término"),
                          color=s.values, color_continuous_scale=[SECONDARY, ACCENT, DARK])
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No se encontraron tokens suficientes tras eliminar stopwords.")
    else:
        st.info("No hay columna 'text'.")

st.markdown("---")
st.header("Clasificador de texto (¿Desastre o no?)")

@st.cache_resource(show_spinner=True)
def load_model(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                pass
    return None

model = load_model(["tweet_classifier_with_neg.joblib", "model.joblib", "clf.joblib"])

colL, colR = st.columns([2,1])
with colL:
    user_text = st.text_area("Escribe un texto para clasificar:", height=120, placeholder="Ej.: 'Forest fire near Los Angeles, people evacuated'")
    run = st.button("Clasificar", type="primary")
    if run:
        if model is None:
            st.error("No se encontró el modelo (.joblib). Colócalo junto a app.py.")
        else:
            try:
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([user_text])[0]
                    pred = int(np.argmax(proba))
                else:
                    pred = int(model.predict([user_text])[0])
                label = "Desastre" if pred == 1 else "No desastre"
                st.success(f"Predicción: **{label}**")
                if proba is not None and len(proba) > 1:
                    st.metric("Confianza (clase 1)", f"{proba[1]*100:.2f}%")
                    bar = go.Figure(go.Bar(x=["No desastre (0)", "Desastre (1)"], y=proba,
                                           marker_color=[PRIMARY, ACCENT]))
                    bar.update_layout(yaxis_title="Probabilidad", xaxis_title="Clase")
                    st.plotly_chart(bar, use_container_width=True)
            except Exception as e:
                st.error(f"Error al correr el modelo: {e}")

with colR:
    st.info("""**Notas**
- La app intenta cargar `train.csv` y `tweet_classifier_with_neg.joblib` automáticamente.
- Si no están, sube el CSV desde la barra lateral.
- El clasificador devuelve la clase y si es posible, la probabilidad.
""")
