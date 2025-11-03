import os
import re
import math
import glob
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

PALETTE = ["#FFB200", "#EB5B00", "#D91656", "#640D5F"]
PRIMARY, SECONDARY, ACCENT, DARK = PALETTE

st.set_page_config(page_title="Desastres Naturales - Dashboard", layout="wide")

st.sidebar.title("Controles")

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    for c in ["keyword", "location", "text"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")
    df["text_len"] = df.get("text", pd.Series(dtype=str)).astype(str).str.len()
    df["word_count"] = df.get("text", pd.Series(dtype=str)).astype(str).str.split().apply(len)
    return df

clean_candidates = [
    "Data/train_clean.parquet",
    "Data/train_clean.csv",
    "artifacts/train_clean.parquet",
    "artifacts/train_clean.csv",
]
raw_candidates = [
    "Data/train.csv",
    "train.csv",
    "data.csv",
    "disasters.csv",
]

df = None
loaded_path = None
is_preclean = False

def _normalize_after_load(df0: pd.DataFrame) -> pd.DataFrame:
    # Ensure essential dtypes and derived columns
    if "target" in df0.columns:
        df0["target"] = pd.to_numeric(df0["target"], errors="coerce").fillna(0).astype(int)
    for c in ["keyword", "location", "text"]:
        if c in df0.columns:
            df0[c] = df0[c].astype(str).fillna("")
    if "text" in df0.columns and "text_len" not in df0.columns:
        df0["text_len"] = df0["text"].astype(str).str.len()
    if "text" in df0.columns and "word_count" not in df0.columns:
        df0["word_count"] = df0["text"].astype(str).str.split().apply(len)
    return df0

# Try cleaned first
for p in clean_candidates + raw_candidates:
    if os.path.exists(p):
        try:
            if p.lower().endswith(".parquet"):
                tmp = pd.read_parquet(p)
            else:
                tmp = pd.read_csv(p)
            df = _normalize_after_load(tmp)
            loaded_path = p
            is_preclean = ("clean_text" in df.columns) or ("train_clean" in os.path.basename(p))
            break
        except Exception:
            continue

if df is None:
    st.sidebar.info("Sube el dataset (CSV) con columnas 'text', 'keyword', 'location' y 'target'.")
    uploaded = st.sidebar.file_uploader("Cargar CSV", type=["csv"])
    if uploaded:
        df = load_data(uploaded)
if df is None:
    st.stop()

st.sidebar.header("Filtros y opciones")
if loaded_path:
    st.sidebar.caption(f"Fuente de datos: {loaded_path} — {'pre-limpia' if is_preclean else 'cruda'}")
if "target" in df.columns:
    clase = st.sidebar.multiselect(
        "Clase (target)",
        options=sorted(df["target"].dropna().unique().tolist()),
        default=sorted(df["target"].dropna().unique().tolist()),
        help="1=desastre, 0=no desastre",
    )
else:
    clase = None

has_keyword_only = st.sidebar.checkbox("Solo filas con 'keyword'", value=False)
topn_kw = st.sidebar.slider("Top N keywords", 5, 50, 20, 1)
bins = st.sidebar.slider("Bins histograma longitud", 5, 120, 40)

# Visualizaciones a mostrar
viz_options = {
    "dist_clases": "Distribución de clases",
    "top_keywords": "Top palabras clave (keyword)",
    "hist_len": "Histograma longitud del texto",
    "freq_terms": "Frecuencia de términos (texto)",
    "sentiment_box": "Sentimiento por clase (box)",
    "sentiment_hist": "Histograma de sentimiento",
    "sent_len_scatter": "Sentimiento vs longitud de texto",
}
st.sidebar.subheader("Visualizaciones")
viz_selected = st.sidebar.multiselect(
    "Selecciona qué ver",
    options=list(viz_options.keys()),
    default=list(viz_options.keys()),
    format_func=lambda k: viz_options[k],
)

# Controles de entrenamiento (velocidad y vectorización)
st.sidebar.subheader("Entrenamiento")
speed_mode = st.sidebar.selectbox(
    "Velocidad", ["Rápido", "Completo"], index=0,
    help="Rápido: muestrea y usa TF-IDF reducido. Completo: todos los datos y bi-gramas."
)
max_rows = int(len(df)) if df is not None else 0
default_sample = min(3000, max_rows) if max_rows else 3000
sample_size = st.sidebar.slider(
    "Muestras para entrenar", 500, max(1000, max_rows) if max_rows else 10000,
    default_sample, step=500,
    help="Número de filas para entrenamiento (aplica en modo Rápido)"
)
tfidf_max = st.sidebar.slider(
    "TF-IDF max_features", 1000, 30000,
    8000 if speed_mode == "Rápido" else 20000, step=1000
)
ngram_max = 1 if speed_mode == "Rápido" else 2

STOPWORDS = set(
    "a an the of and or in on at to for with from by about as into like through after over between out against during without before under around among i you he she it we they me him her them my your his its our their this that these those is are was were be been being have has had do does did not no nor only just rt http https amp via & : ; , . ! ? ( ) [ ] { } ' \" \\n+ \\t 0 1 2 3 4 5 6 7 8 9".split()
)
token_pattern = re.compile(r"[\w']+")

URL_PATTERN = r"http\S+|www\S+|https\S+"
EMO_POS = {":)", ";)", ":d", "xd", ":p", ":3"}
EMO_NEG = {":(", ":'(", ">:(", ":'c"}
NEGATION_WORDS = r"\b(no|not|never|n't|cannot|can't|won't|don't|didn't|isn't|aren't|ain't)\b"

def tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in token_pattern.findall(text or "")]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def clean_text_enriched(text: str) -> Tuple[str, List[str], List[str]]:
    text = "" if text is None else str(text)
    text = re.sub(URL_PATTERN, " ", text)
    hashtags = re.findall(r"#(\w+)", text, flags=re.IGNORECASE)
    low = text.lower()
    emoticons_found = []
    for emo in EMO_POS | EMO_NEG:
        if emo in low:
            emoticons_found.append(emo)
    excl_runs = re.findall(r"!+", text)
    ques_runs = re.findall(r"\?+", text)
    n_excl = min(3, sum(len(r) for r in excl_runs))
    n_ques = min(3, sum(len(r) for r in ques_runs))
    words = re.findall(r"[A-Za-z]+", text)
    has_allcaps = any(w.isupper() and len(w) >= 3 for w in words)
    punct_to_remove = "".join(ch for ch in re.escape("!?'\"#$%&()*+,-./:;<=>@[\\]^_`{|}~") if ch not in "!?'")
    text = re.sub(f"[{punct_to_remove}]", " ", text)
    text = re.sub(NEGATION_WORDS, r" \1 NEG ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b(?!\s*911)", " ", text)
    text = text.replace("911", "nine eleven")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = text.lower()
    if has_allcaps:
        text += " ALLCAPS"
    text += " " + " EXCL" * n_excl
    text += " " + " QUES" * n_ques
    if any(emo in low for emo in EMO_POS):
        text += " EMO_POS"
    if any(emo in low for emo in EMO_NEG):
        text += " EMO_NEG"
    text = re.sub(r"[@#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, hashtags, emoticons_found

@st.cache_data(show_spinner=True)
def enrich_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    df2 = df_in.copy()
    if "text" not in df2.columns:
        return df2
    # Clean text and derive tokens/hashtags
    triplets = df2["text"].apply(clean_text_enriched)
    df2["clean_text"], df2["hashtags"], df2["emoticons"] = zip(*triplets)
    # Sentiment
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        df2["sentiment_scores"] = df2["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])  # type: ignore
    except Exception:
        df2["sentiment_scores"] = 0.0
    df2["negativity"] = -df2["sentiment_scores"]
    # Derived counts
    df2["text_len"] = df2["text"].astype(str).str.len()
    df2["word_count"] = df2["text"].astype(str).str.split().apply(len)
    return df2

# Si ya viene limpio (clean_text+sentiment), evitamos limpiar de nuevo
if ("clean_text" in df.columns) and ("sentiment_scores" in df.columns) and ("negativity" in df.columns):
    work = df.copy()
else:
    work = enrich_dataframe(df)
if has_keyword_only and "keyword" in work.columns:
    work = work[work["keyword"].str.len() > 0]
if clase is not None and "target" in work.columns:
    work = work[work["target"].isin(clase)]
keyword_filter = None
if "keyword" in work.columns and work["keyword"].str.len().sum() > 0:
    kw_options = (
        work["keyword"].value_counts().head(100).index.tolist()
    )
    keyword_filter = st.sidebar.multiselect(
        "Filtrar por keyword específica (opcional)", options=kw_options, default=[]
    )
    if keyword_filter:
        work = work[work["keyword"].isin(keyword_filter)]

st.markdown("# Dashboard Interactivo — Desastres Naturales")
st.caption("Explora, entrena modelos y compara resultados con gráficos interactivos.")

col1, col2 = st.columns(2)
with col1:
    if "dist_clases" in viz_selected:
        st.subheader("1) Distribución de clases")
        if "target" in work.columns:
            counts = work["target"].value_counts().rename({0: "No desastre", 1: "Desastre"})
            fig = px.bar(
                counts,
                x=counts.index,
                y=counts.values,
                labels=dict(x="Clase", y="Cantidad"),
                color=counts.index,
                color_discrete_sequence=[PRIMARY, ACCENT],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No existe columna 'target' en el dataset.")

with col2:
    if "top_keywords" in viz_selected:
        st.subheader("2) Top palabras clave (keyword)")
        if "keyword" in work.columns and work["keyword"].str.len().sum() > 0:
            top_kw = work["keyword"].value_counts().head(topn_kw).sort_values(ascending=True)
            fig2 = px.bar(
                top_kw,
                x=top_kw.values,
                y=top_kw.index,
                orientation="h",
                labels=dict(x="Frecuencia", y="keyword"),
                color=top_kw.values,
                color_continuous_scale=[PRIMARY, SECONDARY, ACCENT, DARK],
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay columna 'keyword' o está vacía.")

col3, col4 = st.columns(2)
with col3:
    if "hist_len" in viz_selected:
        st.subheader("3) Longitud del texto")
        if "text_len" in work.columns and "target" in work.columns:
            fig3 = px.histogram(
                work,
                x="text_len",
                nbins=bins,
                color="target",
                color_discrete_map={0: PRIMARY, 1: ACCENT},
                labels=dict(text_len="Caracteres", target="Clase"),
            )
            st.plotly_chart(fig3, use_container_width=True)
        elif "text_len" in work.columns:
            fig3 = px.histogram(work, x="text_len", nbins=bins, labels=dict(text_len="Caracteres"))
            st.plotly_chart(fig3, use_container_width=True)

with col4:
    if "freq_terms" in viz_selected:
        st.subheader("4) Frecuencia de palabras")
        if "text" in work.columns:
            tokens = []
            sample = work["text"]
            if len(sample) > 6000:
                sample = sample.sample(6000, random_state=42)
            for t in sample:
                tokens.extend(tokenize(t))
            if tokens:
                s = pd.Series(tokens).value_counts().head(25).sort_values(ascending=True)
                fig4 = px.bar(
                    s,
                    x=s.values,
                    y=s.index,
                    orientation="h",
                    labels=dict(x="Frecuencia", y="Término"),
                    color=s.values,
                    color_continuous_scale=[SECONDARY, ACCENT, DARK],
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No se encontraron tokens suficientes tras eliminar stopwords.")
        else:
            st.info("No hay columna 'text'.")

st.markdown("---")
st.header("Modelos y clasificación")
st.caption("Entrena y compara al menos 3 modelos. Visualiza matrices de confusión y curvas ROC/PR.")

@st.cache_resource(show_spinner=True)
def train_models(df_in: pd.DataFrame, random_state: int = 42,
                 speed: str = "Rápido", sample_n: int = 3000,
                 tfidf_features: int = 8000, ngram_max: int = 1):
    if "clean_text" not in df_in.columns:
        df_in = enrich_dataframe(df_in)
    # Muestreo rápido
    if speed == "Rápido" and len(df_in) > sample_n:
        df_use = df_in.sample(sample_n, random_state=random_state)
    else:
        df_use = df_in

    # Split
    X = df_use["clean_text"].astype(str).values
    y = df_use["target"].astype(int).values if "target" in df_use.columns else np.zeros(len(df_use), dtype=int)
    X_train_txt, X_valid_txt, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y if len(np.unique(y)) > 1 else None, random_state=random_state
    )

    # Progreso
    info_box = st.empty()
    prog = st.progress(0, text="Preparando datos…")

    # TF-IDF único
    vec = TfidfVectorizer(max_features=tfidf_features, ngram_range=(1, ngram_max),
                          stop_words="english", sublinear_tf=True)
    info_box.write(f"TF-IDF: max_features={tfidf_features}, ngram=(1,{ngram_max}) — ajustando…")
    X_train = vec.fit_transform(X_train_txt)
    prog.progress(30, text="Vectorizador listo. Transformando validación…")
    X_valid = vec.transform(X_valid_txt)
    prog.progress(45, text="Datos vectorizados. Entrenando modelos…")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200 if speed=="Rápido" else 400, random_state=random_state),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(random_state=random_state, max_iter=2000 if speed=="Rápido" else 4000),
    }
    results = {}
    total = len(models)
    done = 0
    for name, clf in models.items():
        info_box.write(f"Entrenando {name}…")
        clf.fit(X_train, y_train)
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        y_pred = clf.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)
        # Scores for ROC/PR si aplica
        y_score = None
        try:
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_valid)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_valid)
        except Exception:
            y_score = None
        roc = roc_auc_score(y_valid, y_score) if y_score is not None and len(np.unique(y_valid)) > 1 else None
        cm = confusion_matrix(y_valid, y_pred, labels=[0, 1])
        results[name] = {
            "pipeline": pipe,
            "metrics": {"accuracy": acc, "f1": f1, "roc_auc": roc},
            "y_valid": y_valid,
            "y_pred": y_pred,
            "y_score": y_score,
            "cm": cm,
        }
        done += 1
        pct = 45 + int(50 * done / total)
        prog.progress(min(pct, 95), text=f"{name} entrenado. ({done}/{total})")
    prog.progress(100, text="Entrenamiento finalizado")
    info_box.write("Modelos listos ✔")
    return results

with st.spinner("Entrenando modelos…"):
    model_results = train_models(work, speed=speed_mode, sample_n=sample_size,
                                 tfidf_features=tfidf_max, ngram_max=ngram_max)

st.subheader("Comparativa de modelos")
models_available = list(model_results.keys())
models_to_show = st.multiselect(
    "Selecciona modelos para comparar", options=models_available, default=models_available
)
if models_to_show:
    rows = []
    for name in models_to_show:
        m = model_results[name]["metrics"]
        rows.append({"Modelo": name, "Accuracy": m["accuracy"], "F1": m["f1"], "ROC AUC": m["roc_auc"]})
    table = pd.DataFrame(rows).sort_values("F1", ascending=False)
    st.dataframe(table, use_container_width=True)

    # Gráfica comparativa
    fig_cmp = px.bar(
        table.melt(id_vars=["Modelo"], value_vars=["Accuracy", "F1", "ROC AUC"], var_name="Métrica", value_name="Valor"),
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT],
        text_auto=".2f",
    )
    fig_cmp.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_cmp, use_container_width=True)

st.subheader("Métricas detalladas y curvas")
cols = st.columns(len(models_to_show) if models_to_show else 1)
for i, name in enumerate(models_to_show):
    with cols[i % len(cols)]:
        st.markdown(f"**{name}**")
        res = model_results[name]
        cm = res["cm"]
        z = cm.astype(int)
        fig_cm = go.Figure(data=go.Heatmap(z=z, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], colorscale=[[0, "#f7fbff"], [1, ACCENT]]))
        fig_cm.update_layout(title="Matriz de confusión", xaxis_title="Predicción", yaxis_title="Real")
        st.plotly_chart(fig_cm, use_container_width=True)
        # ROC / PR
        y_valid, y_score = res["y_valid"], res["y_score"]
        if y_score is not None and len(np.unique(y_valid)) > 1:
            fpr, tpr, _ = roc_curve(y_valid, y_score)
            prec, rec, _ = precision_recall_curve(y_valid, y_score)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color=PRIMARY)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color="#999", dash="dash")))
            fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR", yaxis_range=[0, 1])
            st.plotly_chart(fig_roc, use_container_width=True)

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR", line=dict(color=SECONDARY)))
            fig_pr.update_layout(title="Curva Precisión-Recall", xaxis_title="Recall", yaxis_title="Precisión", yaxis_range=[0, 1])
            st.plotly_chart(fig_pr, use_container_width=True)

st.subheader("Clasificador interactivo")
colL, colR = st.columns([2, 1])
with colL:
    user_text = st.text_area(
        "Escribe un texto para clasificar:", height=120, placeholder="Ej.: 'Forest fire near Los Angeles, people evacuated'"
    )
    model_choice = st.selectbox("Modelo a usar", options=models_available, index=0)
    run = st.button("Clasificar", type="primary")
    if run:
        try:
            pipe = model_results[model_choice]["pipeline"]
            proba = None
            pred = int(pipe.predict([user_text])[0])
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                proba = pipe.predict_proba([user_text])[0]
            label = "Desastre" if pred == 1 else "No desastre"
            st.success(f"Predicción: **{label}**")
            if proba is not None and len(proba) > 1:
                st.metric("Confianza (clase 1)", f"{proba[1]*100:.2f}%")
                bar = go.Figure(
                    go.Bar(x=["No desastre (0)", "Desastre (1)"], y=proba, marker_color=[PRIMARY, ACCENT])
                )
                bar.update_layout(yaxis_title="Probabilidad", xaxis_title="Clase")
                st.plotly_chart(bar, use_container_width=True)
        except Exception as e:
            st.error(f"Error al correr el modelo: {e}")

with colR:
    st.info(
        """Consejos de uso
- Usa los filtros de la barra lateral para enlazar visualizaciones (clase y keyword).
- Cambia los modelos a comparar en la sección de 'Comparativa'.
- Haz zoom y hover en las gráficas para ver detalles (Plotly)."""
    )

# Sección de sentimiento y gráficos enlazados adicionales
st.markdown("---")
st.header("Exploración de sentimiento y relaciones")
colS1, colS2, colS3 = st.columns(3)
with colS1:
    if "sentiment_box" in viz_selected and "sentiment_scores" in work.columns and "target" in work.columns:
        st.subheader("5) Sentimiento por clase")
        fig5 = px.box(
            work,
            x="target",
            y="sentiment_scores",
            color="target",
            color_discrete_map={0: PRIMARY, 1: ACCENT},
            labels=dict(target="Clase", sentiment_scores="Sentimiento (VADER)"),
        )
        st.plotly_chart(fig5, use_container_width=True)
with colS2:
    if "sentiment_hist" in viz_selected and "sentiment_scores" in work.columns:
        st.subheader("6) Histograma de sentimiento")
        fig6 = px.histogram(
            work,
            x="sentiment_scores",
            nbins=40,
            color="target" if "target" in work.columns else None,
            color_discrete_map={0: PRIMARY, 1: ACCENT},
            labels=dict(sentiment_scores="Sentimiento"),
        )
        st.plotly_chart(fig6, use_container_width=True)
with colS3:
    if "sent_len_scatter" in viz_selected and "sentiment_scores" in work.columns:
        st.subheader("7) Sentimiento vs longitud")
        fig7 = px.scatter(
            work.sample(min(len(work), 3000), random_state=42),
            x="text_len",
            y="sentiment_scores",
            color="target" if "target" in work.columns else None,
            color_discrete_map={0: PRIMARY, 1: ACCENT},
            opacity=0.6,
            labels=dict(text_len="Longitud", sentiment_scores="Sentimiento"),
        )
        st.plotly_chart(fig7, use_container_width=True)

# 8) Visualización enlazada adicional: Top hashtags
st.subheader("8) Top Hashtags")
if "hashtags" in work.columns:
    try:
        # Expand hashtags list into rows
        series = work["hashtags"].explode().dropna().astype(str)
        top_hashtags = series[series.str.len() > 0].value_counts().head(25).sort_values(ascending=True)
        fig8 = px.bar(
            top_hashtags,
            x=top_hashtags.values,
            y=top_hashtags.index,
            orientation="h",
            labels=dict(x="Frecuencia", y="Hashtag"),
            color=top_hashtags.values,
            color_continuous_scale=[PRIMARY, SECONDARY, ACCENT, DARK],
        )
        st.plotly_chart(fig8, use_container_width=True)
    except Exception:
        st.info("No se pudieron extraer hashtags del texto limpio.")
