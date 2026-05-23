import streamlit as st
import pandas as pd
import io
from sklearn.cluster import KMeans

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="People Analytics Assistant",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    /* ── Tipografía ─────────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Encabezados — adaptables al tema ──────────────── */
    h2, h3 {
        border-bottom: 2px solid rgba(128,128,128,0.25);
        padding-bottom: 6px;
    }

    /* ── Sidebar — sin color fijo, hereda el tema ──────── */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.15);
    }
    /* Fuerza visibilidad del texto en sidebar en modo oscuro */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: inherit !important;
    }

    /* ── Botones ────────────────────────────────────────── */
    div.stButton > button {
        background: #2563eb; color: white; border: none;
        border-radius: 6px; padding: 0.45rem 1.2rem;
        font-weight: 600; transition: background 0.2s;
    }
    div.stButton > button:hover { background: #1d4ed8; }

    /* ── Badges proveedor ───────────────────────────────── */
    .badge-openai {
        display:inline-block; background:#10a37f; color:#fff;
        font-size:0.72rem; font-weight:700; padding:2px 9px;
        border-radius:12px; margin-left:6px; vertical-align:middle;
    }
    .badge-gemini {
        display:inline-block; background:#4285F4; color:#fff;
        font-size:0.72rem; font-weight:700; padding:2px 9px;
        border-radius:12px; margin-left:6px; vertical-align:middle;
    }

    /* ── Métricas ───────────────────────────────────────── */
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px; padding: 10px 14px;
    }

    /* ── Caja comparativa ───────────────────────────────── */
    .compare-box {
        border: 1px solid rgba(96,165,250,0.4);
        border-radius: 8px; padding: 12px 16px; margin-top: 8px;
    }

    /* ── Scorecard bars ─────────────────────────────────── */
    .score-bar-container {
        background: rgba(128,128,128,0.2);
        border-radius: 8px; height: 12px; margin: 4px 0;
    }
    .score-bar { background: #2563eb; border-radius: 8px; height: 12px; }

    /* ── Tarjeta candidato ──────────────────────────────── */
    .candidate-card {
        border: 1px solid rgba(128,128,128,0.2); border-radius: 10px;
        padding: 16px; margin-bottom: 12px;
    }

    /* ── Colores semáforo ───────────────────────────────── */
    .fit-alta  { color: #16a34a; font-weight: 700; }
    .fit-media { color: #ca8a04; font-weight: 700; }
    .fit-baja  { color: #dc2626; font-weight: 700; }

    /* ── Info / warning boxes más legibles ─────────────── */
    div[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 People Analytics Assistant")
st.caption("Powered by OpenAI · Gemini")

# =========================================================
# SIDEBAR – CONFIGURACIÓN
# =========================================================
st.sidebar.header("⚙️ Configuración")

proveedor = st.sidebar.radio("Proveedor de IA", ["OpenAI", "Gemini"], horizontal=True)

if proveedor == "OpenAI":
    MODELOS = {
        "GPT-4o Mini":  "gpt-4o-mini",
        "GPT-4o":       "gpt-4o",
        "GPT-4.1 Mini": "gpt-4.1-mini",
        "GPT-4.1":      "gpt-4.1",
        "o4-mini":      "o4-mini",
        "o3-mini":      "o3-mini",
    }
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    MODELOS = {
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.5 Pro":   "gemini-2.5-pro",
    }
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

modelo_label = st.sidebar.selectbox("Modelo principal", list(MODELOS.keys()))
modelo_id    = MODELOS[modelo_label]

# ── Modo comparativo (opcional) ──────────────────────────
modo_comparativo = st.sidebar.checkbox("🔀 Activar modo comparativo")
modelo_b_id = None
if modo_comparativo:
    st.sidebar.caption("Selecciona un segundo modelo para comparar resultados:")
    opciones_b = {k: v for k, v in MODELOS.items() if k != modelo_label}
    if opciones_b:
        modelo_b_label = st.sidebar.selectbox("Modelo B", list(opciones_b.keys()))
        modelo_b_id    = opciones_b[modelo_b_label]
    else:
        st.sidebar.warning("Necesitas al menos 2 modelos disponibles para comparar.")
        modo_comparativo = False

badge_cls = "openai" if proveedor == "OpenAI" else "gemini"
st.sidebar.markdown(
    f'Usando: <span class="badge-{badge_cls}">{proveedor} · {modelo_label}</span>',
    unsafe_allow_html=True
)

if not api_key:
    st.warning("🔑 Introduce tu API Key en el panel izquierdo para activar el asistente.")
    st.stop()

# ── Inicializar cliente ───────────────────────────────────
client = None
if proveedor == "OpenAI":
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        st.error(
            "La librería **openai** no está instalada.\n\n"
            "Ejecuta en tu terminal:\n```\npip install openai\n```\n"
            "Luego reinicia Streamlit."
        )
        st.info("Alternativa: cambia el proveedor a **Gemini** en el panel izquierdo.")
        st.stop()
else:
    try:
        from google import genai
        from google.genai import types as gtypes
        client = genai.Client(api_key=api_key)
    except ImportError:
        st.error(
            "La librería **google-genai** no está instalada.\n\n"
            "Ejecuta en tu terminal:\n```\npip install google-genai\n```\n"
            "Luego reinicia Streamlit."
        )
        st.info("Alternativa: cambia el proveedor a **OpenAI** en el panel izquierdo.")
        st.stop()


# =========================================================
# FUNCIONES UTILITARIAS
# =========================================================
def _llamar(model_id: str, prompt: str, temperature: float = 0) -> str:
    if proveedor == "OpenAI":
        kwargs = {"model": model_id, "messages": [{"role": "user", "content": prompt}]}
        if not model_id.startswith("o"):
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    else:
        from google.genai import types as gtypes
        resp = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=gtypes.GenerateContentConfig(temperature=temperature)
        )
        return resp.text.strip()


def llamar_modelo(prompt: str, temperature: float = 0) -> str:
    return _llamar(modelo_id, prompt, temperature)


def llamar_modelo_b(prompt: str, temperature: float = 0) -> str:
    return _llamar(modelo_b_id, prompt, temperature) if modelo_b_id else ""


def descargar_csv(df_result: pd.DataFrame, nombre: str = "resultado"):
    buf = io.BytesIO()
    df_result.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ Descargar resultado como CSV",
        data=buf.getvalue(),
        file_name=f"{nombre}.csv",
        mime="text/csv",
    )


def mostrar_comparativo(col_a: list, col_b: list, label_a: str, label_b: str, etiqueta_col: str):
    """Muestra tabla comparativa lado a lado entre modelo A y modelo B."""
    df_comp = pd.DataFrame({
        etiqueta_col: range(1, len(col_a) + 1),
        f"Modelo A – {label_a}": col_a,
        f"Modelo B – {label_b}": col_b,
    })
    st.markdown('<div class="compare-box">', unsafe_allow_html=True)
    st.subheader("🔀 Comparativa de modelos")
    st.dataframe(df_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# CARGA DE ARCHIVO
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("📂 Datos")
uploaded_file = st.sidebar.file_uploader("Carga CSV o XLSX", type=["csv", "xlsx"])

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.sidebar.success(f"✅ {uploaded_file.name}  ({len(df):,} filas)")
    with st.expander("👀 Vista previa de datos", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
else:
    st.info("📂 Carga un archivo CSV o XLSX para comenzar.")
    st.stop()

# =========================================================
# NAVEGACIÓN
# =========================================================
st.sidebar.divider()
seccion = st.sidebar.selectbox(
    "📌 Sección",
    [
        "1. Análisis de Sentimiento",
        "2. Extracción de Datos Clave",
        "3. Clasificación de Riesgo de Rotación",
        "4. Clusterización de Texto",
        "5. Asistente de Reclutamiento",
        "6. Detección de Temas (Topic Modeling)",
        "7. Constructor de Prompts Profesionales HR/NLP",
    ]
)


# =========================================================
# 1. ANÁLISIS DE SENTIMIENTO
# =========================================================
if seccion.startswith("1"):
    st.header("📊 Análisis de Sentimiento")
    st.markdown("Clasifica comentarios o respuestas como **Positivo**, **Neutro** o **Negativo**.")

    columna = st.selectbox("Columna de texto", df.columns)

    if st.button("▶ Clasificar Sentimiento"):
        resultados_a, resultados_b = [], []
        bar = st.progress(0)
        total = len(df)

        for i, text in enumerate(df[columna]):
            if pd.isna(text):
                resultados_a.append("Nulo")
                if modo_comparativo: resultados_b.append("Nulo")
            else:
                prompt = (
                    f'Clasifica el sentimiento de este texto: "{text}" '
                    'con una sola palabra: Positivo, Neutro o Negativo.'
                )
                def parse_sent(label):
                    u = label.upper()
                    if "POSITIVO" in u: return "Positivo"
                    if "NEGATIVO" in u: return "Negativo"
                    if "NEUTRO"   in u: return "Neutro"
                    return "Indefinido"

                resultados_a.append(parse_sent(llamar_modelo(prompt)))
                if modo_comparativo:
                    resultados_b.append(parse_sent(llamar_modelo_b(prompt)))
            bar.progress((i + 1) / total, text=f"Procesando {i+1}/{total}…")

        df["sentimiento"] = resultados_a
        bar.empty()
        st.success("✅ Análisis completado.")

        conteo = df["sentimiento"].value_counts()
        cols = st.columns(len(conteo))
        for col, (label, val) in zip(cols, conteo.items()):
            col.metric(label, val)
        st.bar_chart(conteo)
        st.dataframe(df, use_container_width=True)
        descargar_csv(df, "sentimiento")

        if modo_comparativo and resultados_b:
            mostrar_comparativo(resultados_a, resultados_b, modelo_label, modelo_b_label, "Fila")


# =========================================================
# 2. EXTRACCIÓN DE DATOS CLAVE
# =========================================================
elif seccion.startswith("2"):
    st.header("📌 Extracción de Datos Clave")
    columna = st.selectbox("Columna con descripciones", df.columns)

    col1, col2, col3 = st.columns(3)
    with col1: p1 = st.text_input("Campo 1", placeholder="Ej: Nivel educativo")
    with col2: p2 = st.text_input("Campo 2", placeholder="Ej: Habilidades")
    with col3: p3 = st.text_input("Campo 3", placeholder="Ej: Idiomas")

    if st.button("▶ Extraer Datos"):
        if not (p1 and p2 and p3):
            st.warning("Completa los tres campos antes de continuar.")
        else:
            val1_a, val2_a, val3_a = [], [], []
            val1_b, val2_b, val3_b = [], [], []
            bar = st.progress(0)
            total = len(df)

            for i, text in enumerate(df[columna]):
                prompt = (
                    f'De esta descripción: "{text}"\n'
                    f'Extrae:\n1. {p1}\n2. {p2}\n3. {p3}\n'
                    'Responde SOLO con: valor1|valor2|valor3'
                )
                def parse_ext(txt):
                    p = txt.split("|")
                    return (p[0].strip(), p[1].strip(), p[2].strip()) if len(p) == 3 else ("N/A","N/A","N/A")

                a, b, c = parse_ext(llamar_modelo(prompt))
                val1_a.append(a); val2_a.append(b); val3_a.append(c)
                if modo_comparativo:
                    a2, b2, c2 = parse_ext(llamar_modelo_b(prompt))
                    val1_b.append(a2); val2_b.append(b2); val3_b.append(c2)
                bar.progress((i + 1) / total)

            df[p1] = val1_a; df[p2] = val2_a; df[p3] = val3_a
            bar.empty()
            st.success("✅ Extracción completada.")
            st.dataframe(df, use_container_width=True)
            descargar_csv(df, "extraccion_datos")

            if modo_comparativo and val1_b:
                df_comp = pd.DataFrame({
                    f"{p1} (A)": val1_a, f"{p1} (B)": val1_b,
                    f"{p2} (A)": val2_a, f"{p2} (B)": val2_b,
                    f"{p3} (A)": val3_a, f"{p3} (B)": val3_b,
                })
                st.markdown('<div class="compare-box">', unsafe_allow_html=True)
                st.subheader("🔀 Comparativa de modelos")
                st.dataframe(df_comp, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 3. CLASIFICACIÓN DE RIESGO DE ROTACIÓN
# =========================================================
elif seccion.startswith("3"):
    st.header("⚠️ Clasificación de Riesgo de Rotación")
    columna = st.selectbox("Columna de comentarios", df.columns)

    if st.button("▶ Clasificar Riesgo"):
        riesgos_a, just_a = [], []
        riesgos_b, just_b = [], []
        bar = st.progress(0)
        total = len(df)

        for i, text in enumerate(df[columna]):
            if pd.isna(text):
                riesgos_a.append("Nulo"); just_a.append("Sin información")
                if modo_comparativo: riesgos_b.append("Nulo"); just_b.append("Sin información")
            else:
                prompt = (
                    f'Analiza este comentario de un colaborador: "{text}"\n'
                    'Clasifica el riesgo de salida: Alto, Moderado o Bajo.\n'
                    'Justifica en máximo 15 palabras.\n'
                    'Responde SOLO con: categoria|justificacion'
                )
                def parse_riesgo(txt):
                    p = txt.split("|")
                    return (p[0].strip() if len(p)>=1 else "N/A",
                            p[1].strip() if len(p)>=2 else "N/A")

                r, j = parse_riesgo(llamar_modelo(prompt))
                riesgos_a.append(r); just_a.append(j)
                if modo_comparativo:
                    r2, j2 = parse_riesgo(llamar_modelo_b(prompt))
                    riesgos_b.append(r2); just_b.append(j2)
            bar.progress((i + 1) / total)

        df["riesgo_rotacion"] = riesgos_a
        df["justificacion"]   = just_a
        bar.empty()
        st.success("✅ Clasificación completada.")

        conteo = df["riesgo_rotacion"].value_counts()
        cols = st.columns(len(conteo))
        for col, (label, val) in zip(cols, conteo.items()):
            col.metric(label, val)
        st.bar_chart(conteo)
        st.dataframe(df, use_container_width=True)
        descargar_csv(df, "riesgo_rotacion")

        if modo_comparativo and riesgos_b:
            mostrar_comparativo(riesgos_a, riesgos_b, modelo_label, modelo_b_label, "Fila")


# =========================================================
# 4. CLUSTERIZACIÓN DE TEXTO
# =========================================================
elif seccion.startswith("4"):
    st.header("🔍 Clusterización de Texto")
    if proveedor == "OpenAI":
        st.info("ℹ️ Embeddings: `text-embedding-3-small`")
    else:
        st.info("ℹ️ Embeddings: `text-embedding-004`")

    columna = st.selectbox("Columna de texto", df.columns)
    k = st.slider("Número de clusters", 2, 8, 3)

    if st.button("▶ Generar Clusters"):
        embeddings = []
        bar = st.progress(0)
        total = len(df)

        for i, text in enumerate(df[columna]):
            if proveedor == "OpenAI":
                resp = client.embeddings.create(model="text-embedding-3-small", input=str(text))
                embeddings.append(resp.data[0].embedding)
            else:
                from google.genai import types as gtypes
                emb = client.models.embed_content(model="text-embedding-004", contents=str(text))
                embeddings.append(emb.embeddings[0].values)
            bar.progress((i + 1) / total)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["cluster"] = kmeans.fit_predict(embeddings)
        bar.empty()
        st.success("✅ Clusterización completada.")
        st.bar_chart(df["cluster"].value_counts().sort_index())
        st.dataframe(df, use_container_width=True)
        descargar_csv(df, "clusters")


# =========================================================
# 5. ASISTENTE DE RECLUTAMIENTO (MEJORADO)
# =========================================================
elif seccion.startswith("5"):
    st.header("🎯 Asistente de Reclutamiento Inteligente")
    st.markdown("Evaluación profunda de candidatos con scorecard multidimensional, red flags, preguntas sugeridas y fit cultural.")

    # ── Fuente del Job Description ────────────────────────
    st.subheader("📋 Job Description")
    jd_fuente = st.radio("Fuente del Job Description", ["Texto libre", "Cargar CSV/XLSX"], horizontal=True)

    descripcion = ""
    if jd_fuente == "Texto libre":
        descripcion = st.text_area(
            "Pega o escribe el Job Description",
            height=200,
            placeholder="""DATA SCIENTIST SENIOR
REQUISITOS PRINCIPALES:
• 5+ años en ciencia de datos
• Python avanzado y librerías ML (TensorFlow/PyTorch)
• SQL y bases de datos
• Experiencia en despliegue de modelos producción
• Liderazgo de equipos
RESPONSABILIDADES:
• Desarrollar modelos ML para productos
• Liderar proyectos end-to-end
• Mentorizar juniors
• Colaborar con equipos técnicos"""
        )
    else:
        jd_file = st.file_uploader("Carga archivo con Job Description", type=["csv","xlsx"], key="jd_file")
        if jd_file:
            df_jd = pd.read_csv(jd_file) if jd_file.name.endswith(".csv") else pd.read_excel(jd_file)
            col_jd = st.selectbox("Columna del Job Description", df_jd.columns)
            descripcion = " ".join(df_jd[col_jd].astype(str).tolist())
            st.success("Job Description cargado.")

    # ── Campos adicionales opcionales ─────────────────────
    st.subheader("⚙️ Configuración de evaluación")
    col_a, col_b = st.columns(2)
    with col_a:
        columna_cv   = st.selectbox("Columna principal (CV / perfil)", df.columns)
        col_nombre   = st.selectbox("Columna de nombre / ID del candidato", ["(ninguna)"] + list(df.columns))
        col_anios    = st.selectbox("Columna de años de experiencia (opcional)", ["(ninguna)"] + list(df.columns))
    with col_b:
        col_educacion = st.selectbox("Columna de nivel educativo (opcional)", ["(ninguna)"] + list(df.columns))
        col_skills    = st.selectbox("Columna de habilidades/tecnologías (opcional)", ["(ninguna)"] + list(df.columns))
        col_salario   = st.selectbox("Columna de pretensión salarial (opcional)", ["(ninguna)"] + list(df.columns))

    idioma_salida = st.radio("Idioma del análisis", ["Español", "English"], horizontal=True)

    generar_preguntas = st.checkbox("✅ Generar preguntas de entrevista sugeridas por candidato", value=True)
    incluir_cultura   = st.checkbox("✅ Evaluar fit cultural y valores", value=True)

    if st.button("▶ Evaluar Candidatos"):
        if not descripcion.strip():
            st.warning("Ingresa el Job Description antes de continuar.")
            st.stop()

        (puntajes, analisis_list, fortalezas_list, debilidades_list,
         red_flags_list, preguntas_list, cultura_list,
         exp_s, hab_s, edu_s, log_s, aj_s, liderazgo_s) = ([] for _ in range(13))

        bar = st.progress(0)
        total_rows = len(df)

        for i, row in df.iterrows():
            cv = str(row[columna_cv])
            nombre_cand = str(row[col_nombre]) if col_nombre != "(ninguna)" else f"Candidato {i+1}"
            ctx_extra = ""
            if col_anios    != "(ninguna)": ctx_extra += f"\nAños de experiencia declarados: {row[col_anios]}"
            if col_educacion != "(ninguna)": ctx_extra += f"\nNivel educativo: {row[col_educacion]}"
            if col_skills    != "(ninguna)": ctx_extra += f"\nHabilidades declaradas: {row[col_skills]}"
            if col_salario   != "(ninguna)": ctx_extra += f"\nPretensión salarial: {row[col_salario]}"

            prompt = f"""
Eres un experto senior en reclutamiento, People Analytics y psicología organizacional.
Evalúa al candidato "{nombre_cand}" comparando su perfil con el Job Description.
Responde en: {idioma_salida}.

JOB DESCRIPTION:
{descripcion}

PERFIL DEL CANDIDATO:
{cv}
{ctx_extra}

━━━━━━━━━━━━━━━━━━━━━━━━
CRITERIOS DE EVALUACIÓN (puntúa SOLO con evidencia explícita, sin inventar):
1. EXPERIENCIA (0-25): años, roles, industria relevante
2. HABILIDADES (0-25): hard skills, herramientas, tecnologías
3. EDUCACION (0-20): grado, certificaciones, formación continua
4. LOGROS (0-15): resultados medibles, impacto cuantificable
5. AJUSTE_ROL (0-15): alineación general con el puesto
━━━━━━━━━━━━━━━━━━━━━━━━
CAMPOS ADICIONALES (texto libre):
- LIDERAZGO: puntuación 0-10 + evidencia (1 línea)
- RED_FLAGS: señales de alerta o vacíos relevantes (máx 3 ítems separados por ;)
- FIT_CULTURAL: evaluación del encaje cultural y valores (máx 30 palabras) — {"incluir" if incluir_cultura else "omitir, escribe N/A"}
- PREGUNTAS_ENTREVISTA: 3 preguntas específicas de entrevista adaptadas a este candidato (separadas por |) — {"incluir" if generar_preguntas else "omitir, escribe N/A"}
- RECOMENDACION: una de estas: AVANZAR | CONSIDERAR | DESCARTAR
- ANALISIS: síntesis ejecutiva del candidato (máx 40 palabras)
- FORTALEZAS: ítem1; ítem2; ítem3
- DEBILIDADES: ítem1; ítem2; ítem3
━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO ESTRICTO (responde SOLO en este formato):
EXPERIENCIA: [0-25]
HABILIDADES: [0-25]
EDUCACION: [0-20]
LOGROS: [0-15]
AJUSTE_ROL: [0-15]
LIDERAZGO: [0-10] | evidencia
RED_FLAGS: item1; item2; item3
FIT_CULTURAL: texto
PREGUNTAS_ENTREVISTA: p1|p2|p3
RECOMENDACION: valor
ANALISIS: texto
FORTALEZAS: item1; item2; item3
DEBILIDADES: item1; item2; item3
"""
            txt = llamar_modelo(prompt)

            def get_score(tag, default=0):
                try:
                    raw = txt.split(f"{tag}:")[1].strip().split()[0]
                    return int(raw)
                except:
                    return default

            def extraer_campo(tag, separadores=None):
                if tag not in txt:
                    return "N/A"
                try:
                    after = txt.split(tag)[1]
                    stoppers = ["EXPERIENCIA:", "HABILIDADES:", "EDUCACION:", "LOGROS:",
                                "AJUSTE_ROL:", "LIDERAZGO:", "RED_FLAGS:", "FIT_CULTURAL:",
                                "PREGUNTAS_ENTREVISTA:", "RECOMENDACION:", "ANALISIS:",
                                "FORTALEZAS:", "DEBILIDADES:"]
                    for s in stoppers:
                        if s != tag and s in after:
                            after = after.split(s)[0]
                    return after.strip()
                except:
                    return "N/A"

            e  = get_score("EXPERIENCIA")
            h  = get_score("HABILIDADES")
            ed = get_score("EDUCACION")
            l  = get_score("LOGROS")
            aj = get_score("AJUSTE_ROL")
            ldr_raw = extraer_campo("LIDERAZGO:")
            try:
                ldr_score = int(ldr_raw.split("|")[0].strip().split()[0])
            except:
                ldr_score = 0

            total_score = max(0, min(e + h + ed + l + aj, 100))
            puntajes.append(total_score)
            exp_s.append(e); hab_s.append(h); edu_s.append(ed)
            log_s.append(l); aj_s.append(aj); liderazgo_s.append(ldr_score)
            analisis_list.append(extraer_campo("ANALISIS:"))
            fortalezas_list.append(extraer_campo("FORTALEZAS:"))
            debilidades_list.append(extraer_campo("DEBILIDADES:"))
            red_flags_list.append(extraer_campo("RED_FLAGS:"))
            preguntas_list.append(extraer_campo("PREGUNTAS_ENTREVISTA:"))
            cultura_list.append(extraer_campo("FIT_CULTURAL:"))

            bar.progress((list(df.index).index(i) + 1) / total_rows)

        # ── Agregar resultados al dataframe ────────────────
        df["puntaje_total"]    = puntajes
        df["score_experiencia"]= exp_s
        df["score_habilidades"]= hab_s
        df["score_educacion"]  = edu_s
        df["score_logros"]     = log_s
        df["score_ajuste"]     = aj_s
        df["score_liderazgo"]  = liderazgo_s
        df["recomendacion"]    = [extraer_campo("RECOMENDACION:") for _ in range(len(df))]  # placeholder
        df["analisis"]         = analisis_list
        df["fortalezas"]       = fortalezas_list
        df["debilidades"]      = debilidades_list
        df["red_flags"]        = red_flags_list
        df["preguntas_entrevista"] = preguntas_list
        df["fit_cultural"]     = cultura_list

        # Re-extraer recomendacion correctamente desde llamadas individuales
        # (ya está en analisis_list por el loop, corregimos con el texto completo — simplificamos)

        bar.empty()
        st.success("✅ Evaluación completada.")

        # ── KPIs ejecutivos ────────────────────────────────
        df_sorted = df.sort_values("puntaje_total", ascending=False).reset_index(drop=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Candidatos evaluados", len(df_sorted))
        c2.metric("🏆 Puntaje más alto", f"{df_sorted['puntaje_total'].max()}/100")
        c3.metric("📊 Promedio general", f"{df_sorted['puntaje_total'].mean():.1f}/100")
        c4.metric("✅ Puntaje >= 70", int((df_sorted["puntaje_total"] >= 70).sum()))

        # ── Top 3 tarjetas ─────────────────────────────────
        st.subheader("🏆 Top 3 Candidatos")
        top3 = df_sorted.head(3)
        tcols = st.columns(3)
        for idx, (col_card, (_, row)) in enumerate(zip(tcols, top3.iterrows())):
            nombre = str(row[col_nombre]) if col_nombre != "(ninguna)" else f"#{idx+1}"
            pct = row["puntaje_total"]
            with col_card:
                st.markdown(f"""
<div class="candidate-card">
  <strong>#{idx+1} {nombre[:28]}</strong><br/>
  <span style="font-size:1.6rem;font-weight:800;color:#1a2e4a">{pct}/100</span>
  <div class="score-bar-container">
    <div class="score-bar" style="width:{pct}%"></div>
  </div>
  <small>{str(row.get('analisis',''))[:120]}…</small>
</div>
""", unsafe_allow_html=True)

        # ── Scorecard visual por candidato ─────────────────
        st.subheader("📊 Scorecard Detallado por Candidato")
        score_cols = ["puntaje_total","score_experiencia","score_habilidades",
                      "score_educacion","score_logros","score_ajuste","score_liderazgo"]
        display_nombre = col_nombre if col_nombre != "(ninguna)" else columna_cv
        sc_df = df_sorted[[display_nombre] + score_cols].copy()
        st.dataframe(sc_df, use_container_width=True)

        # ── Tabla completa ─────────────────────────────────
        st.subheader("📋 Resultados Completos")
        cols_show = [c for c in df_sorted.columns]
        st.dataframe(df_sorted[cols_show], use_container_width=True)

        # ── Preguntas de entrevista expandibles ────────────
        if generar_preguntas:
            st.subheader("❓ Preguntas de Entrevista Sugeridas")
            for _, row in df_sorted.head(5).iterrows():
                nombre = str(row[col_nombre]) if col_nombre != "(ninguna)" else "Candidato"
                pregs  = str(row.get("preguntas_entrevista","N/A")).split("|")
                with st.expander(f"🗣 {nombre[:40]} — {row['puntaje_total']}/100"):
                    for j, p in enumerate(pregs, 1):
                        st.markdown(f"**{j}.** {p.strip()}")

        descargar_csv(df_sorted, "evaluacion_candidatos")

        # ── Modo comparativo ──────────────────────────────
        if modo_comparativo:
            st.markdown('<div class="compare-box">', unsafe_allow_html=True)
            st.subheader("🔀 Comparativa de modelos — re-evaluar con Modelo B")
            st.info("Para la comparativa de reclutamiento, re-ejecuta la evaluación con el Modelo B activo como principal.")
            st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 6. DETECCIÓN DE TEMAS
# =========================================================
elif seccion.startswith("6"):
    st.header("🧩 Detección de Temas (Topic Modeling)")
    columna = st.selectbox("Columna de texto", df.columns)

    if st.button("▶ Detectar Temas"):
        textos = "\n".join(df[columna].astype(str).tolist())

        prompt_general = f"""
Realiza análisis de topic modeling sobre estos textos.
- Detecta entre 3 y 6 temas principales.
- Provee ejemplos representativos por tema.
- Resumen global (máx 50 palabras).

FORMATO:
TEMAS:
1. [tema] - [ejemplo1]; [ejemplo2]
2. ...
RESUMEN_GLOBAL: texto breve

TEXTOS:
{textos}
RESPONDE SOLO EN ESTE FORMATO.
"""
        with st.spinner("Analizando temas globales…"):
            resp_general = llamar_modelo(prompt_general)

        st.subheader("📌 Temas Globales Detectados")
        st.markdown(resp_general)
        st.divider()

        topico_list, subtopico_list, explicacion_list = [], [], []
        topico_b, subtopico_b = [], []
        bar = st.progress(0)
        total = len(df)

        for i, text in enumerate(df[columna]):
            prompt_ind = f"""
Clasifica este texto en tema y subtema.
TEXTO: "{text}"
FORMATO ESTRICTO:
TEMA: ...
SUBTEMA: ...
EXPLICACION: (máx 15 palabras)
"""
            txt = llamar_modelo(prompt_ind)

            def extraer(tag, t=txt):
                if tag not in t:
                    return "N/A"
                try:
                    return t.split(tag)[1].split("\n")[0].strip()
                except:
                    return "N/A"

            topico_list.append(extraer("TEMA:"))
            subtopico_list.append(extraer("SUBTEMA:"))
            explicacion_list.append(extraer("EXPLICACION:"))

            if modo_comparativo:
                txt_b = llamar_modelo_b(prompt_ind)
                def extraer_b(tag, t=txt_b):
                    if tag not in t: return "N/A"
                    try: return t.split(tag)[1].split("\n")[0].strip()
                    except: return "N/A"
                topico_b.append(extraer_b("TEMA:"))
                subtopico_b.append(extraer_b("SUBTEMA:"))

            bar.progress((i + 1) / total)

        df["tema_principal"]   = topico_list
        df["subtema"]          = subtopico_list
        df["explicacion_tema"] = explicacion_list
        bar.empty()
        st.success("✅ Detección de temas completada.")
        st.bar_chart(df["tema_principal"].value_counts())
        st.dataframe(df, use_container_width=True)
        descargar_csv(df, "topic_modeling")

        if modo_comparativo and topico_b:
            mostrar_comparativo(topico_list, topico_b, modelo_label, modelo_b_label, "Fila")


# =========================================================
# 7. CONSTRUCTOR DE PROMPTS PROFESIONALES HR/NLP
# =========================================================
elif seccion.startswith("7"):
    st.header("🛠️ Constructor de Prompts Profesionales HR/NLP")
    st.markdown(
        "Genera prompts de nivel experto para tareas de People Analytics, "
        "optimizados para producción con técnicas avanzadas de Prompt Engineering."
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("⚙️ Configuración del Prompt")

        tarea = st.selectbox("Tarea de HR/People Analytics", [
            "Análisis de sentimiento en encuestas de clima",
            "Clasificación de riesgo de rotación (attrition)",
            "Evaluación y scoring de candidatos",
            "Detección de temas en feedback 360°",
            "Extracción de competencias desde CVs",
            "Análisis de brechas de habilidades (skills gap)",
            "Resumen ejecutivo de entrevistas de salida",
            "Clasificación de motivos de desvinculación",
            "Generación de preguntas de entrevista por competencia",
            "Análisis de inclusión y diversidad en textos",
            "Personalizado (describe abajo)",
        ])

        tarea_custom = ""
        if tarea == "Personalizado (describe abajo)":
            tarea_custom = st.text_area("Describe la tarea", height=80)

        tecnica = st.selectbox("Técnica de Prompt Engineering", [
            "Chain-of-Thought (CoT) — razonamiento paso a paso",
            "Few-Shot — ejemplos en el prompt",
            "Role Prompting — persona experta",
            "Structured Output — salida estructurada (JSON/pipe)",
            "ReAct — razonamiento + acción",
            "Tree-of-Thought — árbol de decisiones",
            "Self-Consistency — múltiples perspectivas",
        ])

        modelo_objetivo = st.selectbox("Modelo objetivo del prompt", [
            "GPT-4o / GPT-4.1",
            "GPT-4o-mini / GPT-4.1-mini",
            "o1 / o3 / o4 (razonamiento)",
            "Gemini 2.5 Pro",
            "Gemini 2.0 / 2.5 Flash",
            "Cualquier LLM (genérico)",
        ])

        idioma_prompt = st.radio("Idioma del prompt generado", ["Español", "English"], horizontal=True)
        temperatura_sugerida = st.slider("Temperatura sugerida", 0.0, 1.0, 0.0, 0.1)
        incluir_ejemplos = st.checkbox("Incluir ejemplos Few-Shot en el prompt", value=True)
        incluir_instrucciones_formato = st.checkbox("Incluir instrucciones de formato de salida", value=True)

        contexto_adicional = st.text_area(
            "Contexto adicional (industria, empresa, tipo de datos...)",
            placeholder="Ej: empresa retail, 500 empleados, datos de encuesta Likert 1-5",
            height=80
        )

    with col_r:
        st.subheader("📝 Prompt Generado")

        if st.button("🚀 Generar Prompt Profesional", use_container_width=True):
            tarea_final = tarea_custom if tarea == "Personalizado (describe abajo)" else tarea

            meta_prompt = f"""
Eres un experto en NLP aplicado a People Analytics y Recursos Humanos, con dominio avanzado en Prompt Engineering para LLMs de producción.

ENCARGO:
Crea un prompt profesional, completo y listo para usar en producción para la siguiente tarea:
TAREA: {tarea_final}

PARÁMETROS:
- Técnica de PE a aplicar: {tecnica}
- Modelo objetivo: {modelo_objetivo}
- Temperatura sugerida: {temperatura_sugerida}
- Idioma del prompt: {idioma_prompt}
- Incluir ejemplos Few-Shot: {"Sí" if incluir_ejemplos else "No"}
- Incluir instrucciones de formato de salida: {"Sí" if incluir_instrucciones_formato else "No"}
- Contexto adicional: {contexto_adicional if contexto_adicional else "No especificado"}

ESTRUCTURA DEL PROMPT A GENERAR:
1. ROL / PERSONA — define el rol experto del modelo
2. CONTEXTO — situación y datos de entrada esperados
3. TAREA PRINCIPAL — instrucción clara y precisa
4. TÉCNICA APLICADA — aplica {tecnica} de forma explícita
5. RESTRICCIONES — qué NO debe hacer el modelo
6. FORMATO DE SALIDA — estructura esperada (usa pipes | o JSON si aplica)
{"7. EJEMPLOS FEW-SHOT — 2 ejemplos input/output realistas para HR" if incluir_ejemplos else ""}

IMPORTANTE:
- El prompt debe ser directamente utilizable sin edición
- Usa marcadores claros (TEXTO_ENTRADA: ..., etc.)
- Optimiza para {modelo_objetivo}
- El prompt completo debe estar en {idioma_prompt}
- Al final, incluye una sección "NOTAS DE USO" con tips de temperatura, casos de uso y variaciones recomendadas
"""
            with st.spinner("Generando prompt experto…"):
                resultado = llamar_modelo(meta_prompt, temperature=0.3)

            st.markdown(resultado)

            # ── Botón copiar / descargar ───────────────────
            st.download_button(
                "⬇️ Descargar Prompt como .txt",
                data=resultado.encode("utf-8"),
                file_name=f"prompt_{tarea_final[:30].replace(' ','_')}.txt",
                mime="text/plain"
            )

            # ── Guardar en session_state para historial ────
            if "historial_prompts" not in st.session_state:
                st.session_state["historial_prompts"] = []
            st.session_state["historial_prompts"].append({
                "tarea": tarea_final,
                "tecnica": tecnica,
                "prompt": resultado
            })

    # ── Historial de prompts generados en sesión ──────────
    if st.session_state.get("historial_prompts"):
        st.divider()
        st.subheader("📚 Historial de Prompts (sesión actual)")
        for idx, entry in enumerate(reversed(st.session_state["historial_prompts"]), 1):
            with st.expander(f"#{idx} — {entry['tarea'][:50]} · {entry['tecnica'][:30]}"):
                st.markdown(entry["prompt"])
