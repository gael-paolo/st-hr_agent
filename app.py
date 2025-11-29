import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from sklearn.cluster import KMeans

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(page_title="People Analytics Assistant", layout="wide")

st.title("🧠 People Analytics Assistant con Gemini")

# Panel para introducir API KEY
st.sidebar.header("🔑 Configuración")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    client = genai.Client(api_key=api_key)
else:
    st.warning("Introduce tu API Key para activar el asistente.")
    st.stop()

# Cargar archivo CSV/XLSX
uploaded_file = st.sidebar.file_uploader("Carga un archivo CSV o XLSX", type=["csv", "xlsx"])

df = None
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.success("Archivo cargado correctamente.")
    st.write("### Vista previa de datos")
    st.dataframe(df.head())
else:
    st.info("Carga un archivo para comenzar.")
    st.stop()

# =========================================================
# SECCIONES
# =========================================================
seccion = st.sidebar.selectbox(
    "Selecciona una sección",
    [
        "1. Análisis de Sentimiento",
        "2. Extracción de Datos Clave",
        "3. Clasificación de Riesgo",
        "4. Clusterización de Texto",
        "5. Asistente de Reclutamiento",
        "6. Detección de Temas (Topic Modeling con Gemini)"
    ]
)

# =========================================================
# 1. ANÁLISIS DE SENTIMIENTO
# =========================================================
if seccion.startswith("1"):
    st.header("📊 Análisis de Sentimiento")

    columna = st.selectbox("Selecciona la columna de texto", df.columns)

    if st.button("Clasificar Sentimiento"):
        resultados = []
        for text in df[columna]:
            if pd.isna(text):
                resultados.append("NULO")
                continue

            prompt = f'Clasifica el sentimiento de este texto: "{text}" con una palabra: Positivo, Neutro o Negativo.'

            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0)
            )

            label = resp.text.strip().upper()

            if "POSITIVO" in label:
                resultados.append("Positivo")
            elif "NEGATIVO" in label:
                resultados.append("Negativo")
            elif "NEUTRO" in label:
                resultados.append("Neutro")
            else:
                resultados.append("Indefinido")

        df["sentiment"] = resultados

        st.success("Análisis completado.")
        st.dataframe(df)
        st.bar_chart(df["sentiment"].value_counts())

# =========================================================
# 2. EXTRACCIÓN DE DATOS CLAVE (DINÁMICA)
# =========================================================
elif seccion.startswith("2"):
    st.header("📌 Extracción de Datos Clave")

    columna = st.selectbox("Columna con descripciones", df.columns)

    st.markdown("### Define los parámetros a extraer")
    col1, col2, col3 = st.columns(3)
    with col1:
        p1 = st.text_input("Parámetro 1", placeholder="Ej: Nivel educativo")
    with col2:
        p2 = st.text_input("Parámetro 2", placeholder="Ej: Habilidades")
    with col3:
        p3 = st.text_input("Parámetro 3", placeholder="Ej: Idiomas")

    if st.button("Extraer Datos"):
        if not p1 or not p2 or not p3:
            st.warning("Completa los tres parámetros antes de continuar.")
        else:
            val1, val2, val3 = [], [], []

            for text in df[columna]:
                prompt = f'''
                De esta descripción: "{text}"
                Extrae los siguientes datos:
                1. {p1}
                2. {p2}
                3. {p3}

                Formato de salida estrictamente:
                valor1|valor2|valor3
                '''

                resp = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0)
                )

                partes = resp.text.strip().split("|")
                if len(partes) == 3:
                    val1.append(partes[0].strip())
                    val2.append(partes[1].strip())
                    val3.append(partes[2].strip())
                else:
                    val1.append("N/A")
                    val2.append("N/A")
                    val3.append("N/A")

            df[p1] = val1
            df[p2] = val2
            df[p3] = val3

            st.success("Extracción completada.")
            st.dataframe(df)

# =========================================================
# 3. CLASIFICACIÓN DE RIESGO DE ROTACIÓN
# =========================================================
elif seccion.startswith("3"):
    st.header("⚠️ Clasificación de Riesgo de Rotación")

    columna = st.selectbox("Columna de comentarios", df.columns)

    if st.button("Clasificar Riesgo"):
        riesgos = []
        justificaciones = []

        for text in df[columna]:
            if pd.isna(text):
                riesgos.append("NULO")
                justificaciones.append("Sin información disponible")
                continue

            prompt = f'''
            Analiza el siguiente comentario de un colaborador:

            "{text}"

            Clasifica el nivel de riesgo de salida únicamente en una de estas categorías:
            - Alto
            - Moderado
            - Bajo

            También entrega una justificación muy breve, con un máximo de 15 palabras.

            FORMATO DE RESPUESTA (estricto):
            categoria|justificacion
            '''

            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0)
            )

            partes = resp.text.strip().split("|")

            if len(partes) == 2:
                riesgos.append(partes[0].strip())
                justificaciones.append(partes[1].strip())
            else:
                riesgos.append("N/A")
                justificaciones.append("No se pudo generar justificación")

        df["riesgo_rotacion"] = riesgos
        df["justificacion_riesgo"] = justificaciones

        st.success("Clasificación completada.")
        st.dataframe(df)

        st.bar_chart(df["riesgo_rotacion"].value_counts())

# =========================================================
# 4. CLUSTERIZACIÓN DE TEXTO
# =========================================================
elif seccion.startswith("4"):
    st.header("🔍 Clusterización de Texto con Embeddings")

    columna = st.selectbox("Columna de texto", df.columns)
    k = st.slider("Número de clusters", 2, 8, 3)

    if st.button("Generar Clusters"):
        embeddings = []

        for text in df[columna]:
            emb = client.models.embed_content(
                model="text-embedding-004",
                contents=str(text)
            )
            embeddings.append(emb.embeddings[0].values)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["cluster"] = kmeans.fit_predict(embeddings)

        st.success("Clusterización completada.")
        st.dataframe(df)
        st.bar_chart(df["cluster"].value_counts().sort_index())


# =========================================================
# 5. ASISTENTE DE RECLUTAMIENTO
# =========================================================
elif seccion.startswith("5"):
    st.header("🎯 Asistente de Reclutamiento con Gemini")

    descripcion = st.text_area("Descripción del puesto")
    columna = st.selectbox("Columna con CVs", df.columns)

    if st.button("Evaluar Candidatos"):
        puntajes = []
        analisis_list = []
        fortalezas_list = []
        debilidades_list = []

        for cv in df[columna]:

            prompt = f"""
            Evalúa el siguiente CV comparándolo estrictamente con la descripción del puesto.

            DESCRIPCIÓN DEL PUESTO:
            {descripcion}

            CV DEL CANDIDATO:
            {cv}

            INSTRUCCIONES:
            Evalúa estos criterios por separado:
            1. EXPERIENCIA RELEVANTE (0-25)
            2. HABILIDADES Y TÉCNICAS CLAVE (0-25)
            3. EDUCACIÓN / CERTIFICACIONES (0-20)
            4. LOGROS MEDIBLES (0-15)
            5. AJUSTE GENERAL AL ROL (0-15)

            Reglas:
            - No inventes información que no esté en el CV.
            - Puntúa cada criterio basado únicamente en evidencia explícita.
            - Explica de forma breve (máx. 25 palabras) el ajuste general.
            - Entrega fortalezas y debilidades como listas separadas por punto y coma.

            FORMATO ESTRICTO DE RESPUESTA:
            EXPERIENCIA: [0-25]
            HABILIDADES: [0-25]
            EDUCACION: [0-20]
            LOGROS: [0-15]
            AJUSTE: [0-15]

            PUNTAJE_FINAL: [suma total de los anteriores]

            ANALISIS: texto breve (máx 25 palabras)
            FORTALEZAS: item1; item2; item3
            DEBILIDADES: item1; item2; item3

            RESPONDE SOLO EN ESTE FORMATO:
            """

            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0)
            )

            txt = resp.text

            # -------------------------
            # EXTRACCIÓN ROBUSTA DE PUNTAJES
            # -------------------------
            def get_score(tag, default=0):
                try:
                    seg = txt.split(f"{tag}:")[1].strip().split()[0]
                    return int(seg)
                except:
                    return default

            exp_score = get_score("EXPERIENCIA")
            hab_score = get_score("HABILIDADES")
            edu_score = get_score("EDUCACION")
            log_score = get_score("LOGROS")
            aj_score  = get_score("AJUSTE")

            total = exp_score + hab_score + edu_score + log_score + aj_score

            # fallback por si falla el modelo
            if total > 100 or total < 0:
                total = max(0, min(total, 100))

            puntajes.append(total)

            # -------------------------
            # EXTRACCIÓN DE SECCIONES TEXTUALES
            # -------------------------
            def extraer(sec):
                if sec not in txt:
                    return "N/A"
                after = txt.split(sec)[1]
                for nxt in ["FORTALEZAS:", "DEBILIDADES:", "EXPERIENCIA:", "HABILIDADES:", "EDUCACION:", "LOGROS:", "AJUSTE:", "PUNTAJE_FINAL:"]:
                    if nxt != sec and nxt in after:
                        after = after.split(nxt)[0]
                return after.strip()

            analisis_list.append(extraer("ANALISIS:"))
            fortalezas_list.append(extraer("FORTALEZAS:"))
            debilidades_list.append(extraer("DEBILIDADES:"))

        df["puntaje"] = puntajes
        df["analisis"] = analisis_list
        df["fortalezas"] = fortalezas_list
        df["debilidades"] = debilidades_list

        st.success("Evaluación completada.")
        st.dataframe(df.sort_values("puntaje", ascending=False))

# =========================================================
# 6. DETECCIÓN DE TEMAS (Topic Modeling con Gemini)
# =========================================================
elif seccion.startswith("6"):
    st.header("🧩 Detección de Temas (Topic Modeling con Gemini)")

    columna = st.selectbox("Columna de texto", df.columns)

    if st.button("Detectar Temas"):
        textos = "\n".join(df[columna].astype(str).tolist())

        # ----------------------------------------------------
        # PROMPT GENERAL PARA IDENTIFICAR TEMAS GLOBALES
        # ----------------------------------------------------
        prompt_general = f"""
        Realiza un análisis de topic modeling sobre el siguiente conjunto de textos.

        OBJETIVO:
        - Detectar entre 3 y 6 temas principales.
        - Proveer ejemplos representativos por tema.
        - Generar un resumen global muy conciso (máx 50 palabras).

        FORMATO EXACTO DE RESPUESTA:
        TEMAS:
        1. [tema] - [ejemplo1]; [ejemplo2]
        2. [tema] - [ejemplo1]; [ejemplo2]
        ...

        RESUMEN_GLOBAL: texto breve

        TEXTOS:
        {textos}

        RESPONDE SOLO EN ESTE FORMATO.
        """

        resp_general = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_general,
            config=types.GenerateContentConfig(temperature=0)
        )

        st.subheader("📌 Resultados Globales de Topic Modeling")
        st.write(resp_general.text)

        # ----------------------------------------------------
        # ASIGNACIÓN DE TEMA INDIVIDUAL POR FILA
        # ----------------------------------------------------
        topico_list = []
        subtopico_list = []
        explicacion_list = []

        st.subheader("📌 Asignación de temas por texto")

        for text in df[columna]:

            prompt_individual = f"""
            Clasifica este texto en un tema y subtema, basándote en análisis semántico.

            TEXTO:
            "{text}"

            INSTRUCCIONES:
            - Identifica un TEMA principal (una sola frase).
            - Identifica un SUBTEMA más específico.
            - Explica la razón en máximo 15 palabras.
            - No inventes información que no esté en el texto.
            - Mantén una estructura clara y simple.

            FORMATO ESTRICTO:
            TEMA: ...
            SUBTEMA: ...
            EXPLICACION: ...

            RESPONDE SOLO EN ESTE FORMATO.
            """

            resp_individual = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_individual,
                config=types.GenerateContentConfig(temperature=0)
            )

            txt = resp_individual.text

            def extraer(tag):
                if tag not in txt:
                    return "N/A"
                try:
                    val = txt.split(tag)[1].split("\n")[0]
                    return val.strip()
                except:
                    return "N/A"

            topico_list.append(extraer("TEMA:"))
            subtopico_list.append(extraer("SUBTEMA:"))
            explicacion_list.append(extraer("EXPLICACION:"))

        # ----------------------------------------------------
        # AGREGAR AL DATAFRAME
        # ----------------------------------------------------
        df["topico_principal"] = topico_list
        df["sub_topico"] = subtopico_list
        df["explicacion_topico"] = explicacion_list

        st.success("Detección de temas completada.")
        st.dataframe(df)
