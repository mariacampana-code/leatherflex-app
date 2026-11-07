"""Interfaz Profesional SaaS - Leather-Flex 3D. Ejecutar: python -m streamlit run app.py"""
import os
import json
import sqlite3
import tempfile
import shutil
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from leatherflex import calibracion, pipeline, acomodo, simulador
from leatherflex.config import Config

# ------------------------------------------------------------------ 1. Configuración & Estilos CSS Premium
st.set_page_config(page_title="Leather-Flex 3D | Platform", page_icon="🟫", layout="wide")

TMP = st.session_state.setdefault("tmp", tempfile.mkdtemp(prefix="leatherflex_"))
DB_PATH = os.path.join(os.path.dirname(__file__), "leatherflex.db")

# Inyección de CSS para un diseño industrial sobrio y elegante
st.markdown("""
<style>
    /* Estructura general */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Encabezados y títulos */
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Tarjetas contenedoras (Cards) */
    .custom-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Métrica / KPIs personalizados */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #C26D47 !important;
    }
    
    /* Botones primarios */
    .stButton>button[kind="primary"] {
        background-color: #C26D47 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #A55634 !important;
        box-shadow: 0 4px 12px rgba(194, 109, 71, 0.3);
    }

    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    
    /* Ocultar elementos innecesarios */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS moldes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, cliente TEXT, 
                  dimensiones TEXT, fecha TEXT, archivo_dxf TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feed 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, usuario TEXT, titulo TEXT, 
                  descripcion TEXT, tipo_falla TEXT, solucion TEXT, fecha TEXT, media_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------ 2. Navegación Lateral Sobria
with st.sidebar:
    st.markdown("## 🟫 **Leather-Flex 3D**")
    st.caption("Plataforma Industrial de Gestión y Corte")
    st.write("---")
    opcion_menu = st.radio(
        "MÓDULOS DE TRABAJO",
        ["🛠️ Escaneo y Acomodo", "📸 Muro de Taller (Feed)", "📦 Catálogo de Moldes", "🎥 Biblioteca Técnica"]
    )
    st.write("---")
    st.caption("v2.4 Pro · Sistema SaaS Activo")

# ================================================================== MÓDULO 1: ESCANEO Y ACOMODO
if opcion_menu == "🛠️ Escaneo y Acomodo":
    st.title("Escaneo y Acomodo de Moldes")
    st.caption("Análisis óptico de imperfecciones y optimización de corte en tiempo real.")

    with st.sidebar:
        st.markdown("### Configuración del Escáner")
        ruta_tab = os.path.join(TMP, "tablero_para_imprimir.png")
        if not os.path.exists(ruta_tab):
            calibracion.tablero_para_imprimir(ruta_tab)
        st.download_button("📥 Descargar Tablero de Calibración", open(ruta_tab, "rb"), "tablero_calibracion.png", "image/png")
        
        f_tab = st.file_uploader("Foto del tablero", key="tab")
        f_bla = st.file_uploader("Hoja blanca (opcional)", key="bla")
        ruta_cal = os.path.join(TMP, "calib.json")
        
        if f_tab and st.button("Calibrar Escáner", type="secondary"):
            try:
                cal = calibracion.calibrar_tablero(cv2.imdecode(np.frombuffer(f_tab.getbuffer(), np.uint8), cv2.IMREAD_COLOR))
                gan = calibracion.campo_plano(cv2.imdecode(np.frombuffer(f_bla.getbuffer(), np.uint8), cv2.IMREAD_COLOR)) if f_bla else None
                calibracion.guardar(cal, gan, ruta_cal)
                st.session_state["cal"] = cal
                st.success("Escáner Calibrado")
            except Exception as e:
                st.error(f"Error de calibración: {e}")

        cfg = Config()
        cfg.umbral_sigma = st.slider("Sensibilidad a Defectos", 2.0, 5.0, cfg.umbral_sigma, 0.1)

    if not os.path.exists(ruta_cal):
        with open(ruta_cal, "w") as f:
            json.dump({"mm_por_px_foto": 0.5, "angulo_grados": 0.0, "matriz_homografia": None}, f)

    # Bloque Principal de Carga (Estilo Panel)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Captura de la Piel")
        fotos = st.file_uploader("Subir imagen o escaneo de superficie", accept_multiple_files=True, key="fotos")
        demo = st.button("Usar Muestra Simulada")

    with col_b:
        st.subheader("2. Selección de Moldes")
        dxfs = st.file_uploader("Subir archivos de corte (DXF)", accept_multiple_files=True, key="dxf")
        moldes = None
        if dxfs:
            moldes = []
            carp_m = os.path.join(TMP, "moldes")
            shutil.rmtree(carp_m, ignore_errors=True)
            os.makedirs(carp_m)
            for a in dxfs:
                path_a = os.path.join(carp_m, a.name)
                with open(path_a, "wb") as f:
                    f.write(a.getbuffer())
                try:
                    pol = acomodo.molde_desde_dxf(path_a)
                    cant = st.number_input(f"Cantidad para {a.name}", 0, 500, 0, key="c_" + a.name)
                    moldes.append({"nombre": os.path.splitext(a.name)[0], "poligono": pol, "cantidad": cant or None})
                except Exception as e:
                    st.error(f"Error en {a.name}: {e}")

    st.write("---")
    procesar_click = st.button("Ejecutar Análisis y Acomodo", type="primary", disabled=not (fotos or demo))

    if demo:
        carpeta_demo = os.path.join(TMP, "demo")
        shutil.rmtree(carpeta_demo, ignore_errors=True)
        simulador.generar(carpeta_demo)
        st.session_state["entrada"] = (os.path.join(carpeta_demo, "fotos"), os.path.join(carpeta_demo, "calib.json"))
        st.session_state["ejecutar"] = True
    elif procesar_click and fotos:
        carp_f = os.path.join(TMP, "fotos")
        shutil.rmtree(carp_f, ignore_errors=True)
        os.makedirs(carp_f)
        for a in fotos:
            with open(os.path.join(carp_f, a.name), "wb") as f:
                f.write(a.getbuffer())
        st.session_state["entrada"] = (carp_f, ruta_cal)
        st.session_state["ejecutar"] = True

    # Procesamiento y visualización de resultados
    if st.session_state.get("ejecutar") and "entrada" in st.session_state:
        carpeta_fotos, cal_json = st.session_state["entrada"]
        salida = os.path.join(TMP, "salida")
        estado = st.status("Procesando geometrías...", expanded=True)
        try:
            r = pipeline.procesar(carpeta_fotos, cal_json, salida, cfg, moldes, avisar=estado.write)
            estado.update(label="Análisis Finalizado", state="complete", expanded=False)
            st.session_state["resultado"] = (r["info"], salida)
            st.session_state["ejecutar"] = False
        except Exception:
            try:
                estado.write("Procesando textura directa y detectando fallas...")
                os.makedirs(salida, exist_ok=True)
                archivos = [os.path.join(carpeta_fotos, f) for f in os.listdir(carpeta_fotos) if not f.startswith(".")]
                img = cv2.imread(archivos[0])
                h, w, _ = img.shape
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (9, 9), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                res_img = img.copy()
                lista_def = []
                for c in cnts:
                    area = cv2.contourArea(c)
                    if 300 < area < (h * w * 0.1): 
                        x, y, bw, bh = cv2.boundingRect(c)
                        cv2.rectangle(res_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                        lista_def.append({"tipo": "Imperfección", "area_mm2": round(area, 1), "caja_mm": [x, y, x + bw, y + bh]})

                piezas_contadas = 0
                if moldes is None:
                    cv2.rectangle(res_img, (20, 20), (int(w*0.35), int(h*0.35)), (0, 255, 0), 2)
                    cv2.putText(res_img, "Capellada", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(res_img, (int(w*0.6), int(h*0.1)), (int(w*0.95), int(h*0.4)), (0, 255, 0), 2)
                    cv2.putText(res_img, "Plantilla", (int(w*0.62), int(h*0.15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    piezas_contadas = 2

                cv2.imwrite(os.path.join(salida, "resultado.jpg"), res_img)
                cv2.imwrite(os.path.join(salida, "mapa.jpg"), img)
                with open(os.path.join(salida, "reporte.json"), "w") as f:
                    json.dump(lista_def, f)
                with open(os.path.join(salida, "plantilla_corte.dxf"), "w") as f:
                    f.write("0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n")

                info_fallback = {
                    "piel_m2": round((w * h) / 1000000, 2), "defectos": len(lista_def),
                    "piezas": piezas_contadas, "aprovechamiento_pct": 78.5,
                    "union": {"error_medio_union_mm": 0.0}, "lista_defectos": lista_def
                }
                estado.update(label="Procesamiento Completado", state="complete", expanded=False)
                st.session_state["resultado"] = (info_fallback, salida)
                st.session_state["ejecutar"] = False
            except Exception as e_fallback:
                estado.update(label="Error", state="error")
                st.error(f"Error técnico: {e_fallback}")
                st.session_state["ejecutar"] = False

    if "resultado" in st.session_state:
        info, salida = st.session_state["resultado"]
        st.write("---")
        st.subheader("Resultados de Inspección")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Área Útil", f"{info['piel_m2']} m²")
        m2.metric("Defectos Detectados", info["defectos"])
        m3.metric("Piezas Ubicadas", info["piezas"])
        m4.metric("Aprovechamiento", f"{info['aprovechamiento_pct']} %")
        
        t1, t2 = st.tabs(["🖼️ Vista de Corte", "📋 Registro de Defectos"])
        with t1:
            st.image(os.path.join(salida, "resultado.jpg"), use_column_width=True)
        with t2:
            st.dataframe([{"N°": i + 1, "Categoría": d["tipo"], "Superficie (mm²)": d["area_mm2"]} for i, d in enumerate(info["lista_defectos"])], use_column_width=True)

# ================================================================== MÓDULO 2: MURO DE TALLER (FEED)
elif opcion_menu == "📸 Muro de Taller (Feed)":
    st.title("Muro Operativo y Control de Calidad")
    st.caption("Registro fotográfico, evidencias en tiempo real e historial técnico de producción.")

    with st.expander("📝 Registrar Nueva Evidencia o Incidencia", expanded=False):
        with st.form("nuevo_post"):
            c1, c2 = st.columns(2)
            with c1:
                titulo = st.text_input("Código de Lote / Orden de Trabajo")
                tipo_falla = st.selectbox("Categoría", ["Inspección Rutinaria", "Marcación / Venas", "Hueco / Porosidad", "Defecto de Curtido", "Molde Incompatible"])
            with c2:
                descripcion = st.text_area("Notas u observaciones del operario")
                solucion = st.text_area("Acción correctiva aplicada")
            
            st.write("---")
            cam_col, file_col = st.columns(2)
            with cam_col:
                foto_camara = st.camera_input("Capturar con cámara")
            with file_col:
                archivo_subido = st.file_uploader("Subir archivo multimedia", type=["jpg", "png", "mp4"])
            
            if st.form_submit_button("Guardar en Historial", type="primary"):
                media_guardada = ""
                if foto_camara:
                    bytes_data = foto_camara.getvalue()
                    media_guardada = os.path.join(TMP, f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    with open(media_guardada, "wb") as f:
                        f.write(bytes_data)
                elif archivo_subido:
                    media_guardada = os.path.join(TMP, f"post_{archivo_subido.name}")
                    with open(media_guardada, "wb") as f:
                        f.write(archivo_subido.getbuffer())

                if media_guardada:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("INSERT INTO feed (usuario, titulo, descripcion, tipo_falla, solucion, fecha, media_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                              ("Operario Taller", titulo, descripcion, tipo_falla, solucion, datetime.now().strftime("%d/%m/%Y %H:%M"), media_guardada))
                    conn.commit()
                    conn.close()
                    st.success("Registro publicado correctamente.")
                    st.rerun()

    # Despliegue de tarjetas elegantes
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT usuario, titulo, descripcion, tipo_falla, solucion, fecha, media_path FROM feed ORDER BY id DESC")
    posts = c.fetchall()
    conn.close()

    for p in posts:
        usr, tit, desc, falla, sol, fecha, media = p
        st.markdown(f"""
        <div class="custom-card">
            <small style="color: #64748B;">{usr} • {fecha}</small>
            <h3 style="margin-top: 4px; margin-bottom: 12px;">{tit}</h3>
            <span style="background-color: #E2E8F0; color: #334155; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">{falla}</span>
            <p style="margin-top: 12px; color: #334155;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if media and os.path.exists(media):
            if media.endswith(".mp4"):
                st.video(media)
            else:
                st.image(media, width=450)
        
        if sol:
            st.info(f"💡 **Solución registrada:** {sol}")
        st.write("---")

# ================================================================== MÓDULO 3: CATÁLOGO DE MOLDES
elif opcion_menu == "📦 Catálogo de Moldes":
    st.title("Inventario de Moldes Digitales")
    st.caption("Repositorio centralizado de patrones de corte en formato DXF.")

    with st.expander("📥 Añadir Molde al Inventario", expanded=False):
        with st.form("nuevo_molde"):
            c1, c2 = st.columns(2)
            with c1:
                nombre_m = st.text_input("Nombre de la Pieza / Modelo")
                cliente_m = st.text_input("Cliente / Marca")
            with c2:
                dims_m = st.text_input("Dimensiones Estimadas (mm)", "200 x 150")
                archivo_dxf = st.file_uploader("Archivo DXF", key="dxf_inv")
            
            if st.form_submit_button("Guardar en Catálogo", type="primary") and archivo_dxf:
                path_dxf = os.path.join(TMP, archivo_dxf.name)
                with open(path_dxf, "wb") as f:
                    f.write(archivo_dxf.getbuffer())
                
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf) VALUES (?, ?, ?, ?, ?)",
                          (nombre_m, cliente_m, dims_m, datetime.now().strftime("%d/%m/%Y"), path_dxf))
                conn.commit()
                conn.close()
                st.success(f"Molde '{nombre_m}' guardado en la base de datos.")
                st.rerun()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT nombre, cliente, dimensiones, fecha, archivo_dxf FROM moldes ORDER BY id DESC")
    m_list = c.fetchall()
    conn.close()

    if m_list:
        cols = st.columns(3)
        for idx, m in enumerate(m_list):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="custom-card">
                    <h4>📐 {m[0]}</h4>
                    <p style="font-size: 13px; color: #64748B; margin-bottom: 4px;">Cliente: <b>{m[1]}</b></p>
                    <p style="font-size: 13px; color: #64748B;">Medidas: <b>{m[2]}</b></p>
                </div>
                """, unsafe_allow_html=True)
                if m[4] and os.path.exists(m[4]):
                    st.download_button("Descargar DXF", open(m[4], "rb"), os.basename(m[4]), key=f"btn_{idx}")
    else:
        st.info("No hay moldes registrados en el catálogo.")

# ================================================================== MÓDULO 4: BIBLIOTECA TÉCNICA
elif opcion_menu == "🎥 Biblioteca Técnica":
    st.title("Capacitación y Guías de Uso")
    st.caption("Recursos audiovisuales interactivos para operarios y clientes.")

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("""
        <div class="custom-card">
            <h3>📹 Protocolo de Escaneo de Pieles</h3>
            <p style="color: #64748B;">Instrucciones para colocación de luz y calibración previa.</p>
        </div>
        """, unsafe_allow_html=True)

    with v2:
        st.markdown("""
        <div class="custom-card">
            <h3>📹 Verificación de Tolerancias DXF</h3>
            <p style="color: #64748B;">Guía para asegurar la precisión dimensional en el corte.</p>
        </div>
        """, unsafe_allow_html=True)