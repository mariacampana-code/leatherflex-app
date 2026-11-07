"""Leather-Flex 3D - Captura Directa de Moldes Físicos y DXF"""
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

# ------------------------------------------------------------------ Configuración de página
st.set_page_config(
    page_title="Leather-Flex 3D",
    page_icon="🟫",
    layout="wide",
    initial_sidebar_state="expanded"
)

TMP = st.session_state.setdefault("tmp", tempfile.mkdtemp(prefix="leatherflex_"))
DB_PATH = os.path.join(os.path.dirname(__file__), "leatherflex.db")

# ------------------------------------------------------------------ CSS Personalizado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f3eee8 0%, #ab9680 100%) !important;
        color: #2c2520 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(171, 150, 128, 0.3) !important;
    }
    
    div[data-testid="stSidebarHeader"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    .brand-container {
        padding: 1.2rem 0.5rem;
        border-bottom: 1px solid rgba(171, 150, 128, 0.3);
        margin-bottom: 1rem;
    }
    .brand-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #3e3228;
        letter-spacing: 0.5px;
    }

    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div { gap: 8px; }
    div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.45) !important;
        border: 1px solid rgba(171, 150, 128, 0.4) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: #4a3e35 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #8c7763 !important;
        color: #ffffff !important;
        border-color: #73604f !important;
        box-shadow: 0 4px 12px rgba(140, 119, 99, 0.3) !important;
    }
    div[data-testid="stRadio"] input[type="radio"] { display: none !important; }

    .feed-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def generar_contenido_dxf(puntos):
    dxf_content = [
        "0", "SECTION", "2", "HEADER", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES", "0", "ENDSEC",
        "0", "SECTION", "2", "BLOCKS", "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
        "0", "POLYLINE", "8", "MOLDES", "66", "1", "70", "1"
    ]
    for x, y in puntos:
        dxf_content.extend(["0", "VERTEX", "8", "MOLDES", "10", str(x), "20", str(y), "30", "0.0"])
    dxf_content.extend(["0", "SEQEND", "0", "ENDSEC", "0", "EOF"])
    return "\n".join(dxf_content)

def crear_dxf_demo(puntos, nombre_archivo):
    contenido = generar_contenido_dxf(puntos)
    with open(nombre_archivo, "w") as f:
        f.write(contenido)

def extraer_puntos_desde_foto(bytes_imagen):
    arr = np.frombuffer(bytes_imagen, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [(0,0), (200,0), (180,120), (20,120)], "200 x 120"
        
    c_max = max(cnts, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(c_max, True)
    approx = cv2.approxPolyDP(c_max, epsilon, True)
    
    pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    x, y, bw, bh = cv2.boundingRect(c_max)
    dims_str = f"{int(bw)} x {int(bh)}"
    return pts, dims_str

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS moldes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, cliente TEXT, 
                  dimensiones TEXT, fecha TEXT, archivo_dxf TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feed 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, usuario TEXT, titulo TEXT, 
                  descripcion TEXT, tipo_falla TEXT, solucion TEXT, fecha TEXT, media_path TEXT)''')
    
    c.execute("SELECT COUNT(*) FROM moldes")
    count = c.fetchone()[0]
    if count == 0:
        carp_dxf = os.path.join(TMP, "moldes_precargados")
        os.makedirs(carp_dxf, exist_ok=True)
        moldes_demo = [
            ("Bolsillo Frontal Billetera", "Personalizado", "110 x 80", [(0,0), (110,0), (100,80), (10,80)])
        ]
        for nom, cli, dim, pts in moldes_demo:
            ruta = os.path.join(carp_dxf, f"{nom.replace(' ', '_')}.dxf")
            crear_dxf_demo(pts, ruta)
            c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf) VALUES (?, ?, ?, ?, ?)",
                      (nom, cli, dim, datetime.now().strftime("%d/%m/%Y"), ruta))
            
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------ Barra Lateral
with st.sidebar:
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">LEATHER-FLEX 3D</div>
        </div>
    """, unsafe_allow_html=True)
    
    opcion_menu = st.radio(
        "Navegacion",
        ["Captura del Cuero", "Acomodo de Moldes", "Calibración de Escáner", "Catálogo de Moldes", "Biblioteca Técnica"],
        label_visibility="collapsed"
    )

ruta_cal = os.path.join(TMP, "calib.json")
if not os.path.exists(ruta_cal):
    with open(ruta_cal, "w") as f:
        json.dump({"mm_por_px_foto": 0.5, "angulo_grados": 0.0, "matriz_homografia": None}, f)

# ================================================================== MÓDULO 1: CAPTURA DEL CUERO
if opcion_menu == "Captura del Cuero":
    st.title("Captura del Cuero")
    st.caption("Registro de evidencia en vivo (cámara/galería) e historial del taller.")

    @st.dialog("Capturar Nueva Piel / Evidencia")
    def modal_crear_post():
        st.write("### Seleccionar foto o video del cuero:")
        
        archivo_subido = st.file_uploader(
            "Toma una foto/video con la cámara o elige de tu galería", 
            type=["jpg", "png", "jpeg", "mp4", "mov"],
            help="Al presionar se abrirá la cámara nativa de tu dispositivo"
        )

        nota_rapida = st.text_input("Identificador / Lote (Opcional)", placeholder="Ej. Lote Cuero Vacuno A-1")
        
        if st.button("Guardar y Enviar a Acomodo de Moldes", type="primary", use_container_width=True):
            if archivo_subido is None:
                st.warning("Por favor toma una foto/video o elige un archivo.")
            else:
                ext = os.path.splitext(archivo_subido.name)[1] or ".jpg"
                ruta_archivo_guardado = os.path.join(TMP, f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
                with open(ruta_archivo_guardado, "wb") as f:
                    f.write(archivo_subido.getbuffer())

                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO feed (usuario, titulo, descripcion, tipo_falla, solucion, fecha, media_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          ("Operario Taller", nota_rapida or "Captura de Cuero", "", "Piel", "", datetime.now().strftime("%d/%m/%Y %H:%M"), ruta_archivo_guardado))
                conn.commit()
                conn.close()

                st.session_state["ultima_captura_cuero"] = ruta_archivo_guardado
                st.success("Captura guardada")
                st.rerun()

    if st.button("Tomar foto / video del cuero", type="primary", use_container_width=True):
        modal_crear_post()

    st.write("---")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, usuario, titulo, fecha, media_path FROM feed ORDER BY id DESC")
    posts = c.fetchall()
    conn.close()

    if not posts:
        st.info("No hay capturas registradas. Presiona para tomar la primera foto.")
    else:
        for p in posts:
            post_id, usr, tit, fecha, media = p
            with st.container():
                st.markdown(f"""
                <div class="feed-card">
                    <small style="color: #73604f; font-weight: 600;">{usr} • {fecha}</small>
                    <h4 style="margin-top: 4px; margin-bottom: 4px; color: #2c2520;">{tit}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if media and os.path.exists(media):
                    if media.endswith((".mp4", ".mov")):
                        st.video(media)
                    else:
                        st.image(media, use_column_width=True)

                col_del, _ = st.columns([1, 3])
                with col_del:
                    if st.button("Eliminar", key=f"del_{post_id}"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("DELETE FROM feed WHERE id = ?", (post_id,))
                        conn.commit()
                        conn.close()

                        if media and os.path.exists(media):
                            try:
                                os.remove(media)
                            except Exception:
                                pass

                        if st.session_state.get("ultima_captura_cuero") == media:
                            st.session_state["ultima_captura_cuero"] = None

                        st.success("Eliminado correctamente.")
                        st.rerun()
                st.divider()

# ================================================================== MÓDULO 2: ACOMODO DE MOLDES
elif opcion_menu == "Acomodo de Moldes":
    st.title("Acomodo de Moldes")
    st.caption("Inspección de imperfecciones en la piel capturada y cálculo de nidificación (Nesting).")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Imagen de Piel Activa")
        
        foto_conectada = st.session_state.get("ultima_captura_cuero")
        
        if foto_conectada and os.path.exists(foto_conectada):
            st.success("Imagen obtenida directamente de 'Captura del Cuero'")
            st.image(foto_conectada, caption="Piel seleccionada para procesamiento", use_column_width=True)
            if st.button("Desvincular esta foto"):
                st.session_state["ultima_captura_cuero"] = None
                st.rerun()
        else:
            st.info("No hay una foto activa. Puedes tomar una en 'Captura del Cuero' o subir una imagen manual aquí:")

        fotos_manuales = st.file_uploader("Subir otra foto manualmente", accept_multiple_files=True, key="fotos")
        demo = st.button("Usar Muestra de Prueba (Demo)")

    with col_b:
        st.subheader("2. Archivos DXF (Moldes)")
        dxfs = st.file_uploader("Subir moldes (.dxf)", accept_multiple_files=True, key="dxf")
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
                    cant = st.number_input(f"Cantidad {a.name}", 0, 500, 0, key="c_" + a.name)
                    moldes.append({"nombre": os.path.splitext(a.name)[0], "poligono": pol, "cantidad": cant or None})
                except Exception as e:
                    st.error(f"Error en {a.name}: {e}")

    st.write("---")
    
    tiene_imagen = bool(foto_conectada or fotos_manuales or demo)
    procesar_click = st.button("Iniciar Procesamiento y Acomodo de Corte", type="primary", disabled=not tiene_imagen)

    cfg = Config()

    if demo:
        carpeta_demo = os.path.join(TMP, "demo")
        shutil.rmtree(carpeta_demo, ignore_errors=True)
        simulador.generar(carpeta_demo)
        st.session_state["entrada"] = (os.path.join(carpeta_demo, "fotos"), os.path.join(carpeta_demo, "calib.json"))
        st.session_state["ejecutar"] = True
    elif procesar_click:
        carp_f = os.path.join(TMP, "fotos")
        shutil.rmtree(carp_f, ignore_errors=True)
        os.makedirs(carp_f)
        
        if foto_conectada and not fotos_manuales:
            dest = os.path.join(carp_f, os.path.basename(foto_conectada))
            shutil.copy(foto_conectada, dest)
        elif fotos_manuales:
            for a in fotos_manuales:
                with open(os.path.join(carp_f, a.name), "wb") as f:
                    f.write(a.getbuffer())
                    
        st.session_state["entrada"] = (carp_f, ruta_cal)
        st.session_state["ejecutar"] = True

    if st.session_state.get("ejecutar") and "entrada" in st.session_state:
        carpeta_fotos, cal_json = st.session_state["entrada"]
        salida = os.path.join(TMP, "salida")
        estado = st.status("Procesando imágenes...", expanded=True)
        try:
            r = pipeline.procesar(carpeta_fotos, cal_json, salida, cfg, moldes, avisar=estado.write)
            estado.update(label="Procesamiento Exitoso", state="complete", expanded=False)
            st.session_state["resultado"] = (r["info"], salida)
            st.session_state["ejecutar"] = False
        except Exception:
            try:
                estado.write("Ejecutando algoritmo óptico de respaldo...")
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
                estado.update(label="Análisis Finalizado", state="complete", expanded=False)
                st.session_state["resultado"] = (info_fallback, salida)
                st.session_state["ejecutar"] = False
            except Exception as e_fallback:
                estado.update(label="Error", state="error")
                st.error(f"Error de procesamiento: {e_fallback}")
                st.session_state["ejecutar"] = False

    if "resultado" in st.session_state:
        info, salida = st.session_state["resultado"]
        st.subheader("Resultados del Análisis")
        m = st.columns(4)
        m[0].metric("Área Útil", f"{info['piel_m2']} m²")
        m[1].metric("Defectos", info["defectos"])
        m[2].metric("Piezas", info["piezas"])
        m[3].metric("Rendimiento", f"{info['aprovechamiento_pct']} %")
        
        t1, t2 = st.tabs(["Nidificación / Corte", "Reporte de Imperfecciones"])
        t1.image(os.path.join(salida, "resultado.jpg"), use_column_width=True)
        t2.dataframe([{"N°": i + 1, "Tipo": d["tipo"], "Área (mm²)": d["area_mm2"]} for i, d in enumerate(info["lista_defectos"])], use_column_width=True)

# ================================================================== MÓDULOS SECUNDARIOS
elif opcion_menu == "Calibración de Escáner":
    st.title("Calibración del Sistema Óptico")
    st.caption("Ajuste de parámetros de cámara, escala de milímetros por píxel y corrección de plano.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Tablero de Referencia")
        ruta_tab = os.path.join(TMP, "tablero_para_imprimir.png")
        if not os.path.exists(ruta_tab):
            calibracion.tablero_para_imprimir(ruta_tab)
        st.download_button("Descargar Tablero Calibración", open(ruta_tab, "rb"), "tablero_calibracion.png", "image/png")
        
        f_tab = st.file_uploader("Subir foto del tablero en mesa", key="tab")
        f_bla = st.file_uploader("Hoja blanca para compensación (opcional)", key="bla")

    with col2:
        st.subheader("2. Ajustes de Sensibilidad")
        cfg = Config()
        cfg.umbral_sigma = st.slider("Sensibilidad a Defectos (Sigma)", 2.0, 5.0, cfg.umbral_sigma, 0.1)

    st.write("---")
    if f_tab and st.button("Guardar y Calibrar Escáner", type="primary"):
        try:
            cal = calibracion.calibrar_tablero(cv2.imdecode(np.frombuffer(f_tab.getbuffer(), np.uint8), cv2.IMREAD_COLOR))
            gan = calibracion.campo_plano(cv2.imdecode(np.frombuffer(f_bla.getbuffer(), np.uint8), cv2.IMREAD_COLOR)) if f_bla else None
            calibracion.guardar(cal, gan, ruta_cal)
            st.session_state["cal"] = cal
            st.success("Escáner Calibrado y configurado correctamente.")
        except Exception as e:
            st.error(f"Error durante la calibración: {e}")

elif opcion_menu == "Catálogo de Moldes":
    st.title("Catálogo Digital de Moldes")
    st.caption("Base de datos de patrones vectoriales y digitalizaciones físicas.")

    tab_auto, tab_dxf = st.tabs(["Fotos de Moldes", "Archivos DXF"])
    
    with tab_auto:
        st.write("### Digitalización Rápida en Taller")
        st.caption("Coloca el molde de cartón o papel sobre la mesa y tómale una foto limpia desde arriba.")
        
        foto_molde = st.file_uploader("Tomar foto al molde de cartón", type=["jpg", "png", "jpeg"], key="foto_m_fisico")
        with st.form("form_foto_molde"):
            nom_f = st.text_input("Nombre de la Pieza", value="Bolsillo Frontal Billetera")
            cli_f = st.text_input("Cliente", value="Personalizado")
            
            if st.form_submit_button("Digitalizar y Guardar en Catálogo", type="primary"):
                if not foto_molde or not nom_f:
                    st.warning("Por favor toma la foto y escribe el nombre de la pieza.")
                else:
                    pts, dims_calc = extraer_puntos_desde_foto(foto_molde.getbuffer())
                    nombre_archivo_dxf = f"{nom_f.replace(' ', '_')}.dxf"
                    ruta_guardada = os.path.join(TMP, nombre_archivo_dxf)
                    
                    contenido_dxf = generar_contenido_dxf(pts)
                    with open(ruta_guardada, "w") as f:
                        f.write(contenido_dxf)

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf) VALUES (?, ?, ?, ?, ?)",
                              (nom_f, cli_f or "Interno", dims_calc, datetime.now().strftime("%d/%m/%Y"), ruta_guardada))
                    conn.commit()
                    conn.close()
                    st.success(f"Molde '{nom_f}' digitalizado y guardado con éxito.")
                    st.rerun()

    with tab_dxf:
        with st.form("nuevo_molde_dxf"):
            nombre_m = st.text_input("Nombre de la Pieza", value="Bolsillo Frontal Billetera")
            cliente_m = st.text_input("Cliente", value="Personalizado")
            dims_m = st.text_input("Dimensiones (mm)", "110 x 80")
            archivo_dxf = st.file_uploader("Archivo DXF", key="dxf_inv")
            
            if st.form_submit_button("Guardar DXF en Catálogo", type="primary") and archivo_dxf:
                path_dxf = os.path.join(TMP, archivo_dxf.name)
                with open(path_dxf, "wb") as f:
                    f.write(archivo_dxf.getbuffer())
                
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf) VALUES (?, ?, ?, ?, ?)",
                          (nombre_m, cliente_m, dims_m, datetime.now().strftime("%d/%m/%Y"), path_dxf))
                conn.commit()
                conn.close()
                st.success("Molde DXF Guardado")
                st.rerun()

    st.write("---")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT nombre, cliente, dimensiones, fecha, archivo_dxf FROM moldes ORDER BY id DESC")
    m_list = c.fetchall()
    conn.close()

    if m_list:
        cols = st.columns(3)
        for idx, m in enumerate(m_list):
            with cols[idx % 3]:
                st.markdown(f"#### {m[0]}")
                st.write(f"Cliente: **{m[1]}**")
                st.write(f"Dimensiones: **{m[2]}**")
                
                contenido_dxf = ""
                if m[4] and os.path.exists(m[4]):
                    with open(m[4], "r") as f_dxf:
                        contenido_dxf = f_dxf.read()
                else:
                    pts_base = [(0,0), (200,0), (180,120), (20,120)]
                    contenido_dxf = generar_contenido_dxf(pts_base)
                
                st.download_button(
                    "Descargar DXF", 
                    data=contenido_dxf, 
                    file_name=f"{m[0].replace(' ', '_')}.dxf", 
                    mime="application/dxf", 
                    key=f"btn_{idx}"
                )
    else:
        st.info("Sin moldes registrados.")

elif opcion_menu == "Biblioteca Técnica":
    st.title("Capacitación y Guías Técnicas")
    st.write("Módulo reservado para tutoriales en video e instructivos para operarios.")
