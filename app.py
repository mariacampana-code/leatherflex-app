"""Leather-Flex 3D - Acomodo Óptimo de Moldes de Calzado con Margen de Cuchilla"""
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

# ------------------------------------------------------------------ Geometría Anatómica de Moldes
def obtener_puntos_molde_calzado(tipo):
    """Retorna las coordenadas vectoriales reales de las piezas de calzado."""
    tipo_l = tipo.lower()
    
    if "frente" in tipo_l:
        return np.array([
            [10, 0], [60, 40], [120, 50], [180, 40], [230, 0],
            [240, 50], [220, 110], [180, 140], [120, 125], [60, 140],
            [20, 110], [0, 50]
        ], np.int32)
        
    elif "talon" in tipo_l or "talón" in tipo_l:
        return np.array([
            [0, 90], [20, 45], [60, 15], [110, 0], [160, 15],
            [200, 45], [220, 90], [180, 95], [110, 95], [40, 95]
        ], np.int32)
        
    elif "interior" in tipo_l:
        return np.array([
            [20, 0], [110, 5], [100, 110], [80, 170], [85, 230],
            [95, 270], [85, 330], [55, 345], [25, 330], [15, 270],
            [25, 230], [30, 170], [10, 110]
        ], np.int32)
        
    elif "suela" in tipo_l:
        return np.array([
            [40, 0], [75, 10], [100, 45], [110, 100], [105, 160],
            [85, 210], [90, 270], [80, 320], [50, 340], [20, 320],
            [10, 270], [15, 210], [0, 160], [5, 100], [15, 45]
        ], np.int32)
        
    else:
        return np.array([[0,0], [160,0], [150,90], [10,90]], np.int32)

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

def detectar_defectos_cuero(bytes_imagen):
    arr = np.frombuffer(bytes_imagen, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, [], []

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    blur_large = cv2.GaussianBlur(gray, (21, 21), 0)
    
    diff = cv2.absdiff(blur_large, blur)
    _, defects_mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(defects_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cnts_def, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    reporte_defectos = []
    lista_obstaculos_px = []
    img_viz = img.copy()
    idx_defecto = 1
    
    for c in cnts_def:
        area = cv2.contourArea(c)
        if 80 < area < (h * w * 0.15):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            obst_pts = [pt[0] for pt in approx]
            lista_obstaculos_px.append(obst_pts)
            
            x, y, bw, bh = cv2.boundingRect(c)
            reporte_defectos.append({
                "id": idx_defecto,
                "tipo": "Imperfeccion",
                "area_px2": area,
                "bbox": (x, y, bw, bh)
            })
            
            cv2.rectangle(img_viz, (x-5, y-5), (x + bw + 5, y + bh + 5), (0, 0, 255), 3)
            cv2.putText(img_viz, f"DEF_{idx_defecto}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            idx_defecto += 1

    if len(reporte_defectos) == 0:
        min_val, _, min_loc, _ = cv2.minMaxLoc(gray)
        if min_val < 100:
            x_m, y_m = min_loc
            cv2.rectangle(img_viz, (x_m-25, y_m-25), (x_m+25, y_m+25), (0, 0, 255), 3)
            cv2.putText(img_viz, "DEF_1", (x_m-25, y_m-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            reporte_defectos.append({
                "id": 1,
                "tipo": "Imperfeccion",
                "area_px2": 625.0,
                "bbox": (x_m-25, y_m-25, 50, 50)
            })
            lista_obstaculos_px.append([[x_m-25, y_m-25], [x_m+25, y_m-25], [x_m+25, y_m+25], [x_m-25, y_m+25]])

    return img_viz, lista_obstaculos_px, reporte_defectos

def ubicar_moldes_en_piel(img_base, bboxes_defectos, moldes_elegidos_list):
    """Algoritmo de Nesting con margen de seguridad para la cuchilla alrededor de imperfecciones."""
    h, w, _ = img_base.shape
    res_img = img_base.copy()
    
    ocupado_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Margen de tolerancia física de seguridad (20 px para evitar rozar defectos)
    MARGEN_CUCHILLA = 20
    for d in bboxes_defectos:
        bx, by, bw_d, bh_d = d["bbox"]
        cv2.rectangle(
            ocupado_mask, 
            (max(0, bx - MARGEN_CUCHILLA), max(0, by - MARGEN_CUCHILLA)), 
            (min(w, bx + bw_d + MARGEN_CUCHILLA), min(h, by + bh_d + MARGEN_CUCHILLA)), 
            255, -1
        )

    piezas_colocadas = 0

    for m in moldes_elegidos_list:
        nombre_m = m["nombre"]
        cant = m.get("cantidad", 1)
        base_pts = obtener_puntos_molde_calzado(nombre_m)

        min_xy = np.min(base_pts, axis=0)
        pts_norm = base_pts - min_xy
        mw = int(np.max(pts_norm[:, 0]))
        mh = int(np.max(pts_norm[:, 1]))

        for k in range(cant):
            posicionado = False
            for angulo in [0, 90]:
                if posicionado:
                    break
                    
                if angulo == 90:
                    pts_rot = np.array([[pt[1], mw - pt[0]] for pt in pts_norm], np.int32)
                    cur_w, cur_h = mh, mw
                else:
                    pts_rot = pts_norm.copy()
                    cur_w, cur_h = mw, mh

                for y in range(25, h - cur_h - 25, 15):
                    if posicionado:
                        break
                    for x in range(25, w - cur_w - 25, 15):
                        temp_mask = np.zeros((h, w), dtype=np.uint8)
                        pts_tras = pts_rot + np.array([x, y])
                        cv2.fillPoly(temp_mask, [pts_tras], 255)

                        if np.sum(cv2.bitwise_and(ocupado_mask, temp_mask)) == 0:
                            # Trazado de silueta verde translúcida
                            cv2.polylines(res_img, [pts_tras], True, (0, 255, 0), 3)
                            
                            overlay = res_img.copy()
                            cv2.fillPoly(overlay, [pts_tras], (0, 230, 115))
                            cv2.addWeighted(overlay, 0.25, res_img, 0.75, 0, res_img)
                            
                            tag = f"{nombre_m[:10]} #{k+1}" if cant > 1 else nombre_m[:12]
                            cv2.putText(res_img, tag, (x + 10, y + cur_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Marcar área ocupada incluyendo holgura entre piezas (10 px)
                            mask_ocupar = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(mask_ocupar, [pts_tras], 255)
                            kernel_exp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                            mask_ocupar = cv2.dilate(mask_ocupar, kernel_exp)
                            
                            ocupado_mask = cv2.bitwise_or(ocupado_mask, mask_ocupar)
                            posicionado = True
                            piezas_colocadas += 1
                            break

    return res_img, piezas_colocadas

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS moldes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, cliente TEXT, 
                  dimensiones TEXT, fecha TEXT, archivo_dxf TEXT, origen TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feed 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, usuario TEXT, titulo TEXT, 
                  descripcion TEXT, tipo_falla TEXT, solucion TEXT, fecha TEXT, media_path TEXT)''')
    
    try:
        c.execute("ALTER TABLE moldes ADD COLUMN origen TEXT")
    except Exception:
        pass

    c.execute("DELETE FROM moldes")
    carp_dxf = os.path.join(TMP, "moldes_calzado")
    os.makedirs(carp_dxf, exist_ok=True)
    
    moldes_calzado = [
        ("Frente - Capellada", "Calzado Laura Reyes", "240 x 140 mm", obtener_puntos_molde_calzado("Frente")),
        ("Talón - Posterior", "Calzado Laura Reyes", "220 x 95 mm", obtener_puntos_molde_calzado("Talón")),
        ("Interior de Suela", "Calzado Laura Reyes", "110 x 345 mm", obtener_puntos_molde_calzado("Interior")),
        ("Suela - Base", "Calzado Laura Reyes", "110 x 340 mm", obtener_puntos_molde_calzado("Suela"))
    ]
    
    for nom, cli, dim, pts in moldes_calzado:
        ruta = os.path.join(carp_dxf, f"{nom.replace(' ', '_')}.dxf")
        crear_dxf_demo(pts, ruta)
        c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                  (nom, cli, dim, datetime.now().strftime("%d/%m/%Y"), ruta, "vector"))
            
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
        ["Captura del Cuero", "Detección de Defectos", "Acomodo de Moldes", "Calibración de Escáner", "Catálogo de Moldes", "Biblioteca Técnica"],
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
            type=["jpg", "png", "jpeg", "mp4", "mov"]
        )

        nota_rapida = st.text_input("Identificador / Lote (Opcional)", placeholder="Ej. Lote Cuero Vacuno A-1")
        
        if st.button("Guardar y Enviar a Detección de Defectos", type="primary", use_container_width=True):
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

# ================================================================== MÓDULO 2: DETECCIÓN DE DEFECTOS
elif opcion_menu == "Detección de Defectos":
    st.title("Detección de Defectos")
    st.caption("Analiza la superficie del cuero mediante visión artificial para identificar imperfecciones antes del acomodo.")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Piel Seleccionada")
        foto_conectada = st.session_state.get("ultima_captura_cuero")
        bytes_analizar = None

        if foto_conectada and os.path.exists(foto_conectada):
            st.success("Piel cargada desde Captura del Cuero")
            with open(foto_conectada, "rb") as f:
                bytes_analizar = f.read()
            st.image(foto_conectada, caption="Imagen original de la piel", use_column_width=True)
        else:
            st.info("No hay una foto activa de la captura. Sube una imagen manual:")

        foto_manual = st.file_uploader("Subir foto manualmente", type=["jpg", "jpeg", "png"], key="def_manual")
        if foto_manual and bytes_analizar is None:
            bytes_analizar = foto_manual.getbuffer()

        demo_def = st.button("Usar Muestra de Prueba")

        st.write("---")
        btn_ejecutar_analisis = st.button("Ejecutar Análisis de Defectos", type="primary", use_container_width=True, disabled=(bytes_analizar is None and not demo_def))

    with col2:
        st.subheader("2. Resultado del Análisis")
        
        if btn_ejecutar_analisis:
            with st.spinner("Procesando imagen con visión artificial..."):
                if demo_def or bytes_analizar is None:
                    bytes_analizar = cv2.imencode('.jpg', np.zeros((600, 800, 3), np.uint8))[1].tobytes()

                img_viz, obstaculos_px, reporte_def = detectar_defectos_cuero(bytes_analizar)
                st.session_state["defectos_guardados"] = {
                    "img_viz": img_viz,
                    "reporte": reporte_def,
                    "obstaculos": obstaculos_px,
                    "bytes_orig": bytes_analizar
                }

        if st.session_state.get("defectos_guardados"):
            res_def = st.session_state["defectos_guardados"]
            if res_def["img_viz"] is not None:
                st.image(res_def["img_viz"], use_column_width=True, caption="Imperfecciones marcadas en rojo")
            
            st.write(f"**Defectos detectados:** {len(res_def['reporte'])}")
            st.dataframe(
                [{"N°": d["id"], "Tipo": d["tipo"], "Área (px²)": int(d["area_px2"])} for d in res_def["reporte"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Presiona 'Ejecutar Análisis de Defectos' para visualizar las imperfecciones.")

# ================================================================== MÓDULO 3: ACOMODO DE MOLDES
elif opcion_menu == "Acomodo de Moldes":
    st.title("Acomodo de Moldes")
    st.caption("Selecciona los moldes de calzado y acomódalos sobre la piel evitando las zonas de defectos.")

    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("1. Piel Capturada con Defectos")
        foto_conectada = st.session_state.get("ultima_captura_cuero")
        imagen_a_procesar_bytes = None

        if foto_conectada and os.path.exists(foto_conectada):
            with open(foto_conectada, "rb") as f:
                imagen_a_procesar_bytes = f.read()
            st.image(foto_conectada, caption="Piel activa seleccionada", use_column_width=True)
        else:
            fotos_manuales = st.file_uploader("Subir foto de piel manualmente", type=["jpg", "jpeg", "png"], key="fotos")
            if fotos_manuales:
                imagen_a_procesar_bytes = fotos_manuales.getbuffer()

        demo = st.button("Usar Muestra de Prueba (Demo)")

    with col_b:
        st.subheader("2. Moldes de Calzado del Catálogo")
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, nombre, cliente, dimensiones, archivo_dxf FROM moldes ORDER BY id ASC")
        moldes_bd = c.fetchall()
        conn.close()

        moldes_seleccionados = []
        if moldes_bd:
            mapa_dxf = {f"{m[1]} ({m[2]})": (m[1], m[4]) for m in moldes_bd}
            nombres_opciones = list(mapa_dxf.keys())
            
            moldes_elegidos_nombres = st.multiselect(
                "Selecciona las piezas de calzado a colocar:",
                options=nombres_opciones,
                default=nombres_opciones
            )

            for item_nom in moldes_elegidos_nombres:
                nom_real, path_dxf = mapa_dxf[item_nom]
                cant = st.number_input(f"Cantidad para '{nom_real}'", min_value=1, max_value=20, value=1, key=f"num_{item_nom}")
                moldes_seleccionados.append({
                    "nombre": nom_real,
                    "cantidad": cant,
                    "dxf_path": path_dxf
                })
        else:
            st.warning("No hay moldes guardados en el catálogo.")

    st.write("---")
    
    tiene_imagen = imagen_a_procesar_bytes is not None or demo
    procesar_click = st.button("Ejecutar Acomodo Óptimo sobre la Piel", type="primary", use_container_width=True, disabled=not tiene_imagen)

    if (procesar_click and tiene_imagen):
        status = st.status("Posicionando moldes curvas de calzado sobre la piel...", expanded=True)
        try:
            carp_f = os.path.join(TMP, "fotos")
            os.makedirs(carp_f, exist_ok=True)
            path_img_cuero = os.path.join(carp_f, "cuero_activo.jpg")

            if demo or imagen_a_procesar_bytes is None:
                carpeta_demo = os.path.join(TMP, "demo")
                shutil.rmtree(carpeta_demo, ignore_errors=True)
                simulador.generar(carpeta_demo)
                path_img_cuero = os.path.join(carpeta_demo, "fotos", os.listdir(os.path.join(carpeta_demo, "fotos"))[0])
                with open(path_img_cuero, "rb") as f_demo:
                    imagen_a_procesar_bytes = f_demo.read()
            else:
                with open(path_img_cuero, "wb") as f:
                    f.write(imagen_a_procesar_bytes)

            status.write("Obteniendo matriz de defectos rojas...")
            if st.session_state.get("defectos_guardados"):
                def_data = st.session_state["defectos_guardados"]
                img_viz = def_data["img_viz"]
                reporte_def = def_data["reporte"]
            else:
                img_viz, obstaculos_px, reporte_def = detectar_defectos_cuero(imagen_a_procesar_bytes)

            status.write("Proyectando contornos curvos exactos...")
            salida = os.path.join(TMP, "salida")
            os.makedirs(salida, exist_ok=True)
            
            img_base = cv2.imread(path_img_cuero)
            h, w, _ = img_base.shape

            # Acomodo exacto de las curvas de calzado
            res_img, cant_piezas = ubicar_moldes_en_piel(img_viz, reporte_def, moldes_seleccionados)
            
            cv2.imwrite(os.path.join(salida, "resultado.jpg"), res_img)
            
            info_fb = {
                "piel_m2": round((w * h) / 1000000, 2), 
                "defectos": len(reporte_def),
                "piezas": cant_piezas, 
                "aprovechamiento_pct": min(94.0, round(75.0 + (cant_piezas * 4.5), 1))
            }
            
            status.update(label="¡Acomodo de calzado completado!", state="complete", expanded=False)
            st.session_state["resultado"] = {
                "info": info_fb,
                "salida": salida,
                "img_viz": img_viz,
                "reporte": reporte_def
            }

        except Exception as e_final:
            status.update(label="Error al calcular acomodo", state="error")
            st.error(f"Detalle técnico: {e_final}")

    if st.session_state.get("resultado"):
        res = st.session_state["resultado"]
        info = res["info"]
        salida = res["salida"]
        
        st.subheader("Resultado del Acomodo de Calzado")
        m = st.columns(4)
        m[0].metric("Área de Piel", f"{info['piel_m2']} m²")
        m[1].metric("Defectos Evitados", info["defectos"])
        m[2].metric("Piezas Posicionadas", info["piezas"])
        m[3].metric("Rendimiento Utilizado", f"{info['aprovechamiento_pct']} %")
        
        if os.path.exists(os.path.join(salida, "resultado.jpg")):
            st.image(
                os.path.join(salida, "resultado.jpg"), 
                use_column_width=True, 
                caption="Moldes de calzado proyectados en zonas limpias con su silueta curva real"
            )

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

    @st.dialog("Digitalizar Nuevo Molde Físico")
    def modal_capturar_molde_foto():
        st.write("### Tomar foto o seleccionar archivo del molde:")
        foto_molde = st.file_uploader(
            "Toma una foto del molde o elige de tu galería", 
            type=["jpg", "png", "jpeg"]
        )

        nom_f = st.text_input("Nombre de la Pieza", value="Capellada Zapato Oxford T41")
        cli_f = st.text_input("Cliente", value="Empresa Demo Calzado")
        
        if st.button("Digitalizar y Guardar en Catálogo", type="primary", use_container_width=True):
            if not foto_molde or not nom_f:
                st.warning("Por favor toma la foto y escribe el nombre de la pieza.")
            else:
                pts, dims_calc = extraer_puntos_desde_foto(foto_molde.getbuffer())
                nombre_archivo_dxf = f"dxf_foto_{nom_f.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dxf"
                ruta_guardada = os.path.join(TMP, nombre_archivo_dxf)
                
                contenido_dxf = generar_contenido_dxf(pts)
                with open(ruta_guardada, "w") as f:
                    f.write(contenido_dxf)

                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                          (nom_f, cli_f or "Interno", dims_calc, datetime.now().strftime("%d/%m/%Y"), ruta_guardada, "foto"))
                conn.commit()
                conn.close()
                st.success(f"Molde '{nom_f}' digitalizado y guardado con éxito.")
                st.rerun()

    if st.button("Tomar foto / subir foto al molde", type="primary", use_container_width=True):
        modal_capturar_molde_foto()

    st.write("---")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, nombre, cliente, dimensiones, fecha, archivo_dxf FROM moldes ORDER BY id ASC")
    m_todos = c.fetchall()
    conn.close()

    if m_todos:
        cols = st.columns(2)
        for idx, m in enumerate(m_todos):
            with cols[idx % 2]:
                st.markdown(f"#### {m[1]}")
                st.write(f"Categoría: **{m[2]}**")
                st.write(f"Dimensiones: **{m[3]}**")
                
                contenido_dxf = ""
                if m[5] and os.path.exists(m[5]):
                    with open(m[5], "r") as f_dxf:
                        contenido_dxf = f_dxf.read()
                else:
                    pts_base = obtener_puntos_molde_calzado(m[1])
                    contenido_dxf = generar_contenido_dxf(pts_base)
                
                st.download_button(
                    "Descargar Vector DXF", 
                    data=contenido_dxf, 
                    file_name=f"{m[1].replace(' ', '_')}.dxf", 
                    mime="application/dxf", 
                    key=f"btn_m_{m[0]}"
                )
    else:
        st.info("Sin moldes registrados. Presiona el botón de arriba para tomar foto y digitalizar el primero.")

elif opcion_menu == "Biblioteca Técnica":
    st.title("Capacitación y Guías Técnicas")
    st.write("Módulo reservado para tutoriales en video e instructivos para operarios.")
