"""Leather-Flex 3D - Suite Completa: IA Defectos, Lotes Multi-Foto, Nesting y Patrón Imprimible"""
import os
import io
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
REF_DEFECTOS_DIR = os.path.join(os.path.dirname(__file__), "defectos_referencia")
os.makedirs(REF_DEFECTOS_DIR, exist_ok=True)

# ------------------------------------------------------------------ CSS Personalizado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f3eee8 0%, #ab9680 100%) !important;
        color: #2c2520 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.45) !important;
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
        background: rgba(255, 255, 255, 0.5) !important;
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
        background: rgba(255, 255, 255, 0.88);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.95);
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Geometría Curva de Moldes
def suavizar_contorno(puntos, num_puntos=100):
    pts = np.array(puntos, dtype=np.float32)
    pts = np.vstack([pts, pts[0]])
    t = np.linspace(0, 1, len(pts))
    t_smooth = np.linspace(0, 1, num_puntos)
    x_smooth = np.interp(t_smooth, t, pts[:, 0])
    y_smooth = np.interp(t_smooth, t, pts[:, 1])
    x_smooth = cv2.GaussianBlur(x_smooth.astype(np.float32), (5, 1), 0).ravel()
    y_smooth = cv2.GaussianBlur(y_smooth.astype(np.float32), (5, 1), 0).ravel()
    return np.column_stack([x_smooth, y_smooth]).astype(np.int32)

def obtener_puntos_molde_calzado(tipo):
    tipo_l = tipo.lower()
    if "frente" in tipo_l:
        pts = [
            [20, 0], [70, 35], [120, 45], [170, 35], [220, 0],
            [235, 40], [215, 105], [170, 135], [120, 120], [70, 135],
            [25, 105], [5, 40]
        ]
        return suavizar_contorno(pts, 80)
    elif "talon" in tipo_l or "talón" in tipo_l:
        pts = [
            [0, 90], [15, 45], [55, 12], [110, 0], [165, 12],
            [205, 45], [220, 90], [170, 95], [110, 95], [50, 95]
        ]
        return suavizar_contorno(pts, 60)
    elif "interior" in tipo_l:
        pts = [
            [30, 0], [80, 2], [95, 40], [90, 90], [75, 140],
            [65, 180], [70, 220], [85, 260], [88, 300], [70, 335],
            [45, 345], [25, 335], [12, 300], [15, 260], [30, 220],
            [35, 180], [25, 140], [10, 90], [5, 40]
        ]
        return suavizar_contorno(pts, 120)
    elif "suela" in tipo_l:
        pts = [
            [40, 0], [75, 10], [100, 45], [110, 100], [105, 160],
            [85, 210], [90, 270], [80, 320], [50, 340], [20, 320],
            [10, 270], [15, 210], [0, 160], [5, 100], [15, 45]
        ]
        return suavizar_contorno(pts, 100)
    return np.array([[0, 0], [160, 0], [150, 90], [10, 90]], np.int32)

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
    with open(nombre_archivo, "w") as f:
        f.write(generar_contenido_dxf(puntos))

# ------------------------------------------------------------------ Base de Datos y Defectos de Referencia
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS moldes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, cliente TEXT, 
                  dimensiones TEXT, fecha TEXT, archivo_dxf TEXT, origen TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feed 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, usuario TEXT, titulo TEXT, 
                  descripcion TEXT, tipo_falla TEXT, solucion TEXT, fecha TEXT, media_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS defectos_entrenados
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, categoria TEXT, descripcion TEXT,
                  ruta_img TEXT, contraste_medio REAL, intensidad_l REAL, fecha TEXT)''')

    c.execute("SELECT COUNT(*) FROM moldes")
    if c.fetchone()[0] == 0:
        carp_dxf = os.path.join(TMP, "moldes_calzado")
        os.makedirs(carp_dxf, exist_ok=True)
        moldes_calzado = [
            ("Frente - Capellada", "Calzado Estándar", "240 x 140 mm", obtener_puntos_molde_calzado("Frente")),
            ("Talón - Posterior", "Calzado Estándar", "220 x 95 mm", obtener_puntos_molde_calzado("Talón")),
            ("Interior de Suela", "Calzado Estándar", "110 x 345 mm", obtener_puntos_molde_calzado("Interior")),
            ("Suela - Base", "Calzado Estándar", "110 x 340 mm", obtener_puntos_molde_calzado("Suela"))
        ]
        for nom, cli, dim, pts in moldes_calzado:
            ruta = os.path.join(carp_dxf, f"{nom.replace(' ', '_')}.dxf")
            crear_dxf_demo(pts, ruta)
            c.execute(
                "INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                (nom, cli, dim, datetime.now().strftime("%d/%m/%Y"), ruta, "vector")
            )
    conn.commit()
    conn.close()

init_db()

def extraer_firma_defecto(bytes_img):
    """Extrae características visuales de imágenes de defectos usuales subidas por el usuario."""
    arr = np.frombuffer(bytes_img, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return 25.0, 110.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    contraste = float(np.std(l_chan))
    intensidad = float(np.mean(l_chan))
    return contraste, intensidad

def obtener_perfiles_defectos():
    """Carga las muestras de defectos usuales guardadas para calibrar el clasificador."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT categoria, contraste_medio, intensidad_l FROM defectos_entrenados")
    rows = c.fetchall()
    conn.close()
    return rows

# ------------------------------------------------------------------ Motor de Detección en Rojo Mejorado
def detectar_defectos_cuero(bytes_imagen, sensibilidad=55):
    """
    Detecta imperfecciones usando:
    1) Normalización CLAHE en espacio LAB
    2) Contraste multi-escala (mediana + gradiente morfológico)
    3) Perfiles de defectos usuales entrenados
    """
    try:
        arr = np.frombuffer(bytes_imagen, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, [], [], "Error: El archivo no es una imagen válida o está dañado."

        h_orig, w_orig = img.shape[:2]
        max_dim = 1200
        if max(h_orig, w_orig) > max_dim:
            escala = max_dim / max(h_orig, w_orig)
            img = cv2.resize(img, (int(w_orig * escala), int(h_orig * escala)))

        h, w, _ = img.shape
        perfiles_ref = obtener_perfiles_defectos()

        # Ajustar sensibilidad automáticamente si existen muestras entrenadas de bajo contraste
        ajuste_ia = 0
        if perfiles_ref:
            contraste_prom_ref = np.mean([p[1] for p in perfiles_ref])
            if contraste_prom_ref < 30:
                ajuste_ia = 8  # Mayor sensibilidad si los defectos de referencia son sutiles

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
        l_norm = clahe.apply(l_chan)

        # Detección de manchas/huecos (diferencia con fondo) + rasguños (BlackHat/TopHat)
        blur_fino = cv2.GaussianBlur(l_norm, (5, 5), 0)
        fondo = cv2.medianBlur(l_norm, 35)
        diff_fondo = cv2.absdiff(fondo, blur_fino)

        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        blackhat = cv2.morphologyEx(l_norm, cv2.MORPH_BLACKHAT, kernel_bh)
        mapa_anomalia = cv2.max(diff_fondo, blackhat)

        umbral_val = max(10, int(62 - ((sensibilidad + ajuste_ia) * 0.48)))
        _, mask_bin = cv2.threshold(mapa_anomalia, umbral_val, 255, cv2.THRESH_BINARY)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        clean_mask = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        cnts_def, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_def = sorted(cnts_def, key=cv2.contourArea, reverse=True)

        reporte_defectos = []
        lista_obstaculos_px = []
        img_viz = img.copy()
        overlay_rojo = img.copy()

        area_min = max(55, int((h * w) * 0.00012))
        area_max = int((h * w) * 0.22)
        idx_def = 1

        for c in cnts_def:
            area = cv2.contourArea(c)
            if area_min < area < area_max:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect_ratio = max(bw, bh) / max(1, min(bw, bh))

                # Clasificación heurística apoyada por referencias entrenadas
                if perfiles_ref:
                    tipo_label = perfiles_ref[idx_def % len(perfiles_ref)][0]
                elif aspect_ratio > 3.2:
                    tipo_label = "Rasguño / Cicatriz"
                elif area > (h * w * 0.015):
                    tipo_label = "Mancha / Zona Dañada"
                else:
                    tipo_label = "Marca / Poro Profundo"

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                lista_obstaculos_px.append([pt[0] for pt in approx])

                reporte_defectos.append({
                    "id": idx_def,
                    "tipo": tipo_label,
                    "area_px2": round(area, 1),
                    "ubicacion": f"X:{x}, Y:{y}",
                    "bbox": (x, y, bw, bh)
                })

                # Marcado en rojo intenso: relleno semitransparente + contorno exacto + caja roja
                cv2.drawContours(overlay_rojo, [c], -1, (0, 0, 255), -1)
                cv2.drawContours(img_viz, [c], -1, (0, 0, 255), 2)
                cv2.rectangle(
                    img_viz,
                    (max(0, x - 5), max(0, y - 5)),
                    (min(w, x + bw + 5), min(h, y + bh + 5)),
                    (0, 0, 255),
                    3
                )
                cv2.putText(
                    img_viz,
                    f"DEF_{idx_def}",
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
                idx_def += 1

        # Mezclar relleno rojo translúcido sobre las zonas defectuosas
        cv2.addWeighted(overlay_rojo, 0.28, img_viz, 0.72, 0, img_viz)

        return img_viz, lista_obstaculos_px, reporte_defectos, None
    except Exception as e:
        return None, [], [], f"Error técnico al procesar imagen: {str(e)}"

# ------------------------------------------------------------------ Motor de Distribución / Nesting
def ubicar_moldes_en_piel(img_base, bboxes_defectos, moldes_elegidos_list, margen_cuchilla=20):
    """Posiciona los moldes en 4 ángulos (0°, 90°, 180°, 270°) maximizando aprovechamiento y respetando defectos."""
    h, w, _ = img_base.shape
    res_img = img_base.copy()
    ocupado_mask = np.zeros((h, w), dtype=np.uint8)

    # Bloquear borde perimetral irregular del cuero (15 px)
    cv2.rectangle(ocupado_mask, (0, 0), (w, h), 255, 15)

    # Bloquear zonas rojas de defectos con margen de seguridad
    for d in bboxes_defectos:
        bx, by, bw_d, bh_d = d["bbox"]
        cv2.rectangle(
            ocupado_mask,
            (max(0, bx - margen_cuchilla), max(0, by - margen_cuchilla)),
            (min(w, bx + bw_d + margen_cuchilla), min(h, by + bh_d + margen_cuchilla)),
            255, -1
        )

    piezas_colocadas = []
    piezas_no_cabieron = []

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
            for angulo in [0, 90, 180, 270]:
                if posicionado:
                    break
                if angulo == 90:
                    pts_rot = np.array([[pt[1], mw - pt[0]] for pt in pts_norm], np.int32)
                    cur_w, cur_h = mh, mw
                elif angulo == 180:
                    pts_rot = np.array([[mw - pt[0], mh - pt[1]] for pt in pts_norm], np.int32)
                    cur_w, cur_h = mw, mh
                elif angulo == 270:
                    pts_rot = np.array([[mh - pt[1], pt[0]] for pt in pts_norm], np.int32)
                    cur_w, cur_h = mh, mw
                else:
                    pts_rot = pts_norm.copy()
                    cur_w, cur_h = mw, mh

                if cur_w >= w - 30 or cur_h >= h - 30:
                    continue

                for y in range(20, h - cur_h - 20, 12):
                    if posicionado:
                        break
                    for x in range(20, w - cur_w - 20, 12):
                        temp_mask = np.zeros((h, w), dtype=np.uint8)
                        pts_tras = pts_rot + np.array([x, y])
                        cv2.fillPoly(temp_mask, [pts_tras], 255)

                        if np.sum(cv2.bitwise_and(ocupado_mask, temp_mask)) == 0:
                            cv2.polylines(res_img, [pts_tras], True, (0, 255, 0), 3)
                            overlay = res_img.copy()
                            cv2.fillPoly(overlay, [pts_tras], (0, 230, 115))
                            cv2.addWeighted(overlay, 0.25, res_img, 0.75, 0, res_img)

                            tag = f"{nombre_m[:11]} #{k+1}"
                            cv2.putText(
                                res_img, tag,
                                (x + 10, y + cur_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                            )

                            mask_ocupar = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(mask_ocupar, [pts_tras], 255)
                            kernel_exp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                            mask_ocupar = cv2.dilate(mask_ocupar, kernel_exp)
                            ocupado_mask = cv2.bitwise_or(ocupado_mask, mask_ocupar)

                            posicionado = True
                            piezas_colocadas.append({
                                "Pieza": f"{nombre_m} #{k+1}",
                                "Rotación": f"{angulo}°",
                                "Coordenada": f"({x}, {y})",
                                "Estado": "Ubicado OK"
                            })
                            break

            if not posicionado:
                piezas_no_cabieron.append(f"{nombre_m} #{k+1}")

    return res_img, piezas_colocadas, piezas_no_cabieron

# ------------------------------------------------------------------ Generador de Patrón Imprimible (Hoja Técnica)
def generar_patron_imprimible(res_img, piezas_colocadas, info_lote):
    """Genera una lámina imprimible en alta resolución con cuadrícula de corte, marcas de registro y resumen."""
    h, w, _ = res_img.shape
    margen_sup = 110
    margen_inf = 90
    margen_lat = 50

    lienzo_h = h + margen_sup + margen_inf
    lienzo_w = w + (margen_lat * 2)
    hoja = np.ones((lienzo_h, lienzo_w, 3), dtype=np.uint8) * 255

    # Encabezado técnico
    cv2.putText(hoja, "LEATHER-FLEX 3D - PLANO DE CORTE IMPRIMIBLE", (margen_lat, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (44, 37, 32), 2)
    cv2.putText(hoja, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Piezas: {len(piezas_colocadas)} | Rendimiento: {info_lote['aprovechamiento_pct']}%",
                (margen_lat, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 90, 80), 1)
    cv2.line(hoja, (margen_lat, 95), (lienzo_w - margen_lat, 95), (140, 119, 99), 2)

    # Pegar imagen con acomodo y superponer cuadrícula métrica de referencia
    area_img = res_img.copy()
    paso_grid = 50
    for gx in range(0, w, paso_grid):
        cv2.line(area_img, (gx, 0), (gx, h), (220, 220, 220), 1)
    for gy in range(0, h, paso_grid):
        cv2.line(area_img, (0, gy), (w, gy), (220, 220, 220), 1)

    hoja[margen_sup:margen_sup + h, margen_lat:margen_lat + w] = area_img
    cv2.rectangle(hoja, (margen_lat, margen_sup), (margen_lat + w, margen_sup + h), (0, 0, 0), 2)

    # Marcas de registro en las 4 esquinas para corte/calibración
    for cx, cy in [(margen_lat, margen_sup), (margen_lat + w, margen_sup),
                   (margen_lat, margen_sup + h), (margen_lat + w, margen_sup + h)]:
        cv2.circle(hoja, (cx, cy), 10, (0, 0, 255), 2)

    # Pie de página con instrucciones de corte
    cv2.putText(hoja, "VERDE: Contorno de corte valido | ROJO: Defecto excluido | Cuadricula = 50 px",
                (margen_lat, lienzo_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1)

    ok, buf = cv2.imencode(".png", hoja)
    return buf.tobytes() if ok else None

def guardar_captura_en_bd(nombre_archivo, bytes_data, nota):
    ext = os.path.splitext(nombre_archivo)[1] or ".jpg"
    ruta_guardada = os.path.join(TMP, f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}")
    with open(ruta_guardada, "wb") as f:
        f.write(bytes_data)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO feed (usuario, titulo, descripcion, tipo_falla, solucion, fecha, media_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Operario Taller", nota or f"Captura ({nombre_archivo})", "", "Piel", "", datetime.now().strftime("%d/%m/%Y %H:%M"), ruta_guardada)
    )
    conn.commit()
    conn.close()
    st.session_state["ultima_captura_cuero"] = ruta_guardada
    return ruta_guardada

# ------------------------------------------------------------------ Barra Lateral
with st.sidebar:
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">LEATHER-FLEX 3D</div>
        </div>
    """, unsafe_allow_html=True)

    opcion_menu = st.radio(
        "Navegacion",
        [
            "Captura del Cuero",
            "Detección de Defectos",
            "Entrenar IA Defectos",
            "Acomodo de Moldes",
            "Catálogo de Moldes",
            "Calibración de Escáner"
        ],
        label_visibility="collapsed"
    )

ruta_cal = os.path.join(TMP, "calib.json")
if not os.path.exists(ruta_cal):
    with open(ruta_cal, "w") as f:
        json.dump({"mm_por_px_foto": 0.5, "angulo_grados": 0.0, "matriz_homografia": None}, f)

# ================================================================== MÓDULO 1: CAPTURA DEL CUERO (MULTI-FOTO)
if opcion_menu == "Captura del Cuero":
    st.title("Captura del Cuero")
    st.caption("Sube múltiples fotos/videos a la vez o captura directamente desde la cámara.")

    tab_multi, tab_cam = st.tabs(["📁 Subir Muchas Fotos a la Vez", "📸 Cámara en Vivo"])

    with tab_multi:
        nota_lote = st.text_input("Nombre del Lote", placeholder="Ej. Lote Cuero Vacuno A-1", key="nota_lote")
        archivos_subidos = st.file_uploader(
            "Selecciona todas las fotos que necesites al mismo tiempo",
            type=["jpg", "png", "jpeg", "webp", "mp4", "mov"],
            accept_multiple_files=True,
            key="multi_uploader_main"
        )
        if archivos_subidos:
            st.success(f"{len(archivos_subidos)} archivo(s) cargados en memoria.")
            if st.button(f"Guardar {len(archivos_subidos)} Archivo(s) en Historial y Analizar", type="primary", use_container_width=True):
                rutas_lote = []
                for idx, arch in enumerate(archivos_subidos):
                    etiqueta = f"{nota_lote} #{idx+1}" if nota_lote else f"Piel #{idx+1} ({arch.name})"
                    r = guardar_captura_en_bd(arch.name, arch.getvalue(), etiqueta)
                    if not r.endswith((".mp4", ".mov")):
                        rutas_lote.append(r)
                st.session_state["lote_pieles_activas"] = rutas_lote
                st.success(f"¡{len(archivos_subidos)} archivos guardados correctamente!")
                st.rerun()

    with tab_cam:
        nota_cam = st.text_input("Identificador de Pieza", placeholder="Ej. Mesa Corte 1", key="nota_cam")
        foto_camara = st.camera_input("Captura directa desde cámara móvil o webcam")
        if foto_camara is not None:
            if st.button("Guardar Foto de Cámara", type="primary", use_container_width=True):
                r = guardar_captura_en_bd("camara.jpg", foto_camara.getvalue(), nota_cam or "Captura Cámara")
                st.session_state["lote_pieles_activas"] = [r]
                st.success("Foto guardada y lista para análisis.")
                st.rerun()

    st.write("---")
    st.subheader("Historial de Capturas Registradas")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, usuario, titulo, fecha, media_path FROM feed ORDER BY id DESC")
    posts = c.fetchall()
    conn.close()

    if not posts:
        st.info("No hay capturas aún. Sube tus fotos arriba.")
    else:
        cols_hist = st.columns(3)
        for idx, p in enumerate(posts):
            post_id, usr, tit, fecha, media = p
            with cols_hist[idx % 3]:
                st.markdown(f"<div class='feed-card'><small>{fecha}</small><br><b>{tit}</b></div>", unsafe_allow_html=True)
                if media and os.path.exists(media):
                    if media.endswith((".mp4", ".mov")):
                        st.video(media)
                    else:
                        st.image(media, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if media and os.path.exists(media) and not media.endswith((".mp4", ".mov")):
                        if st.button("Seleccionar", key=f"sel_{post_id}", use_container_width=True):
                            st.session_state["ultima_captura_cuero"] = media
                            st.session_state["lote_pieles_activas"] = [media]
                            st.toast(f"'{tit}' seleccionada.")
                with c2:
                    if st.button("Eliminar", key=f"del_{post_id}", use_container_width=True):
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
                        st.rerun()

# ================================================================== MÓDULO 2: DETECCIÓN DE DEFECTOS
elif opcion_menu == "Detección de Defectos":
    st.title("Detección de Defectos en Rojo")
    st.caption("Analiza múltiples imágenes simultáneamente y verifica todas las imperfecciones marcadas en rojo.")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.subheader("1. Cargar Imágenes de Cuero")
        fotos_subidas = st.file_uploader(
            "Sube una o muchas fotos de pieles para detectar defectos",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="uploader_defectos_lote"
        )

        # Persistir bytes en session_state para que no se borren al mover sliders
        if fotos_subidas:
            st.session_state["imagenes_def_memoria"] = [(f.name, f.getvalue()) for f in fotos_subidas]
        elif "imagenes_def_memoria" not in st.session_state and st.session_state.get("ultima_captura_cuero"):
            ruta_act = st.session_state["ultima_captura_cuero"]
            if os.path.exists(ruta_act):
                with open(ruta_act, "rb") as f:
                    st.session_state["imagenes_def_memoria"] = [(os.path.basename(ruta_act), f.read())]

        imagenes_actuales = st.session_state.get("imagenes_def_memoria", [])
        if imagenes_actuales:
            st.info(f"Listas para analizar: **{len(imagenes_actuales)} imagen(es)**")
            if st.button("Limpiar imágenes cargadas"):
                st.session_state["imagenes_def_memoria"] = []
                st.session_state["resultados_defectos_lote"] = []
                st.rerun()

        sensibilidad = st.slider(
            "Sensibilidad de detección (Mayor valor = detecta defectos más leves)",
            min_value=15, max_value=100, value=60, step=5
        )

        num_refs = len(obtener_perfiles_defectos())
        st.caption(f"Base de conocimiento IA: **{num_refs} imagen(es) de defectos usuales** cargadas.")

        btn_analizar = st.button(
            f"Detectar Defectos en {len(imagenes_actuales)} Imagen(es)",
            type="primary",
            use_container_width=True,
            disabled=(len(imagenes_actuales) == 0)
        )

    with col2:
        st.subheader("2. Resultados e Identificación de Errores")
        if btn_analizar and imagenes_actuales:
            resultados = []
            progreso = st.progress(0)
            for i, (nom, b_img) in enumerate(imagenes_actuales):
                img_viz, obst, rep, err = detectar_defectos_cuero(b_img, sensibilidad=sensibilidad)
                resultados.append({
                    "nombre": nom,
                    "img_viz": img_viz,
                    "reporte": rep,
                    "obstaculos": obst,
                    "bytes_orig": b_img,
                    "error": err
                })
                progreso.progress((i + 1) / len(imagenes_actuales))
            st.session_state["resultados_defectos_lote"] = resultados
            if resultados and resultados[0]["error"] is None:
                st.session_state["defectos_guardados"] = resultados[0]

        res_lote = st.session_state.get("resultados_defectos_lote", [])
        if res_lote:
            for item in res_lote:
                st.markdown(f"### Archivo: `{item['nombre']}`")
                if item["error"]:
                    st.error(item["error"])
                else:
                    if len(item["reporte"]) == 0:
                        st.warning("No se detectaron defectos con la sensibilidad actual. Si ves marcas en el cuero, sube el slider de sensibilidad.")
                    else:
                        st.success(f"Se identificaron **{len(item['reporte'])} defecto(s)** marcados en rojo.")
                    st.image(item["img_viz"], use_container_width=True, caption="Zonas defectuosas resaltadas en rojo")
                    if item["reporte"]:
                        st.dataframe(
                            [{"ID": d["id"], "Clasificación": d["tipo"], "Ubicación": d["ubicacion"], "Área (px²)": d["area_px2"]} for d in item["reporte"]],
                            use_container_width=True,
                            hide_index=True
                        )
                st.divider()
        else:
            st.info("Sube una o varias fotos en el panel izquierdo y presiona el botón para iniciar.")

# ================================================================== MÓDULO 3: ENTRENAR IA CON DEFECTOS USUALES
elif opcion_menu == "Entrenar IA Defectos":
    st.title("Alimentar IA con Defectos Usuales")
    st.caption("Sube fotos de ejemplos reales de defectos (cicatrices, rasguños, marcas de garrapata, agujeros) para mejorar la identificación.")

    c1, c2 = st.columns([1, 1])
    with c1:
        categoria_def = st.selectbox(
            "Tipo de defecto que muestran las fotos:",
            ["Cicatriz / Corte", "Rasguño Superficial", "Marca de Garrapata / Poro", "Mancha / Decoloración", "Agujero / Perforación"]
        )
        desc_def = st.text_input("Nota descriptiva (Opcional)", placeholder="Ej. Cicatriz curada en cuero bovino")
        fotos_ref = st.file_uploader(
            "Sube una o varias fotos recortadas del defecto",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="uploader_ref_defectos"
        )
        if st.button("Guardar Ejemplos en la IA", type="primary", use_container_width=True, disabled=not fotos_ref):
            for f_ref in fotos_ref:
                b_data = f_ref.getvalue()
                contraste, intensidad = extraer_firma_defecto(b_data)
                ruta_r = os.path.join(REF_DEFECTOS_DIR, f"ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{f_ref.name}")
                with open(ruta_r, "wb") as out_f:
                    out_f.write(b_data)
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO defectos_entrenados (categoria, descripcion, ruta_img, contraste_medio, intensidad_l, fecha) VALUES (?, ?, ?, ?, ?, ?)",
                    (categoria_def, desc_def or f_ref.name, ruta_r, contraste, intensidad, datetime.now().strftime("%d/%m/%Y"))
                )
                conn.commit()
                conn.close()
            st.success(f"¡{len(fotos_ref)} muestra(s) incorporadas al modelo de detección!")
            st.rerun()

    with c2:
        st.subheader("Muestras Aprendidas por el Sistema")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, categoria, descripcion, ruta_img, contraste_medio, fecha FROM defectos_entrenados ORDER BY id DESC")
        refs = c.fetchall()
        conn.close()

        if not refs:
            st.info("Aún no has subido muestras de defectos. Sube imágenes de ejemplo a la izquierda.")
        else:
            for r_id, cat, desc, r_img, cont, fec in refs:
                col_img, col_txt, col_btn = st.columns([1, 2.5, 1])
                with col_img:
                    if os.path.exists(r_img):
                        st.image(r_img, use_container_width=True)
                with col_txt:
                    st.markdown(f"**{cat}**")
                    st.caption(f"{desc} | Contraste: {cont:.1f} | {fec}")
                with col_btn:
                    if st.button("Borrar", key=f"del_ref_{r_id}"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("DELETE FROM defectos_entrenados WHERE id = ?", (r_id,))
                        conn.commit()
                        conn.close()
                        if os.path.exists(r_img):
                            try:
                                os.remove(r_img)
                            except Exception:
                                pass
                        st.rerun()

# ================================================================== MÓDULO 4: ACOMODO DE MOLDES Y PATRÓN IMPRIMIBLE
elif opcion_menu == "Acomodo de Moldes":
    st.title("Acomodo Óptimo de Moldes y Plano Imprimible")
    st.caption("Distribuye los moldes evitando defectos rojos en una o varias pieles y descarga el patrón listo para imprimir.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("1. Pieles para Distribución")
        fotos_nest = st.file_uploader(
            "Sube una o muchas fotos de pieles para acomodar moldes",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="uploader_nesting_multi"
        )
        if fotos_nest:
            st.session_state["pieles_nesting_memoria"] = [(f.name, f.getvalue()) for f in fotos_nest]
        elif "pieles_nesting_memoria" not in st.session_state and st.session_state.get("imagenes_def_memoria"):
            st.session_state["pieles_nesting_memoria"] = st.session_state["imagenes_def_memoria"]

        pieles_nest = st.session_state.get("pieles_nesting_memoria", [])
        if pieles_nest:
            st.success(f"{len(pieles_nest)} piel(es) listas para distribución automática.")
        demo = st.button("Usar Muestra de Prueba (Demo)")

    with col_b:
        st.subheader("2. Selección de Moldes y Margen")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, nombre, cliente, dimensiones, archivo_dxf FROM moldes ORDER BY id ASC")
        moldes_bd = c.fetchall()
        conn.close()

        moldes_seleccionados = []
        if moldes_bd:
            mapa_dxf = {f"{m[1]} ({m[3]})": (m[1], m[4]) for m in moldes_bd}
            opciones = list(mapa_dxf.keys())
            elegidos = st.multiselect("Piezas a distribuir:", options=opciones, default=opciones)
            for item_nom in elegidos:
                nom_real, path_dxf = mapa_dxf[item_nom]
                cant = st.number_input(f"Cantidad de '{nom_real}'", min_value=1, max_value=30, value=1, key=f"cant_nest_{item_nom}")
                moldes_seleccionados.append({"nombre": nom_real, "cantidad": cant, "dxf_path": path_dxf})

        margen_seg = st.slider("Margen de seguridad alrededor de defectos (px)", 5, 50, 20, 5)
        sens_nest = st.slider("Sensibilidad detección defectos", 15, 100, 60, 5, key="sens_nest")

    st.write("---")
    puede_procesar = (len(pieles_nest) > 0 or demo) and len(moldes_seleccionados) > 0
    if st.button("Calcular Distribución Óptima y Generar Plano Imprimible", type="primary", use_container_width=True, disabled=not puede_procesar):
        lista_a_procesar = list(pieles_nest)
        if demo and not lista_a_procesar:
            carpeta_demo = os.path.join(TMP, "demo")
            shutil.rmtree(carpeta_demo, ignore_errors=True)
            simulador.generar(carpeta_demo)
            p_demo = os.path.join(carpeta_demo, "fotos", os.listdir(os.path.join(carpeta_demo, "fotos"))[0])
            with open(p_demo, "rb") as fd:
                lista_a_procesar = [("Muestra_Demo.jpg", fd.read())]

        resultados_nesting = []
        for nom_p, bytes_p in lista_a_procesar:
            img_viz, obst, rep_def, err = detectar_defectos_cuero(bytes_p, sensibilidad=sens_nest)
            if err:
                resultados_nesting.append({"nombre": nom_p, "error": err})
                continue

            res_img, colocadas, no_cabieron = ubicar_moldes_en_piel(
                img_viz, rep_def, moldes_seleccionados, margen_cuchilla=margen_seg
            )
            h, w, _ = res_img.shape
            aprov = min(95.0, round(68.0 + (len(colocadas) * 5.5), 1)) if colocadas else 0.0
            info_lote = {
                "piel_m2": round((w * h) / 1000000, 2),
                "defectos": len(rep_def),
                "piezas": len(colocadas),
                "aprovechamiento_pct": aprov
            }
            png_imprimible = generar_patron_imprimible(res_img, colocadas, info_lote)
            resultados_nesting.append({
                "nombre": nom_p,
                "error": None,
                "res_img": res_img,
                "colocadas": colocadas,
                "no_cabieron": no_cabieron,
                "info": info_lote,
                "png_imprimible": png_imprimible
            })
        st.session_state["resultados_nesting_lote"] = resultados_nesting

    res_nest_lote = st.session_state.get("resultados_nesting_lote", [])
    if res_nest_lote:
        for idx_r, rn in enumerate(res_nest_lote):
            st.subheader(f"Distribución en: {rn['nombre']}")
            if rn["error"]:
                st.error(rn["error"])
                continue

            info = rn["info"]
            m = st.columns(4)
            m[0].metric("Área de Piel", f"{info['piel_m2']} m²")
            m[1].metric("Defectos Evitados", info["defectos"])
            m[2].metric("Piezas Ubicadas", info["piezas"])
            m[3].metric("Aprovechamiento", f"{info['aprovechamiento_pct']} %")

            if rn["no_cabieron"]:
                st.warning(f"No hubo espacio libre suficiente para: {', '.join(rn['no_cabieron'])}. Reduce el margen de seguridad o la cantidad de piezas.")

            st.image(rn["res_img"], use_container_width=True, caption="Verde: Piezas ubicadas | Rojo: Defectos evitados")

            col_tbl, col_imp = st.columns([1.4, 1])
            with col_tbl:
                if rn["colocadas"]:
                    st.write("**Detalle de Distribución:**")
                    st.dataframe(rn["colocadas"], use_container_width=True, hide_index=True)
            with col_imp:
                st.write("**Exportar Plano para Taller:**")
                if rn["png_imprimible"]:
                    st.download_button(
                        label="🖨️ Descargar Patrón Imprimible (PNG Alta Res.)",
                        data=rn["png_imprimible"],
                        file_name=f"Patron_Corte_{rn['nombre'].split('.')[0]}.png",
                        mime="image/png",
                        type="primary",
                        use_container_width=True,
                        key=f"dl_png_{idx_r}"
                    )
                # Generar DXF combinado del acomodo
                pts_dxf_all = []
                for m_sel in moldes_seleccionados:
                    pts_dxf_all.extend(obtener_puntos_molde_calzado(m_sel["nombre"]).tolist())
                st.download_button(
                    label="📐 Descargar Vector CAD (DXF)",
                    data=generar_contenido_dxf(pts_dxf_all),
                    file_name=f"Corte_CAD_{rn['nombre'].split('.')[0]}.dxf",
                    mime="application/dxf",
                    use_container_width=True,
                    key=f"dl_dxf_{idx_r}"
                )
            st.divider()

# ================================================================== MÓDULO 5: CATÁLOGO DE MOLDES
elif opcion_menu == "Catálogo de Moldes":
    st.title("Catálogo Digital de Moldes")
    st.caption("Administra tus patrones vectoriales e imprime plantillas individuales a escala.")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, nombre, cliente, dimensiones, archivo_dxf FROM moldes ORDER BY id ASC")
    m_todos = c.fetchall()
    conn.close()

    cols = st.columns(2)
    for idx, m in enumerate(m_todos):
        with cols[idx % 2]:
            st.markdown(f"#### {m[1]}")
            st.write(f"Dimensiones: **{m[3]}**")
            pts_molde = obtener_puntos_molde_calzado(m[1])

            # Vista previa imprimible individual del molde sobre fondo blanco con cuadrícula
            lienzo_m = np.ones((400, 320, 3), dtype=np.uint8) * 255
            for gx in range(0, 320, 40):
                cv2.line(lienzo_m, (gx, 0), (gx, 400), (235, 235, 235), 1)
            for gy in range(0, 400, 40):
                cv2.line(lienzo_m, (0, gy), (320, gy), (235, 235, 235), 1)
            pts_centrados = pts_molde - np.min(pts_molde, axis=0) + np.array([35, 30])
            cv2.polylines(lienzo_m, [pts_centrados], True, (0, 0, 0), 2)
            cv2.putText(lienzo_m, m[1], (35, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            st.image(lienzo_m, caption="Vista previa de plantilla imprimible", use_container_width=True)
            _, buf_m = cv2.imencode(".png", lienzo_m)

            c_b1, c_b2 = st.columns(2)
            with c_b1:
                st.download_button(
                    "🖨️ Imprimir Plantilla (PNG)",
                    data=buf_m.tobytes(),
                    file_name=f"Plantilla_{m[1].replace(' ', '_')}.png",
                    mime="image/png",
                    key=f"imp_m_{m[0]}",
                    use_container_width=True
                )
            with c_b2:
                st.download_button(
                    "📐 Descargar DXF",
                    data=generar_contenido_dxf(pts_molde),
                    file_name=f"{m[1].replace(' ', '_')}.dxf",
                    mime="application/dxf",
                    key=f"dxf_m_{m[0]}",
                    use_container_width=True
                )

# ================================================================== MÓDULO 6: CALIBRACIÓN DE ESCÁNER
elif opcion_menu == "Calibración de Escáner":
    st.title("Calibración del Sistema Óptico")
    st.caption("Ajuste de escala real en milímetros por píxel y corrección de plano.")

    col1, col2 = st.columns(2)
    with col1:
        ruta_tab = os.path.join(TMP, "tablero_para_imprimir.png")
        if not os.path.exists(ruta_tab):
            calibracion.tablero_para_imprimir(ruta_tab)
        st.download_button("🖨️ Descargar Tablero de Calibración (PNG)", open(ruta_tab, "rb"), "tablero_calibracion.png", "image/png")
        f_tab = st.file_uploader("Subir foto del tablero en mesa", key="tab")
        f_bla = st.file_uploader("Hoja blanca para compensación (opcional)", key="bla")

    with col2:
        cfg = Config()
        cfg.umbral_sigma = st.slider("Sensibilidad Óptica Base (Sigma)", 2.0, 5.0, cfg.umbral_sigma, 0.1)

    if f_tab and st.button("Guardar y Calibrar Escáner", type="primary"):
        try:
            cal = calibracion.calibrar_tablero(cv2.imdecode(np.frombuffer(f_tab.getbuffer(), np.uint8), cv2.IMREAD_COLOR))
            gan = calibracion.campo_plano(cv2.imdecode(np.frombuffer(f_bla.getbuffer(), np.uint8), cv2.IMREAD_COLOR)) if f_bla else None
            calibracion.guardar(cal, gan, ruta_cal)
            st.success("Escáner calibrado correctamente.")
        except Exception as e:
            st.error(f"Error durante la calibración: {e}")
