"""Leather-Flex 3D - Ensamblaje de Plancha Completa (Mosaico/Stitching), IA de Defectos y Nesting Integral"""
import os
import math
import json
import sqlite3
import tempfile
import shutil
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_DISPONIBLE = True
except ImportError:
    SKLEARN_DISPONIBLE = False

from leatherflex import calibracion, pipeline, acomodo, simulador
from leatherflex.config import Config

# ------------------------------------------------------------------ Configuración de página
st.set_page_config(
    page_title="Leather-Flex 3D",
    layout="wide",
    initial_sidebar_state="expanded"
)

TMP = st.session_state.setdefault("tmp", tempfile.mkdtemp(prefix="leatherflex_"))
DB_PATH = os.path.join(os.path.dirname(__file__), "leatherflex.db")
EXTENSIONES_VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# ------------------------------------------------------------------ Motor de IA Embebido (Aprendizaje Automático)
CLASES_DEFECTOS = [
    "Veta Natural (Sano)",
    "Picadura / Hueco",
    "Cicatriz / Rayón Profundo",
    "Rayón Superficial",
    "Agujero / Perforación",
    "Mancha / Variación de Curtido"
]

@st.cache_resource
def inicializar_modelo_ia():
    rng = np.random.default_rng(42)
    X_train = []
    y_train = []

    perfiles = {
        0: ([3.5, 1.2, 1.5, 5.0, 1.4, 0.65, 0.15, 12.0, 3.0, 0.5], [2.0, 0.8, 0.8, 2.0, 0.4, 0.15, 0.10, 5.0, 1.5, 0.5]),
        1: ([22.0, 4.5, 5.0, 14.0, 1.25, 0.82, 0.35, 38.0, 24.0, 1.0], [6.0, 2.0, 2.0, 4.0, 0.25, 0.10, 0.25, 10.0, 6.0, 0.1]),
        2: ([28.0, 5.0, 6.0, 18.0, 4.5, 0.28, 1.80, 45.0, 26.0, 1.0], [7.0, 2.0, 2.0, 5.0, 1.2, 0.10, 1.00, 12.0, 7.0, 0.2]),
        3: ([15.0, 3.0, 3.5, 11.0, 3.4, 0.35, 0.90, 28.0, 16.0, 0.7], [4.5, 1.5, 1.5, 3.5, 0.9, 0.12, 0.60, 8.0, 5.0, 0.4]),
        4: ([48.0, 8.0, 9.0, 26.0, 1.35, 0.78, 1.20, 68.0, 46.0, 1.0], [9.0, 3.0, 3.0, 6.0, 0.3, 0.12, 0.80, 14.0, 9.0, 0.1]),
        5: ([18.0, 7.5, 8.0, 12.0, 1.60, 0.55, 8.50, 22.0, 17.0, 0.6], [5.0, 2.5, 2.5, 4.0, 0.5, 0.15, 4.00, 7.0, 5.0, 0.4])
    }

    for clase_id, (medias, stds) in perfiles.items():
        muestras = rng.normal(loc=medias, scale=stds, size=(250, 10))
        muestras = np.clip(muestras, 0.01, None)
        X_train.append(muestras)
        y_train.extend([clase_id] * 250)

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    if SKLEARN_DISPONIBLE:
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42
        )
        clf.fit(X_train, y_train)
        return clf
    else:
        modelo_np = {}
        for c_id in range(len(CLASES_DEFECTOS)):
            sub = X_train[y_train == c_id]
            modelo_np[c_id] = (np.mean(sub, axis=0), np.std(sub, axis=0) + 1e-4)
        return modelo_np

MODELO_IA_CUERO = inicializar_modelo_ia()

def predecir_defecto_ia(vector_caracteristicas):
    vec = np.array(vector_caracteristicas, dtype=np.float32).reshape(1, -1)
    if SKLEARN_DISPONIBLE:
        probs = MODELO_IA_CUERO.predict_proba(vec)[0]
        clase_id = int(np.argmax(probs))
        confianza = float(probs[clase_id] * 100.0)
        return clase_id, CLASES_DEFECTOS[clase_id], round(confianza, 1)
    else:
        scores = []
        for c_id, (mu, sigma) in MODELO_IA_CUERO.items():
            z = np.sum(((vec[0] - mu) / sigma) ** 2)
            scores.append(np.exp(-0.15 * z))
        probs = np.array(scores) / (np.sum(scores) + 1e-8)
        clase_id = int(np.argmax(probs))
        return clase_id, CLASES_DEFECTOS[clase_id], round(float(probs[clase_id] * 100.0), 1)

# ------------------------------------------------------------------ Extractor de Fotogramas de Video
def extraer_fotogramas_nitidos_video(bytes_video, nombre_video="video.mp4", max_fotogramas=6):
    ext = os.path.splitext(nombre_video)[1] or ".mp4"
    tmp_vid = os.path.join(TMP, f"vid_proc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}")
    with open(tmp_vid, "wb") as f:
        f.write(bytes_video)

    cap = cv2.VideoCapture(tmp_vid)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total_frames <= 0:
        total_frames = 300

    paso = max(1, total_frames // 45)
    candidatos = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % paso == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nitidez = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            tiempo_seg = round(frame_idx / fps, 1)
            candidatos.append({
                "frame": frame,
                "nitidez": nitidez,
                "hist": hist,
                "tiempo": tiempo_seg
            })
        frame_idx += 1

    cap.release()
    try:
        os.remove(tmp_vid)
    except Exception:
        pass

    if not candidatos:
        return []

    num_segmentos = min(max_fotogramas, len(candidatos))
    segmentos = np.array_split(candidatos, num_segmentos)
    seleccionados = []
    ultimo_hist = None

    for seg in segmentos:
        if len(seg) == 0:
            continue
        mejor = max(seg, key=lambda item: item["nitidez"])
        if ultimo_hist is not None:
            similitud = cv2.compareHist(ultimo_hist, mejor["hist"], cv2.HISTCMP_CORREL)
            if similitud > 0.992 and len(seleccionados) > 0:
                continue
        ultimo_hist = mejor["hist"]
        ok, buf = cv2.imencode(".jpg", mejor["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            etiqueta = f"{os.path.splitext(nombre_video)[0]}_seg_{mejor['tiempo']}s.jpg"
            seleccionados.append((etiqueta, buf.tobytes()))

    return seleccionados

def expandir_archivos_a_imagenes(lista_archivos):
    imagenes_listas = []
    for nom, b_data in lista_archivos:
        if nom.lower().endswith(EXTENSIONES_VIDEO):
            frames_vid = extraer_fotogramas_nitidos_video(b_data, nombre_video=nom, max_fotogramas=6)
            imagenes_listas.extend(frames_vid)
        else:
            imagenes_listas.append((nom, b_data))
    return imagenes_listas

# ------------------------------------------------------------------ Ensamblador de Plancha Completa de Cuero (Mosaico + Stitching)
def ensamblar_plancha_completa(lista_imagenes_bytes, columnas_cuadricula=0):
    """
    Une múltiples fotos parciales de una plancha de cuero en UNA SOLA PLANCHA ENTERA.
    - Si las fotos tienen solapamiento óptico (panorámica), intenta OpenCV Stitcher.
    - Si son tomas en cuadrícula/sectores (como 3x4 = 12 fotos o cualquier cantidad N),
      las ensambla limpiamente en una única plancha unificada sin bordes divisorios
      para que la IA detecte defectos y acomode los moldes sobre toda la superficie.
    """
    imgs_bgr = []
    for _, b_img in lista_imagenes_bytes:
        arr = np.frombuffer(b_img, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is not None:
            imgs_bgr.append(im)

    if not imgs_bgr:
        return None, "No se pudieron decodificar las imágenes."

    if len(imgs_bgr) == 1:
        im_unica = imgs_bgr[0]
        h0, w0 = im_unica.shape[:2]
        if max(h0, w0) > 1400:
            esc = 1400 / max(h0, w0)
            im_unica = cv2.resize(im_unica, (int(w0 * esc), int(h0 * esc)))
        ok, buf = cv2.imencode(".jpg", im_unica, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        return buf.tobytes(), f"Plancha individual cargada ({im_unica.shape[1]}x{im_unica.shape[0]} px)"

    # Intento 1: Panorámica automática por puntos clave si hay solapamiento directo (<= 8 fotos)
    if columnas_cuadricula == 0 and 2 <= len(imgs_bgr) <= 8:
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            status, pano = stitcher.stitch(imgs_bgr)
            if status == cv2.Stitcher_OK and pano is not None:
                hp, wp = pano.shape[:2]
                if max(hp, wp) > 1600:
                    esc = 1600 / max(hp, wp)
                    pano = cv2.resize(pano, (int(wp * esc), int(hp * esc)))
                ok, buf = cv2.imencode(".jpg", pano, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                return buf.tobytes(), f"Plancha unida automáticamente por visión panorámica ({len(imgs_bgr)} tomas)"
        except Exception:
            pass

    # Intento 2: Ensamblaje de Plancha Completa en Cuadrícula Continua (para N fotos: 4, 6, 9, 12, etc.)
    n = len(imgs_bgr)
    if columnas_cuadricula > 0:
        cols = columnas_cuadricula
    else:
        # Proporción horizontal natural de plancha de cuero (ej. 12 fotos -> 4 cols x 3 filas)
        cols = math.ceil(math.sqrt(n * 1.35))
    filas = math.ceil(n / cols)

    # Normalizar tamaño de cada sector para unión limpia y uniforme
    sector_w = max(240, min(480, 1440 // cols))
    sector_h = max(200, min(400, 1080 // filas))

    # Color medio del cuero para rellenar espacios vacíos si N no llena el rectángulo exacto
    muestra_centro = imgs_bgr[0][
        imgs_bgr[0].shape[0] // 4 : 3 * imgs_bgr[0].shape[0] // 4,
        imgs_bgr[0].shape[1] // 4 : 3 * imgs_bgr[0].shape[1] // 4
    ]
    color_base_piel = tuple(map(int, np.median(muestra_centro.reshape(-1, 3), axis=0))) if muestra_centro.size > 0 else (140, 100, 70)

    plancha = np.full((filas * sector_h, cols * sector_w, 3), color_base_piel, dtype=np.uint8)

    for idx, im in enumerate(imgs_bgr):
        r = idx // cols
        c = idx % cols
        im_res = cv2.resize(im, (sector_w, sector_h), interpolation=cv2.INTER_AREA)
        y1, y2 = r * sector_h, (r + 1) * sector_h
        x1, x2 = c * sector_w, (c + 1) * sector_w
        plancha[y1:y2, x1:x2] = im_res

    # Suavizar sutilmente las uniones entre sectores para evitar falsas líneas de costura
    for c in range(1, cols):
        x_seam = c * sector_w
        x_a, x_b = max(0, x_seam - 4), min(plancha.shape[1], x_seam + 4)
        plancha[:, x_a:x_b] = cv2.GaussianBlur(plancha[:, x_a:x_b], (7, 1), 0)
    for r in range(1, filas):
        y_seam = r * sector_h
        y_a, y_b = max(0, y_seam - 4), min(plancha.shape[0], y_seam + 4)
        plancha[y_a:y_b, :] = cv2.GaussianBlur(plancha[y_a:y_b, :], (1, 7), 0)

    ok, buf = cv2.imencode(".jpg", plancha, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return buf.tobytes(), f"Plancha entera reconstruida desde {n} fotos ({cols} columnas x {filas} filas = {plancha.shape[1]}x{plancha.shape[0]} px)"

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

# ------------------------------------------------------------------ Base de Datos
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS moldes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, cliente TEXT, 
                  dimensiones TEXT, fecha TEXT, archivo_dxf TEXT, origen TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feed 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, usuario TEXT, titulo TEXT, 
                  descripcion TEXT, tipo_falla TEXT, solucion TEXT, fecha TEXT, media_path TEXT)''')

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

# ------------------------------------------------------------------ Segmentación de Piel y Detección con IA
def aislar_piel_del_fondo(img_bgr):
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, a_chan, b_chan = cv2.split(lab)

    sat_ab = cv2.absdiff(a_chan, 128) + cv2.absdiff(b_chan, 128)
    blur_sat = cv2.GaussianBlur(sat_ab, (15, 15), 0)
    _, mask_piel = cv2.threshold(blur_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Si la foto o plancha reconstruida es casi todo piel (ej. cierre de plancha completa)
    if np.mean(mask_piel == 255) < 0.12 or np.mean(mask_piel == 255) > 0.92:
        mask_completa = np.ones((h, w), dtype=np.uint8) * 255
        cv2.rectangle(mask_completa, (0, 0), (w, h), 0, 12)
        return mask_completa

    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_piel = cv2.morphologyEx(mask_piel, cv2.MORPH_CLOSE, kernel_fill, iterations=2)
    mask_piel = cv2.morphologyEx(mask_piel, cv2.MORPH_OPEN, kernel_fill, iterations=1)

    cnts, _ = cv2.findContours(mask_piel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones((h, w), dtype=np.uint8) * 255

    c_mayor = max(cnts, key=cv2.contourArea)
    mask_limpia = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_limpia, [c_mayor], -1, 255, -1)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
    mask_interior = cv2.erode(mask_limpia, kernel_erode, iterations=1)
    return mask_interior

def extraer_vector_region(contorno, area, area_total, lab_img, grad_mag):
    h, w = lab_img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contorno)

    pad = max(6, int(max(bw, bh) * 0.45))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

    roi_lab = lab_img[y1:y2, x1:x2]
    mask_def = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cnt_shift = contorno - np.array([x1, y1])
    cv2.drawContours(mask_def, [cnt_shift], -1, 255, -1)
    mask_anillo = cv2.bitwise_not(mask_def)

    if np.sum(mask_def == 255) == 0 or np.sum(mask_anillo == 255) == 0:
        return [15.0, 3.0, 3.0, 10.0, 1.5, 0.6, 0.5, 25.0, 15.0, 1.0]

    l_def = roi_lab[:, :, 0][mask_def == 255]
    a_def = roi_lab[:, :, 1][mask_def == 255]
    b_def = roi_lab[:, :, 2][mask_def == 255]

    l_ani = roi_lab[:, :, 0][mask_anillo == 255]
    a_ani = roi_lab[:, :, 1][mask_anillo == 255]
    b_ani = roi_lab[:, :, 2][mask_anillo == 255]

    delta_l = abs(float(np.mean(l_ani)) - float(np.mean(l_def)))
    delta_a = abs(float(np.mean(a_ani)) - float(np.mean(a_def)))
    delta_b = abs(float(np.mean(b_ani)) - float(np.mean(b_def)))
    std_l = float(np.std(l_def))

    elongacion = max(bw, bh) / max(1.0, float(min(bw, bh)))
    perimetro = max(1.0, cv2.arcLength(contorno, True))
    circularidad = min(1.0, (4.0 * np.pi * area) / (perimetro * perimetro))
    area_rel_1000 = (area / max(1.0, float(area_total))) * 1000.0

    roi_grad = grad_mag[y:y + bh, x:x + bw]
    grad_medio = float(np.mean(roi_grad)) if roi_grad.size > 0 else 20.0

    contraste_anillo = delta_l + (0.5 * (delta_a + delta_b))
    es_oscuro = 1.0 if np.mean(l_def) < np.mean(l_ani) else 0.0

    return [
        delta_l, delta_a, delta_b, std_l,
        elongacion, circularidad, area_rel_1000,
        grad_medio, contraste_anillo, es_oscuro
    ]

def detectar_defectos_cuero(bytes_imagen, sensibilidad=75):
    try:
        arr = np.frombuffer(bytes_imagen, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, [], [], "Error: El archivo no es una imagen válida o está dañado."

        h_orig, w_orig = img.shape[:2]
        max_dim = 1400
        if max(h_orig, w_orig) > max_dim:
            escala = max_dim / max(h_orig, w_orig)
            img = cv2.resize(img, (int(w_orig * escala), int(h_orig * escala)))

        h, w, _ = img.shape
        area_total = h * w

        mask_piel = aislar_piel_del_fondo(img)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, _, _ = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
        l_norm = clahe.apply(l_chan)

        gx = cv2.Sobel(l_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_norm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        g_fino = cv2.GaussianBlur(l_norm, (3, 3), 0)
        g_medio = cv2.GaussianBlur(l_norm, (15, 15), 0)
        dog_oscuro = cv2.subtract(g_medio, g_fino)
        dog_claro = cv2.subtract(g_fino, g_medio)

        kernel_bh_fino = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_bh_medio = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        bh_fino = cv2.morphologyEx(l_norm, cv2.MORPH_BLACKHAT, kernel_bh_fino)
        bh_medio = cv2.morphologyEx(l_norm, cv2.MORPH_BLACKHAT, kernel_bh_medio)
        th_medio = cv2.morphologyEx(l_norm, cv2.MORPH_TOPHAT, kernel_bh_medio)

        mapa_oscuro = cv2.max(dog_oscuro, cv2.max(bh_fino, bh_medio))
        mapa_claro = cv2.max(dog_claro, th_medio)

        umbral_osc = max(7, int(34 - (sensibilidad * 0.27)))
        umbral_cla = max(9, int(40 - (sensibilidad * 0.29)))

        _, mask_osc = cv2.threshold(mapa_oscuro, umbral_osc, 255, cv2.THRESH_BINARY)
        _, mask_cla = cv2.threshold(mapa_claro, umbral_cla, 255, cv2.THRESH_BINARY)

        mask_total = cv2.bitwise_or(mask_osc, mask_cla)
        mask_total = cv2.bitwise_and(mask_total, mask_piel)

        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_join = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel_join, iterations=1)

        cnts_def, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_def = sorted(cnts_def, key=cv2.contourArea, reverse=True)

        reporte_defectos = []
        lista_obstaculos_px = []
        img_viz_bgr = img.copy()
        overlay_rojo = img.copy()

        area_min = 9
        area_max = int(area_total * 0.15)
        idx_def = 1

        for c in cnts_def:
            area = cv2.contourArea(c)
            if area_min <= area <= area_max:
                vec_feat = extraer_vector_region(c, area, area_total, lab, grad_mag)
                clase_id, tipo_label, confianza = predecir_defecto_ia(vec_feat)

                if clase_id == 0 and sensibilidad < 85:
                    continue

                if clase_id == 0:
                    tipo_label = "Marca Leve de Superficie"

                x, y, bw, bh = cv2.boundingRect(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                lista_obstaculos_px.append([pt[0] for pt in approx])

                reporte_defectos.append({
                    "id": idx_def,
                    "tipo": tipo_label,
                    "confianza": f"{confianza}%",
                    "area_px2": round(area, 1),
                    "ubicacion": f"X:{x}, Y:{y}",
                    "bbox": (x, y, bw, bh)
                })

                cv2.drawContours(overlay_rojo, [c], -1, (0, 0, 255), -1)
                cx, cy = x + bw // 2, y + bh // 2
                radio = max(14, int(max(bw, bh) * 0.65))
                cv2.circle(img_viz_bgr, (cx, cy), radio, (0, 0, 255), 3)
                cv2.putText(
                    img_viz_bgr,
                    f"D{idx_def}: {tipo_label.split('/')[0].strip()}",
                    (max(4, cx - radio), max(18, cy - radio - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255),
                    2
                )
                idx_def += 1

        cv2.addWeighted(overlay_rojo, 0.32, img_viz_bgr, 0.68, 0, img_viz_bgr)
        img_viz_rgb = cv2.cvtColor(img_viz_bgr, cv2.COLOR_BGR2RGB)
        return img_viz_rgb, lista_obstaculos_px, reporte_defectos, None
    except Exception as e:
        return None, [], [], f"Error técnico al procesar imagen: {str(e)}"

# ------------------------------------------------------------------ Motor de Distribución Integral en TODA la Plancha
def ubicar_moldes_en_toda_la_plancha(bytes_imagen, bboxes_defectos, moldes_elegidos_list, margen_cuchilla=16, llenar_toda_plancha=True, escala_pieza_pct=100):
    """
    Acomoda los moldes abarcando TODA la plancha de cuero (no solo una esquina),
    escalando las piezas proporcionalmente al área útil de la piel y repitiendo
    el set seleccionado si 'llenar_toda_plancha' está activo, siempre evitando los defectos rojos.
    """
    arr = np.frombuffer(bytes_imagen, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h_orig, w_orig = img.shape[:2]
    max_dim = 1400
    if max(h_orig, w_orig) > max_dim:
        escala = max_dim / max(h_orig, w_orig)
        img = cv2.resize(img, (int(w_orig * escala), int(h_orig * escala)))

    h, w, _ = img.shape
    res_img = img.copy()

    mask_piel = aislar_piel_del_fondo(img)
    area_piel_px = max(1, int(np.sum(mask_piel == 255)))
    ocupado_mask = cv2.bitwise_not(mask_piel)

    # Marcar zonas prohibidas (defectos en rojo + margen de seguridad)
    for d in bboxes_defectos:
        bx, by, bw_d, bh_d = d["bbox"]
        cx, cy = bx + bw_d // 2, by + bh_d // 2
        radio_vis = max(14, int(max(bw_d, bh_d) * 0.65))
        radio_seg = radio_vis + margen_cuchilla

        cv2.circle(res_img, (cx, cy), radio_vis, (0, 0, 255), 3)
        cv2.putText(
            res_img,
            d["tipo"].split("/")[0].strip(),
            (max(4, cx - radio_vis), max(18, cy - radio_vis - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2
        )
        cv2.circle(ocupado_mask, (cx, cy), radio_seg, 255, -1)

    # Preparar lista de piezas base escaladas proporcionalmente al tamaño de la plancha entera
    # Un molde estándar ocupa ~1/8 del ancho de la plancha (ajustable con escala_pieza_pct)
    factor_escala_plancha = (min(w, h) / 680.0) * (escala_pieza_pct / 100.0)
    factor_escala_plancha = max(0.45, factor_escala_plancha)

    cola_piezas = []
    for m in moldes_elegidos_list:
        nombre_m = m["nombre"]
        cant = m.get("cantidad", 1)
        pts_raw = obtener_puntos_molde_calzado(nombre_m)
        pts_scaled = (pts_raw * factor_escala_plancha).astype(np.int32)
        min_xy = np.min(pts_scaled, axis=0)
        pts_norm = pts_scaled - min_xy
        area_m = cv2.contourArea(pts_norm)
        for _ in range(cant):
            cola_piezas.append({
                "nombre": nombre_m,
                "pts": pts_norm,
                "area": area_m
            })

    # Ordenar de mayor a menor área (estrategia Best-Fit Decreasing para aprovechar mejor la plancha)
    cola_piezas.sort(key=lambda item: item["area"], reverse=True)

    # Si se activa "Llenar toda la plancha", repetir el juego de moldes hasta cubrir toda la piel disponible
    if llenar_toda_plancha and len(cola_piezas) > 0:
        area_promedio_pieza = max(1000.0, np.mean([p["area"] for p in cola_piezas]))
        max_piezas_estimadas = min(45, max(len(cola_piezas), int((area_piel_px * 0.72) / area_promedio_pieza)))
        cola_expandida = []
        idx_ciclo = 0
        while len(cola_expandida) < max_piezas_estimadas:
            cola_expandida.append(cola_piezas[idx_ciclo % len(cola_piezas)])
            idx_ciclo += 1
        cola_piezas = cola_expandida

    piezas_colocadas = []
    piezas_no_cabieron = []
    conteo_por_nombre = {}
    area_ocupada_moldes = 0.0

    # Paso de barrido fino para aprovechar cada hueco de toda la plancha
    paso_x = max(8, w // 75)
    paso_y = max(8, h // 65)

    for pieza in cola_piezas:
        nombre_m = pieza["nombre"]
        pts_norm = pieza["pts"]
        mw = int(np.max(pts_norm[:, 0]))
        mh = int(np.max(pts_norm[:, 1]))

        conteo_por_nombre[nombre_m] = conteo_por_nombre.get(nombre_m, 0) + 1
        num_instancia = conteo_por_nombre[nombre_m]

        posicionado = False
        # Probar 4 rotaciones + espejo horizontal para encajar en curvas de toda la plancha
        variantes = [
            (pts_norm, mw, mh, "0°"),
            (np.array([[pt[1], mw - pt[0]] for pt in pts_norm], np.int32), mh, mw, "90°"),
            (np.array([[mw - pt[0], mh - pt[1]] for pt in pts_norm], np.int32), mw, mh, "180°"),
            (np.array([[mh - pt[1], pt[0]] for pt in pts_norm], np.int32), mh, mw, "270°"),
            (np.array([[mw - pt[0], pt[1]] for pt in pts_norm], np.int32), mw, mh, "Espejo")
        ]

        for pts_rot, cur_w, cur_h, etiq_rot in variantes:
            if posicionado or cur_w >= w - 16 or cur_h >= h - 16:
                continue

            for y in range(12, h - cur_h - 12, paso_y):
                if posicionado:
                    break
                for x in range(12, w - cur_w - 12, paso_x):
                    # Verificación rápida de esquinas antes de construir máscara completa
                    if (ocupado_mask[y, x] > 0 or
                        ocupado_mask[y + cur_h // 2, x + cur_w // 2] > 0 or
                        ocupado_mask[min(h - 1, y + cur_h), min(w - 1, x + cur_w)] > 0):
                        continue

                    pts_tras = pts_rot + np.array([x, y])
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(temp_mask, [pts_tras], 255)

                    if np.sum(cv2.bitwise_and(ocupado_mask, temp_mask)) == 0:
                        # Dibujar pieza ubicada con estilo de patrón de corte limpio
                        overlay = res_img.copy()
                        cv2.fillPoly(overlay, [pts_tras], (35, 55, 45))
                        cv2.addWeighted(overlay, 0.65, res_img, 0.35, 0, res_img)
                        cv2.polylines(res_img, [pts_tras], True, (0, 230, 115), 3)

                        tag = f"{nombre_m.split('-')[0].strip()} #{num_instancia}"
                        tx = x + max(6, cur_w // 6)
                        ty = y + cur_h // 2
                        cv2.putText(
                            res_img, tag, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
                        )

                        # Reservar espacio con separación de cuchilla entre moldes
                        kernel_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
                        mask_ocupar = cv2.dilate(temp_mask, kernel_sep)
                        ocupado_mask = cv2.bitwise_or(ocupado_mask, mask_ocupar)

                        area_ocupada_moldes += cv2.contourArea(pts_tras)
                        posicionado = True
                        piezas_colocadas.append({
                            "Pieza": f"{nombre_m} #{num_instancia}",
                            "Rotación": etiq_rot,
                            "Coordenada": f"({x}, {y})",
                            "Estado": "Ubicado OK"
                        })
                        break

        if not posicionado and not llenar_toda_plancha:
            piezas_no_cabieron.append(f"{nombre_m} #{num_instancia}")

    aprovechamiento_real = min(94.5, round((area_ocupada_moldes / float(area_piel_px)) * 100.0, 1)) if piezas_colocadas else 0.0
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    return res_img_rgb, res_img, piezas_colocadas, piezas_no_cabieron, aprovechamiento_real

# ------------------------------------------------------------------ Generador de Patrón Imprimible
def generar_patron_imprimible(res_img_bgr, piezas_colocadas, info_lote):
    h, w, _ = res_img_bgr.shape
    margen_sup = 110
    margen_inf = 90
    margen_lat = 50

    lienzo_h = h + margen_sup + margen_inf
    lienzo_w = w + (margen_lat * 2)
    hoja = np.ones((lienzo_h, lienzo_w, 3), dtype=np.uint8) * 255

    cv2.putText(hoja, "LEATHER-FLEX 3D - PLANO DE CORTE DE PLANCHA COMPLETA", (margen_lat, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (44, 37, 32), 2)
    cv2.putText(hoja, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Moldes: {len(piezas_colocadas)} | Defectos Evitados: {info_lote['defectos']} | Aprovechamiento: {info_lote['aprovechamiento_pct']}%",
                (margen_lat, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (100, 90, 80), 1)
    cv2.line(hoja, (margen_lat, 95), (lienzo_w - margen_lat, 95), (140, 119, 99), 2)

    area_img = res_img_bgr.copy()
    paso_grid = 50
    for gx in range(0, w, paso_grid):
        cv2.line(area_img, (gx, 0), (gx, h), (220, 220, 220), 1)
    for gy in range(0, h, paso_grid):
        cv2.line(area_img, (0, gy), (w, gy), (220, 220, 220), 1)

    hoja[margen_sup:margen_sup + h, margen_lat:margen_lat + w] = area_img
    cv2.rectangle(hoja, (margen_lat, margen_sup), (margen_lat + w, margen_sup + h), (0, 0, 0), 2)

    for cx, cy in [(margen_lat, margen_sup), (margen_lat + w, margen_sup),
                   (margen_lat, margen_sup + h), (margen_lat + w, margen_sup + h)]:
        cv2.circle(hoja, (cx, cy), 10, (0, 0, 255), 2)

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

# ================================================================== MÓDULO 1: CAPTURA DEL CUERO
if opcion_menu == "Captura del Cuero":
    st.title("Captura del Cuero")
    st.caption("Sube todas las fotos o videos de las secciones de la plancha de cuero. El sistema las unirá en una plancha entera.")

    if "bandeja_capturas" not in st.session_state:
        st.session_state["bandeja_capturas"] = []
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    with st.container(border=True):
        nota_lote = st.text_input("Identificador / Nombre de la Plancha (Opcional)", placeholder="Ej. Plancha Cuero Vacuno Completa #1", key="nota_unica")

        archivos_nuevos = st.file_uploader(
            "Seleccionar de la biblioteca, tomar foto o grabar video (puedes elegir todas las fotos de la plancha)",
            type=["jpg", "png", "jpeg", "webp", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            key=f"uploader_unico_{st.session_state['uploader_key']}"
        )

        if archivos_nuevos:
            for arch in archivos_nuevos:
                st.session_state["bandeja_capturas"].append((arch.name, arch.getvalue()))
            st.session_state["uploader_key"] += 1
            st.rerun()

        bandeja = st.session_state["bandeja_capturas"]

        if bandeja:
            st.write("---")
            st.markdown(f"**Tomas acumuladas para formar la plancha ({len(bandeja)}):**")
            cols_prev = st.columns(4)
            for idx_p, (nom_p, bytes_p) in enumerate(bandeja):
                with cols_prev[idx_p % 4]:
                    if nom_p.lower().endswith(EXTENSIONES_VIDEO):
                        st.video(bytes_p)
                        st.caption(nom_p)
                    else:
                        st.image(bytes_p, caption=nom_p, use_container_width=True)

            col_g1, col_g2 = st.columns([3, 1])
            with col_g1:
                if st.button(f"Guardar y Unir {len(bandeja)} Toma(s) en Plancha Entera", type="primary", use_container_width=True):
                    rutas_lote = []
                    for idx, (nom_item, bytes_item) in enumerate(bandeja):
                        etiqueta = f"{nota_lote} - Toma #{idx+1}" if nota_lote else f"Toma #{idx+1} ({nom_item})"
                        ruta_g = guardar_captura_en_bd(nom_item, bytes_item, etiqueta)
                        rutas_lote.append(ruta_g)

                    with st.spinner("Extrayendo vistas y ensamblando plancha de cuero completa..."):
                        imagenes_memoria = expandir_archivos_a_imagenes(bandeja)
                        bytes_plancha, msg_plancha = ensamblar_plancha_completa(imagenes_memoria)

                    if imagenes_memoria:
                        st.session_state["lote_pieles_activas"] = rutas_lote
                        st.session_state["imagenes_def_memoria"] = imagenes_memoria
                        st.session_state["pieles_nesting_memoria"] = imagenes_memoria
                    if bytes_plancha:
                        st.session_state["plancha_completa_bytes"] = bytes_plancha
                        st.session_state["plancha_completa_info"] = msg_plancha

                    st.session_state["bandeja_capturas"] = []
                    st.session_state["uploader_key"] += 1
                    st.success(f"Plancha ensamblada: {msg_plancha}")
                    st.rerun()

            with col_g2:
                if st.button("Limpiar Todo", use_container_width=True):
                    st.session_state["bandeja_capturas"] = []
                    st.session_state["uploader_key"] += 1
                    st.rerun()

    if st.session_state.get("plancha_completa_bytes"):
        st.write("---")
        st.subheader("Vista Previa de Plancha de Cuero Ensamblada")
        st.caption(st.session_state.get("plancha_completa_info", ""))
        st.image(st.session_state["plancha_completa_bytes"], use_container_width=True)

    st.write("---")
    st.subheader("Historial de Capturas Registradas")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, usuario, titulo, fecha, media_path FROM feed ORDER BY id DESC")
    posts = c.fetchall()
    conn.close()

    if not posts:
        st.info("No hay capturas aún. Usa el selector de arriba para subir las fotos de la plancha.")
    else:
        cols_hist = st.columns(3)
        for idx, p in enumerate(posts):
            post_id, usr, tit, fecha, media = p
            with cols_hist[idx % 3]:
                st.markdown(f"<div class='feed-card'><small>{fecha}</small><br><b>{tit}</b></div>", unsafe_allow_html=True)
                if media and os.path.exists(media):
                    if media.lower().endswith(EXTENSIONES_VIDEO):
                        st.video(media)
                    else:
                        st.image(media, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if media and os.path.exists(media):
                        if st.button("Seleccionar", key=f"sel_{post_id}", use_container_width=True):
                            st.session_state["ultima_captura_cuero"] = media
                            with open(media, "rb") as f_sel:
                                b_sel = f_sel.read()
                            imgs_sel = expandir_archivos_a_imagenes([(os.path.basename(media), b_sel)])
                            st.session_state["imagenes_def_memoria"] = imgs_sel
                            st.session_state["pieles_nesting_memoria"] = imgs_sel
                            b_plancha, m_plancha = ensamblar_plancha_completa(imgs_sel)
                            if b_plancha:
                                st.session_state["plancha_completa_bytes"] = b_plancha
                                st.session_state["plancha_completa_info"] = m_plancha
                            st.toast(f"'{tit}' cargado.")
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
    st.caption("Une todas las fotos en una plancha entera y detecta cada defecto (Rayón, Mancha, Hueco, Picadura) en rojo.")

    col1, col2 = st.columns([1, 1.35])
    with col1:
        st.subheader("1. Fotos / Videos de la Plancha")
        archivos_subidos = st.file_uploader(
            "Sube todas las fotos o videos de la plancha de cuero",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            key="uploader_defectos_lote"
        )

        if archivos_subidos:
            with st.spinner("Extrayendo vistas y uniendo plancha..."):
                raw_files = [(f.name, f.getvalue()) for f in archivos_subidos]
                imgs_exp = expandir_archivos_a_imagenes(raw_files)
                st.session_state["imagenes_def_memoria"] = imgs_exp
                st.session_state["pieles_nesting_memoria"] = imgs_exp

        imagenes_actuales = st.session_state.get("imagenes_def_memoria", [])

        cols_cuadricula = st.number_input(
            "Columnas para unir las fotos (0 = Automático según cantidad de fotos)",
            min_value=0, max_value=12, value=0, step=1
        )

        if imagenes_actuales:
            st.info(f"Tomas cargadas: **{len(imagenes_actuales)} foto(s)** listas para unir en una plancha entera.")
            if st.button("Limpiar fotos cargadas"):
                st.session_state["imagenes_def_memoria"] = []
                st.session_state["resultados_defectos_lote"] = []
                st.session_state["plancha_completa_bytes"] = None
                st.rerun()

        sensibilidad = st.slider(
            "Sensibilidad de detección IA (75 recomendado)",
            min_value=20, max_value=100, value=75, step=5
        )

        btn_analizar = st.button(
            f"Unir {len(imagenes_actuales)} Foto(s) y Detectar Defectos en Plancha Completa",
            type="primary",
            use_container_width=True,
            disabled=(len(imagenes_actuales) == 0)
        )

    with col2:
        st.subheader("2. Plancha Completa y Defectos Detectados")
        if btn_analizar and imagenes_actuales:
            with st.spinner("Uniendo fotos en plancha completa y ejecutando IA de detección..."):
                bytes_plancha, info_ensamble = ensamblar_plancha_completa(imagenes_actuales, columnas_cuadricula=cols_cuadricula)
                st.session_state["plancha_completa_bytes"] = bytes_plancha
                st.session_state["plancha_completa_info"] = info_ensamble

                img_viz_rgb, obst, rep, err = detectar_defectos_cuero(bytes_plancha, sensibilidad=sensibilidad)
                st.session_state["resultado_plancha_defectos"] = {
                    "info_ensamble": info_ensamble,
                    "num_fotos": len(imagenes_actuales),
                    "img_viz": img_viz_rgb,
                    "reporte": rep,
                    "obstaculos": obst,
                    "bytes_plancha": bytes_plancha,
                    "error": err
                }

        res_plancha = st.session_state.get("resultado_plancha_defectos")
        if res_plancha:
            if res_plancha["error"]:
                st.error(res_plancha["error"])
            else:
                st.caption(res_plancha["info_ensamble"])
                m_def = st.columns(2)
                m_def[0].metric("Fotos Unidas", res_plancha["num_fotos"])
                m_def[1].metric("Defectos Detectados", len(res_plancha["reporte"]))

                if len(res_plancha["reporte"]) == 0:
                    st.warning("No se detectaron defectos con la sensibilidad actual. Sube el slider a 80-90.")
                else:
                    st.success(f"IA identificó **{len(res_plancha['reporte'])} defecto(s)** marcados en rojo sobre la plancha completa.")

                st.image(res_plancha["img_viz"], use_container_width=True, caption="Plancha completa con defectos marcados en rojo")

                if res_plancha["reporte"]:
                    st.dataframe(
                        [{
                            "ID": f"D{d['id']}",
                            "Diagnóstico IA": d["tipo"],
                            "Confianza": d["confianza"],
                            "Ubicación": d["ubicacion"],
                            "Área (px²)": d["area_px2"]
                        } for d in res_plancha["reporte"]],
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.info("Carga las fotos de la plancha en el panel izquierdo y presiona el botón para unirlas y detectar defectos.")

# ================================================================== MÓDULO 3: ACOMODO DE MOLDES EN TODA LA PLANCHA
elif opcion_menu == "Acomodo de Moldes":
    st.title("Acomodo de Moldes en Toda la Plancha")
    st.caption("Junta todas las fotos en una sola plancha de cuero, identifica los defectos en rojo y distribuye los moldes abarcando TODA la superficie disponible.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("1. Fotos de la Plancha Completa")
        fotos_nest = st.file_uploader(
            "Sube todas las fotos o videos que forman la plancha de cuero",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            key="uploader_nesting_multi"
        )
        if fotos_nest:
            with st.spinner("Preparando tomas de la plancha..."):
                imgs_n = expandir_archivos_a_imagenes([(f.name, f.getvalue()) for f in fotos_nest])
                st.session_state["pieles_nesting_memoria"] = imgs_n
                st.session_state["imagenes_def_memoria"] = imgs_n
        elif "pieles_nesting_memoria" not in st.session_state and st.session_state.get("imagenes_def_memoria"):
            st.session_state["pieles_nesting_memoria"] = st.session_state["imagenes_def_memoria"]

        pieles_nest = st.session_state.get("pieles_nesting_memoria", [])
        if pieles_nest:
            st.success(f"**{len(pieles_nest)} foto(s)** listas para unirse en una sola plancha entera.")

        cols_cuad_nest = st.number_input(
            "Columnas de unión de plancha (0 = Automático)",
            min_value=0, max_value=12, value=0, step=1, key="cols_cuad_nest"
        )
        demo = st.button("Usar Muestra de Prueba (Demo)")

    with col_b:
        st.subheader("2. Moldes y Aprovechamiento de Plancha")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, nombre, cliente, dimensiones, archivo_dxf FROM moldes ORDER BY id ASC")
        moldes_bd = c.fetchall()
        conn.close()

        moldes_seleccionados = []
        if moldes_bd:
            mapa_dxf = {f"{m[1]} ({m[3]})": (m[1], m[4]) for m in moldes_bd}
            opciones = list(mapa_dxf.keys())
            elegidos = st.multiselect("Piezas a distribuir en la plancha:", options=opciones, default=opciones)
            for item_nom in elegidos:
                nom_real, path_dxf = mapa_dxf[item_nom]
                cant = st.number_input(f"Cantidad mínima de '{nom_real}'", min_value=1, max_value=30, value=1, key=f"cant_nest_{item_nom}")
                moldes_seleccionados.append({"nombre": nom_real, "cantidad": cant, "dxf_path": path_dxf})

        llenar_toda = st.checkbox(
            "Abarcar toda la plancha automáticamente (repetir moldes hasta llenar toda la piel sana)",
            value=True
        )
        escala_pieza = st.slider("Tamaño relativo de los moldes en la plancha (%)", 60, 150, 100, 10)
        margen_seg = st.slider("Distancia mínima alrededor de defectos (px)", 5, 45, 16, 2)
        sens_nest = st.slider("Sensibilidad detección IA", 20, 100, 75, 5, key="sens_nest")

    st.write("---")
    puede_procesar = (len(pieles_nest) > 0 or demo) and len(moldes_seleccionados) > 0
    if st.button("Unir Plancha Completa, Detectar Defectos y Acomodar Moldes", type="primary", use_container_width=True, disabled=not puede_procesar):
        lista_a_procesar = list(pieles_nest)
        if demo and not lista_a_procesar:
            carpeta_demo = os.path.join(TMP, "demo")
            shutil.rmtree(carpeta_demo, ignore_errors=True)
            simulador.generar(carpeta_demo)
            p_demo = os.path.join(carpeta_demo, "fotos", os.listdir(os.path.join(carpeta_demo, "fotos"))[0])
            with open(p_demo, "rb") as fd:
                lista_a_procesar = [("Muestra_Demo.jpg", fd.read())]

        with st.spinner("Reconstruyendo plancha entera, detectando defectos y distribuyendo moldes en toda la superficie..."):
            bytes_plancha_entera, info_ensamble = ensamblar_plancha_completa(lista_a_procesar, columnas_cuadricula=cols_cuad_nest)
            _, obst, rep_def, err = detectar_defectos_cuero(bytes_plancha_entera, sensibilidad=sens_nest)

            if err:
                st.session_state["resultado_nesting_plancha"] = {"error": err}
            else:
                res_img_rgb, res_img_bgr, colocadas, no_cabieron, aprov_real = ubicar_moldes_en_toda_la_plancha(
                    bytes_plancha_entera,
                    rep_def,
                    moldes_seleccionados,
                    margen_cuchilla=margen_seg,
                    llenar_toda_plancha=llenar_toda,
                    escala_pieza_pct=escala_pieza
                )
                h, w, _ = res_img_rgb.shape
                info_lote = {
                    "fotos_unidas": len(lista_a_procesar),
                    "info_ensamble": info_ensamble,
                    "piel_m2": round((w * h) / 1000000, 2),
                    "defectos": len(rep_def),
                    "piezas": len(colocadas),
                    "aprovechamiento_pct": aprov_real
                }
                png_imprimible = generar_patron_imprimible(res_img_bgr, colocadas, info_lote)
                st.session_state["resultado_nesting_plancha"] = {
                    "error": None,
                    "res_img": res_img_rgb,
                    "colocadas": colocadas,
                    "no_cabieron": no_cabieron,
                    "info": info_lote,
                    "png_imprimible": png_imprimible
                }

    rn = st.session_state.get("resultado_nesting_plancha")
    if rn:
        if rn["error"]:
            st.error(rn["error"])
        else:
            info = rn["info"]
            st.caption(info["info_ensamble"])
            st.success(f"Listo: **{info['piezas']} moldes ubicados** abarcando toda la plancha, ninguno sobre un defecto.")

            m = st.columns(4)
            m[0].metric("Fotos Unidas", info["fotos_unidas"])
            m[1].metric("Defectos Evitados", info["defectos"])
            m[2].metric("Moldes Ubicados", info["piezas"])
            m[3].metric("Aprovechamiento", f"{info['aprovechamiento_pct']} %")

            if rn["no_cabieron"]:
                st.warning(f"No hubo espacio libre suficiente para: {', '.join(rn['no_cabieron'])}. Reduce el tamaño relativo o el margen de seguridad.")

            st.image(rn["res_img"], use_container_width=True, caption="Moldes ubicados lejos de los defectos abarcando toda la plancha")

            col_tbl, col_imp = st.columns([1.4, 1])
            with col_tbl:
                if rn["colocadas"]:
                    st.write("**Detalle de Distribución en Plancha:**")
                    st.dataframe(rn["colocadas"], use_container_width=True, hide_index=True)
            with col_imp:
                st.write("**Exportar Plan de Corte:**")
                if rn["png_imprimible"]:
                    st.download_button(
                        label="Descargar Plan de Corte Imprimible (PNG)",
                        data=rn["png_imprimible"],
                        file_name="Plan_Corte_Plancha_Completa.png",
                        mime="image/png",
                        type="primary",
                        use_container_width=True
                    )
                pts_dxf_all = []
                for m_sel in moldes_seleccionados:
                    pts_dxf_all.extend(obtener_puntos_molde_calzado(m_sel["nombre"]).tolist())
                st.download_button(
                    label="Descargar Vector CAD (DXF)",
                    data=generar_contenido_dxf(pts_dxf_all),
                    file_name="Plan_Corte_Plancha.dxf",
                    mime="application/dxf",
                    use_container_width=True
                )

# ================================================================== MÓDULO 4: CATÁLOGO DE MOLDES
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
                    "Imprimir Plantilla (PNG)",
                    data=buf_m.tobytes(),
                    file_name=f"Plantilla_{m[1].replace(' ', '_')}.png",
                    mime="image/png",
                    key=f"imp_m_{m[0]}",
                    use_container_width=True
                )
            with c_b2:
                st.download_button(
                    "Descargar DXF",
                    data=generar_contenido_dxf(pts_molde),
                    file_name=f"{m[1].replace(' ', '_')}.dxf",
                    mime="application/dxf",
                    key=f"dxf_m_{m[0]}",
                    use_container_width=True
                )

# ================================================================== MÓDULO 5: CALIBRACIÓN DE ESCÁNER
elif opcion_menu == "Calibración de Escáner":
    st.title("Calibración del Sistema Óptico")
    st.caption("Ajuste de escala real en milímetros por píxel y corrección de plano.")

    col1, col2 = st.columns(2)
    with col1:
        ruta_tab = os.path.join(TMP, "tablero_para_imprimir.png")
        if not os.path.exists(ruta_tab):
            calibracion.tablero_para_imprimir(ruta_tab)
        st.download_button("Descargar Tablero de Calibración (PNG)", open(ruta_tab, "rb"), "tablero_calibracion.png", "image/png")
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
