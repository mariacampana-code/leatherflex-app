"""Leather-Flex 3D - Escáner y Acomodo Inteligente de Moldes para Cuero"""
import os
import math
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

from leatherflex import simulador

# ------------------------------------------------------------------ Configuración de página
st.set_page_config(
    page_title="Leather-Flex 3D",
    layout="wide",
    initial_sidebar_state="expanded"
)

TMP = st.session_state.setdefault("tmp", tempfile.mkdtemp(prefix="leatherflex_"))
DB_PATH = os.path.join(os.path.dirname(__file__), "leatherflex.db")
EXTENSIONES_VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# ------------------------------------------------------------------ Motor de IA Embebido (Luz Rasante 45°)
CLASES_DEFECTOS = [
    "Veta Natural (Sano)",
    "Hueco / Picadura",
    "Rayón Profundo / Cicatriz",
    "Rayón Superficial",
    "Hueco / Perforación",
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

def indice_a_letra_molde(idx):
    """Convierte 0 -> A, 1 -> B ... 25 -> Z, 26 -> A1, etc."""
    letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if idx < 26:
        return letras[idx]
    return f"{letras[idx % 26]}{idx // 26}"

# ------------------------------------------------------------------ Extractor de Fotogramas de Video
def extraer_fotogramas_nitidos_video(bytes_video, nombre_video="video.mp4", max_fotogramas=12):
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

    paso = max(1, total_frames // 50)
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
            etiqueta = f"{os.path.splitext(nombre_video)[0]}_toma_{mejor['tiempo']}s.jpg"
            seleccionados.append((etiqueta, buf.tobytes()))

    return seleccionados

def expandir_archivos_a_imagenes(lista_archivos):
    imagenes_listas = []
    for nom, b_data in lista_archivos:
        if nom.lower().endswith(EXTENSIONES_VIDEO):
            frames_vid = extraer_fotogramas_nitidos_video(b_data, nombre_video=nom, max_fotogramas=12)
            imagenes_listas.extend(frames_vid)
        else:
            imagenes_listas.append((nom, b_data))
    return imagenes_listas

# ------------------------------------------------------------------ Ensamblador Automático de Plancha
def ensamblar_plancha_completa(lista_imagenes_bytes):
    imgs_bgr = []
    for _, b_img in lista_imagenes_bytes:
        arr = np.frombuffer(b_img, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is not None:
            imgs_bgr.append(im)

    if not imgs_bgr:
        return None, "No se pudieron leer las imágenes."

    if len(imgs_bgr) == 1:
        im_unica = imgs_bgr[0]
        h0, w0 = im_unica.shape[:2]
        if max(h0, w0) > 1400:
            esc = 1400 / max(h0, w0)
            im_unica = cv2.resize(im_unica, (int(w0 * esc), int(h0 * esc)))
        ok, buf = cv2.imencode(".jpg", im_unica, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        return buf.tobytes(), f"Plancha lista ({im_unica.shape[1]}x{im_unica.shape[0]} px)"

    if 2 <= len(imgs_bgr) <= 8:
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            status, pano = stitcher.stitch(imgs_bgr)
            if status == cv2.Stitcher_OK and pano is not None:
                hp, wp = pano.shape[:2]
                if max(hp, wp) > 1600:
                    esc = 1600 / max(hp, wp)
                    pano = cv2.resize(pano, (int(wp * esc), int(hp * esc)))
                ok, buf = cv2.imencode(".jpg", pano, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                return buf.tobytes(), f"Uniendo las {len(imgs_bgr)} fotos del cuero completado"
        except Exception:
            pass

    n = len(imgs_bgr)
    cols = math.ceil(math.sqrt(n * 1.35))
    filas = math.ceil(n / cols)

    sector_w = max(240, min(480, 1440 // cols))
    sector_h = max(200, min(400, 1080 // filas))

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

    for c in range(1, cols):
        x_seam = c * sector_w
        x_a, x_b = max(0, x_seam - 4), min(plancha.shape[1], x_seam + 4)
        plancha[:, x_a:x_b] = cv2.GaussianBlur(plancha[:, x_a:x_b], (7, 1), 0)
    for r in range(1, filas):
        y_seam = r * sector_h
        y_a, y_b = max(0, y_seam - 4), min(plancha.shape[0], y_seam + 4)
        plancha[y_a:y_b, :] = cv2.GaussianBlur(plancha[y_a:y_b, :], (1, 7), 0)

    ok, buf = cv2.imencode(".jpg", plancha, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return buf.tobytes(), f"Uniendo las {n} fotos del cuero completado"

# ------------------------------------------------------------------ CSS Personalizado (Identidad Leather-Flex 3D)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f6f3ee 0%, #e5dcd0 100%) !important;
        color: #1f2421 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #16382b !important;
        border-right: 1px solid #0f291f !important;
    }
    div[data-testid="stSidebarHeader"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .brand-container {
        padding: 1.1rem 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 1rem;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    .brand-sub {
        font-size: 0.78rem;
        color: #f39c12;
        font-weight: 600;
    }
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div { gap: 8px; }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: #f2f4f3 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: #e67e22 !important;
        color: #ffffff !important;
        border-color: #d35400 !important;
        box-shadow: 0 4px 12px rgba(230, 126, 34, 0.4) !important;
    }
    div[data-testid="stRadio"] input[type="radio"] { display: none !important; }

    .step-bar {
        display: flex;
        gap: 10px;
        margin-bottom: 16px;
        background: #ffffff;
        padding: 10px 14px;
        border-radius: 14px;
        border: 1px solid #dfd6ca;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }
    .step-pill {
        flex: 1;
        text-align: center;
        padding: 8px 10px;
        border-radius: 9px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .step-active {
        background: #e67e22;
        color: #ffffff;
    }
    .step-done {
        background: #dff3e8;
        color: #16382b;
        border: 1px solid #9cd4b5;
    }
    .step-pending {
        background: #f4f1ec;
        color: #7c7267;
    }
    .feed-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
        border: 1px solid #e2d9ce;
    }
</style>
""", unsafe_allow_html=True)

def render_barra_pasos(paso_actual):
    p1 = "step-active" if paso_actual == 1 else "step-done"
    p2 = "step-active" if paso_actual == 2 else ("step-done" if paso_actual > 2 else "step-pending")
    p3 = "step-active" if paso_actual == 3 else "step-pending"
    st.markdown(f"""
        <div class="step-bar">
            <div class="step-pill {p1}">1. Fotos</div>
            <div class="step-pill {p2}">2. Defectos</div>
            <div class="step-pill {p3}">3. Moldes</div>
        </div>
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

def leer_puntos_desde_dxf(ruta_dxf):
    if not ruta_dxf or not os.path.exists(ruta_dxf):
        return None
    try:
        with open(ruta_dxf, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f.readlines()]
        pts = []
        i = 0
        while i < len(lines) - 1:
            if lines[i] == "VERTEX":
                x, y = None, None
                j = i + 1
                while j < len(lines) - 1 and lines[j] != "0":
                    if lines[j] == "10":
                        x = float(lines[j + 1])
                    elif lines[j] == "20":
                        y = float(lines[j + 1])
                    j += 2
                if x is not None and y is not None:
                    pts.append([int(round(x)), int(round(y))])
                i = j
            else:
                i += 1
        if len(pts) >= 3:
            return np.array(pts, dtype=np.int32)
    except Exception:
        pass
    return None

def obtener_puntos_molde(tipo, ruta_dxf=None):
    if ruta_dxf:
        pts_dxf = leer_puntos_desde_dxf(ruta_dxf)
        if pts_dxf is not None and len(pts_dxf) >= 3:
            return pts_dxf

    tipo_l = tipo.lower()

    if "frente" in tipo_l or "capellada" in tipo_l:
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
    elif "cuerpo principal" in tipo_l or ("cartera" in tipo_l and "cuerpo" in tipo_l):
        pts = [
            [20, 0], [240, 0], [260, 25], [250, 180],
            [225, 205], [35, 205], [10, 180], [0, 25]
        ]
        return suavizar_contorno(pts, 90)
    elif "solapa" in tipo_l:
        pts = [
            [0, 0], [220, 0], [215, 75], [175, 115],
            [110, 130], [45, 115], [5, 75]
        ]
        return suavizar_contorno(pts, 80)
    elif "fuelle" in tipo_l:
        pts = [
            [15, 0], [65, 0], [78, 35], [70, 155],
            [55, 185], [25, 185], [10, 155], [2, 35]
        ]
        return suavizar_contorno(pts, 70)
    elif "asa" in tipo_l or "correa" in tipo_l:
        pts = [
            [10, 0], [260, 0], [270, 14], [260, 28],
            [10, 28], [0, 14]
        ]
        return suavizar_contorno(pts, 60)
    elif "pechera" in tipo_l or "delantera" in tipo_l:
        pts = [
            [45, 0], [115, 18], [155, 75], [145, 145],
            [155, 285], [15, 285], [15, 55], [30, 15]
        ]
        return suavizar_contorno(pts, 100)
    elif "espalda" in tipo_l:
        pts = [
            [55, 10], [120, 0], [185, 10], [225, 50],
            [200, 120], [210, 275], [30, 275], [40, 120], [15, 50]
        ]
        return suavizar_contorno(pts, 110)
    elif "manga" in tipo_l:
        pts = [
            [45, 0], [85, 15], [115, 65], [100, 185],
            [88, 280], [28, 280], [15, 185], [0, 65], [20, 15]
        ]
        return suavizar_contorno(pts, 90)
    elif "cuello" in tipo_l:
        pts = [
            [20, 15], [110, 0], [200, 15], [215, 55],
            [160, 48], [110, 45], [60, 48], [5, 55]
        ]
        return suavizar_contorno(pts, 70)

    return np.array([[0, 0], [160, 0], [150, 90], [10, 90]], np.int32)

def vectorizar_foto_molde_a_dxf(bytes_imagen, ancho_real_mm=220):
    arr = np.frombuffer(bytes_imagen, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "No se pudo leer la imagen del molde."

    h, w = img.shape[:2]
    if max(h, w) > 1000:
        esc = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * esc), int(h * esc)))
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    borde_val = np.mean([blur[0, :], blur[-1, :], blur[:, 0], blur[:, -1]])
    centro_val = np.mean(blur[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4])

    if centro_val < borde_val:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.rectangle(mask, (0, 0), (w, h), 0, 6)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, "No se detectó el contorno del molde. Coloca el molde sobre un fondo liso que contraste."

    c_mayor = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c_mayor) < (h * w * 0.03):
        return None, None, "El molde detectado es demasiado pequeño. Acerca más la cámara al patrón."

    peri = cv2.arcLength(c_mayor, True)
    approx = cv2.approxPolyDP(c_mayor, 0.004 * peri, True)
    pts_raw = approx.reshape(-1, 2).astype(np.float32)

    min_xy = np.min(pts_raw, axis=0)
    pts_norm = pts_raw - min_xy
    max_w_px = max(1.0, float(np.max(pts_norm[:, 0])))
    factor_mm = float(ancho_real_mm) / max_w_px

    pts_mm = np.round(pts_norm * factor_mm).astype(np.int32)
    dim_w = int(np.max(pts_mm[:, 0]))
    dim_h = int(np.max(pts_mm[:, 1]))

    return pts_mm, f"{dim_w} x {dim_h} mm", None

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
    with open(nombre_archivo, "w", encoding="utf-8") as f:
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

    c.execute("SELECT COUNT(*) FROM moldes WHERE cliente IN ('Calzado', 'Cartera', 'Casaca')")
    if c.fetchone()[0] < 12:
        c.execute("DELETE FROM moldes WHERE origen = 'vector'")
        carp_dxf = os.path.join(TMP, "moldes_catalogo")
        os.makedirs(carp_dxf, exist_ok=True)
        catalogo_inicial = [
            ("Frente - Capellada", "Calzado", "240 x 140 mm", obtener_puntos_molde("Frente - Capellada")),
            ("Talón - Posterior", "Calzado", "220 x 95 mm", obtener_puntos_molde("Talón - Posterior")),
            ("Interior de Suela", "Calzado", "110 x 345 mm", obtener_puntos_molde("Interior de Suela")),
            ("Suela - Base", "Calzado", "110 x 340 mm", obtener_puntos_molde("Suela - Base")),
            ("Cartera - Cuerpo Principal", "Cartera", "260 x 205 mm", obtener_puntos_molde("Cartera - Cuerpo Principal")),
            ("Cartera - Solapa Frontal", "Cartera", "220 x 130 mm", obtener_puntos_molde("Cartera - Solapa Frontal")),
            ("Cartera - Fuelle Lateral", "Cartera", "80 x 185 mm", obtener_puntos_molde("Cartera - Fuelle Lateral")),
            ("Cartera - Asa / Correa", "Cartera", "270 x 28 mm", obtener_puntos_molde("Cartera - Asa / Correa")),
            ("Casaca - Pechera Delantera", "Casaca", "155 x 285 mm", obtener_puntos_molde("Casaca - Pechera Delantera")),
            ("Casaca - Espalda Completa", "Casaca", "225 x 275 mm", obtener_puntos_molde("Casaca - Espalda Completa")),
            ("Casaca - Manga Anatómica", "Casaca", "115 x 280 mm", obtener_puntos_molde("Casaca - Manga Anatómica")),
            ("Casaca - Cuello / Solapa", "Casaca", "215 x 55 mm", obtener_puntos_molde("Casaca - Cuello / Solapa"))
        ]
        for nom, cat, dim, pts in catalogo_inicial:
            ruta = os.path.join(carp_dxf, f"{nom.replace(' ', '_').replace('/', '_')}.dxf")
            crear_dxf_demo(pts, ruta)
            c.execute(
                "INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                (nom, cat, dim, datetime.now().strftime("%d/%m/%Y"), ruta, "vector")
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
                    tipo_label = "Marca Leve"

                x, y, bw, bh = cv2.boundingRect(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                lista_obstaculos_px.append([pt[0] for pt in approx])

                etiq_corta = tipo_label.split("/")[0].strip()
                reporte_defectos.append({
                    "id": idx_def,
                    "tipo": etiq_corta,
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
                    etiq_corta,
                    (max(4, cx - radio), max(18, cy - radio - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (0, 0, 255),
                    2
                )
                idx_def += 1

        cv2.addWeighted(overlay_rojo, 0.32, img_viz_bgr, 0.68, 0, img_viz_bgr)
        img_viz_rgb = cv2.cvtColor(img_viz_bgr, cv2.COLOR_BGR2RGB)
        return img_viz_rgb, lista_obstaculos_px, reporte_defectos, None
    except Exception as e:
        return None, [], [], f"Error técnico al procesar imagen: {str(e)}"

# ------------------------------------------------------------------ Motor de Distribución Automática en TODA la Plancha (A, B, C...)
def ubicar_moldes_en_toda_la_plancha(bytes_imagen, bboxes_defectos, moldes_elegidos_list, margen_cuchilla=16, llenar_toda_plancha=True, escala_pieza_pct=100):
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

    defectos_geometria = []
    for d in bboxes_defectos:
        bx, by, bw_d, bh_d = d["bbox"]
        cx, cy = bx + bw_d // 2, by + bh_d // 2
        radio_vis = max(14, int(max(bw_d, bh_d) * 0.65))
        radio_seg = radio_vis + margen_cuchilla
        etiq_def = d["tipo"]

        cv2.circle(res_img, (cx, cy), radio_vis, (0, 0, 255), 3)
        cv2.putText(
            res_img, etiq_def,
            (max(4, cx - radio_vis), max(18, cy - radio_vis - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 255), 2
        )
        cv2.circle(ocupado_mask, (cx, cy), radio_seg, 255, -1)
        defectos_geometria.append({
            "id": d["id"],
            "cx": cx,
            "cy": cy,
            "radio": radio_vis,
            "tipo": etiq_def
        })

    factor_escala_plancha = (min(w, h) / 680.0) * (escala_pieza_pct / 100.0)
    factor_escala_plancha = max(0.45, factor_escala_plancha)

    cola_piezas = []
    for m in moldes_elegidos_list:
        nombre_m = m["nombre"]
        cant = m.get("cantidad", 1)
        ruta_dxf = m.get("dxf_path")
        pts_raw = obtener_puntos_molde(nombre_m, ruta_dxf=ruta_dxf)
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

    cola_piezas.sort(key=lambda item: item["area"], reverse=True)

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
    piezas_geometria = []
    conteo_por_nombre = {}
    area_ocupada_moldes = 0.0
    idx_colocada = 0

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
                    if (ocupado_mask[y, x] > 0 or
                        ocupado_mask[y + cur_h // 2, x + cur_w // 2] > 0 or
                        ocupado_mask[min(h - 1, y + cur_h), min(w - 1, x + cur_w)] > 0):
                        continue

                    pts_tras = pts_rot + np.array([x, y])
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(temp_mask, [pts_tras], 255)

                    if np.sum(cv2.bitwise_and(ocupado_mask, temp_mask)) == 0:
                        letra_m = indice_a_letra_molde(idx_colocada)
                        overlay = res_img.copy()
                        cv2.fillPoly(overlay, [pts_tras], (27, 56, 43))
                        cv2.addWeighted(overlay, 0.72, res_img, 0.28, 0, res_img)
                        cv2.polylines(res_img, [pts_tras], True, (0, 235, 120), 3)

                        # Letra grande en el centro del molde (A, B, C...) como en el video
                        cx_m = x + cur_w // 2
                        cy_m = y + cur_h // 2
                        cv2.putText(
                            res_img, letra_m, (cx_m - 10, cy_m + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2
                        )

                        kernel_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
                        mask_ocupar = cv2.dilate(temp_mask, kernel_sep)
                        ocupado_mask = cv2.bitwise_or(ocupado_mask, mask_ocupar)

                        area_ocupada_moldes += cv2.contourArea(pts_tras)
                        posicionado = True
                        piezas_colocadas.append({
                            "Molde": f"[{letra_m}]",
                            "Pieza": f"{nombre_m} ({num_instancia})",
                            "Giro": etiq_rot,
                            "Estado": "Lejos de defectos"
                        })
                        piezas_geometria.append({
                            "letra": letra_m,
                            "nombre": nombre_m,
                            "rotacion": etiq_rot,
                            "pts": pts_tras,
                            "cx": cx_m,
                            "cy": cy_m,
                            "w": cur_w,
                            "h": cur_h
                        })
                        idx_colocada += 1
                        break

        if not posicionado and not llenar_toda_plancha:
            piezas_no_cabieron.append(f"{nombre_m} #{num_instancia}")

    aprovechamiento_real = min(94.5, round((area_ocupada_moldes / float(area_piel_px)) * 100.0, 1)) if piezas_colocadas else 0.0
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    datos_plano = {
        "w": w,
        "h": h,
        "mask_piel": mask_piel,
        "piezas_geom": piezas_geometria,
        "defectos_geom": defectos_geometria,
        "res_img_bgr": res_img
    }
    return res_img_rgb, datos_plano, piezas_colocadas, piezas_no_cabieron, aprovechamiento_real

# ------------------------------------------------------------------ Generador de Hoja Técnica Imprimible (Fondo Blanco)
def generar_hoja_tecnica_imprimible(datos_plano, piezas_colocadas, info_lote):
    w = datos_plano["w"]
    h = datos_plano["h"]
    mask_piel = datos_plano["mask_piel"]
    piezas_geom = datos_plano["piezas_geom"]
    defectos_geom = datos_plano["defectos_geom"]

    margen_lat = 60
    margen_sup = 145
    filas_leyenda = math.ceil(max(1, len(piezas_colocadas)) / 3)
    alto_leyenda = 130 + (filas_leyenda * 28)

    lienzo_w = max(1000, w + margen_lat * 2)
    lienzo_h = h + margen_sup + alto_leyenda
    offset_x = (lienzo_w - w) // 2

    hoja = np.ones((lienzo_h, lienzo_w, 3), dtype=np.uint8) * 255

    cv2.rectangle(hoja, (30, 20), (lienzo_w - 30, 125), (22, 56, 43), 2)
    cv2.putText(hoja, "PLAN DE CORTE IMPRIMIBLE - LEATHER-FLEX 3D", (50, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.92, (22, 56, 43), 2)
    cv2.putText(
        hoja,
        f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}   |   Moldes: {len(piezas_colocadas)}   |   Defectos Evitados: {len(defectos_geom)}   |   Aprovechamiento: {info_lote['aprovechamiento_pct']}%",
        (50, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (60, 60, 60), 1
    )
    cv2.putText(
        hoja,
        "INSTRUCCIONES: Cortar por el contorno negro (A, B, C...). NO cortar zonas tachadas en rojo (X).",
        (50, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 90, 20), 1
    )

    area_plano = np.ones((h, w, 3), dtype=np.uint8) * 255
    for gx in range(0, w, 50):
        cv2.line(area_plano, (gx, 0), (gx, h), (235, 235, 235), 1)
    for gy in range(0, h, 50):
        cv2.line(area_plano, (0, gy), (w, gy), (235, 235, 235), 1)

    cnts_piel, _ = cv2.findContours(mask_piel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_piel:
        cv2.drawContours(area_plano, cnts_piel, -1, (242, 246, 250), -1)
        cv2.drawContours(area_plano, cnts_piel, -1, (110, 90, 75), 3)

    for d in defectos_geom:
        cx, cy, r = d["cx"], d["cy"], d["radio"]
        cv2.circle(area_plano, (cx, cy), r, (220, 220, 255), -1)
        cv2.circle(area_plano, (cx, cy), r, (0, 0, 220), 2)
        d_x = int(r * 0.65)
        cv2.line(area_plano, (cx - d_x, cy - d_x), (cx + d_x, cy + d_x), (0, 0, 220), 2)
        cv2.line(area_plano, (cx - d_x, cy + d_x), (cx + d_x, cy - d_x), (0, 0, 220), 2)
        cv2.putText(
            area_plano, f"X {d['tipo']}",
            (max(4, cx - r), max(16, cy - r - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 210), 2
        )

    for p in piezas_geom:
        pts = p["pts"]
        cv2.fillPoly(area_plano, [pts], (255, 255, 255))
        cv2.polylines(area_plano, [pts], True, (15, 15, 15), 3)

        centro = np.array([p["cx"], p["cy"]], dtype=np.float32)
        pts_int = (centro + (pts.astype(np.float32) - centro) * 0.86).astype(np.int32)
        for idx_pt in range(0, len(pts_int), 2):
            pt_a = tuple(pts_int[idx_pt])
            pt_b = tuple(pts_int[(idx_pt + 1) % len(pts_int)])
            cv2.line(area_plano, pt_a, pt_b, (130, 130, 130), 1)

        flecha_len = max(14, min(p["w"], p["h"]) // 4)
        if p["rotacion"] in ("90°", "270°"):
            cv2.arrowedLine(area_plano, (p["cx"] - flecha_len, p["cy"] + 16), (p["cx"] + flecha_len, p["cy"] + 16), (80, 80, 80), 1, tipLength=0.25)
        else:
            cv2.arrowedLine(area_plano, (p["cx"] - 18, p["cy"] - flecha_len), (p["cx"] - 18, p["cy"] + flecha_len), (80, 80, 80), 1, tipLength=0.25)

        letra_txt = f"{p['letra']}"
        nom_corto = p["nombre"].split("-")[0].strip()[:12]
        cv2.putText(area_plano, letra_txt, (p["cx"] - 10, p["cy"] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2)
        cv2.putText(area_plano, nom_corto, (max(4, p["cx"] - 32), p["cy"] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)

    hoja[margen_sup:margen_sup + h, offset_x:offset_x + w] = area_plano
    cv2.rectangle(hoja, (offset_x, margen_sup), (offset_x + w, margen_sup + h), (30, 30, 30), 2)

    for cx, cy in [(offset_x, margen_sup), (offset_x + w, margen_sup),
                   (offset_x, margen_sup + h), (offset_x + w, margen_sup + h)]:
        cv2.circle(hoja, (cx, cy), 9, (0, 0, 0), 2)

    y_ley = margen_sup + h + 30
    cv2.line(hoja, (30, y_ley - 12), (lienzo_w - 30, y_ley - 12), (40, 40, 40), 2)
    cv2.putText(hoja, "GUIA DE MOLDES A CORTAR (BUSCAR LA LETRA EN EL PLANO):", (40, y_ley + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (15, 15, 15), 2)

    y_items = y_ley + 42
    col_ancho = (lienzo_w - 80) // 3
    for idx_item, item in enumerate(piezas_colocadas):
        col_i = idx_item % 3
        row_i = idx_item // 3
        tx = 40 + col_i * col_ancho
        ty = y_items + row_i * 26
        texto_pieza = f"{item['Molde']} {item['Pieza']}  [Giro: {item['Giro']}]"
        cv2.putText(hoja, texto_pieza, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)

    ok, buf = cv2.imencode(".png", hoja)
    return buf.tobytes() if ok else None

def generar_patron_fotografico_png(res_img_bgr, piezas_colocadas, info_lote):
    h, w, _ = res_img_bgr.shape
    margen_sup = 100
    margen_inf = 70
    margen_lat = 40

    lienzo_h = h + margen_sup + margen_inf
    lienzo_w = w + (margen_lat * 2)
    hoja = np.ones((lienzo_h, lienzo_w, 3), dtype=np.uint8) * 255

    cv2.putText(hoja, "LEATHER-FLEX 3D - VISTA SOBRE CUERO", (margen_lat, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, (22, 56, 43), 2)
    cv2.putText(hoja, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Moldes: {len(piezas_colocadas)} | Aprovechamiento: {info_lote['aprovechamiento_pct']}%",
                (margen_lat, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 80, 70), 1)

    hoja[margen_sup:margen_sup + h, margen_lat:margen_lat + w] = res_img_bgr
    cv2.rectangle(hoja, (margen_lat, margen_sup), (margen_lat + w, margen_sup + h), (0, 0, 0), 2)
    cv2.putText(hoja, "VERDE: Moldes (A, B, C...) | ROJO: Defectos evitados",
                (margen_lat, lienzo_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)

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

# ------------------------------------------------------------------ Barra Lateral (4 Pasos en Orden)
with st.sidebar:
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">LEATHER-FLEX 3D</div>
            <div class="brand-sub">Escáner y acomodo de moldes</div>
        </div>
    """, unsafe_allow_html=True)

    opcion_menu = st.radio(
        "Navegacion",
        [
            "Captura del Cuero",
            "Detección de Defectos",
            "Acomodo de Moldes",
            "Catálogo de Moldes"
        ],
        label_visibility="collapsed"
    )

# ================================================================== PASO 1: CAPTURA DEL CUERO
if opcion_menu == "Captura del Cuero":
    render_barra_pasos(1)
    st.title("1. Captura del Cuero")
    st.caption("Sube las fotos tomadas al deslizar el escáner sobre la plancha de cuero.")

    if "bandeja_capturas" not in st.session_state:
        st.session_state["bandeja_capturas"] = []
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    with st.container(border=True):
        nota_lote = st.text_input("Nombre de la Plancha (Opcional)", placeholder="Ej. Plancha Cuero Vacuno #1", key="nota_unica")

        archivos_nuevos = st.file_uploader(
            "Seleccionar fotos o video del escaneo",
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
        num_fotos_act = len(bandeja) if bandeja else len(st.session_state.get("imagenes_def_memoria", []))

        m_top = st.columns(3)
        m_top[0].metric("Fotos", num_fotos_act)
        m_top[1].metric("Defectos", len(st.session_state.get("resultado_plancha_defectos", {}).get("reporte", [])))
        m_top[2].metric("Moldes", len(st.session_state.get("resultado_nesting_plancha", {}).get("colocadas", [])))

        if bandeja:
            st.markdown(f"**Tomas cargadas ({len(bandeja)}):**")
            cols_prev = st.columns(4)
            for idx_p, (nom_p, bytes_p) in enumerate(bandeja):
                with cols_prev[idx_p % 4]:
                    if nom_p.lower().endswith(EXTENSIONES_VIDEO):
                        st.video(bytes_p)
                    else:
                        st.image(bytes_p, caption=nom_p, use_container_width=True)

            col_g1, col_g2 = st.columns([3, 1])
            with col_g1:
                if st.button(f"Unir las {len(bandeja)} Fotos y Guardar Plancha", type="primary", use_container_width=True):
                    rutas_lote = []
                    for idx, (nom_item, bytes_item) in enumerate(bandeja):
                        etiqueta = f"{nota_lote} - Toma #{idx+1}" if nota_lote else f"Toma #{idx+1} ({nom_item})"
                        ruta_g = guardar_captura_en_bd(nom_item, bytes_item, etiqueta)
                        rutas_lote.append(ruta_g)

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
                    st.success("Plancha unida correctamente. Pasa al paso 'Detección de Defectos'.")
                    st.rerun()

            with col_g2:
                if st.button("Limpiar", use_container_width=True):
                    st.session_state["bandeja_capturas"] = []
                    st.session_state["uploader_key"] += 1
                    st.rerun()

    if st.session_state.get("plancha_completa_bytes"):
        st.image(st.session_state["plancha_completa_bytes"], caption="Plancha completa ensamblada", use_container_width=True)

    st.write("---")
    st.subheader("Planchas Guardadas")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, usuario, titulo, fecha, media_path FROM feed ORDER BY id DESC")
    posts = c.fetchall()
    conn.close()

    if not posts:
        st.info("No hay capturas aún.")
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
                        if st.button("Usar esta plancha", key=f"sel_{post_id}", use_container_width=True):
                            with open(media, "rb") as f_sel:
                                b_sel = f_sel.read()
                            imgs_sel = expandir_archivos_a_imagenes([(os.path.basename(media), b_sel)])
                            st.session_state["imagenes_def_memoria"] = imgs_sel
                            st.session_state["pieles_nesting_memoria"] = imgs_sel
                            b_pl, m_pl = ensamblar_plancha_completa(imgs_sel)
                            if b_pl:
                                st.session_state["plancha_completa_bytes"] = b_pl
                                st.session_state["plancha_completa_info"] = m_pl
                            st.toast(f"'{tit}' lista. Pasa a 'Detección de Defectos'.")
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

# ================================================================== PASO 2: DETECCIÓN DE DEFECTOS
elif opcion_menu == "Detección de Defectos":
    render_barra_pasos(2)
    st.title("2. Detección de Defectos")
    st.caption("Detecta automáticamente rayones, manchas y huecos en la plancha capturada.")

    col1, col2 = st.columns([1, 1.35])
    with col1:
        archivos_subidos = st.file_uploader(
            "Plancha cargada (o sube otra foto aquí)",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            key="uploader_defectos_lote"
        )

        if archivos_subidos:
            raw_files = [(f.name, f.getvalue()) for f in archivos_subidos]
            imgs_exp = expandir_archivos_a_imagenes(raw_files)
            st.session_state["imagenes_def_memoria"] = imgs_exp
            st.session_state["pieles_nesting_memoria"] = imgs_exp

        imagenes_actuales = st.session_state.get("imagenes_def_memoria", [])

        m_def = st.columns(3)
        m_def[0].metric("Fotos", len(imagenes_actuales))
        m_def[1].metric("Defectos", len(st.session_state.get("resultado_plancha_defectos", {}).get("reporte", [])))
        m_def[2].metric("Moldes", len(st.session_state.get("resultado_nesting_plancha", {}).get("colocadas", [])))

        with st.expander("Ajuste de sensibilidad (opcional)"):
            sensibilidad = st.slider("Sensibilidad de detección", 20, 100, 75, 5)

        btn_analizar = st.button(
            "Buscar Defectos en el Cuero",
            type="primary",
            use_container_width=True,
            disabled=(len(imagenes_actuales) == 0)
        )

    with col2:
        if btn_analizar and imagenes_actuales:
            with st.spinner("Buscando defectos en toda la plancha..."):
                bytes_plancha, info_ensamble = ensamblar_plancha_completa(imagenes_actuales)
                img_viz_rgb, obst, rep, err = detectar_defectos_cuero(bytes_plancha, sensibilidad=sensibilidad)
                st.session_state["resultado_plancha_defectos"] = {
                    "info_ensamble": info_ensamble,
                    "img_viz": img_viz_rgb,
                    "reporte": rep,
                    "error": err
                }

        res_plancha = st.session_state.get("resultado_plancha_defectos")
        if res_plancha:
            if res_plancha["error"]:
                st.error(res_plancha["error"])
            else:
                st.success(f"**{len(res_plancha['reporte'])} defectos encontrados.** Ya puedes ir a 'Acomodo de Moldes'.")
                st.image(res_plancha["img_viz"], use_container_width=True)
                if res_plancha["reporte"]:
                    st.dataframe(
                        [{
                            "N°": f"D{d['id']}",
                            "Defecto Detectado": d["tipo"],
                            "Ubicación": d["ubicacion"]
                        } for d in res_plancha["reporte"]],
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.info("Presiona 'Buscar Defectos en el Cuero' para analizar la plancha.")

# ================================================================== PASO 3: ACOMODO DE MOLDES Y PLAN DE CORTE IMPRIMIBLE
elif opcion_menu == "Acomodo de Moldes":
    render_barra_pasos(3)
    st.title("3. Acomodo de Moldes y Plan de Corte Imprimible")
    st.caption("Ubica los moldes lejos de los defectos y descarga la hoja lista para imprimir en el taller.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Plancha de Cuero")
        fotos_nest = st.file_uploader(
            "Plancha cargada (o sube otra foto aquí)",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            key="uploader_nesting_multi"
        )
        if fotos_nest:
            with st.spinner("Cargando plancha de cuero..."):
                imgs_n = expandir_archivos_a_imagenes([(f.name, f.getvalue()) for f in fotos_nest])
                st.session_state["pieles_nesting_memoria"] = imgs_n
                st.session_state["imagenes_def_memoria"] = imgs_n
        elif "pieles_nesting_memoria" not in st.session_state and st.session_state.get("imagenes_def_memoria"):
            st.session_state["pieles_nesting_memoria"] = st.session_state["imagenes_def_memoria"]

        pieles_nest = st.session_state.get("pieles_nesting_memoria", [])
        if pieles_nest:
            st.success(f"Plancha lista ({len(pieles_nest)} toma/s).")
            st.image(pieles_nest[0][1], caption="Vista previa de la plancha", use_container_width=True)
        else:
            if st.button("Probar con un Cuero de Ejemplo (Demo)", use_container_width=True):
                carpeta_demo = os.path.join(TMP, "demo")
                shutil.rmtree(carpeta_demo, ignore_errors=True)
                simulador.generar(carpeta_demo)
                p_demo = os.path.join(carpeta_demo, "fotos", os.listdir(os.path.join(carpeta_demo, "fotos"))[0])
                with open(p_demo, "rb") as fd:
                    demo_data = [("Cuero_Ejemplo.jpg", fd.read())]
                st.session_state["pieles_nesting_memoria"] = demo_data
                st.session_state["imagenes_def_memoria"] = demo_data
                st.rerun()

    with col_b:
        st.subheader("Elige los Moldes")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, nombre, cliente, dimensiones, archivo_dxf FROM moldes ORDER BY id ASC")
        moldes_bd = c.fetchall()
        conn.close()

        moldes_seleccionados = []
        if moldes_bd:
            cat_filtro_nest = st.radio(
                "Tipo de producto:",
                ["Todos", "Calzado", "Cartera", "Casaca", "Personalizados"],
                horizontal=True
            )
            moldes_filtrados_nest = [
                m for m in moldes_bd
                if cat_filtro_nest == "Todos" or m[2].lower() == cat_filtro_nest.lower()
            ]
            mapa_dxf = {f"{m[1]} ({m[3]})": (m[1], m[4]) for m in moldes_filtrados_nest}
            opciones = list(mapa_dxf.keys())
            elegidos = st.multiselect(
                "Selecciona las piezas que necesitas cortar:",
                options=opciones,
                default=opciones[:3] if len(opciones) >= 3 else opciones
            )
            for item_nom in elegidos:
                nom_real, path_dxf = mapa_dxf[item_nom]
                moldes_seleccionados.append({"nombre": nom_real, "cantidad": 1, "dxf_path": path_dxf})

        with st.expander("Ajustes opcionales"):
            llenar_toda = st.checkbox(
                "Llenar toda la plancha automáticamente con los moldes elegidos",
                value=True
            )
            escala_pieza = st.slider("Tamaño de los moldes (%)", 60, 150, 100, 10)
            margen_seg = st.slider("Separación alrededor de defectos (px)", 5, 45, 16, 2)
            sens_nest = st.slider("Sensibilidad detección de defectos", 20, 100, 75, 5)

    st.write("---")
    puede_procesar = len(pieles_nest) > 0 and len(moldes_seleccionados) > 0
    if st.button("Acomodar Moldes y Ver Plan de Corte", type="primary", use_container_width=True, disabled=not puede_procesar):
        with st.spinner("Ubicando moldes lejos de los defectos..."):
            bytes_plancha_entera, info_ensamble = ensamblar_plancha_completa(pieles_nest)
            _, obst, rep_def, err = detectar_defectos_cuero(bytes_plancha_entera, sensibilidad=sens_nest)

            if err:
                st.session_state["resultado_nesting_plancha"] = {"error": err}
            else:
                res_img_rgb, datos_plano, colocadas, no_cabieron, aprov_real = ubicar_moldes_en_toda_la_plancha(
                    bytes_plancha_entera,
                    rep_def,
                    moldes_seleccionados,
                    margen_cuchilla=margen_seg,
                    llenar_toda_plancha=llenar_toda,
                    escala_pieza_pct=escala_pieza
                )
                h, w, _ = res_img_rgb.shape
                info_lote = {
                    "fotos_unidas": len(pieles_nest),
                    "info_ensamble": info_ensamble,
                    "piel_m2": round((w * h) / 1000000, 2),
                    "defectos": len(rep_def),
                    "piezas": len(colocadas),
                    "aprovechamiento_pct": aprov_real
                }
                hoja_tecnica_png = generar_hoja_tecnica_imprimible(datos_plano, colocadas, info_lote)
                foto_plano_png = generar_patron_fotografico_png(datos_plano["res_img_bgr"], colocadas, info_lote)
                st.session_state["resultado_nesting_plancha"] = {
                    "error": None,
                    "res_img": res_img_rgb,
                    "colocadas": colocadas,
                    "no_cabieron": no_cabieron,
                    "info": info_lote,
                    "hoja_tecnica_png": hoja_tecnica_png,
                    "foto_plano_png": foto_plano_png
                }

    rn = st.session_state.get("resultado_nesting_plancha")
    if rn:
        if rn["error"]:
            st.error(rn["error"])
        else:
            info = rn["info"]
            st.success(f"**¡Listo! {info['piezas']} moldes, ninguno sobre un defecto** ({info['aprovechamiento_pct']}% de aprovechamiento).")

            m_res = st.columns(3)
            m_res[0].metric("Fotos", info["fotos_unidas"])
            m_res[1].metric("Defectos", info["defectos"])
            m_res[2].metric("Moldes", info["piezas"])

            col_v1, col_v2 = st.columns([1.25, 1])
            with col_v1:
                st.subheader("Plan de Corte Imprimible (Fondo Blanco)")
                st.caption("Hoja limpia para imprimir en papel sin gastar tinta: moldes A, B, C... y defectos tachados con X.")
                if rn["hoja_tecnica_png"]:
                    st.image(rn["hoja_tecnica_png"], use_container_width=True)
            with col_v2:
                st.subheader("Vista sobre el Cuero")
                st.caption("Moldes ubicados lejos de los defectos sobre tu plancha.")
                st.image(rn["res_img"], use_container_width=True)

            st.write("---")
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if rn["hoja_tecnica_png"]:
                    st.download_button(
                        label="IMPRIMIR PLAN DE CORTE (PNG)",
                        data=rn["hoja_tecnica_png"],
                        file_name="Plan_Corte_Imprimible_Taller.png",
                        mime="image/png",
                        type="primary",
                        use_container_width=True
                    )
            with col_btn2:
                if rn["foto_plano_png"]:
                    st.download_button(
                        label="Descargar Vista sobre Cuero (PNG)",
                        data=rn["foto_plano_png"],
                        file_name="Vista_Sobre_Cuero.png",
                        mime="image/png",
                        use_container_width=True
                    )
            with col_btn3:
                pts_dxf_all = []
                for m_sel in moldes_seleccionados:
                    pts_dxf_all.extend(obtener_puntos_molde(m_sel["nombre"], ruta_dxf=m_sel.get("dxf_path")).tolist())
                st.download_button(
                    label="Descargar Vector CAD (DXF)",
                    data=generar_contenido_dxf(pts_dxf_all),
                    file_name="Plan_Corte_Plancha.dxf",
                    mime="application/dxf",
                    use_container_width=True
                )

            if rn["colocadas"]:
                with st.expander("Ver guía de moldes (A, B, C...)"):
                    st.dataframe(rn["colocadas"], use_container_width=True, hide_index=True)

# ================================================================== PASO 4: CATÁLOGO DE MOLDES + FOTO -> DXF
elif opcion_menu == "Catálogo de Moldes":
    st.title("4. Catálogo Digital de Moldes")
    st.caption("Sube fotos de tus moldes físicos para convertirlos a DXF, o usa nuestros moldes de Calzado, Cartera y Casaca.")

    with st.container(border=True):
        st.subheader("Subir Molde del Marroquinero (Foto a DXF)")
        c_up1, c_up2 = st.columns([1.2, 1])
        with c_up1:
            nombre_molde_nuevo = st.text_input("Nombre del Molde", placeholder="Ej. Bolsillo Cartera o Solapa Casaca")
            archivo_molde_fisico = st.file_uploader(
                "Sube foto del molde físico (JPG, PNG) o archivo DXF",
                type=["jpg", "jpeg", "png", "webp", "dxf"],
                key="uploader_molde_dxf"
            )
        with c_up2:
            categoria_molde_nuevo = st.selectbox("Categoría", ["Personalizados", "Calzado", "Cartera", "Casaca"])
            ancho_real_mm = st.number_input("Ancho real del molde (mm)", min_value=20, max_value=1500, value=220, step=10)

        if archivo_molde_fisico is not None:
            nom_arch = archivo_molde_fisico.name
            bytes_arch = archivo_molde_fisico.getvalue()
            nombre_final = nombre_molde_nuevo.strip() or os.path.splitext(nom_arch)[0]

            if nom_arch.lower().endswith(".dxf"):
                carp_dxf = os.path.join(TMP, "moldes_personalizados")
                os.makedirs(carp_dxf, exist_ok=True)
                ruta_dxf_guardado = os.path.join(carp_dxf, f"{nombre_final.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.dxf")
                with open(ruta_dxf_guardado, "wb") as f_out:
                    f_out.write(bytes_arch)
                pts_leidos = leer_puntos_desde_dxf(ruta_dxf_guardado)
                dim_texto = f"{int(np.max(pts_leidos[:,0]))} x {int(np.max(pts_leidos[:,1]))} mm" if pts_leidos is not None else f"{ancho_real_mm} mm"

                if st.button("Guardar Molde DXF", type="primary", use_container_width=True):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                        (nombre_final, categoria_molde_nuevo, dim_texto, datetime.now().strftime("%d/%m/%Y"), ruta_dxf_guardado, "subido")
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"Molde '{nombre_final}' guardado.")
                    st.rerun()
            else:
                pts_vec, dim_str, err_vec = vectorizar_foto_molde_a_dxf(bytes_arch, ancho_real_mm=ancho_real_mm)
                if err_vec:
                    st.error(err_vec)
                else:
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.image(bytes_arch, caption="Foto original", use_container_width=True)
                    with col_v2:
                        lienzo_prev = np.ones((300, 320, 3), dtype=np.uint8) * 255
                        for gx in range(0, 320, 32):
                            cv2.line(lienzo_prev, (gx, 0), (gx, 300), (235, 235, 235), 1)
                        for gy in range(0, 300, 32):
                            cv2.line(lienzo_prev, (0, gy), (320, gy), (235, 235, 235), 1)
                        esc_p = min(260.0 / max(1, np.max(pts_vec[:, 0])), 230.0 / max(1, np.max(pts_vec[:, 1])))
                        pts_p = (pts_vec * esc_p).astype(np.int32) + np.array([30, 30])
                        cv2.polylines(lienzo_prev, [pts_p], True, (0, 150, 70), 2)
                        st.image(lienzo_prev, caption=f"Molde Vectorizado DXF ({dim_str})", use_container_width=True)

                    dxf_str = generar_contenido_dxf(pts_vec)
                    c_btn1, c_btn2 = st.columns(2)
                    with c_btn1:
                        if st.button("Guardar en Catálogo", type="primary", use_container_width=True):
                            carp_dxf = os.path.join(TMP, "moldes_personalizados")
                            os.makedirs(carp_dxf, exist_ok=True)
                            ruta_dxf_guardado = os.path.join(carp_dxf, f"{nombre_final.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.dxf")
                            with open(ruta_dxf_guardado, "w", encoding="utf-8") as f_out:
                                f_out.write(dxf_str)
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute(
                                "INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                                (nombre_final, categoria_molde_nuevo, dim_str, datetime.now().strftime("%d/%m/%Y"), ruta_dxf_guardado, "subido")
                            )
                            conn.commit()
                            conn.close()
                            st.success(f"Molde '{nombre_final}' guardado.")
                            st.rerun()
                    with c_btn2:
                        st.download_button(
                            "Descargar DXF",
                            data=dxf_str,
                            file_name=f"{nombre_final.replace(' ', '_')}.dxf",
                            mime="application/dxf",
                            use_container_width=True
                        )

    st.write("---")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, nombre, cliente, dimensiones, archivo_dxf, origen FROM moldes ORDER BY id DESC")
    m_todos = c.fetchall()
    conn.close()

    cat_vista = st.radio(
        "Ver moldes de:",
        ["Todos", "Calzado", "Cartera", "Casaca", "Personalizados"],
        horizontal=True
    )

    m_filtrados = [
        m for m in m_todos
        if cat_vista == "Todos" or m[2].lower() == cat_vista.lower()
    ]

    if not m_filtrados:
        st.info(f"No hay moldes en '{cat_vista}'.")
    else:
        cols = st.columns(2)
        for idx, m in enumerate(m_filtrados):
            m_id, m_nom, m_cat, m_dim, m_dxf, m_origen = m
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"#### {m_nom}")
                    st.write(f"Categoría: **{m_cat}** | Medidas: **{m_dim}**")
                    pts_molde = obtener_puntos_molde(m_nom, ruta_dxf=m_dxf)

                    lienzo_m = np.ones((360, 320, 3), dtype=np.uint8) * 255
                    for gx in range(0, 320, 40):
                        cv2.line(lienzo_m, (gx, 0), (gx, 360), (235, 235, 235), 1)
                    for gy in range(0, 320, 40):
                        cv2.line(lienzo_m, (0, gy), (320, gy), (235, 235, 235), 1)

                    min_xy = np.min(pts_molde, axis=0)
                    pts_zero = pts_molde - min_xy
                    max_w = max(1, int(np.max(pts_zero[:, 0])))
                    max_h = max(1, int(np.max(pts_zero[:, 1])))
                    esc_fit = min(250.0 / max_w, 270.0 / max_h)
                    pts_centrados = (pts_zero * esc_fit).astype(np.int32) + np.array([35, 30])

                    cv2.polylines(lienzo_m, [pts_centrados], True, (0, 0, 0), 2)
                    cv2.putText(lienzo_m, f"{m_nom} ({m_dim})", (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

                    st.image(lienzo_m, use_container_width=True)
                    _, buf_m = cv2.imencode(".png", lienzo_m)

                    c_b1, c_b2 = st.columns(2)
                    with c_b1:
                        st.download_button(
                            "Imprimir Plantilla (PNG)",
                            data=buf_m.tobytes(),
                            file_name=f"Plantilla_{m_nom.replace(' ', '_')}.png",
                            mime="image/png",
                            key=f"imp_m_{m_id}",
                            use_container_width=True
                        )
                    with c_b2:
                        st.download_button(
                            "Descargar DXF",
                            data=generar_contenido_dxf(pts_molde),
                            file_name=f"{m_nom.replace(' ', '_')}.dxf",
                            mime="application/dxf",
                            key=f"dxf_m_{m_id}",
                            use_container_width=True
                        )

                    if m_origen == "subido":
                        if st.button("Eliminar Molde", key=f"del_molde_{m_id}", use_container_width=True):
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute("DELETE FROM moldes WHERE id = ?", (m_id,))
                            conn.commit()
                            conn.close()
                            st.rerun()
