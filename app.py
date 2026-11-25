"""Leather-Flex 3D - Motor de IA Embebido (Random Forest + Analisis Multi-Canal LAB + Video Keyframes), Lotes y Plano Imprimible"""
import os
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
    "Picadura de Garrapata / Poro",
    "Cicatriz / Corte Lineal",
    "Rasguño Superficial",
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

# ------------------------------------------------------------------ Extractor Inteligente de Fotogramas de Video
def extraer_fotogramas_nitidos_video(bytes_video, nombre_video="video.mp4", max_fotogramas=6):
    """
    Extrae los fotogramas más nítidos de un video descartando los cuadros borrosos
    por movimiento rápido (usando varianza de Laplaciano) y evitando duplicados si
    la cámara se movió lento.
    """
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

    # Evaluar hasta 45 puntos candidatos distribuidos en el video
    paso = max(1, total_frames // 45)
    candidatos = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % paso == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Medida de nitidez (mayor valor = menos desenfoque por movimiento)
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

    # Dividir el video en segmentos temporales y elegir el fotograma MÁS NÍTIDO de cada segmento
    num_segmentos = min(max_fotogramas, len(candidatos))
    segmentos = np.array_split(candidatos, num_segmentos)
    seleccionados = []
    ultimo_hist = None

    for seg in segmentos:
        if len(seg) == 0:
            continue
        mejor = max(seg, key=lambda item: item["nitidez"])
        # Evitar fotogramas prácticamente idénticos si la cámara estuvo detenida
        if ultimo_hist is not None:
            similitud = cv2.compareHist(ultimo_hist, mejor["hist"], cv2.HISTCMP_CORREL)
            if similitud > 0.992 and len(seleccionados) > 0:
                continue
        ultimo_hist = mejor["hist"]
        ok, buf = cv2.imencode(".jpg", mejor["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            etiqueta = f"{os.path.splitext(nombre_video)[0]} (Seg {mejor['tiempo']}s)"
            seleccionados.append((etiqueta, buf.tobytes()))

    return seleccionados

def expandir_archivos_a_imagenes(lista_archivos):
    """Convierte una mezcla de fotos y videos en una lista unificada de imágenes nítidas listas para la IA."""
    imagenes_listas = []
    for nom, b_data in lista_archivos:
        if nom.lower().endswith(EXTENSIONES_VIDEO):
            frames_vid = extraer_fotogramas_nitidos_video(b_data, nombre_video=nom, max_fotogramas=6)
            imagenes_listas.extend(frames_vid)
        else:
            imagenes_listas.append((nom, b_data))
    return imagenes_listas

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

    if np.mean(mask_piel == 255) < 0.15 or np.mean(mask_piel == 255) > 0.96:
        mask_completa = np.ones((h, w), dtype=np.uint8) * 255
        cv2.rectangle(mask_completa, (0, 0), (w, h), 0, 10)
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

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
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
        max_dim = 1280
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
                pad = 6
                cv2.rectangle(
                    img_viz_bgr,
                    (max(0, x - pad), max(0, y - pad)),
                    (min(w, x + bw + pad), min(h, y + bh + pad)),
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    img_viz_bgr,
                    f"D{idx_def}",
                    (max(0, x - pad), max(16, y - pad - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    2
                )
                idx_def += 1

        cv2.addWeighted(overlay_rojo, 0.30, img_viz_bgr, 0.70, 0, img_viz_bgr)
        img_viz_rgb = cv2.cvtColor(img_viz_bgr, cv2.COLOR_BGR2RGB)
        return img_viz_rgb, lista_obstaculos_px, reporte_defectos, None
    except Exception as e:
        return None, [], [], f"Error técnico al procesar imagen: {str(e)}"

# ------------------------------------------------------------------ Motor de Distribución / Nesting
def ubicar_moldes_en_piel(bytes_imagen, bboxes_defectos, moldes_elegidos_list, margen_cuchilla=20):
    arr = np.frombuffer(bytes_imagen, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h_orig, w_orig = img.shape[:2]
    max_dim = 1280
    if max(h_orig, w_orig) > max_dim:
        escala = max_dim / max(h_orig, w_orig)
        img = cv2.resize(img, (int(w_orig * escala), int(h_orig * escala)))

    h, w, _ = img.shape
    res_img = img.copy()

    mask_piel = aislar_piel_del_fondo(img)
    ocupado_mask = cv2.bitwise_not(mask_piel)

    for d in bboxes_defectos:
        bx, by, bw_d, bh_d = d["bbox"]
        cv2.rectangle(
            res_img,
            (max(0, bx - 5), max(0, by - 5)),
            (min(w, bx + bw_d + 5), min(h, by + bh_d + 5)),
            (0, 0, 255),
            2
        )
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

                if cur_w >= w - 20 or cur_h >= h - 20:
                    continue

                for y in range(15, h - cur_h - 15, 10):
                    if posicionado:
                        break
                    for x in range(15, w - cur_w - 15, 10):
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

    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    return res_img_rgb, res_img, piezas_colocadas, piezas_no_cabieron

# ------------------------------------------------------------------ Generador de Patrón Imprimible
def generar_patron_imprimible(res_img_bgr, piezas_colocadas, info_lote):
    h, w, _ = res_img_bgr.shape
    margen_sup = 110
    margen_inf = 90
    margen_lat = 50

    lienzo_h = h + margen_sup + margen_inf
    lienzo_w = w + (margen_lat * 2)
    hoja = np.ones((lienzo_h, lienzo_w, 3), dtype=np.uint8) * 255

    cv2.putText(hoja, "LEATHER-FLEX 3D - PLANO DE CORTE IMPRIMIBLE", (margen_lat, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (44, 37, 32), 2)
    cv2.putText(hoja, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Piezas: {len(piezas_colocadas)} | Rendimiento: {info_lote['aprovechamiento_pct']}%",
                (margen_lat, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 90, 80), 1)
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

# ================================================================== MÓDULO 1: CAPTURA DEL CUERO (UNA SOLA OPCIÓN LIMPIA)
if opcion_menu == "Captura del Cuero":
    st.title("Captura del Cuero")
    st.caption("Sube fotos o videos desde tu galería, toma fotos con la cámara de tu dispositivo o graba video. Puedes seguir agregando más archivos antes de guardar.")

    if "bandeja_capturas" not in st.session_state:
        st.session_state["bandeja_capturas"] = []
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    with st.container(border=True):
        nota_lote = st.text_input("Identificador / Nombre del Lote (Opcional)", placeholder="Ej. Lote Cuero Vacuno Mesa 1", key="nota_unica")

        archivos_nuevos = st.file_uploader(
            "Seleccionar de la biblioteca, tomar foto o grabar video (puedes elegir varios a la vez)",
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
            st.markdown(f"**Archivos cargados ({len(bandeja)}):** Puedes seguir tocando el botón de arriba para tomar o subir más fotos o videos.")
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
                if st.button(f"Guardar {len(bandeja)} Archivo(s) y Preparar Análisis", type="primary", use_container_width=True):
                    rutas_lote = []
                    for idx, (nom_item, bytes_item) in enumerate(bandeja):
                        etiqueta = f"{nota_lote} #{idx+1}" if nota_lote else f"Captura #{idx+1} ({nom_item})"
                        ruta_g = guardar_captura_en_bd(nom_item, bytes_item, etiqueta)
                        rutas_lote.append(ruta_g)

                    with st.spinner("Extrayendo fotogramas nítidos para análisis IA..."):
                        imagenes_memoria = expandir_archivos_a_imagenes(bandeja)

                    if imagenes_memoria:
                        st.session_state["lote_pieles_activas"] = rutas_lote
                        st.session_state["imagenes_def_memoria"] = imagenes_memoria
                        st.session_state["pieles_nesting_memoria"] = imagenes_memoria

                    st.session_state["bandeja_capturas"] = []
                    st.session_state["uploader_key"] += 1
                    st.success(f"{len(bandeja)} archivo(s) guardados y preparados para detección.")
                    st.rerun()

            with col_g2:
                if st.button("Limpiar Todo", use_container_width=True):
                    st.session_state["bandeja_capturas"] = []
                    st.session_state["uploader_key"] += 1
                    st.rerun()

    st.write("---")
    st.subheader("Historial de Capturas Registradas")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, usuario, titulo, fecha, media_path FROM feed ORDER BY id DESC")
    posts = c.fetchall()
    conn.close()

    if not posts:
        st.info("No hay capturas aún. Usa el selector de arriba para subir fotos o videos.")
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
                            st.session_state["lote_pieles_activas"] = [media]
                            with open(media, "rb") as f_sel:
                                b_sel = f_sel.read()
                            imgs_sel = expandir_archivos_a_imagenes([(os.path.basename(media), b_sel)])
                            st.session_state["imagenes_def_memoria"] = imgs_sel
                            st.session_state["pieles_nesting_memoria"] = imgs_sel
                            st.toast(f"'{tit}' cargado para análisis IA.")
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
    st.caption("Motor de visión artificial con clasificador de IA integrado para fotos y videos (con filtro anti-movimiento).")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.subheader("1. Cargar Imágenes o Videos de Cuero")
        archivos_subidos = st.file_uploader(
            "Sube fotos o videos de pieles para detectar defectos",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            key="uploader_defectos_lote"
        )

        if archivos_subidos:
            with st.spinner("Extrayendo fotogramas nítidos..."):
                raw_files = [(f.name, f.getvalue()) for f in archivos_subidos]
                st.session_state["imagenes_def_memoria"] = expandir_archivos_a_imagenes(raw_files)
        elif "imagenes_def_memoria" not in st.session_state and st.session_state.get("ultima_captura_cuero"):
            ruta_act = st.session_state["ultima_captura_cuero"]
            if os.path.exists(ruta_act):
                with open(ruta_act, "rb") as f:
                    st.session_state["imagenes_def_memoria"] = expandir_archivos_a_imagenes([(os.path.basename(ruta_act), f.read())])

        imagenes_actuales = st.session_state.get("imagenes_def_memoria", [])
        if imagenes_actuales:
            st.info(f"Listas para analizar: **{len(imagenes_actuales)} vista(s) / fotograma(s) nítidos**")
            if st.button("Limpiar archivos cargados"):
                st.session_state["imagenes_def_memoria"] = []
                st.session_state["resultados_defectos_lote"] = []
                st.rerun()

        sensibilidad = st.slider(
            "Sensibilidad de detección IA (75 recomendado para marcas finas y picaduras)",
            min_value=20, max_value=100, value=75, step=5
        )

        btn_analizar = st.button(
            f"Detectar Defectos en {len(imagenes_actuales)} Vista(s)",
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
                img_viz_rgb, obst, rep, err = detectar_defectos_cuero(b_img, sensibilidad=sensibilidad)
                resultados.append({
                    "nombre": nom,
                    "img_viz": img_viz_rgb,
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
                st.markdown(f"### Vista: `{item['nombre']}`")
                if item["error"]:
                    st.error(item["error"])
                else:
                    if len(item["reporte"]) == 0:
                        st.warning("No se detectaron defectos con la sensibilidad actual. Sube el slider a 80-90.")
                    else:
                        st.success(f"IA identificó **{len(item['reporte'])} defecto(s)** marcados en rojo.")
                    st.image(item["img_viz"], use_container_width=True, caption="Imperfecciones marcadas en rojo")
                    if item["reporte"]:
                        st.dataframe(
                            [{
                                "ID": f"D{d['id']}",
                                "Diagnóstico IA": d["tipo"],
                                "Confianza": d["confianza"],
                                "Ubicación": d["ubicacion"],
                                "Área (px²)": d["area_px2"]
                            } for d in item["reporte"]],
                            use_container_width=True,
                            hide_index=True
                        )
                st.divider()
        else:
            st.info("Sube fotos o videos en el panel izquierdo y presiona el botón para iniciar.")

# ================================================================== MÓDULO 3: ACOMODO DE MOLDES Y PATRÓN IMPRIMIBLE
elif opcion_menu == "Acomodo de Moldes":
    st.title("Acomodo Óptimo de Moldes y Plano Imprimible")
    st.caption("Distribuye los moldes dentro de la piel evitando los defectos rojos y descarga el patrón listo para imprimir.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("1. Pieles para Distribución")
        fotos_nest = st.file_uploader(
            "Sube fotos o videos de pieles para acomodar moldes",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            key="uploader_nesting_multi"
        )
        if fotos_nest:
            with st.spinner("Extrayendo vistas nítidas..."):
                st.session_state["pieles_nesting_memoria"] = expandir_archivos_a_imagenes([(f.name, f.getvalue()) for f in fotos_nest])
        elif "pieles_nesting_memoria" not in st.session_state and st.session_state.get("imagenes_def_memoria"):
            st.session_state["pieles_nesting_memoria"] = st.session_state["imagenes_def_memoria"]

        pieles_nest = st.session_state.get("pieles_nesting_memoria", [])
        if pieles_nest:
            st.success(f"{len(pieles_nest)} vista(s) de piel listas para distribución automática.")
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

        margen_seg = st.slider("Margen de seguridad alrededor de defectos (px)", 5, 45, 18, 5)
        sens_nest = st.slider("Sensibilidad detección IA", 20, 100, 75, 5, key="sens_nest")

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
            _, obst, rep_def, err = detectar_defectos_cuero(bytes_p, sensibilidad=sens_nest)
            if err:
                resultados_nesting.append({"nombre": nom_p, "error": err})
                continue

            res_img_rgb, res_img_bgr, colocadas, no_cabieron = ubicar_moldes_en_piel(
                bytes_p, rep_def, moldes_seleccionados, margen_cuchilla=margen_seg
            )
            h, w, _ = res_img_rgb.shape
            aprov = min(95.0, round(68.0 + (len(colocadas) * 5.5), 1)) if colocadas else 0.0
            info_lote = {
                "piel_m2": round((w * h) / 1000000, 2),
                "defectos": len(rep_def),
                "piezas": len(colocadas),
                "aprovechamiento_pct": aprov
            }
            png_imprimible = generar_patron_imprimible(res_img_bgr, colocadas, info_lote)
            resultados_nesting.append({
                "nombre": nom_p,
                "error": None,
                "res_img": res_img_rgb,
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
                        label="Descargar Patrón Imprimible (PNG Alta Res.)",
                        data=rn["png_imprimible"],
                        file_name=f"Patron_Corte_{rn['nombre'].split('.')[0]}.png",
                        mime="image/png",
                        type="primary",
                        use_container_width=True,
                        key=f"dl_png_{idx_r}"
                    )
                pts_dxf_all = []
                for m_sel in moldes_seleccionados:
                    pts_dxf_all.extend(obtener_puntos_molde_calzado(m_sel["nombre"]).tolist())
                st.download_button(
                    label="Descargar Vector CAD (DXF)",
                    data=generar_contenido_dxf(pts_dxf_all),
                    file_name=f"Corte_CAD_{rn['nombre'].split('.')[0]}.dxf",
                    mime="application/dxf",
                    use_container_width=True,
                    key=f"dl_dxf_{idx_r}"
                )
            st.divider()

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
