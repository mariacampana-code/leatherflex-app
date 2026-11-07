"""Genera un escaneo SINTETICO para probar el software sin cuero ni celular.

Crea: fotos/ (una por posicion de la traba), calibracion/tablero.jpg, calibracion/blanco.jpg y verdad.json
con las posiciones reales de cada foto y los defectos colocados. Las fotos imitan al celular:
resolucion 0,1 mm/px, camara girada, error de posicion de la traba, vineteo y ruido.
"""
import os
import json
import math
import numpy as np
import cv2

MM_MUNDO = 0.2          # resolucion del "mundo" simulado
MM_FOTO = 0.1           # resolucion de la camara simulada
FOTO_MM = (181.0, 134.0)


def _mundo(ancho_mm, alto_mm, rng):
    W, H = int(ancho_mm / MM_MUNDO), int(alto_mm / MM_MUNDO)
    mesa = np.full((H, W, 3), (196, 199, 203), np.float32)
    # piel: blob irregular centrado
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rad = 1 + sum(a * np.cos(k * ang + rng.uniform(0, 6.28)) for k, a in [(2, .07), (3, .05), (5, .03), (7, .015)])
    pts = np.stack([W / 2 + W * .40 * rad * np.cos(ang), H / 2 + H * .38 * rad * np.sin(ang)], 1).astype(np.int32)
    mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [pts], 255)
    # grano del cuero
    n = int(W * H / 180)
    semillas = np.zeros((H, W), np.uint8); semillas[rng.integers(0, H, n), rng.integers(0, W, n)] = 255
    celdas = np.clip(cv2.distanceTransform(255 - semillas, cv2.DIST_L2, 3) / 5.0, 0, 1)
    bajo = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (H // 10, W // 10)).astype(np.float32), (0, 0), 3), (W, H))
    g = 9 * (celdas - .5) + 9 * bajo + 2.5 * rng.normal(0, 1, (H, W))
    cuero = np.array([62, 98, 150], np.float32) + g[..., None] * np.array([.55, .75, 1.0], np.float32)
    verdad = []
    interior = cv2.erode(mask, np.ones((int(40 / MM_MUNDO),) * 2, np.uint8))
    ys, xs = np.nonzero(interior)
    def punto():
        i = rng.integers(len(xs)); return int(xs[i]), int(ys[i])
    def anotar(tipo, m):
        yy, xx = np.nonzero(m)
        cont, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        verdad.append({"tipo": tipo, "caja_mm": [float(xx.min() * MM_MUNDO), float(yy.min() * MM_MUNDO),
                                                  float(xx.max() * MM_MUNDO), float(yy.max() * MM_MUNDO)],
                       "contornos_mm": [(c.reshape(-1, 2) * MM_MUNDO).tolist() for c in cont]})
    for _ in range(2):                                    # rayas
        x, y = punto(); m = np.zeros((H, W), np.uint8); a = rng.uniform(0, np.pi); camino = []
        for _ in range(25):
            a += rng.normal(0, .06); x += 4 / MM_MUNDO * math.cos(a); y += 4 / MM_MUNDO * math.sin(a); camino.append((int(x), int(y)))
        cv2.polylines(m, [np.array(camino, np.int32)], False, 255, max(2, int(1.0 / MM_MUNDO)))
        cuero[m > 0] -= 32; anotar("raya", m)
    x0, y0 = punto(); m = np.zeros((H, W), np.uint8)       # picaduras
    for _ in range(9):
        cv2.circle(m, (int(x0 + rng.normal(0, 20)), int(y0 + rng.normal(0, 20))), int(rng.integers(5, 8)), 255, -1)
    cuero[m > 0] -= 40; anotar("picadura", m)
    x0, y0 = punto(); m = np.zeros((H, W), np.uint8)       # mancha
    cv2.ellipse(m, (x0, y0), (int(28 / MM_MUNDO), int(18 / MM_MUNDO)), rng.uniform(0, 180), 0, 360, 255, -1)
    cuero -= (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6 / MM_MUNDO) / 255 * 30)[..., None]; anotar("mancha", m)
    x0, y0 = punto(); m = np.zeros((H, W), np.uint8)       # arruga clara
    cv2.line(m, (x0, y0), (int(x0 + 70 / MM_MUNDO), int(y0 + 12 / MM_MUNDO)), 255, int(1.5 / MM_MUNDO))
    cuero[m > 0] += 28; anotar("arruga", m)
    mundo = np.where(mask[..., None] > 0, cuero, mesa)
    x0, y0 = punto(); m = np.zeros((H, W), np.uint8)       # agujero: se ve la mesa
    cv2.circle(m, (x0, y0), int(5 / MM_MUNDO), 255, -1); mundo[m > 0] = (196, 199, 203); anotar("agujero", m)
    return np.clip(mundo, 0, 255).astype(np.uint8), verdad


def _foto(mundo, cx_mm, cy_mm, ang_grados, vig, rng, fondo=None):
    """Fotografia el mundo con el centro de la camara en (cx, cy) mm y la camara girada."""
    wf, hf = int(FOTO_MM[0] / MM_FOTO), int(FOTO_MM[1] / MM_FOTO)
    t = math.radians(ang_grados); c, s = math.cos(t), math.sin(t)
    k = MM_FOTO / MM_MUNDO
    # pixel de la foto (u, v) -> pixel del mundo
    M = np.array([[k * c, -k * s, cx_mm / MM_MUNDO - k * (c * wf / 2 - s * hf / 2)],
                  [k * s, k * c, cy_mm / MM_MUNDO - k * (s * wf / 2 + c * hf / 2)]], np.float32)
    src = mundo if fondo is None else fondo
    img = cv2.warpAffine(src, M, (wf, hf), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(196, 199, 203))
    img = img.astype(np.float32) * vig[..., None] + rng.normal(0, 1.8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def generar(carpeta, columnas=6, filas=4, paso=100.0, error_traba_mm=1.5, giro_camara=1.2, semilla=3, nombrar=True):
    rng = np.random.default_rng(semilla)
    os.makedirs(os.path.join(carpeta, "fotos"), exist_ok=True); os.makedirs(os.path.join(carpeta, "calibracion"), exist_ok=True)
    margen = 70.0
    ancho = (columnas - 1) * paso + FOTO_MM[0] + 2 * margen
    alto = (filas - 1) * paso + FOTO_MM[1] + 2 * margen
    mundo, defectos = _mundo(ancho, alto, rng)
    wf, hf = int(FOTO_MM[0] / MM_FOTO), int(FOTO_MM[1] / MM_FOTO)
    yy, xx = np.mgrid[0:hf, 0:wf]
    vig = (1 - .20 * (((xx - wf / 2) / (wf / 2)) ** 2 + ((yy - hf / 2) / (hf / 2)) ** 2) / 2).astype(np.float32)

    pos = []
    k = 0
    for c in range(columnas):
        for r in range(filas):
            cx = margen + FOTO_MM[0] / 2 + c * paso + rng.uniform(-error_traba_mm, error_traba_mm)
            cy = margen + FOTO_MM[1] / 2 + r * paso + rng.uniform(-error_traba_mm, error_traba_mm)
            nombre = f"{chr(65 + c)}{r + 1}.jpg" if nombrar else f"IMG_{1000 + k:04d}.jpg"
            cv2.imwrite(os.path.join(carpeta, "fotos", nombre), _foto(mundo, cx, cy, giro_camara, vig, rng), [cv2.IMWRITE_JPEG_QUALITY, 93])
            pos.append({"nombre": nombre, "col": c, "fila": r, "cx_mm": cx, "cy_mm": cy}); k += 1

    # tablero de 10 mm (9 x 6 esquinas interiores) y hoja blanca, con la misma camara
    tab = np.full((int(134 / MM_MUNDO), int(181 / MM_MUNDO), 3), 255, np.uint8)
    px = 10 / MM_MUNDO; ox, oy = (181 - 100) / 2 / MM_MUNDO, (134 - 70) / 2 / MM_MUNDO
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                tab[int(oy + r * px):int(oy + (r + 1) * px), int(ox + c * px):int(ox + (c + 1) * px)] = 0
    cv2.imwrite(os.path.join(carpeta, "calibracion", "tablero.jpg"), _foto(None, 90.5, 67, giro_camara, vig, rng, fondo=tab))
    blanco = np.full((int(200 / MM_MUNDO), int(250 / MM_MUNDO), 3), 232, np.uint8)
    cv2.imwrite(os.path.join(carpeta, "calibracion", "blanco.jpg"), _foto(None, 125, 100, giro_camara, vig, rng, fondo=blanco))
    with open(os.path.join(carpeta, "verdad.json"), "w") as f:
        json.dump({"fotos": pos, "defectos": defectos, "giro_camara": giro_camara, "mm_por_px_foto": MM_FOTO,
                   "foto_mm": FOTO_MM}, f, indent=1)
    return carpeta
