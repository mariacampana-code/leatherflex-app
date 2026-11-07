"""Calibracion de la camara dentro del escaner.

- Escala y giro: con una foto de un tablero de ajedrez impreso (cuadros de 10 mm).
- Correccion de luz (campo plano): con una foto de una hoja blanca mate.
Se hace UNA vez por escaner/celular y se guarda en un archivo .json + .png.
"""
import json
import numpy as np
import cv2


def calibrar_tablero(img_bgr, cuadro_mm=10.0, patron=(9, 6)):
    """Devuelve mm por pixel de la foto y el angulo del tablero (grados).

    patron = esquinas INTERIORES (columnas, filas) del tablero impreso.
    """
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ok, esquinas = cv2.findChessboardCorners(
        gris, patron, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        raise ValueError("No se encontro el tablero. Revisa que se vea completo, enfocado y sin reflejos.")
    esquinas = cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    p = esquinas.reshape(patron[1], patron[0], 2)          # filas x columnas
    dx = p[:, 1:, :] - p[:, :-1, :]                         # a lo largo de cada fila
    dy = p[1:, :, :] - p[:-1, :, :]                         # a lo largo de cada columna
    paso_px = np.mean(np.r_[np.linalg.norm(dx, axis=2).ravel(), np.linalg.norm(dy, axis=2).ravel()])
    ang = np.degrees(np.arctan2(dx[..., 1].mean(), dx[..., 0].mean()))
    # el tablero no tiene "frente": se lleva el angulo al rango (-45, 45]
    ang = (ang + 45) % 90 - 45
    return {"mm_por_px_foto": float(cuadro_mm / paso_px), "angulo_grados": float(ang),
            "ancho_px": int(img_bgr.shape[1]), "alto_px": int(img_bgr.shape[0])}


def campo_plano(img_blanca_bgr):
    """Mapa de ganancia para corregir el oscurecimiento de las esquinas (viñeteo) y la luz desigual."""
    g = cv2.cvtColor(img_blanca_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 40)
    return (g.max() / np.maximum(g, 1)).astype(np.float32)


def guardar(calib, ganancia, ruta_json):
    calib = dict(calib)
    if ganancia is not None:
        ruta_png = ruta_json.replace(".json", "_ganancia.npy")
        np.save(ruta_png, ganancia)
        calib["ganancia"] = ruta_png
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)


def cargar(ruta_json):
    with open(ruta_json, encoding="utf-8") as f:
        calib = json.load(f)
    gan = np.load(calib["ganancia"]) if calib.get("ganancia") else None
    return calib, gan


def tablero_para_imprimir(ruta_png, cuadro_mm=10.0, patron=(9, 6), dpi=300):
    """Genera el tablero de calibracion para imprimir a escala 100 % (sin 'ajustar a la pagina')."""
    px = cuadro_mm / 25.4 * dpi
    cols, filas = patron[0] + 1, patron[1] + 1
    margen = int(px)
    w, h = int(cols * px + 2 * margen), int(filas * px + 2 * margen + px)
    img = np.full((h, w), 255, np.uint8)
    for r in range(filas):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0, y0 = int(margen + c * px), int(margen + r * px)
                img[y0:int(y0 + px), x0:int(x0 + px)] = 0
    cv2.putText(img, f"Leather-Flex 3D - tablero de calibracion - cuadros de {cuadro_mm:g} mm - imprimir al 100 %",
                (margen, h - int(px * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, px / 90, 0, max(1, int(px / 40)))
    cv2.imwrite(ruta_png, img)
    return ruta_png
