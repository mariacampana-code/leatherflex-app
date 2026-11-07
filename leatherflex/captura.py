"""Carga las fotos del escaneo y les asigna su posicion en la regla (columna, fila)."""
import os
import re
import numpy as np
import cv2

EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic")


def _leer(ruta):
    if ruta.lower().endswith(".heic"):
        try:
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            return cv2.cvtColor(np.array(Image.open(ruta).convert("RGB")), cv2.COLOR_RGB2BGR)
        except ImportError as e:
            raise ImportError("Foto HEIC: instala 'pillow-heif' o pon el iPhone en Ajustes > Camara > "
                              "Formatos > Mas compatible (JPG).") from e
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer {ruta}")
    return img


def letras_a_numero(s):
    n = 0
    for ch in s.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def numero_a_letras(n):
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def cargar_fotos(carpeta, filas=4, ruta="columnas"):
    """Devuelve una lista de dicts {nombre, col, fila, img}.

    - Si el archivo se llama como su posicion (G5.jpg, b12.png) se usa ese nombre.
    - Si no, se ordenan por nombre (el iPhone numera en orden de disparo) y se asignan
      siguiendo la ruta: 'columnas' = A1, A2.. A{filas}, B1, B2..; 'serpentina' = A1..A4, B4..B1, C1..
    """
    archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXT))
    if not archivos:
        raise ValueError(f"No hay fotos en {carpeta}")
    patron = re.compile(r"^([A-Za-z]{1,2})(\d{1,3})$")
    nombradas = [patron.match(os.path.splitext(f)[0]) for f in archivos]
    fotos = []
    if all(nombradas):
        for f, m in zip(archivos, nombradas):
            fotos.append({"nombre": f, "col": letras_a_numero(m.group(1)), "fila": int(m.group(2)) - 1,
                          "img": _leer(os.path.join(carpeta, f))})
    else:
        for i, f in enumerate(archivos):
            col, fila = divmod(i, filas)
            if ruta == "serpentina" and col % 2 == 1:
                fila = filas - 1 - fila
            fotos.append({"nombre": f, "col": col, "fila": fila, "img": _leer(os.path.join(carpeta, f))})
    return fotos


def preparar(img, calib, ganancia, mm_por_px):
    """Corrige luz, endereza el giro de la camara y lleva la foto a la resolucion de trabajo."""
    f = img.astype(np.float32)
    if ganancia is not None:
        g = ganancia if ganancia.shape == f.shape[:2] else cv2.resize(ganancia, (f.shape[1], f.shape[0]))
        f *= g[..., None]
    f = np.clip(f, 0, 255).astype(np.uint8)
    ang = calib.get("angulo_grados", 0.0)
    if abs(ang) > 0.01:
        h, w = f.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        f = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # se recorta un borde para no usar las esquinas rellenadas por el giro
        m = int(np.ceil(np.tan(np.radians(abs(ang))) * max(h, w) / 2)) + 2
        f = f[m:h - m, m:w - m]
    esc = calib["mm_por_px_foto"] / mm_por_px
    return cv2.resize(f, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
