"""Une las fotos en un solo mapa de la piel.

1. Cada foto parte en la posicion que da la traba (columna x paso_x, fila x paso_y).
2. Para cada par de fotos vecinas se mide el desplazamiento real en la zona de traslape
   con correlacion de fase (usa el grano del cuero).
3. Se resuelve un sistema de minimos cuadrados que combina las mediciones con la posicion
   de la traba, y se pegan las fotos mezclando suavemente los bordes.
"""
import numpy as np
import cv2


def _nominal(f, cfg):
    return np.array([f["col"] * cfg.paso_x_mm, f["fila"] * cfg.paso_y_mm], float)


def _parche(img, x0, y0, w, h):
    return img[y0:y0 + h, x0:x0 + w]


def _desplazamiento(a, b):
    """Desplazamiento (dx, dy) en px de b respecto de a, y la confianza (0..1)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # se realza el grano (pasa-altos) para que la luz no domine la correlacion
    ga -= cv2.GaussianBlur(ga, (0, 0), 6); gb -= cv2.GaussianBlur(gb, (0, 0), 6)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return dx, dy, resp


def medir_pares(fotos, cfg):
    """Mide el desplazamiento entre cada par de fotos vecinas. Devuelve lista de (i, j, dx_mm, dy_mm, peso)."""
    idx = {(f["col"], f["fila"]): i for i, f in enumerate(fotos)}
    s = cfg.mm_por_px
    pares = []
    for (c, r), i in idx.items():
        for dc, dr in ((1, 0), (0, 1)):
            j = idx.get((c + dc, r + dr))
            if j is None:
                continue
            A, B = fotos[i]["prep"], fotos[j]["prep"]
            pa, pb = _nominal(fotos[i], cfg), _nominal(fotos[j], cfg)
            d = (pb - pa) / s                                     # desplazamiento nominal en px
            # traslape en coordenadas de A
            x0, y0 = max(0, int(round(d[0]))), max(0, int(round(d[1])))
            x1 = min(A.shape[1], int(round(d[0])) + B.shape[1]); y1 = min(A.shape[0], int(round(d[1])) + B.shape[0])
            w, h = x1 - x0, y1 - y0
            if w < 40 or h < 40:
                continue
            pa_ = _parche(A, x0, y0, w, h)
            pb_ = _parche(B, x0 - int(round(d[0])), y0 - int(round(d[1])), w, h)
            dx, dy, resp = _desplazamiento(pa_, pb_)
            # la correlacion devuelve cuanto hay que mover pb_ para calzar con pa_ -> posicion real de B
            real = (np.array([int(round(d[0])), int(round(d[1]))]) - np.array([dx, dy])) * s
            if np.linalg.norm(real - (pb - pa)) > cfg.max_correccion_mm or resp < 0.03:
                continue                                          # medicion dudosa: se confia en la traba
            pares.append((i, j, real[0], real[1], float(resp)))
    return pares


def resolver_posiciones(fotos, pares, cfg):
    """Minimos cuadrados: posiciones (mm) que mejor cumplen las mediciones y la traba."""
    n = len(fotos)
    filas_A, b = [], []
    for k in range(2):                                            # x e y por separado
        A_k, b_k = [], []
        for (i, j, dx, dy, w) in pares:
            row = np.zeros(n); row[j] = 1; row[i] = -1
            A_k.append(row * w); b_k.append((dx if k == 0 else dy) * w)
        for i, f in enumerate(fotos):
            row = np.zeros(n); row[i] = 1
            wt = 10.0 if i == 0 else cfg.peso_traba          # la primera foto fija el origen
            A_k.append(row * wt); b_k.append(_nominal(f, cfg)[k] * wt)
        filas_A.append(np.array(A_k)); b.append(np.array(b_k))
    pos = np.stack([np.linalg.lstsq(filas_A[k], b[k], rcond=None)[0] for k in range(2)], 1)
    residuos = [np.hypot(pos[j, 0] - pos[i, 0] - dx, pos[j, 1] - pos[i, 1] - dy) for (i, j, dx, dy, _) in pares]
    return pos, (float(np.mean(residuos)) if residuos else 0.0)


def componer(fotos, pos_mm, cfg):
    """Pega las fotos en un lienzo. Devuelve imagen BGR y mascara de zona cubierta."""
    s = cfg.mm_por_px
    pos_px = np.round((pos_mm - pos_mm.min(0)) / s).astype(int)
    W = int(max(p[0] + f["prep"].shape[1] for p, f in zip(pos_px, fotos)))
    H = int(max(p[1] + f["prep"].shape[0] for p, f in zip(pos_px, fotos)))
    acc = np.zeros((H, W, 3), np.float32); pesos = np.zeros((H, W), np.float32)
    for p, f in zip(pos_px, fotos):
        img = f["prep"].astype(np.float32); h, w = img.shape[:2]
        # peso alto al centro de la foto y bajo en sus bordes (mezcla suave)
        wy = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1; wx = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
        peso = np.minimum.outer(wy, wx).astype(np.float32)
        acc[p[1]:p[1] + h, p[0]:p[0] + w] += img * peso[..., None]
        pesos[p[1]:p[1] + h, p[0]:p[0] + w] += peso
    cubierto = pesos > 0
    mosaico = np.zeros((H, W, 3), np.uint8)
    mosaico[cubierto] = np.clip(acc[cubierto] / pesos[cubierto, None], 0, 255).astype(np.uint8)
    return mosaico, cubierto, pos_px


def unir(fotos, cfg):
    pares = medir_pares(fotos, cfg)
    pos, residuo = resolver_posiciones(fotos, pares, cfg)
    mosaico, cubierto, pos_px = componer(fotos, pos, cfg)
    info = {"fotos": len(fotos), "pares_medidos": len(pares), "error_medio_union_mm": round(residuo, 3)}
    return mosaico, cubierto, pos, info
