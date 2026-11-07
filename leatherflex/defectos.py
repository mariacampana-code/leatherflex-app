"""Separa la piel del fondo y detecta defectos (rayas, picaduras, manchas, arrugas, agujeros)."""
import numpy as np
import cv2


def mascara_piel(mosaico, cubierto):
    """Piel = lo que se diferencia del color de la mesa. El fondo se estima en el borde del mapa."""
    lab = cv2.cvtColor(mosaico, cv2.COLOR_BGR2LAB).astype(np.float32)
    borde = cubierto & ~cv2.erode(cubierto.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    fondo = np.median(lab[borde], axis=0)
    dist = np.linalg.norm(lab - fondo, axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), 2)
    umbral, _ = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    piel = ((dist > umbral) & cubierto).astype(np.uint8)
    piel = cv2.morphologyEx(piel, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(piel)
    if n <= 1:
        raise ValueError("No se encontro la piel en el mapa (¿mesa del mismo color que el cuero?).")
    piel = (lab_ == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contornos, _ = cv2.findContours(piel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    llena = np.zeros_like(piel); cv2.drawContours(llena, contornos, -1, 1, -1)
    agujeros = (llena > 0) & (piel == 0)                   # fondo visible DENTRO de la piel
    return llena.astype(bool), agujeros, max(contornos, key=cv2.contourArea)


def detectar(mosaico, piel, agujeros, cfg):
    s = cfg.mm_por_px
    gris = cv2.cvtColor(mosaico, cv2.COLOR_BGR2GRAY).astype(np.float32)
    interior = cv2.erode(piel.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=max(1, int(cfg.borde_ignorado_mm / s / 1))) > 0
    media = float(np.mean(gris[interior])) if interior.any() else 128.0
    trabajo = np.where(piel, gris, media).astype(np.float32)
    suave = cv2.GaussianBlur(trabajo, (0, 0), cfg.suavizado_mm / s)
    # "cuero normal": mediana en ventana grande, calculada en chico para que sea rapido
    k = 4
    chico = cv2.resize(np.clip(suave, 0, 255).astype(np.uint8), None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
    ventana = max(3, int(cfg.fondo_mm / s / k) | 1)
    fondo = cv2.resize(cv2.medianBlur(chico, ventana), (suave.shape[1], suave.shape[0]), interpolation=cv2.INTER_LINEAR)
    dif = suave - fondo.astype(np.float32)
    ruido = float(np.std(dif[interior])) if interior.any() else 1.0
    # segunda escala: manchas grandes y suaves (se comparan contra una ventana mucho mayor)
    k2 = 8
    chico2 = cv2.resize(cv2.GaussianBlur(trabajo, (0, 0), 3.0 / s), None, fx=1 / k2, fy=1 / k2, interpolation=cv2.INTER_AREA)
    ventana2 = max(3, int(cfg.fondo_manchas_mm / s / k2) | 1)
    fondo2 = cv2.medianBlur(np.clip(chico2, 0, 255).astype(np.uint8), ventana2).astype(np.float32)
    dif2 = cv2.resize(chico2 - fondo2, (suave.shape[1], suave.shape[0]), interpolation=cv2.INTER_LINEAR)
    ruido2 = float(np.std(dif2[interior])) if interior.any() else 1.0
    manchas = (dif2 < -cfg.umbral_sigma * ruido2) & interior
    cand = ((np.abs(dif) > cfg.umbral_sigma * ruido) & interior) | (agujeros & interior) | manchas
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    grupo = cv2.dilate(cand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(cfg.agrupar_mm / s) | 1,) * 2))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grupo)
    lista, mascara = [], np.zeros(piel.shape, np.uint8)
    for i in range(1, n):
        nucleo = (cand > 0) & (lab == i)
        area_mm2 = nucleo.sum() * s * s
        if area_mm2 < cfg.area_minima_mm2:
            continue
        x, y, w, h = st[i, :4]
        signo = float(np.mean(dif[nucleo]))
        alarg = max(w, h) / max(1, min(w, h))
        if (agujeros & nucleo).sum() > 0.3 * nucleo.sum():
            tipo = "agujero"
        elif signo > 0:
            tipo = "arruga"
        elif (manchas & nucleo).sum() > 0.5 * nucleo.sum() or area_mm2 > 900:
            tipo = "mancha"
        elif alarg > 3 or (nucleo.sum() < 0.25 * w * h and max(w, h) * s > 60):
            tipo = "raya"
        else:
            tipo = "picadura"
        mascara[nucleo] = 255
        cont, _ = cv2.findContours(nucleo.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lista.append({"tipo": tipo, "area_mm2": round(area_mm2, 1),
                      "caja_mm": [round(x * s, 1), round(y * s, 1), round((x + w) * s, 1), round((y + h) * s, 1)],
                      "contornos_px": [c.reshape(-1, 2).tolist() for c in cont]})
    lista.sort(key=lambda d: d["caja_mm"][0])
    return lista, mascara > 0, ruido
