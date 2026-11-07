"""Acomodo automatico de moldes sobre la piel, esquivando defectos (anidamiento por rejilla).

Los moldes se leen de archivos DXF (un molde por archivo, en milimetros) o se usan los de ejemplo.
"""
import os
import math
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from shapely import affinity


# ---------------- moldes ----------------
def molde_desde_dxf(ruta):
    import ezdxf
    from ezdxf import path as dxfpath
    doc = ezdxf.readfile(ruta)
    cerrados, abiertos = [], []
    for e in doc.modelspace():
        try:
            p = dxfpath.make_path(e)
        except TypeError:
            continue
        pts = [(round(v.x, 2), round(v.y, 2)) for v in p.flattening(0.3)]   # puntas redondeadas a 0,01 mm para que calcen
        if len(pts) < 2:
            continue
        if len(pts) >= 3 and (np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.05 or getattr(e.dxf, "flags", 0) & 1
                              or e.dxftype() in ("CIRCLE", "ELLIPSE")):
            cerrados.append(Polygon(pts).buffer(0))
        else:
            abiertos.append(LineString(pts))
    if abiertos:
        unidos = unary_union(abiertos)
        nuevos = list(polygonize(unidos))
        if not nuevos:     # puntas que casi se tocan: se cierran engrosando las lineas y tomando el hueco interior
            gordo = unidos.buffer(0.2)
            for g in getattr(gordo, "geoms", [gordo]):
                nuevos += [Polygon(h).buffer(0.2) for h in g.interiors]
        cerrados += nuevos
    cerrados = [c for c in cerrados if c.is_valid and c.area > 1]
    if not cerrados:
        raise ValueError(f"{os.path.basename(ruta)}: no tiene ningun contorno cerrado.")
    pol = max(cerrados, key=lambda c: c.area)
    if pol.geom_type == "MultiPolygon":
        pol = max(pol.geoms, key=lambda g: g.area)
    a, b, _, _ = pol.bounds
    return affinity.translate(Polygon(pol.exterior), -a, -b)


def moldes_ejemplo():
    cap = [(130 + 130 * math.cos(t), 85 + 85 * math.sin(t)) for t in np.linspace(-math.pi / 2, math.pi / 2, 40)]
    cap += [(30, 170), (0, 150), (0, 20), (30, 0)]
    plan = [(135 + 135 * math.cos(t), 47 + (43.5 + 3.5 * math.cos(t)) * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 72)]
    bill = Polygon([(0, 0), (200, 0), (200, 95), (0, 95)]).buffer(-10).buffer(10)
    return [{"nombre": "capellada", "poligono": Polygon(cap), "cantidad": None, "color": (38, 167, 255)},
            {"nombre": "plantilla", "poligono": Polygon(plan), "cantidad": None, "color": (218, 198, 38)},
            {"nombre": "billetera", "poligono": bill, "cantidad": None, "color": (255, 120, 171)}]


# ---------------- rasterizado ----------------
def _raster(pol, r):
    a, b, c, d = pol.bounds
    pol = affinity.translate(pol, -a, -b)
    w, h = int(math.ceil((c - a) / r)) + 1, int(math.ceil((d - b) / r)) + 1
    m = np.zeros((h, w), np.uint8)
    cont = np.array(pol.exterior.coords) / r
    cv2.fillPoly(m, [np.round(cont).astype(np.int32)], 1)
    m = cv2.dilate(m, np.ones((2, 2), np.uint8))          # conservador: toda celda tocada cuenta
    return m, pol


def acomodar(piel, defectos, moldes, cfg):
    """piel, defectos: mascaras booleanas en el mapa (mm_por_px). Devuelve lista de piezas en mm."""
    s, r = cfg.mm_por_px, cfg.resolucion_acomodo_mm
    Hm, Wm = piel.shape
    GW, GH = int(math.ceil(Wm * s / r)), int(math.ceil(Hm * s / r))
    def a_rejilla(m, todo=False):
        g = cv2.resize(m.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
        return g > 0.999 if todo else g > 0.0
    k_borde = max(1, int(round(cfg.margen_borde_mm / s)))
    util = a_rejilla(cv2.erode(piel.astype(np.uint8), np.ones((2 * k_borde + 1,) * 2, np.uint8)), todo=True)
    k_def = max(1, int(round(cfg.margen_defecto_mm / s)))
    malo = a_rejilla(cv2.dilate(defectos.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_def + 1,) * 2)))
    ocupado = ((~util) | malo).astype(np.uint8)
    sep = max(1, int(round(cfg.separacion_mm / r)))

    variantes = {}
    for m in moldes:
        vs = []
        for rot in cfg.rotaciones:
            pr = affinity.rotate(m["poligono"], rot, origin=(0, 0))
            ras, pn = _raster(pr, r)
            con_sep = cv2.dilate(np.pad(ras, sep), np.ones((2 * sep + 1,) * 2, np.uint8))   # pieza + espacio de corte
            vs.append((rot, ras, con_sep, pn))
        variantes[m["nombre"]] = vs

    cola = []
    for m in sorted(moldes, key=lambda m: -m["poligono"].area):
        if m.get("cantidad"):
            cola += [m["nombre"]] * int(m["cantidad"])
    relleno = [m["nombre"] for m in sorted(moldes, key=lambda m: -m["poligono"].area) if not m.get("cantidad")]

    piezas, agotados = [], set()

    def mejor_lugar(nombre):
        mejor = None
        for rot, ras, ras_sep, pn in variantes[nombre]:
            h, w = ras.shape
            if h > GH or w > GW:
                continue
            choque = cv2.filter2D(ocupado.astype(np.float32), -1, ras.astype(np.float32), anchor=(0, 0),
                                  borderType=cv2.BORDER_CONSTANT)
            choque[GH - h + 1:, :] = 1; choque[:, GW - w + 1:] = 1
            ys, xs = np.nonzero(choque < 0.5)
            if len(xs) == 0:
                continue
            i = np.lexsort((ys, xs))[0]                      # lo mas a la izquierda y luego arriba
            if mejor is None or (xs[i], ys[i]) < (mejor[0], mejor[1]):
                mejor = (int(xs[i]), int(ys[i]), rot, ras_sep, pn)
        return mejor

    def fijar(nombre, lugar):
        x, y, rot, ras_sep, pn = lugar
        y0, x0 = max(0, y - sep), max(0, x - sep)
        sub = ras_sep[y0 - (y - sep):, x0 - (x - sep):]
        ocupado[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] |= sub[:GH - y0, :GW - x0]
        piezas.append({"nombre": nombre, "rot": rot, "x_mm": x * r, "y_mm": y * r,
                       "poligono": affinity.translate(pn, x * r, y * r)})

    def colocar(nombre):
        lugar = mejor_lugar(nombre)
        if lugar is None:
            return False
        fijar(nombre, lugar)
        return True

    for nombre in cola:
        colocar(nombre)
    # relleno: dos estrategias y se queda con la que aprovecha mas la piel
    area = {m["nombre"]: m["poligono"].area for m in moldes}
    base_ocupado, base_piezas = ocupado.copy(), list(piezas)
    resultados = []
    for estrategia in ("grande_primero", "izquierda_primero"):
        ocupado[:] = base_ocupado; piezas[:] = base_piezas; agotados.clear()
        for _ in range(5000):
            candidatos = []
            for nombre in relleno:
                if nombre in agotados:
                    continue
                c = mejor_lugar(nombre)
                if c is None:
                    agotados.add(nombre)
                    continue
                candidatos.append((nombre, c))
                if estrategia == "grande_primero":
                    break                       # el molde mas grande que todavia entra
            if not candidatos:
                break
            if estrategia == "izquierda_primero":
                nombre, c = min(candidatos, key=lambda t: (t[1][0], -area[t[0]], t[1][1]))
            else:
                nombre, c = candidatos[0]
            fijar(nombre, c)
        resultados.append((sum(area[p["nombre"]] for p in piezas), list(piezas)))
    piezas = max(resultados, key=lambda t: t[0])[1]
    return piezas
