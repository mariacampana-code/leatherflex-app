"""Salidas: imagen del mapa con defectos y piezas, plantilla de corte DXF y reporte JSON."""
import json
import numpy as np
import cv2

ROJO = (53, 57, 229)


def imagen_resultado(mosaico, contorno_piel, defectos, piezas, moldes, cfg):
    s = cfg.mm_por_px
    img = mosaico.copy()
    capa = img.copy()
    for d in defectos:
        for c in d["contornos_px"]:
            cv2.drawContours(capa, [np.array(c, np.int32)], -1, ROJO, -1)
    colores = {m["nombre"]: m.get("color", (255, 167, 38)) for m in moldes}
    for p in piezas:
        pts = np.round(np.array(p["poligono"].exterior.coords) / s).astype(np.int32)
        cv2.fillPoly(capa, [pts], colores[p["nombre"]])
    img = cv2.addWeighted(capa, 0.35, img, 0.65, 0)
    cv2.drawContours(img, [contorno_piel], -1, (255, 255, 255), 2)
    for d in defectos:
        for c in d["contornos_px"]:
            cv2.drawContours(img, [np.array(c, np.int32)], -1, ROJO, 2)
    for p in piezas:
        pts = np.round(np.array(p["poligono"].exterior.coords) / s).astype(np.int32)
        cv2.polylines(img, [pts], True, colores[p["nombre"]], 3)
    return img


def dxf_corte(ruta, alto_mapa_px, contorno_piel, defectos, piezas, cfg):
    """Plantilla en milimetros. Capas: PIEL, DEFECTOS, PIEZAS, TEXTOS. El eje Y se invierte (DXF mira hacia arriba)."""
    import ezdxf
    s = cfg.mm_por_px
    H = alto_mapa_px * s
    doc = ezdxf.new("R2010"); doc.units = ezdxf.units.MM
    for nombre, color in (("PIEL", 7), ("DEFECTOS", 1), ("PIEZAS", 3), ("TEXTOS", 5)):
        doc.layers.add(nombre, color=color)
    msp = doc.modelspace()
    def mm(pts_px):
        return [(float(x) * s, H - float(y) * s) for x, y in pts_px]
    msp.add_lwpolyline(mm(contorno_piel.reshape(-1, 2)), close=True, dxfattribs={"layer": "PIEL"})
    for d in defectos:
        for c in d["contornos_px"]:
            if len(c) >= 3:
                msp.add_lwpolyline(mm(c), close=True, dxfattribs={"layer": "DEFECTOS"})
    for k, p in enumerate(piezas):
        pts = [(x, H - y) for x, y in list(p["poligono"].exterior.coords)[:-1]]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "PIEZAS"})
        c = p["poligono"].centroid
        msp.add_text(f"{k + 1} {p['nombre']}", height=6,
                     dxfattribs={"layer": "TEXTOS"}).set_placement((c.x, H - c.y), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    doc.saveas(ruta)
    return ruta


def reporte(ruta, info):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return ruta
