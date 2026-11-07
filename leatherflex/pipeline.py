"""Proceso completo: fotos -> mapa -> defectos -> acomodo -> archivos de salida."""
import os
import time
import cv2

from .config import Config
from . import calibracion, captura, montaje, defectos as defectos_mod, acomodo, exportar


def procesar(carpeta_fotos, ruta_calibracion, carpeta_salida, cfg=None, moldes=None, avisar=print):
    cfg = cfg or Config()
    os.makedirs(carpeta_salida, exist_ok=True)
    t0 = time.time()

    calib, ganancia = calibracion.cargar(ruta_calibracion)
    avisar("1/5 Cargando fotos")
    fotos = captura.cargar_fotos(carpeta_fotos, filas=cfg.filas, ruta=cfg.ruta)
    for f in fotos:
        f["prep"] = captura.preparar(f["img"], calib, ganancia, cfg.mm_por_px)
        del f["img"]

    avisar(f"2/5 Uniendo {len(fotos)} fotos")
    mosaico, cubierto, pos_mm, info_union = montaje.unir(fotos, cfg)

    avisar("3/5 Buscando defectos")
    piel, agujeros, contorno = defectos_mod.mascara_piel(mosaico, cubierto)
    lista, mascara_def, ruido = defectos_mod.detectar(mosaico, piel, agujeros, cfg)

    avisar("4/5 Acomodando moldes")
    moldes = moldes or acomodo.moldes_ejemplo()
    piezas = acomodo.acomodar(piel, mascara_def, moldes, cfg)

    avisar("5/5 Guardando resultados")
    s = cfg.mm_por_px
    area_piel = float(piel.sum()) * s * s / 1e6
    area_piezas = sum(p["poligono"].area for p in piezas) / 1e6
    resultado = exportar.imagen_resultado(mosaico, contorno, lista, piezas, moldes, cfg)
    cv2.imwrite(os.path.join(carpeta_salida, "mapa.jpg"), mosaico, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(carpeta_salida, "resultado.jpg"), resultado, [cv2.IMWRITE_JPEG_QUALITY, 92])
    exportar.dxf_corte(os.path.join(carpeta_salida, "plantilla_corte.dxf"), mosaico.shape[0], contorno, lista, piezas, cfg)
    por_tipo = {}
    for d in lista:
        por_tipo[d["tipo"]] = por_tipo.get(d["tipo"], 0) + 1
    info = {
        "union": info_union,
        "piel_m2": round(area_piel, 3),
        "defectos": len(lista), "defectos_por_tipo": por_tipo,
        "piezas": len(piezas), "piezas_por_molde": {m["nombre"]: sum(p["nombre"] == m["nombre"] for p in piezas) for m in moldes},
        "aprovechamiento_pct": round(100 * area_piezas / area_piel, 1) if area_piel else 0,
        "segundos": round(time.time() - t0, 1),
        "lista_defectos": [{k: v for k, v in d.items() if k != "contornos_px"} for d in lista],
    }
    exportar.reporte(os.path.join(carpeta_salida, "reporte.json"), info)
    return {"info": info, "mosaico": mosaico, "resultado": resultado, "piel": piel, "defectos": lista,
            "piezas": piezas, "posiciones_mm": pos_mm, "fotos": fotos}
