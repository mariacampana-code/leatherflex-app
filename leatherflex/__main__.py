"""Uso por consola:

  python -m leatherflex tablero  salida/tablero.png              # tablero para imprimir
  python -m leatherflex calibrar tablero.jpg blanco.jpg calib.json
  python -m leatherflex procesar fotos/ calib.json resultados/ [--molde capellada.dxf:10 ...]
  python -m leatherflex simular  prueba/                          # escaneo sintetico para probar
"""
import argparse
import os
import cv2

from .config import Config
from . import calibracion, pipeline, simulador, acomodo


def main():
    ap = argparse.ArgumentParser(prog="leatherflex", description="Leather-Flex 3D: mapa de piel, defectos y acomodo de moldes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tablero"); t.add_argument("salida")
    c = sub.add_parser("calibrar"); c.add_argument("tablero"); c.add_argument("blanco"); c.add_argument("salida")
    p = sub.add_parser("procesar"); p.add_argument("fotos"); p.add_argument("calibracion"); p.add_argument("salida")
    p.add_argument("--config", help="archivo .json con parametros")
    p.add_argument("--filas", type=int, help="fotos por columna (si las fotos no tienen nombre de posicion)")
    p.add_argument("--molde", action="append", default=[],
                   help="archivo DXF de un molde, opcionalmente con cantidad: capellada.dxf:10 (sin cantidad = rellenar)")
    p.add_argument("--solo-0-180", action="store_true", help="no girar moldes 90 grados (respetar el hilo del cuero)")
    s = sub.add_parser("simular"); s.add_argument("carpeta"); s.add_argument("--sin-nombres", action="store_true")
    a = ap.parse_args()

    if a.cmd == "tablero":
        print("Tablero guardado en", calibracion.tablero_para_imprimir(a.salida))
    elif a.cmd == "calibrar":
        cal = calibracion.calibrar_tablero(cv2.imread(a.tablero))
        gan = calibracion.campo_plano(cv2.imread(a.blanco)) if a.blanco != "-" else None
        calibracion.guardar(cal, gan, a.salida)
        print(f"Escala: {cal['mm_por_px_foto']:.4f} mm/px  ·  giro de la camara: {cal['angulo_grados']:.2f} grados")
    elif a.cmd == "procesar":
        cfg = Config.cargar(a.config) if a.config else Config()
        if a.filas: cfg.filas = a.filas
        if a.solo_0_180: cfg.rotaciones = [0, 180]
        moldes = None
        if a.molde:
            moldes = []
            for spec in a.molde:
                ruta, _, cant = spec.partition(":")
                moldes.append({"nombre": os.path.splitext(os.path.basename(ruta))[0], "poligono": acomodo.molde_desde_dxf(ruta),
                               "cantidad": int(cant) if cant else None})
        r = pipeline.procesar(a.fotos, a.calibracion, a.salida, cfg, moldes)
        i = r["info"]
        print(f"\nPiel: {i['piel_m2']} m²  ·  defectos: {i['defectos']} {i['defectos_por_tipo']}")
        print(f"Piezas: {i['piezas']} {i['piezas_por_molde']}  ·  aprovechamiento: {i['aprovechamiento_pct']} %")
        print(f"Archivos en {a.salida}: mapa.jpg, resultado.jpg, plantilla_corte.dxf, reporte.json")
    elif a.cmd == "simular":
        simulador.generar(a.carpeta, nombrar=not a.sin_nombres)
        print("Escaneo sintetico creado en", a.carpeta)


if __name__ == "__main__":
    main()
