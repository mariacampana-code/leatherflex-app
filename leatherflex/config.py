"""Parametros del sistema Leather-Flex 3D. Todas las medidas en milimetros."""
from dataclasses import dataclass, field, asdict
import json


@dataclass
class Config:
    # --- captura ---
    paso_x_mm: float = 100.0          # distancia entre muescas de la regla (columnas)
    paso_y_mm: float = 100.0          # distancia entre posiciones a lo largo del puente (filas)
    ruta: str = "columnas"            # orden de las fotos si no vienen nombradas: "columnas" o "serpentina"
    filas: int = 4                    # filas por columna (para asignar posiciones por orden)

    # --- montaje ---
    mm_por_px: float = 0.4            # resolucion de trabajo del mapa
    max_correccion_mm: float = 15.0   # cuanto puede corregir el montaje respecto a la posicion de la traba
    peso_traba: float = 0.05          # confianza en la posicion nominal de la traba (vs. correlacion de imagen)

    # --- defectos ---
    suavizado_mm: float = 1.2         # quita el grano del cuero antes de buscar defectos
    fondo_mm: float = 25.0            # tamano de la ventana que estima el "cuero normal"
    fondo_manchas_mm: float = 120.0   # ventana para detectar manchas grandes y suaves
    umbral_sigma: float = 3.2         # cuantas desviaciones del cuero normal = defecto
    area_minima_mm2: float = 3.0      # defectos mas chicos se ignoran
    agrupar_mm: float = 6.0           # manchas cercanas se cuentan como un solo defecto
    borde_ignorado_mm: float = 10.0   # el borde de la piel no se analiza (siempre se descarta)

    # --- acomodo ---
    resolucion_acomodo_mm: float = 2.0
    margen_defecto_mm: float = 5.0    # distancia minima entre una pieza y un defecto
    margen_borde_mm: float = 5.0      # distancia minima entre una pieza y el borde de la piel
    separacion_mm: float = 2.0        # espacio entre piezas (cuchilla / tijera)
    rotaciones: list = field(default_factory=lambda: [0, 90, 180, 270])

    def guardar(self, ruta):
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            return cls(**json.load(f))
