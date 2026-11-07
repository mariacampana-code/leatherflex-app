"""Leather-Flex 3D - Con Detección Automatizada de Defectos (Visión Artificial)"""
import os
import json
import sqlite3
import tempfile
import shutil
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# Funciones simuladas/importadas de la lógica de negocio
from leatherflex import calibracion, pipeline, acomodo, simulador
from leatherflex.config import Config

# ------------------------------------------------------------------ Configuración de página
st.set_page_config(
    page_title="Leather-Flex 3D",
    page_icon="🟫",
    layout="wide",
    initial_sidebar_state="expanded"
)

TMP = st.session_state.setdefault("tmp", tempfile.mkdtemp(prefix="leatherflex_"))
DB_PATH = os.path.join(os.path.dirname(__file__), "leatherflex.db")

# ------------------------------------------------------------------ CSS Personalizado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f3eee8 0%, #ab9680 100%) !important;
        color: #2c2520 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(171, 150, 128, 0.3) !important;
    }
    
    div[data-testid="stSidebarHeader"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    .brand-container {
        padding: 1.2rem 0.5rem;
        border-bottom: 1px solid rgba(171, 150, 128, 0.3);
        margin-bottom: 1rem;
    }
    .brand-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #3e3228;
        letter-spacing: 0.5px;
    }

    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div { gap: 8px; }
    div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.45) !important;
        border: 1px solid rgba(171, 150, 128, 0.4) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: #4a3e35 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #8c7763 !important;
        color: #ffffff !important;
        border-color: #73604f !important;
        box-shadow: 0 4px 12px rgba(140, 119, 99, 0.3) !important;
    }
    div[data-testid="stRadio"] input[type="radio"] { display: none !important; }

    .feed-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Lógica de Negocio (Módulos simulados)
def generar_contenido_dxf(puntos):
    """Genera un contenido DXF básico de polilíneas."""
    dxf_content = [
        "0", "SECTION", "2", "HEADER", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES", "0", "ENDSEC",
        "0", "SECTION", "2", "BLOCKS", "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
        "0", "POLYLINE", "8", "MOLDES", "66", "1", "70", "1"
    ]
    for x, y in puntos:
        dxf_content.extend(["0", "VERTEX", "8", "MOLDES", "10", str(x), "20", str(y), "30", "0.0"])
    dxf_content.extend(["0", "SEQEND", "0", "ENDSEC", "0", "EOF"])
    return "\n".join(dxf_content)

def crear_dxf_demo(puntos, nombre_archivo):
    contenido = generar_contenido_dxf(puntos)
    with open(nombre_archivo, "w") as f:
        f.write(contenido)

def extraer_puntos_desde_foto(bytes_imagen):
    """Detecta el contorno del molde físico y calcula dimensiones."""
    arr = np.frombuffer(bytes_imagen, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # Procesamiento básico para contorno
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contorno más grande
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Molde demo por defecto si falla la detección
        return [(0,0), (200,0), (180,120), (20,120)], "200 x 120"
        
    c_max = max(cnts, key=cv2.contourArea)
    # Aproximar polígono
    epsilon = 0.01 * cv2.arcLength(c_max, True)
    approx = cv2.approxPolyDP(c_max, epsilon, True)
    
    # Convertir a lista de tuplas
    pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    
    # Calcular dimensiones bounding box
    x, y, bw, bh = cv2.boundingRect(c_max)
    # En producción usaríamos la calibración mm/px, aquí simulamos
    dims_str = f"{int(bw)} x {int(bh)}" # Usando px como mm demo
    
    return pts, dims_str

# ================================================================== NUEVA FUNCIÓN: DETECCIÓN DE DEFECTOS
def detectar_defectos_cuero(bytes_imagen):
    """Algoritmo de Visión Artificial para detectar anomalías en la piel."""
    arr = np.frombuffer(bytes_imagen, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, [], None

    h, w, _ = img.shape
    
    # 1. Convertir a Grises y desenfocar para ruido
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0) # Desenfoque fuerte para textura promedio
    
    # 2. Detección por contraste (Thresholding Adaptativo u Otsu)
    # Buscamos desviaciones oscuras (típicas de cicatrices/marcas)
    # Adaptativo funciona mejor con iluminación variable de cámara
    # En su lugar, usaremos una técnica de "segmentación de textura" básica
    # calculando la diferencia absoluta entre el original y el promedio local.
    
    diff = cv2.absdiff(gray, blur)
    
    # Umbralizar la diferencia para resaltar las anomalías
    # Este umbral puede necesitar calibración
    umbral_contraste = 25 # Sensibilidad al defecto (más bajo = más sensible)
    _, defects_mask = cv2.threshold(diff, umbral_contraste, 255, cv2.THRESH_BINARY)
    
    # Limpieza morfológica para agrupar defectos pequeños (picaduras)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(defects_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=3)
    
    # 3. Encontrar contornos de los defectos
    cnts_def, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Generar datos y visualizar
    reporte_defectos = []
    lista_obstaculos_px = []
    
    img_viz = img.copy() # Imagen para la UI
    
    # Definir filtros de tamaño mínimo/máximo para ruido
    min_area_px = 50 # pixeles
    
    for i, c in enumerate(cnts_def):
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
            
        # Simplificar el contorno para obstáculo de Nesting
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        obst_pts = [pt[0] for pt in approx]
        lista_obstaculos_px.append(obst_pts)
        
        # Guardar en reporte
        x, y, bw, bh = cv2.boundingRect(c)
        reporte_defectos.append({
            "id": i+1,
            "tipo": "Imperfección", # Genérico para demo
            "area_px2": area,
            "bbox": (x, y, bw, bh)
        })
        
        # Dibujar contorno rojo en la visualización
        cv2.drawContours(img_viz, [approx], -1, (0, 0, 255), 3) # Rojo, grosor 3
        # Etiqueta
        cv2.putText(img_viz, f"DEF_{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return img_viz, lista_obstaculos_px, reporte_defectos

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS moldes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, cliente TEXT, 
                  dimensiones TEXT, fecha TEXT, archivo_dxf TEXT, origen TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feed 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, usuario TEXT, titulo TEXT, 
                  descripcion TEXT, tipo_falla TEXT, solucion TEXT, fecha TEXT, media_path TEXT)''')
    
    # Intentar añadir columna origen a moldes antiguos
    try:
        c.execute("ALTER TABLE moldes ADD COLUMN origen TEXT")
    except Exception:
        pass

    # Moldes Precargados Demo (para evitar que esté vacía al inicio)
    c.execute("SELECT COUNT(*) FROM moldes")
    count = c.fetchone()[0]
    if count == 0:
        carp_dxf = os.path.join(TMP, "moldes_precargados")
        os.makedirs(carp_dxf, exist_ok=True)
        moldes_demo = [
            ("Bolsillo Frontal Billetera", "Personalizado", "110 x 80", [(0,0), (110,0), (100,80), (10,80)]),
            ("Tirante Mochila Cuero", "Artisan Leather", "450 x 45", [(0,0), (450,0), (440,45), (10,45)]),
            ("Cuerpo Bolso Mano", "Moda & Cuero", "350 x 250", [(0,0), (350,0), (330,250), (20,250)]),
            ("Plantilla Ergonómica Talla 42", "Industrial Boot", "280 x 95", [(0,0), (280,0), (260,95), (20,95)]),
            ("Capellada Zapato Oxford", "Calzados Lujo", "240 x 160", [(0,0), (240,0), (220,160), (20,160)])
        ]
        for nom, cli, dim, pts in moldes_demo:
            ruta = os.path.join(carp_dxf, f"{nom.replace(' ', '_')}.dxf")
            crear_dxf_demo(pts, ruta)
            c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                      (nom, cli, dim, datetime.now().strftime("%d/%m/%Y"), ruta, "foto"))
            
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------ Barra Lateral
with st.sidebar:
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">LEATHER-FLEX 3D</div>
        </div>
    """, unsafe_allow_html=True)
    
    opcion_menu = st.radio(
        "Navegacion",
        ["Captura del Cuero", "Acomodo de Moldes", "Calibración de Escáner", "Catálogo de Moldes", "Biblioteca Técnica"],
        label_visibility="collapsed"
    )

# Cargar calibración (o demo si no existe)
ruta_cal = os.path.join(TMP, "calib.json")
if not os.path.exists(ruta_cal):
    with open(ruta_cal, "w") as f:
        json.dump({"mm_por_px_foto": 0.5, "angulo_grados": 0.0, "matriz_homografia": None}, f)

# ================================================================== MÓDULO 1: CAPTURA DEL CUERO
if opcion_menu == "Captura del Cuero":
    st.title("Captura del Cuero")
    st.caption("Registro de evidencia en vivo (cámara/galería) e historial del taller.")

    @st.dialog("Capturar Nueva Piel / Evidencia")
    def modal_crear_post():
        st.write("### Seleccionar foto o video del cuero:")
        
        archivo_subido = st.file_uploader(
            "Toma una foto/video con la cámara o elige de tu galería", 
            type=["jpg", "png", "jpeg", "mp4", "mov"],
            help="Al presionar se abrirá la cámara nativa de tu dispositivo"
        )

        nota_rapida = st.text_input("Identificador / Lote (Opcional)", placeholder="Ej. Lote Cuero Vacuno A-1")
        
        if st.button("Guardar y Enviar a Acomodo de Moldes", type="primary", use_container_width=True):
            if archivo_subido is None:
                st.warning("Por favor toma una foto/video o elige un archivo.")
            else:
                ext = os.path.splitext(archivo_subido.name)[1] or ".jpg"
                ruta_archivo_guardado = os.path.join(TMP, f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
                with open(ruta_archivo_guardado, "wb") as f:
                    f.write(archivo_subido.getbuffer())

                # Guardar en DB Feed
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO feed (usuario, titulo, descripcion, tipo_falla, solucion, fecha, media_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          ("Operario Taller", nota_rapida or "Captura de Cuero", "", "Piel", "", datetime.now().strftime("%d/%m/%Y %H:%M"), ruta_archivo_guardado))
                conn.commit()
                conn.close()

                # Marcar como captura activa para el módulo de Acomodo
                st.session_state["ultima_captura_cuero"] = ruta_archivo_guardado
                st.success("Captura guardada")
                st.rerun()

    if st.button("Tomar foto / video del cuero", type="primary", use_container_width=True):
        modal_crear_post()

    st.write("---")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, usuario, titulo, fecha, media_path FROM feed ORDER BY id DESC")
    posts = c.fetchall()
    conn.close()

    if not posts:
        st.info("No hay capturas registradas. Presiona para tomar la primera foto.")
    else:
        for p in posts:
            post_id, usr, tit, fecha, media = p
            with st.container():
                st.markdown(f"""
                <div class="feed-card">
                    <small style="color: #73604f; font-weight: 600;">{usr} • {fecha}</small>
                    <h4 style="margin-top: 4px; margin-bottom: 4px; color: #2c2520;">{tit}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if media and os.path.exists(media):
                    if media.endswith((".mp4", ".mov")):
                        st.video(media)
                    else:
                        st.image(media, use_column_width=True)

                col_del, _ = st.columns([1, 3])
                with col_del:
                    if st.button("Eliminar", key=f"del_{post_id}"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("DELETE FROM feed WHERE id = ?", (post_id,))
                        conn.commit()
                        conn.close()

                        # Borrar archivo físico
                        if media and os.path.exists(media):
                            try:
                                os.remove(media)
                            except Exception:
                                pass

                        # Desvincular de captura activa si corresponde
                        if st.session_state.get("ultima_captura_cuero") == media:
                            st.session_state["ultima_captura_cuero"] = None

                        st.success("Eliminado correctamente.")
                        st.rerun()
                st.divider()

# ================================================================== MÓDULO 2: ACOMODO DE MOLDES
elif opcion_menu == "Acomodo de Moldes":
    st.title("Acomodo de Moldes")
    st.caption("Procesamiento de imagen para detección automática de defectos y cálculo de Nesting.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Imagen de Piel Activa")
        
        foto_conectada = st.session_state.get("ultima_captura_cuero")
        
        # Guardar imagen para procesar si viene conectada
        imagen_a_procesar_bytes = None

        if foto_conectada and os.path.exists(foto_conectada):
            st.success("Imagen conectada directamente de 'Captura del Cuero'")
            with open(foto_conectada, "rb") as f:
                imagen_a_procesar_bytes = f.read()
            st.image(foto_conectada, caption="Piel seleccionada para procesamiento", use_column_width=True)
            if st.button("Desvincular esta foto"):
                st.session_state["ultima_captura_cuero"] = None
                st.rerun()
        else:
            st.info("No hay una foto activa. Puedes tomar una en 'Captura del Cuero' o subir una imagen manual aquí:")

        fotos_manuales = st.file_uploader("Subir otra foto manualmente", type=["jpg", "jpeg", "png"], key="fotos")
        if fotos_manuales and not imagen_a_procesar_bytes:
            imagen_a_procesar_bytes = fotos_manuales.getbuffer()

        demo = st.button("Usar Muestra de Prueba (Demo con defectos simulados)")

    with col_b:
        st.subheader("2. Selección de Moldes (Catálogo o Archivo)")
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, nombre, cliente, archivo_dxf FROM moldes ORDER BY id DESC")
        moldes_bd = c.fetchall()
        conn.close()

        moldes_seleccionados = []
        if moldes_bd:
            st.write("**Seleccionar del Catálogo Registrado:**")
            opciones_cat = {f"{m[1]} ({m[2]})": m[3] for m in moldes_bd}
            moldes_elegidos = st.multiselect("Elegir moldes para cortar:", list(opciones_cat.keys()))
            
            for m_nombre in moldes_elegidos:
                path_dxf_cat = opciones_cat[m_nombre]
                if path_dxf_cat and os.path.exists(path_dxf_cat):
                    # Pedir cantidad por cada molde
                    cant = st.number_input(f"Cantidad de '{m_nombre}'", 1, 500, 1, key=f"cat_{m_nombre}")
                    try:
                        pol = acomodo.molde_desde_dxf(path_dxf_cat)
                        moldes_seleccionados.append({"nombre": m_nombre, "poligono": pol, "cantidad": cant})
                    except Exception:
                        pass

        dxfs = st.file_uploader("O subir moldes (.dxf) manualmente:", accept_multiple_files=True, key="dxf")
        if dxfs:
            carp_m = os.path.join(TMP, "moldes")
            shutil.rmtree(carp_m, ignore_errors=True)
            os.makedirs(carp_m)
            for a in dxfs:
                path_a = os.path.join(carp_m, a.name)
                with open(path_a, "wb") as f:
                    f.write(a.getbuffer())
                try:
                    pol = acomodo.molde_desde_dxf(path_a)
                    cant = st.number_input(f"Cantidad {a.name}", 1, 500, 1, key="c_" + a.name)
                    moldes_seleccionados.append({"nombre": os.path.splitext(a.name)[0], "poligono": pol, "cantidad": cant})
                except Exception as e:
                    st.error(f"Error en {a.name}: {e}")

        moldes = moldes_seleccionados if moldes_seleccionados else None

    st.write("---")
    
    # ------------------------------------------------------------------ EJECUTAR DETECCION Y ACOMODO
    tiene_imagen = imagen_a_procesar_bytes is not None or demo
    procesar_click = st.button("Iniciar Procesamiento y Acomodo de Corte", type="primary", disabled=not tiene_imagen)

    if (procesar_click and tiene_imagen):
        status = st.status("Iniciando Visión Artificial y Nesting...", expanded=True)
        
        try:
            # 1. Preparar la imagen del cuero
            piel_bytes = None
            if demo:
                piel_bytes = simulador.get_piel_demo_bytes() # Simular bytes para demo
            else:
                piel_bytes = imagen_a_procesar_bytes
                
            status.write("Procesando imagen del cuero...")
            
            # ========================================================== PASO 1: DETECTAR DEFECTOS
            with st.spinner("Visión Artificial detectando defectos..."):
                img_viz, obstaculos_px, reporte = detectar_defectos_cuero(piel_bytes)
            
            if img_viz is None:
                st.error("Fallo la carga de imagen para detección de defectos.")
                st.session_state["resultado"] = None
            else:
                # ====================================================== PASO 2: ACOMODO (NESTING)
                status.write("Detección finalizada. Calculando Nesting (Nidificación) evitando defectos...")
                with st.spinner("Optimizando acomodo de moldes (Backtracking)..."):
                    # El simulador toma los obstáculos (defectos) para el Nesting
                    cfg = Config()
                    r_pipeline = pipeline.nest_completo(piel_bytes, obstaculos_px, moldes, cfg)
                
                status.update(label="Procesamiento Exitoso", state="complete", expanded=False)
                
                # Guardar info en session state
                st.session_state["resultado"] = {
                    "r_nest": r_pipeline, # Resultados del pipeline simulado
                    "img_viz_defectos": img_viz,
                    "reporte_defectos": reporte
                }

        except Exception as e:
            status.update(label="Error en el procesamiento", state="error")
            st.error(f"Error técnico durante el Nesting: {e}")
            st.session_state["resultado"] = None

    # Mostrar resultados si existen
    if st.session_state.get("resultado"):
        res = st.session_state["resultado"]
        
        st.subheader("Resultados del Análisis y Acomodo")
        m = st.columns(4)
        r_nest = res["r_nest"]
        info_nest = r_nest["info"]
        
        m[0].metric("Área Útil Piel", f"{info_nest['piel_m2']} m²")
        m[1].metric("Imperfecciones Detectadas", info_nest["defectos"])
        m[2].metric("Piezas Acomodadas", info_nest["piezas"])
        m[3].metric("Aprovechamiento", f"{info_nest['aprovechamiento_pct']} %")
        
        t_corte, t_vision = st.tabs(["Nidificación / Plano de Corte DXF", "Detección Óptica de Defectos"])
        
        with t_corte:
            st.write("### Plano de nidificación calculado:")
            st.image(r_nest["imagen_corte"], use_column_width=True, caption="Acomodo final evitando imperfecciones (zonas rojas)")
            st.download_button("Descargar Plano de Corte DXF", r_nest["plano_corte_dxf"], "nidificacion_produccion.dxf", "application/dxf")

        with t_vision:
            st.write("### Visualización Óptica en Taller (Defectos en Rojo)")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(res["img_viz_defectos"], use_column_width=True, caption="Zonas prohibidas para el corte detectadas por Visión Artificial")
            with col2:
                st.write("#### Reporte de Imperfecciones:")
                df_reporte = [{"N°": d["id"], "Tipo": d["tipo"], "Área (px²)": int(d["area_px2"])} for d in res["reporte_defectos"]]
                st.dataframe(df_reporte, use_container_width=True, hide_index=True)

# ================================================================== MÓDULOS SECUNDARIOS
elif opcion_menu == "Calibración de Escáner":
    st.title("Calibración del Sistema Óptico")
    st.caption("Ajuste de parámetros de cámara, escala de milímetros por píxel y corrección de plano.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Tablero de Referencia")
        ruta_tab = os.path.join(TMP, "tablero_para_imprimir.png")
        if not os.path.exists(ruta_tab):
            calibracion.tablero_para_imprimir(ruta_tab)
        st.download_button("Descargar Tablero Calibración", open(ruta_tab, "rb"), "tablero_calibracion.png", "image/png")
        
        f_tab = st.file_uploader("Subir foto del tablero en mesa", type=["jpg", "jpeg", "png"], key="tab")
        f_bla = st.file_uploader("Hoja blanca para compensación (opcional)", type=["jpg", "jpeg", "png"], key="bla")

    with col2:
        st.subheader("2. Ajustes de Sensibilidad Óptica")
        Umbral_Def = st.slider("Sensibilidad de Detección de Defectos (más bajo = más sensible)", 10, 80, 25, 1, help="Calibra el umbral de contraste para cicatrices")

    st.write("---")
    if f_tab and st.button("Guardar y Calibrar Escáner", type="primary"):
        try:
            cal = calibracion.calibrar_tablero(cv2.imdecode(np.frombuffer(f_tab.getbuffer(), np.uint8), cv2.IMREAD_COLOR))
            calibracion.guardar(cal, ruta_cal)
            st.success("Escáner Calibrado y configurado correctamente.")
        except Exception as e:
            st.error(f"Error durante la calibración: {e}")

elif opcion_menu == "Catálogo de Moldes":
    st.title("Catálogo Digital de Moldes")
    st.caption("Base de datos de patrones vectoriales y digitalizaciones físicas.")

    @st.dialog("Digitalizar Nuevo Molde Físico")
    def modal_capturar_molde_foto():
        st.write("### Tomar foto o seleccionar archivo del molde:")
        foto_molde = st.file_uploader(
            "Toma una foto del molde o elige de tu galería", 
            type=["jpg", "png", "jpeg"],
            help="Al presionar se abrirá la cámara nativa de tu dispositivo"
        )

        nom_f = st.text_input("Nombre de la Pieza", value="Capellada Zapato Oxford T41")
        cli_f = st.text_input("Cliente", value="Empresa Demo Calzado")
        
        if st.button("Digitalizar y Guardar en Catálogo", type="primary", use_container_width=True):
            if not foto_molde or not nom_f:
                st.warning("Por favor toma la foto y escribe el nombre de la pieza.")
            else:
                pts, dims_calc = extraer_puntos_desde_foto(foto_molde.getbuffer())
                nombre_archivo_dxf = f"dxf_foto_{nom_f.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dxf"
                ruta_guardada = os.path.join(TMP, nombre_archivo_dxf)
                
                contenido_dxf = generar_contenido_dxf(pts)
                with open(ruta_guardada, "w") as f:
                    f.write(contenido_dxf)

                # Guardar en base de datos
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO moldes (nombre, cliente, dimensiones, fecha, archivo_dxf, origen) VALUES (?, ?, ?, ?, ?, ?)",
                          (nom_f, cli_f or "Interno", dims_calc, datetime.now().strftime("%d/%m/%Y"), ruta_guardada, "foto"))
                conn.commit()
                conn.close()
                st.success(f"Molde '{nom_f}' digitalizado y guardado con éxito.")
                st.rerun()

    if st.button("Tomar foto / subir foto al molde", type="primary", use_container_width=True):
        modal_capturar_molde_foto()

    st.write("---")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, nombre, cliente, dimensiones, fecha, archivo_dxf FROM moldes ORDER BY id DESC")
    m_todos = c.fetchall()
    conn.close()

    if m_todos:
        cols = st.columns(3)
        for idx, m in enumerate(m_todos):
            with cols[idx % 3]:
                st.markdown(f"#### {m[1]}")
                st.write(f"Cliente: **{m[2]}**")
                st.write(f"Dimensiones: **{m[3]}**")
                
                contenido_dxf = ""
                if m[5] and os.path.exists(m[5]):
                    with open(m[5], "r") as f_dxf:
                        contenido_dxf = f_dxf.read()
                else:
                    pts_base = [(0,0), (200,0), (180,120), (20,120)]
                    contenido_dxf = generar_contenido_dxf(pts_base)
                
                st.download_button(
                    "Descargar Vector DXF", 
                    data=contenido_dxf, 
                    file_name=f"{m[1].replace(' ', '_')}.dxf", 
                    mime="application/dxf", 
                    key=f"btn_m_{m[0]}"
                )
    else:
        st.info("Sin moldes registrados. Presiona el botón de arriba para tomar foto y digitalizar el primero.")

elif opcion_menu == "Biblioteca Técnica":
    st.title("Capacitación y Guías Técnicas")
    st.write("Módulo reservado para tutoriales en video e instructivos para operarios.")
