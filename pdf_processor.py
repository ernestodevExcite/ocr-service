"""
Utilidades para procesamiento de PDFs:
  - Convertir páginas de un PDF a imágenes (requiere Poppler en PATH)
  - Generar un PDF searchable con texto OCR incrustado como capa invisible (usando fpdf2)
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

from pdf2image import convert_from_path
from fpdf import FPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Fuentes Unicode candidatas (se elige la primera que exista en el sistema)
_UNICODE_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",                                              # Windows – Arial
    r"C:\Windows\Fonts\calibri.ttf",                                            # Windows – Calibri
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",                         # Linux – DejaVu
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",         # Linux – Liberation
]


def _find_unicode_font() -> Optional[str]:
    for path in _UNICODE_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    fmt: str = "jpg",
    poppler_path: Optional[str] = None,
) -> List[str]:
    """
    Convierte cada página de un PDF a una imagen.

    Args:
        pdf_path:      Ruta al PDF de entrada.
        output_dir:    Directorio donde guardar las imágenes.
        dpi:           Resolución de las imágenes generadas (default 300).
        fmt:           Formato de salida: 'jpg' o 'png' (default 'jpg').
        poppler_path:  Ruta a los binarios de Poppler (opcional, si no está en PATH).

    Returns:
        Lista ordenada de rutas a las imágenes generadas.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(pdf_path).stem

    kwargs: dict = {"dpi": dpi, "fmt": fmt, "output_folder": output_dir, "paths_only": True}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    logger.info(f"Convirtiendo PDF a imágenes: {pdf_path} (dpi={dpi})")
    pages = convert_from_path(pdf_path, **kwargs)

    # Renombramos para mantener el nombre base + número de página
    renamed: List[str] = []
    for i, raw_path in enumerate(pages):
        ext = fmt.lower().replace("jpeg", "jpg")
        dest = os.path.join(output_dir, f"{base_name}_page_{i+1:04d}.{ext}")
        if str(raw_path) != dest:
            os.rename(str(raw_path), dest)
        renamed.append(dest)

    logger.info(f"  → {len(renamed)} páginas extraídas")
    return renamed


def build_searchable_pdf(
    pages_data: List[Dict],
    output_path: str,
) -> bool:
    """
    Genera un PDF searchable superponiendo el texto OCR como capa invisible
    sobre cada imagen de página.

    Args:
        pages_data: Lista de dicts con claves:
            - image_path (str): ruta a la imagen de la página
            - text_lines (list): líneas OCR con x_min, y_min, x_max, y_max, text/refined_text
        output_path: Ruta del PDF de salida.

    Returns:
        True si el PDF fue generado correctamente.
    """
    try:
        pdf = FPDF(unit="pt")
        pdf.set_auto_page_break(auto=False)

        # Cargar fuente Unicode (TTF).
        # Helvetica built-in solo soporta Latin-1; los textos OCR pueden tener
        # cualquier carácter Unicode, así que necesitamos una fuente TTF.
        unicode_font_path = _find_unicode_font()
        if unicode_font_path:
            font_name = "UniFont"
            pdf.add_font(font_name, fname=unicode_font_path)
            logger.info(f"Usando fuente Unicode TTF: {unicode_font_path}")
        else:
            font_name = "Helvetica"
            logger.warning(
                "No se encontró fuente Unicode TTF. "
                "Usando Helvetica con sanitización de caracteres."
            )

        def _sanitize(text: str) -> str:
            """Descarta caracteres fuera de Latin-1 cuando no hay fuente Unicode."""
            if font_name != "Helvetica":
                return text
            return text.encode("latin-1", errors="ignore").decode("latin-1")

        for page_data in pages_data:
            img_path = page_data["image_path"]
            text_lines: List[Dict] = page_data.get("text_lines", [])

            with Image.open(img_path) as img:
                img_w_px, img_h_px = img.size

            # 1 px = 1 pt → las coordenadas OCR mapean directo al PDF
            page_w_pt = float(img_w_px)
            page_h_pt = float(img_h_px)

            pdf.add_page(format=(page_w_pt, page_h_pt))

            # Imagen de fondo
            pdf.image(img_path, x=0, y=0, w=page_w_pt, h=page_h_pt)

            # Texto invisible en blanco encima (buscable en lectores PDF)
            pdf.set_text_color(255, 255, 255)

            for line in text_lines:
                text = line.get("refined_text", line.get("text", "")).strip()
                if not text:
                    continue

                text = _sanitize(text)
                if not text:
                    continue

                x_min = float(line.get("x_min", 0))
                y_min = float(line.get("y_min", 0))
                x_max = float(line.get("x_max", 0))
                y_max = float(line.get("y_max", 0))

                box_w = max(x_max - x_min, 1.0)
                box_h = max(y_max - y_min, 1.0)

                font_size = max(4.0, min(box_h * 0.8, 48.0))
                pdf.set_font(font_name, size=font_size)

                pdf.set_xy(x_min, y_min)
                pdf.cell(w=box_w, h=box_h, text=text, border=0, align="L")

        pdf.output(output_path)
        logger.info(f"✓ PDF searchable generado: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error generando PDF searchable: {e}", exc_info=True)
        return False
