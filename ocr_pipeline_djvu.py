"""
Pipeline de OCR extendido con funcionalidad DJVU
Extiende OCRPipeline para incluir generación de archivos DJVU con texto incrustado
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import logging

from ocr_pipeline_2 import OCRPipeline
from djvu_utils import DJVUGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRPipelineDJVU(OCRPipeline):
    """
    Pipeline de OCR extendido con funcionalidad para generar archivos DJVU.
    Hereda todas las funcionalidades de OCRPipeline y agrega métodos para DJVU.
    """
    
    def __init__(self, *args, djvulibre_path: Optional[str] = None, **kwargs):
        """
        Inicializa el pipeline con soporte DJVU.
        
        Args:
            *args: Argumentos para OCRPipeline
            djvulibre_path: Ruta al directorio de djvulibre (si no está en PATH)
            **kwargs: Argumentos adicionales para OCRPipeline
        """
        super().__init__(*args, **kwargs)
        self.djvu_generator = DJVUGenerator(djvulibre_path=djvulibre_path)
    
    def save_results_with_djvu(self, text: str, text_lines: list, 
                               metadata: dict, output_dir: str, 
                               base_name: str, original_image_path: str = None,
                               create_djvu: bool = True,
                               djvu_quality: int = 85) -> Tuple[str, str, str, Optional[str]]:
        """
        Guarda los resultados incluyendo archivo DJVU con texto incrustado.
        
        Args:
            text: Texto completo extraído
            text_lines: Lista de líneas con metadatos
            metadata: Metadatos del procesamiento
            output_dir: Directorio de salida
            base_name: Nombre base para los archivos
            original_image_path: Ruta a la imagen original
            create_djvu: Si crear archivo DJVU (default: True)
            djvu_quality: Calidad de compresión DJVU (0-100, default: 85)
            
        Returns:
            Tupla con (ruta CSV, ruta TXT, ruta imagen boxes, ruta DJVU)
        """
        # Primero guardar resultados normales
        csv_path, txt_path, img_path = self.save_results(
            text, text_lines, metadata, output_dir, base_name, original_image_path
        )
        
        djvu_path = None
        
        # Crear DJVU si está habilitado y hay imagen original
        if create_djvu and original_image_path and text_lines:
            djvu_path = os.path.join(output_dir, f"{base_name}.djvu")
            
            success, temp_text_file = self.djvu_generator.create_djvu_with_text(
                image_path=original_image_path,
                text_lines=text_lines,
                output_djvu_path=djvu_path,
                quality=djvu_quality,
                page_number=0
            )
            
            if success:
                logger.info(f"✓ Archivo DJVU creado: {djvu_path}")
            else:
                logger.warning(f"⚠️  No se pudo crear el archivo DJVU: {djvu_path}")
                djvu_path = None
        
        return csv_path, txt_path, img_path, djvu_path
    
    def generate_djvu_text_file_only(self, text_lines: list,
                                     image_path: str,
                                     output_path: str,
                                     page_number: int = 0) -> bool:
        """
        Solo genera el archivo de texto DJVU sin crear el archivo DJVU completo.
        Útil si quieres revisar o modificar el archivo antes de incrustarlo.
        
        Args:
            text_lines: Lista de líneas con bounding boxes y texto
            image_path: Ruta a la imagen original
            output_path: Ruta donde guardar el archivo de texto DJVU
            page_number: Número de página (default: 0)
            
        Returns:
            True si se generó correctamente
        """
        return self.djvu_generator.generate_djvu_text_file(
            text_lines, image_path, output_path, page_number
        )

    def merge_djvu_files(self, output_path: str, input_paths_list: list) -> bool:
        """
        Wrapper para unir múltiples archivos DJVU.
        """
        return self.djvu_generator.merge_page_djvus(output_path, input_paths_list)
