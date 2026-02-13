"""
Utilidades para crear archivos DJVU con texto OCR incrustado
Requiere djvulibre instalado en el sistema
"""

import os
import subprocess
import logging
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DJVUGenerator:
    """
    Clase para generar archivos DJVU con texto OCR incrustado.
    """
    
    def __init__(self, djvulibre_path: Optional[str] = None):
        """
        Inicializa el generador de DJVU.
        
        Args:
            djvulibre_path: Ruta al directorio de djvulibre (si no está en PATH)
        """
        self.djvulibre_path = djvulibre_path
        self.c44_cmd = self._find_command('c44')
        self.djvused_cmd = self._find_command('djvused')
        self.djvm_cmd = self._find_command('djvm')
        
        if not self.c44_cmd or not self.djvused_cmd or not self.djvm_cmd:
            logger.warning("Algunas herramientas de djvulibre no fueron encontradas en PATH (c44, djvused, djvm).")
    
    def _find_command(self, cmd: str) -> Optional[str]:
        """
        Busca un comando en el sistema.
        
        Args:
            cmd: Nombre del comando
            
        Returns:
            Ruta completa al comando o None si no se encuentra
        """
        if self.djvulibre_path:
            # Buscar en el directorio especificado
            cmd_path = os.path.join(self.djvulibre_path, cmd)
            if os.path.exists(cmd_path):
                return cmd_path
        
        # Buscar en PATH
        try:
            result = subprocess.run(
                ['where' if os.name == 'nt' else 'which', cmd],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        return None
    
    def generate_djvu_text_file(self, text_lines: List[Dict], 
                                image_path: str,
                                output_path: str,
                                page_number: int = 0) -> bool:
        try:
            # 1. Obtener dimensiones reales de la imagen para el encabezado
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return False
            
            img_height, img_width = image.shape[:2]
            
            def group_lines_by_row(lines, y_threshold=12):
                """
                Une cajas que están en la misma línea visual.
                y_threshold ~ 8-15 funciona bien para 1600x2300
                """

                for l in lines:
                    l['y_center'] = (l['y_min'] + l['y_max']) // 2

                lines = sorted(lines, key=lambda l: l['y_center'])

                groups = []
                current = []

                for l in lines:
                    if not current:
                        current.append(l)
                        continue

                    if abs(l['y_center'] - current[-1]['y_center']) <= y_threshold:
                        current.append(l)
                    else:
                        groups.append(current)
                        current = [l]

                if current:
                    groups.append(current)

                return groups


            with open(output_path, 'w', encoding='utf-8') as f:
                # 2. El encabezado DEBE tener el formato: (page x_offset y_offset ancho alto
                # Usualmente x_offset e y_offset son 0 0
                f.write(f'(page 0 0 {img_width} {img_height}\n')
                
                #for line in ordered_lines:
                #for line in sort_layout_order(text_lines):
                # for line in text_lines:
                #     text = line.get('refined_text', line.get('text', '')).strip()
                #     if not text:
                #         continue
                    
                #     # Coordenadas (top-left de la imagen original)
                #     x_min = int(line.get('x_min', 0))
                #     y_min = int(line.get('y_min', 0))
                #     x_max = int(line.get('x_max', 0))
                #     y_max = int(line.get('y_max', 0))
                    
                #     # 3. Conversión de coordenadas: DjVu mide de abajo hacia arriba
                #     djvu_y1 = img_height - y_max
                #     djvu_y2 = img_height - y_min
                    
                #     text_escaped = text.replace('\\', '\\\\').replace('"', '\\"')
                    
                #     # 4. Escribir la línea tabulada para orden
                #     f.write(f' (line {x_min} {djvu_y1} {x_max} {djvu_y2} "{text_escaped}")\n')
                
                groups = group_lines_by_row(text_lines)

                for group in groups:

                    # ordenar izquierda → derecha
                    group = sorted(group, key=lambda l: l['x_min'])

                    full_text = " ".join(
                        g.get('refined_text', g.get('text', '')).strip()
                        for g in group
                    )

                    x_min = min(g['x_min'] for g in group)
                    y_min = min(g['y_min'] for g in group)
                    x_max = max(g['x_max'] for g in group)
                    y_max = max(g['y_max'] for g in group)

                    djvu_y1 = img_height - y_max
                    djvu_y2 = img_height - y_min

                    text_escaped = full_text.replace('\\', '\\\\').replace('"', '\\"')

                    f.write(f' (line {x_min} {djvu_y1} {x_max} {djvu_y2} "{text_escaped}")\n')
                # 5. Cerrar el paréntesis de (page ...)
                f.write(')\n')
            
            logger.info(f"Archivo de texto DJVU generado correctamente con dimensiones {img_width}x{img_height}")
            return True
            
        except Exception as e:
            logger.error(f"Error generando archivo de texto DJVU: {e}")
            return False

    def convert_image_to_djvu(self, image_path: str, 
                              djvu_path: str,
                              quality: int = 85) -> bool:
        if not self.c44_cmd:
            logger.error("Comando c44 no encontrado.")
            return False
        
        temp_jpg = None
        try:
            # --- PARA LOS ARCHIVOS TIFF ---
            # Si la extensión es .tif o .tiff, convertimos a .jpg temporalmente
            ext = Path(image_path).suffix.lower()
            if ext in ['.tif', '.tiff']:
                logger.info(f"Detectado archivo TIFF. Convirtiendo temporalmente a JPEG...")
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"No se pudo leer el archivo TIFF: {image_path}")
                
                temp_jpg = image_path.replace(ext, "_temp_conv.jpg")
                cv2.imwrite(temp_jpg, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                processing_path = temp_jpg
            else:
                processing_path = image_path

            # Construir comando c44 usando la ruta del archivo (original o convertido)
            cmd = [
                self.c44_cmd,
                '-slice',
                str(quality),
                processing_path,
                djvu_path
            ]
            
            logger.info(f"Convirtiendo a DJVU: {processing_path} -> {djvu_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Limpiar archivo temporal si se creó
            if temp_jpg and os.path.exists(temp_jpg):
                os.remove(temp_jpg)

            return os.path.exists(djvu_path)
                
        except Exception as e:
            if temp_jpg and os.path.exists(temp_jpg):
                os.remove(temp_jpg)
            logger.error(f"Error convirtiendo imagen a DJVU: {e}")
            return False
    def embed_text_in_djvu(self, djvu_path: str, 
                           text_file_path: str) -> bool:
        if not self.djvused_cmd:
            logger.error("Comando djvused no encontrado.")
            return False
        
        try:
            # NORMALIZACIÓN DE RUTAS PARA WINDOWS:
            # Convertimos las '\' en '/' para que djvused no las interprete como comandos
            djvu_path_fixed = Path(djvu_path).as_posix()
            text_file_fixed = Path(text_file_path).as_posix()
            
            # El comando debe usar comillas simples por fuera y 
            # comillas dobles internas para las rutas con espacios.
            # 'select 1; set-txt "C:/ruta/archivo.txt"; save'
            script = f'select 1; set-txt "{text_file_fixed}"; save'
            
            cmd = [
                self.djvused_cmd,
                djvu_path_fixed,
                '-e',
                script
            ]
            
            logger.info(f"Incrustando texto en DJVU: {djvu_path_fixed}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"✓ Texto incrustado exitosamente.")
            return True
            
        except subprocess.CalledProcessError as e:
            # Aquí capturamos el error exacto de djvused si falla
            logger.error(f"Error ejecutando djvused:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            return False

    def create_djvu_with_text(self, image_path: str,
                             text_lines: List[Dict],
                             output_djvu_path: str,
                             quality: int = 85,
                             page_number: int = 0) -> Tuple[bool, Optional[str]]:
        """
        Crea un archivo DJVU completo con texto OCR incrustado.
        
        Este método realiza todos los pasos:
        1. Genera el archivo de texto con coordenadas
        2. Convierte la imagen a DJVU
        3. Incrusta el texto en el DJVU
        
        Args:
            image_path: Ruta a la imagen original
            text_lines: Lista de líneas con bounding boxes y texto
            output_djvu_path: Ruta donde guardar el DJVU final
            quality: Calidad de compresión (0-100, default: 85)
            page_number: Número de página (default: 0)
            
        Returns:
            Tupla (éxito, ruta_al_archivo_txt_temporal)
        """
        try:
            # Crear archivo temporal para el texto DJVU
            base_name = Path(output_djvu_path).stem
            temp_dir = Path(output_djvu_path).parent
            temp_text_file = os.path.join(temp_dir, f"{base_name}_djvu_text.txt")
            
            # Paso 1: Generar archivo de texto DJVU
            if not self.generate_djvu_text_file(text_lines, image_path, temp_text_file, page_number):
                return False, None
            
            # Paso 2: Convertir imagen a DJVU
            if not self.convert_image_to_djvu(image_path, output_djvu_path, quality):
                # Limpiar archivo temporal si falla
                if os.path.exists(temp_text_file):
                    os.remove(temp_text_file)
                return False, None
            
            # Paso 3: Incrustar texto en DJVU
            if not self.embed_text_in_djvu(output_djvu_path, temp_text_file):
                return False, temp_text_file
            
            # Opcional: eliminar archivo temporal de texto
            # if os.path.exists(temp_text_file):
            #     os.remove(temp_text_file)
            
            logger.info(f"✓ DJVU con texto creado exitosamente: {output_djvu_path}")
            return True, temp_text_file
            
        except Exception as e:
            logger.error(f"Error creando DJVU con texto: {e}")
            return False, None

    def merge_page_djvus(self, output_path: str, input_paths_list: List[str]) -> bool:
        """
        Une múltiples archivos DJVU de una sola página en un archivo multi-página.
        
        Args:
            output_path: Ruta del archivo de salida
            input_paths_list: Lista de rutas a los archivos DJVU individuales
            
        Returns:
            True si la unión fue exitosa
        """
        if not self.djvm_cmd:
            logger.error("Comando djvm no encontrado.")
            return False
            
        if not input_paths_list:
            logger.warning("No hay archivos para unir.")
            return False
            
        try:
            # djvm -c output.djvu page1.djvu page2.djvu ...
            cmd = [self.djvm_cmd, '-c', output_path] + input_paths_list
            
            logger.info(f"Uniendo {len(input_paths_list)} archivos en {output_path}...")
            
            # djvm puede tener problemas con listas de argumentos muy largas si hay miles de páginas,
            # pero para usos normales debería estar bien.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"✓ Archivo multi-página creado: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error ejecutando djvm:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error uniendo archivos DJVU: {e}")
            return False
