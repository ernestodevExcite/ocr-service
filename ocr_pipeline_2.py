"""
Pipeline de OCR de alta precisión para documentos antiguos
Integra PaddleOCR con pre-procesamiento avanzado y refinamiento post-OCR
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from paddleocr import PaddleOCR
import cv2
import numpy as np

from document_cleaner import DocumentCleaner
from text_refiner import TextRefiner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRPipeline:
    """
    Pipeline completo de OCR para documentos antiguos.
    Integra pre-procesamiento, extracción OCR y post-procesamiento.
    """
    
    def __init__(self,
                 use_textline_orientation: bool = True,
                 lang: str = 'es',
                 text_det_thresh: float = 0.3,
                 text_det_box_thresh: float = 0.6,
                 text_recognition_batch_size: int = 6,
                 use_gpu: bool = True):
        """
        Inicializa el pipeline de OCR.
        
        Args:
            use_textline_orientation: Usar clasificación de orientación de líneas de texto
            lang: Idioma ('es' para español, 'en' para inglés, etc.)
            text_det_thresh: Umbral para detección de texto (filtra ruido de fondo)
            text_det_box_thresh: Umbral para cajas de texto detectadas
            text_recognition_batch_size: Tamaño de batch para reconocimiento
            use_gpu: Usar GPU si está disponible
        """
        print("Inicializando PaddleOCR...")
        logger.info("Inicializando PaddleOCR...")
        
        # Inicializar PaddleOCR con modelo PP-OCRv4 (servidor de alta precisión)
        # Solo pasar parámetros válidos explícitamente (sin use_pdserving, show_log, etc.)
        # Parámetros específicos de OCR
        ocr_params = {
            'use_textline_orientation': use_textline_orientation,
            'lang': lang,
            'text_det_thresh': text_det_thresh,
            'text_det_box_thresh': text_det_box_thresh,
            'text_recognition_batch_size': text_recognition_batch_size,
        }
        
        # Configurar dispositivo (GPU o CPU) - parámetro común válido
        device_param = 'gpu' if use_gpu else 'cpu'
        
        # Inicializar PaddleOCR pasando solo parámetros válidos
        self.ocr = PaddleOCR(
            use_textline_orientation=ocr_params['use_textline_orientation'],
            lang=ocr_params['lang'],
            # text_det_thresh=ocr_params['text_det_thresh'],
            # text_det_box_thresh=ocr_params['text_det_box_thresh'],
            # text_recognition_batch_size=ocr_params['text_recognition_batch_size'],
            #det_limit_side_len=2300,
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            det_db_unclip_ratio=1.15,
            #det_db_min_size=5,
            device=device_param,
        )
        
        # Inicializar componentes de pre y post-procesamiento
        self.document_cleaner = DocumentCleaner()
        self.text_refiner = TextRefiner()
        
        print("✓ Pipeline de OCR inicializado correctamente\n")
        logger.info("Pipeline de OCR inicializado correctamente")
    def _group_text_by_lines(self, text_lines: List[Dict], line_tolerance: float = 0.2) -> str:
        """
        Agrupa elementos de texto por su posición Y (línea horizontal) y los une correctamente.
        """
        if not text_lines:
            return ""
        
        # Usar solo elementos que sean diccionarios
        dict_lines = [line for line in text_lines if isinstance(line, dict)]
        if not dict_lines:
            # Si por alguna razón todo son strings, devolverlos tal cual en líneas
            return '\n'.join(str(x) for x in text_lines)

        # Calcular altura promedio para determinar tolerancia
        heights = [
            line.get('y_max', 0) - line.get('y_min', 0)
            for line in dict_lines
            if 'y_min' in line and 'y_max' in line
        ]
        avg_height = np.mean(heights) if heights else 50
        tolerance = avg_height * line_tolerance
        
        # Agrupar por línea (usando y_min como referencia)
        lines_dict = {}
        for line_data in dict_lines:
            y_min = line_data.get('y_min', 0)
            y_max = line_data.get('y_max', 0)
            y_center = (y_min + y_max) / 2
            
            # Buscar si ya existe una línea similar (dentro de la tolerancia)
            found_line_key = None
            for line_key in lines_dict.keys():
                if abs(y_center - line_key) <= tolerance:
                    found_line_key = line_key
                    break
            
            if found_line_key is None:
                # Nueva línea
                lines_dict[y_center] = []
            
            # Agregar a la línea correspondiente
            target_key = found_line_key if found_line_key is not None else y_center
            lines_dict[target_key].append(line_data)
        
        # Ordenar líneas de arriba hacia abajo (por Y)
        sorted_line_keys = sorted(lines_dict.keys())
        
        # Construir texto línea por línea
        result_lines = []
        for line_key in sorted_line_keys:
            line_items = lines_dict[line_key]
            # Ordenar elementos de la línea de izquierda a derecha (por x_min)
            line_items.sort(key=lambda x: x.get('x_min', 0))
            
            # Unir palabras de la misma línea con espacios
            line_text = ' '.join(item.get('text', '') for item in line_items)
            result_lines.append(line_text)
        
        # Unir líneas con \n
        return '\n'.join(result_lines)
    def draw_bounding_boxes(self, image_path: str, text_lines: List[Dict], 
                       output_path: str, show_text: bool = True):
        """
        Dibuja los bounding boxes y texto detectado sobre la imagen original.
        
        Args:
            image_path: Ruta a la imagen original
            text_lines: Lista de líneas con bounding boxes
            output_path: Ruta donde guardar la imagen con boxes
            show_text: Si mostrar el texto sobre cada box
        """
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        
        # Cargar imagen original
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"No se pudo cargar la imagen: {image_path}")
            return
        
        # Convertir a RGB para PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Intentar cargar fuente, si no está disponible usar default
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
        
        # Dibujar cada bounding box
        for line in text_lines:
            x_min = int(line['x_min'])
            y_min = int(line['y_min'])
            x_max = int(line['x_max'])
            y_max = int(line['y_max'])
            
            # Dibujar rectángulo
            draw.rectangle([(x_min, y_min), (x_max, y_max)], 
                        outline='green', width=2)
            
            # Mostrar texto si está habilitado
            if show_text and 'text' in line:
                text = line['text'] # Limitar longitud para visualización
                # Dibujar fondo para el texto
                text_bbox = draw.textbbox((x_min, y_min - 15), text, font=font)
                draw.rectangle(text_bbox, fill='green')
                # Dibujar texto
                draw.text((x_min, y_min - 15), text, fill='white', font=font)
        
        # Guardar imagen
        pil_image.save(output_path)
        logger.info(f"Imagen con bounding boxes guardada en: {output_path}")

    def extract_text(self, image_path: str, 
                    preprocess: bool = False,
                    refine_text: bool = False) -> Tuple[str, List[Dict], Dict]:
        """
        Extrae texto de una imagen usando el pipeline completo.
        
        Args:
            image_path: Ruta a la imagen
            preprocess: Si aplicar pre-procesamiento
            refine_text: Si aplicar refinamiento post-OCR
            
        Returns:
            Tupla con (texto completo, resultados detallados, metadatos)
        """
        print(f"Procesando imagen: {image_path}")
        logger.info(f"Procesando imagen: {image_path}")
        
        metadata = {
            'image_path': image_path,
            'preprocessed': preprocess,
            'text_refined': refine_text
        }
        
        # Pre-procesamiento
        if preprocess:
            logger.info("Aplicando pre-procesamiento...")
            processed_image, preprocess_metadata = self.document_cleaner.process_pipeline(
                image_path,
                apply_denoise=True,
                apply_deskew=True,
                apply_contrast=True
            )
            metadata['preprocessing'] = preprocess_metadata
            
            # Guardar imagen procesada temporalmente para OCR
            temp_path = str(Path(image_path).stem + '_processed.png')
            cv2.imwrite(temp_path, processed_image)
            image_for_ocr = temp_path
        else:
            #convertir la imagen a png
            if not image_path.lower().endswith('.png'):
                img = cv2.imread(image_path)
                new_path = str(Path(image_path).stem + '_processed.png')
                cv2.imwrite(new_path, img)
                image_for_ocr = new_path
            #image_for_ocr = image_path
            else:
                image_for_ocr = image_path
        try:
            # Extracción OCR con PaddleOCR
            print("Ejecutando OCR con PaddleOCR...")
            logger.info("Ejecutando OCR con PaddleOCR...")
            ocr_results = self.ocr.predict(image_for_ocr)
            
            # Procesar resultados
            if ocr_results is None or len(ocr_results) == 0:
                logger.warning("No se detectó texto en la imagen")
                return "", [], metadata
            
            # Estructurar resultados
            text_lines = []
            full_text_parts = []
            
            # El resultado puede ser una lista de páginas/documentos
            # Cada elemento puede ser un diccionario con claves como rec_texts, rec_scores, rec_polys
            # O puede ser el formato antiguo: [[[bbox], (texto, confianza)], ...]
            
            for page_result in ocr_results:
                if page_result is None:
                    continue
                
                # Verificar si es un diccionario (nuevo formato)
                if isinstance(page_result, dict):
                    # Formato nuevo: diccionario con rec_texts, rec_scores, rec_polys
                    rec_texts = page_result.get('rec_texts', [])
                    rec_scores = page_result.get('rec_scores', [])
                    rec_polys = page_result.get('rec_polys', [])
                    rec_boxes = page_result.get('rec_boxes', [])
                    
                    # Procesar cada texto reconocido
                    num_texts = len(rec_texts)
                    for i in range(num_texts):
                        text = rec_texts[i] if i < len(rec_texts) else ""
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        bbox = rec_polys[i] if i < len(rec_polys) else (rec_boxes[i] if i < len(rec_boxes) else [])
                        
                        if not text:
                            continue
                        
                        # Calcular coordenadas del bounding box
                        # Manejar arrays de numpy y listas normales
                        if bbox is not None and len(bbox) > 0:
                            try:
                                # Convertir a lista si es array de numpy
                                if hasattr(bbox, 'tolist'):
                                    bbox = bbox.tolist()
                                
                                if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                                    x_coords = [point[0] for point in bbox]
                                    y_coords = [point[1] for point in bbox]
                                    x_min, x_max = min(x_coords), max(x_coords)
                                    y_min, y_max = min(y_coords), max(y_coords)

                                    #shrink bbox (mejorar precisión visual/DJVU)
                                    pad_x = 3
                                    pad_y = 2
                                    x_min += pad_x
                                    y_min += pad_y
                                    x_max -= pad_y
                                    y_max -= pad_x

                                else:
                                    x_min = x_max = y_min = y_max = 0
                            except (TypeError, IndexError):
                                x_min = x_max = y_min = y_max = 0
                        else:
                            x_min = x_max = y_min = y_max = 0
                        
                        text_lines.append({
                            'text': text,
                            'confidence': float(confidence) if confidence is not None else 1.0,
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max,
                            'bbox': bbox
                        })
                        
                        full_text_parts.append(text)
                
                # Formato antiguo: lista de líneas [[[bbox], (texto, confianza)], ...]
                elif isinstance(page_result, list):
                    for line in page_result:
                        if line is None:
                            continue
                        
                        try:
                            # Estructura: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (texto, confianza)]
                            if len(line) >= 2:
                                bbox = line[0]
                                text_info = line[1]
                                
                                # Manejar diferentes formatos de text_info
                                if isinstance(text_info, tuple):
                                    text = text_info[0]
                                    confidence = text_info[1] if len(text_info) > 1 else 1.0
                                elif isinstance(text_info, str):
                                    text = text_info
                                    confidence = 1.0
                                elif isinstance(text_info, list) and len(text_info) >= 1:
                                    text = text_info[0]
                                    confidence = text_info[1] if len(text_info) > 1 else 1.0
                                else:
                                    continue
                                
                                # Calcular coordenadas del bounding box
                                # Manejar arrays de numpy y listas normales
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        # Convertir a lista si es array de numpy
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        
                                        x_coords = [point[0] for point in bbox]
                                        y_coords = [point[1] for point in bbox]
                                        x_min, x_max = min(x_coords), max(x_coords)
                                        y_min, y_max = min(y_coords), max(y_coords)

                                        pad_x = 3
                                        pad_y = 2
                                        x_min += pad_x
                                        y_min += pad_y
                                        x_max -= pad_x
                                        y_max -= pad_y
                                    except (TypeError, IndexError):
                                        x_min = x_max = y_min = y_max = 0
                                else:
                                    x_min = x_max = y_min = y_max = 0
                                
                                text_lines.append({
                                    'text': text,
                                    'confidence': float(confidence) if confidence is not None else 1.0,
                                    'x_min': x_min,
                                    'y_min': y_min,
                                    'x_max': x_max,
                                    'y_max': y_max,
                                    'bbox': bbox
                                })
                                
                                full_text_parts.append(text)
                        except (IndexError, TypeError, KeyError) as e:
                            logger.debug(f"Error procesando línea en formato antiguo: {e}")
                            continue
            
            # Unir texto completo
            #full_text = '\n'.join(full_text_parts)
            full_text = self._group_text_by_lines(text_lines)
            # Refinamiento post-OCR línea por línea
            if refine_text:
                logger.info("Aplicando refinamiento de texto línea por línea...")
                refined_full_text_parts = []
                
                # Inicializar estadísticas globales
                total_stats = {
                    'original_length': 0,
                    'words_corrected': 0,
                    'total_words': 0,
                    'corrections': []
                }
                
                for line_data in text_lines:
                    # Refinar texto de cada línea individualmente
                    original_line_text = line_data['text']
                    refined_line, line_stats = self.text_refiner.refine_text(original_line_text)
                    
                    # Actualizar línea con texto refinado
                    line_data['refined_text'] = refined_line
                    refined_full_text_parts.append(refined_line)
                    
                    # Acumular estadísticas
                    if line_stats:
                        total_stats['original_length'] += line_stats.get('original_length', 0)
                        total_stats['words_corrected'] += line_stats.get('words_corrected', 0)
                        total_stats['total_words'] += line_stats.get('total_words', 0)
                        if 'corrections' in line_stats:
                            total_stats['corrections'].extend(line_stats['corrections'])
                
                # Calcular longitud final y tasa
                total_stats['final_length'] = sum(len(part) for part in refined_full_text_parts) + len(refined_full_text_parts) - 1 # aproximado
                if total_stats['total_words'] > 0:
                    total_stats['correction_rate'] = (total_stats['words_corrected'] / total_stats['total_words'] * 100)
                else:
                    total_stats['correction_rate'] = 0
                
                metadata['refinement_stats'] = total_stats
                #full_text = '\n'.join(refined_full_text_parts)
                for i, line_data in enumerate(text_lines):
                    if i < len(refined_full_text_parts):
                        line_data['text'] = refined_full_text_parts[i]  # Actualizar con texto refinado
                full_text = self._group_text_by_lines(text_lines)
            else:
                # Si no se refina, usar texto original
                for line_data in text_lines:
                    line_data['refined_text'] = line_data['text']
            
            metadata['total_lines'] = len(text_lines)
            metadata['avg_confidence'] = np.mean([line['confidence'] for line in text_lines]) if text_lines else 0.0
            
            print(f"✓ Extracción completada: {len(text_lines)} líneas detectadas")
            logger.info(f"Extracción completada: {len(text_lines)} líneas detectadas")
            
            return full_text, text_lines, metadata
            
        finally:
            # Limpiar imagen temporal si existe
            #if preprocess and os.path.exists(image_for_ocr) and image_for_ocr != image_path:
            try:
                os.remove(image_for_ocr)
            except:
                pass
    
    def save_results(self, text: str, text_lines: List[Dict], 
                metadata: Dict, output_dir: str, 
                base_name: str, original_image_path: str = None) -> Tuple[str, str, str]:
        """
        Guarda los resultados en archivos CSV, TXT e imagen con bounding boxes.
        
        Args:
            text: Texto completo extraído
            text_lines: Lista de líneas con metadatos
            metadata: Metadatos del procesamiento
            output_dir: Directorio de salida
            base_name: Nombre base para los archivos
            original_image_path: Ruta a la imagen original (para dibujar boxes)
            
        Returns:
            Tupla con (ruta CSV, ruta TXT, ruta imagen)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar texto completo en TXT
        txt_path = os.path.join(output_dir, f"{base_name}_text.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Texto guardado en: {txt_path}")
        
        # Crear DataFrame con resultados detallados
        df_data = []
        for i, line in enumerate(text_lines):
            df_data.append({
                'line_number': i + 1,
                'text': line['text'],
                'refined_text': line.get('refined_text', line['text']),
                'confidence': line['confidence'],
                'x_min': line['x_min'],
                'y_min': line['y_min'],
                'x_max': line['x_max'],
                'y_max': line['y_max']
            })
        
        df = pd.DataFrame(df_data)
        
        # Guardar CSV
        csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Resultados CSV guardados en: {csv_path}")
        
        # Guardar metadatos en JSON
        import json
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        metadata_serializable = self._make_json_serializable(metadata)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadatos guardados en: {metadata_path}")
        
        # Generar imagen con bounding boxes si hay imagen original
        image_path = None
        if original_image_path and text_lines:
            image_path = os.path.join(output_dir, f"{base_name}_boxes.jpg")
            self.draw_bounding_boxes(original_image_path, text_lines, image_path)
        
        return csv_path, txt_path, image_path

    def _make_json_serializable(self, obj):
        """Convierte objetos numpy a tipos Python nativos para JSON."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """Función principal para ejecutar el pipeline desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Pipeline de OCR de alta precisión para documentos antiguos'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Ruta a la imagen a procesar'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directorio de salida para los resultados (default: output)'
    )
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Desactivar pre-procesamiento de imagen'
    )
    parser.add_argument(
        '--no-refine',
        action='store_true',
        help='Desactivar refinamiento de texto post-OCR'
    )
    parser.add_argument(
        '--det-thresh',
        type=float,
        default=0.3,
        help='Umbral de detección para filtrar ruido (default: 0.3)'
    )
    parser.add_argument(
        '--det-box-thresh',
        type=float,
        default=0.6,
        help='Umbral de cajas de texto (default: 0.6)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Usar GPU si está disponible'
    )
    
    args = parser.parse_args()
    
    # Validar que la imagen existe
    if not os.path.exists(args.image_path):
        logger.error(f"La imagen no existe: {args.image_path}")
        sys.exit(1)
    
    # Inicializar pipeline
    pipeline = OCRPipeline(
        text_det_thresh=args.det_thresh,
        text_det_box_thresh=args.det_box_thresh,
        use_gpu=args.use_gpu
    )
    
    # Procesar imagen
    text, text_lines, metadata = pipeline.extract_text(
        args.image_path,
        preprocess=not args.no_preprocess,
        refine_text=not args.no_refine
    )
    
    # Guardar resultados
    base_name = Path(args.image_path).stem
    csv_path, txt_path = pipeline.save_results(
        text,
        text_lines,
        metadata,
        args.output_dir,
        base_name,
        original_image_path=args.image_path
    )
    
    # Mostrar resultados tanto en log como en consola
    print("\n" + "=" * 60)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Texto extraído: {len(text)} caracteres")
    print(f"Líneas detectadas: {len(text_lines)}")
    print(f"Confianza promedio: {metadata.get('avg_confidence', 0):.2%}")
    if 'refinement_stats' in metadata:
        stats = metadata['refinement_stats']
        print(f"Palabras corregidas: {stats.get('words_corrected', 0)}/{stats.get('total_words', 0)}")
    print(f"\nArchivos generados:")
    print(f"  - CSV: {csv_path}")
    print(f"  - TXT: {txt_path}")
    print(f"  - JSON: {os.path.join(args.output_dir, f'{base_name}_metadata.json')}")
    print("=" * 60 + "\n")
    
    logger.info("=" * 60)
    logger.info("PROCESAMIENTO COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"Texto extraído: {len(text)} caracteres")
    logger.info(f"Líneas detectadas: {len(text_lines)}")
    logger.info(f"Confianza promedio: {metadata.get('avg_confidence', 0):.2%}")
    #logger.info(f"Archivo CSV: {csv_path}")
    #logger.info(f"Archivo TXT: {txt_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
