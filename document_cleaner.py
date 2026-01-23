"""
DocumentCleaner: Pre-procesamiento avanzado de imágenes para OCR
Implementa técnicas de Computer Vision para mejorar la calidad de documentos antiguos
"""

import cv2
import numpy as np
from skimage import filters, transform
from scipy import ndimage
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentCleaner:
    """
    Clase para pre-procesamiento avanzado de documentos antiguos.
    Implementa técnicas de limpieza, denoising, thresholding adaptativo y deskewing.
    """
    
    def __init__(self, 
                 adaptive_thresh_block_size: int = 11,
                 adaptive_thresh_c: int = 2,
                 denoise_h: float = 10.0,
                 denoise_template_window_size: int = 7,
                 denoise_search_window_size: int = 21):
        """
        Inicializa el DocumentCleaner con parámetros configurables.
        
        Args:
            adaptive_thresh_block_size: Tamaño del bloque para thresholding adaptativo (debe ser impar)
            adaptive_thresh_c: Constante restada de la media para thresholding adaptativo
            denoise_h: Parámetro de filtrado para denoising (mayor = más suave)
            denoise_template_window_size: Tamaño de la ventana de plantilla para denoising
            denoise_search_window_size: Tamaño de la ventana de búsqueda para denoising
        """
        self.adaptive_thresh_block_size = adaptive_thresh_block_size
        self.adaptive_thresh_c = adaptive_thresh_c
        self.denoise_h = denoise_h
        self.denoise_template_window_size = denoise_template_window_size
        self.denoise_search_window_size = denoise_search_window_size
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Carga una imagen desde un archivo.
        
        Args:
            image_path: Ruta al archivo de imagen
            
        Returns:
            Imagen en formato numpy array (BGR)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")
        logger.info(f"Imagen cargada: {image.shape}")
        return image
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convierte una imagen a escala de grises.
        
        Args:
            image: Imagen en formato BGR
            
        Returns:
            Imagen en escala de grises
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray
    
    def apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica denoising usando el algoritmo Non-Local Means de OpenCV.
        Efectivo para eliminar manchas y ruido causado por la edad del papel.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Imagen con ruido reducido
        """
        logger.info("Aplicando denoising...")
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=self.denoise_h,
            templateWindowSize=self.denoise_template_window_size,
            searchWindowSize=self.denoise_search_window_size
        )
        return denoised
    
    def apply_adaptive_thresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica thresholding adaptativo usando método Gaussiano.
        Separa eficientemente el texto del fondo amarillento del papel.
        
        Args:
            image: Imagen en escala de grises (preferiblemente denoised)
            
        Returns:
            Imagen binaria (blanco y negro)
        """
        logger.info("Aplicando thresholding adaptativo...")
        # Asegurar que block_size sea impar
        block_size = self.adaptive_thresh_block_size
        if block_size % 2 == 0:
            block_size += 1
        
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.adaptive_thresh_c
        )
        return binary
    
    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detecta el ángulo de inclinación del documento usando la Transformada de Hough.
        
        Args:
            image: Imagen binaria
            
        Returns:
            Ángulo de inclinación en grados
        """
        logger.info("Detectando ángulo de inclinación...")
        
        # Aplicar detección de bordes usando Canny
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Aplicar Transformada de Hough para detectar líneas
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            logger.warning("No se detectaron líneas. Retornando ángulo 0.")
            return 0.0
        
        # Calcular ángulos de todas las líneas detectadas
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            # Filtrar ángulos razonables (entre -45 y 45 grados)
            if -45 <= angle <= 45:
                angles.append(angle)
        
        if len(angles) == 0:
            logger.warning("No se encontraron ángulos válidos. Retornando ángulo 0.")
            return 0.0
        
        # Retornar la mediana de los ángulos (más robusto que la media)
        median_angle = np.median(angles)
        logger.info(f"Ángulo de inclinación detectado: {median_angle:.2f} grados")
        return median_angle
    
    def apply_deskewing(self, image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Corrige la inclinación del documento (deskewing).
        
        Args:
            image: Imagen a corregir
            angle: Ángulo de inclinación (si es None, se detecta automáticamente)
            
        Returns:
            Tupla con (imagen corregida, ángulo aplicado)
        """
        if angle is None:
            angle = self.detect_skew_angle(image)
        
        if abs(angle) < 0.1:  # Si el ángulo es muy pequeño, no rotar
            logger.info("Ángulo de inclinación despreciable. No se aplica rotación.")
            return image, 0.0
        
        logger.info(f"Aplicando deskewing con ángulo: {angle:.2f} grados")
        
        # Obtener dimensiones de la imagen
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Crear matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calcular nuevas dimensiones para evitar recorte
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Ajustar la matriz de rotación para el centro
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Aplicar rotación
        deskewed = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255  # Fondo blanco
        )
        
        return deskewed, angle
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Mejora el contraste de la imagen.
        
        Args:
            image: Imagen en escala de grises
            alpha: Factor de contraste (1.0 = sin cambio)
            beta: Factor de brillo
            
        Returns:
            Imagen con contraste mejorado
        """
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def process_pipeline(self, image_path: str, 
                        apply_denoise: bool = True,
                        apply_deskew: bool = True,
                        apply_contrast: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Ejecuta el pipeline completo de pre-procesamiento.
        
        Args:
            image_path: Ruta a la imagen a procesar
            apply_denoise: Si aplicar denoising
            apply_deskew: Si aplicar deskewing
            apply_contrast: Si mejorar el contraste
            
        Returns:
            Tupla con (imagen procesada, metadatos del procesamiento)
        """
        logger.info(f"Iniciando pipeline de pre-procesamiento para: {image_path}")
        
        # Cargar imagen
        image = self.load_image(image_path)
        
        # Convertir a escala de grises
        gray = self.convert_to_grayscale(image)
        
        metadata = {
            'original_shape': image.shape,
            'skew_angle': 0.0
        }
        
        # Aplicar mejoras de contraste (opcional, antes del denoising)
        if apply_contrast:
            gray = self.enhance_contrast(gray)
        
        # Aplicar denoising
        if apply_denoise:
            gray = self.apply_denoising(gray)
        
        # Aplicar deskewing
        if apply_deskew:
            gray, angle = self.apply_deskewing(gray)
            metadata['skew_angle'] = angle
        
        # Aplicar thresholding adaptativo (último paso)
        binary = self.apply_adaptive_thresholding(gray)
        
        metadata['final_shape'] = binary.shape
        metadata['is_binary'] = True
        
        logger.info("Pipeline de pre-procesamiento completado")
        
        return binary, metadata
