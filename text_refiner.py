"""
TextRefiner: Refinamiento post-OCR usando técnicas de NLP estadístico
Implementa corrección ortográfica basada en diccionarios y distancia de Levenshtein
"""

import re
import string
from typing import List, Tuple, Optional, Dict
import logging
from Levenshtein import distance as levenshtein_distance
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
import os

# Descargar recursos de NLTK necesarios
try:
    # Intentar encontrar punkt_tab (versión más reciente)
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        # Si no existe, intentar descargar punkt_tab
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        # Fallback a punkt si punkt_tab no está disponible
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextRefiner:
    """
    Clase para refinamiento y corrección de texto extraído por OCR.
    Utiliza técnicas estadísticas de NLP sin LLMs externos.
    """
    
    def __init__(self, 
                 dictionary_path: Optional[str] = None,
                 max_edit_distance: int = 2,
                 prefix_length: int = 7,
                 use_symspell: bool = True):
        """
        Inicializa el TextRefiner.
        
        Args:
            dictionary_path: Ruta al diccionario personalizado (formato: palabra frecuencia)
            max_edit_distance: Distancia máxima de edición para corrección
            prefix_length: Longitud del prefijo para búsqueda en SymSpell
            use_symspell: Si usar SymSpell (True) o pyspellchecker (False)
        """
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self.use_symspell = use_symspell
        
        # Inicializar SymSpell
        if use_symspell:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, 
                                     prefix_length=prefix_length)
            self._load_dictionary(dictionary_path)
        
        # Cargar stopwords en inglés
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            logger.warning("Stopwords no disponibles. Continuando sin filtrado.")
            self.stopwords = set()
        
        # Patrones comunes de errores de OCR
        self.ocr_error_patterns = {
            r'\b0\b': 'O',  # Cero confundido con O
            r'\b5\b': 'S',  # Cinco confundido con S (en contexto)
            r'\bl\b': 'I',  # l minúscula confundida con I mayúscula
            r'\bI\b': 'l',  # I mayúscula confundida con l minúscula (menos común)

        }
    
    def _load_dictionary(self, dictionary_path: Optional[str] = None):
        """
        Carga un diccionario personalizado o usa el diccionario por defecto.
        
        Args:
            dictionary_path: Ruta al archivo de diccionario
        """
        if dictionary_path and os.path.exists(dictionary_path):
            logger.info(f"Cargando diccionario personalizado desde {dictionary_path}")
            if not self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
                logger.warning("No se pudo cargar el diccionario personalizado. Usando diccionario por defecto.")
                self._load_default_dictionary()
        else:
            logger.info("Usando diccionario por defecto de SymSpell")
            self._load_default_dictionary()
    
    def _load_default_dictionary(self):
        """
        Carga el diccionario por defecto de SymSpell.
        Para documentos del siglo XIX, se puede crear un diccionario personalizado.
        """
        # Intentar cargar diccionario por defecto
        # SymSpell viene con un diccionario básico
        # Para mejor precisión, se recomienda crear un diccionario personalizado
        # con vocabulario médico/literario del siglo XIX
        try:
            # Crear un diccionario básico con palabras comunes
            # En producción, esto debería cargarse desde un archivo
            common_words = [
                "the", "of", "and", "to", "a", "in", "that", "it", "with", "for",
                "as", "was", "on", "are", "you", "his", "they", "be", "at", "one",
                "have", "this", "from", "or", "had", "by", "not", "word", "but",
                "what", "some", "we", "can", "out", "other", "were", "all", "there",
                "when", "up", "use", "your", "how", "said", "an", "each", "which",
                "she", "do", "their", "time", "if", "will", "way", "about", "out",
                "many", "then", "them", "these", "so", "some", "her", "would", "make",
                "like", "into", "him", "has", "two", "more", "go", "no", "way", "could",
                "my", "than", "first", "been", "call", "who", "oil", "sit", "now", "find",
                "King", "Know", "FATHER", "Hope", "Soap", "max", "ltd", "new", "York"
            ]
            
            # Agregar palabras al diccionario con frecuencia 1
            for word in common_words:
                self.sym_spell.create_dictionary_entry(word, 1)
            
            logger.info(f"Diccionario básico cargado con {len(common_words)} palabras")
        except Exception as e:
            logger.error(f"Error al cargar diccionario: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto eliminando caracteres no deseados y normalizando espacios.
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        # Eliminar caracteres de control y caracteres especiales del marco decorativo
        # Mantener solo letras, números, espacios y puntuación básica
        text = re.sub(r'[^\w\s\.,;:!?\'"-]', '', text)
        
        # Normalizar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar espacios al inicio y final
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokeniza el texto en palabras.
        
        Args:
            text: Texto a tokenizar
            
        Returns:
            Lista de tokens (palabras)
        """
        try:
            # Intentar usar NLTK tokenizer
            tokens = word_tokenize(text)
            return tokens
        except LookupError as e:
            # Si falta el recurso, intentar descargarlo
            logger.info("Descargando recursos de NLTK necesarios (punkt_tab)...")
            try:
                nltk.download('punkt_tab', quiet=True)
                tokens = word_tokenize(text)
                return tokens
            except Exception:
                logger.warning(f"No se pudo descargar recursos de NLTK: {e}. Usando método simple.")
                return text.split()
        except Exception as e:
            logger.warning(f"Error en tokenización: {e}. Usando método simple.")
            return text.split()
    
    def correct_word_symspell(self, word: str) -> Tuple[str, float]:
        """
        Corrige una palabra usando SymSpell.
        
        Args:
            word: Palabra a corregir
            
        Returns:
            Tupla con (palabra corregida, confianza)
        """
        # Limpiar la palabra de puntuación para la corrección
        clean_word = word.strip(string.punctuation).lower()
        
        if not clean_word or len(clean_word) < 2:
            return word, 1.0
        
        # Buscar sugerencias
        suggestions = self.sym_spell.lookup(
            clean_word,
            Verbosity.CLOSEST,
            max_edit_distance=self.max_edit_distance
        )
        
        if suggestions:
            best_suggestion = suggestions[0]
            corrected = best_suggestion.term
            
            # Calcular confianza basada en la distancia de edición
            edit_distance = best_suggestion.distance
            confidence = 1.0 - (edit_distance / max(len(clean_word), len(corrected)))
            
            # Preservar capitalización original si era mayúscula
            if word[0].isupper():
                corrected = corrected.capitalize()
            if word.isupper():
                corrected = corrected.upper()
            
            return corrected, max(0.0, confidence)
        else:
            # Si no hay sugerencias, retornar palabra original
            return word, 0.5
    
    def correct_word_levenshtein(self, word: str, dictionary: List[str]) -> Tuple[str, float]:
        """
        Corrige una palabra usando distancia de Levenshtein contra un diccionario.
        
        Args:
            word: Palabra a corregir
            dictionary: Lista de palabras válidas del diccionario
            
        Returns:
            Tupla con (palabra corregida, confianza)
        """
        if not word or len(word) < 2:
            return word, 1.0
        
        clean_word = word.strip(string.punctuation).lower()
        min_distance = float('inf')
        best_match = word
        
        for dict_word in dictionary:
            dist = levenshtein_distance(clean_word, dict_word.lower())
            if dist < min_distance and dist <= self.max_edit_distance:
                min_distance = dist
                best_match = dict_word
        
        if min_distance == 0:
            confidence = 1.0
        elif min_distance <= self.max_edit_distance:
            confidence = 1.0 - (min_distance / max(len(clean_word), len(best_match)))
        else:
            confidence = 0.0
            best_match = word
        
        # Preservar capitalización
        if word[0].isupper():
            best_match = best_match.capitalize()
        if word.isupper():
            best_match = best_match.upper()
        
        return best_match, confidence
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Corrige errores comunes de OCR usando patrones regex.
        
        Args:
            text: Texto a corregir
            
        Returns:
            Texto corregido
        """
        # Aplicar correcciones de patrones comunes
        # Nota: Estos son ejemplos básicos, se pueden expandir según necesidades
        corrected = text
        
        # Correcciones contextuales más seguras
        # (comentadas por ser muy agresivas, descomentar si es necesario)
        # for pattern, replacement in self.ocr_error_patterns.items():
        #     corrected = re.sub(pattern, replacement, corrected)
        
        return corrected
    
    def refine_text(self, text: str, 
                   apply_spell_check: bool = True,
                   apply_ocr_fixes: bool = True,
                   min_word_length: int = 2) -> Tuple[str, Dict]:
        """
        Refina el texto completo aplicando todas las correcciones.
        
        Args:
            text: Texto a refinar
            apply_spell_check: Si aplicar corrección ortográfica
            apply_ocr_fixes: Si aplicar correcciones de errores comunes de OCR
            min_word_length: Longitud mínima de palabra para corregir
            
        Returns:
            Tupla con (texto refinado, estadísticas)
        """
        logger.info("Iniciando refinamiento de texto...")
        
        stats = {
            'original_length': len(text),
            'words_corrected': 0,
            'total_words': 0,
            'corrections': []
        }
        
        # Limpiar texto
        cleaned_text = self.clean_text(text)
        
        # Aplicar correcciones de errores comunes de OCR
        if apply_ocr_fixes:
            cleaned_text = self.fix_common_ocr_errors(cleaned_text)
        
        # Tokenizar en oraciones para preservar estructura
        try:
            sentences = sent_tokenize(cleaned_text)
        except LookupError:
            # Si falta el recurso, intentar descargarlo
            try:
                nltk.download('punkt_tab', quiet=True)
                sentences = sent_tokenize(cleaned_text)
            except Exception:
                # Fallback: dividir por puntos y signos de interrogación/exclamación
                sentences = re.split(r'[.!?]+\s+', cleaned_text)
                sentences = [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Fallback: usar el texto completo como una sola oración
            sentences = [cleaned_text]
        
        refined_sentences = []
        
        for sentence in sentences:
            # Tokenizar palabras en la oración
            tokens = self.tokenize_text(sentence)
            corrected_tokens = []
            
            for token in tokens:
                stats['total_words'] += 1
                
                # Saltar puntuación y palabras muy cortas
                if len(token.strip(string.punctuation)) < min_word_length:
                    corrected_tokens.append(token)
                    continue
                
                # Aplicar corrección ortográfica
                if apply_spell_check and self.use_symspell:
                    corrected, confidence = self.correct_word_symspell(token)
                    
                    if corrected.lower() != token.strip(string.punctuation).lower():
                        stats['words_corrected'] += 1
                        stats['corrections'].append({
                            'original': token,
                            'corrected': corrected,
                            'confidence': confidence
                        })
                        corrected_tokens.append(corrected)
                    else:
                        corrected_tokens.append(token)
                else:
                    corrected_tokens.append(token)
            
            # Reconstruir oración
            refined_sentence = ' '.join(corrected_tokens)
            refined_sentences.append(refined_sentence)
        
        # Unir todas las oraciones
        refined_text = ' '.join(refined_sentences)
        
        # Normalizar espacios finales
        refined_text = re.sub(r'\s+', ' ', refined_text).strip()
        
        stats['final_length'] = len(refined_text)
        stats['correction_rate'] = (stats['words_corrected'] / stats['total_words'] * 100) if stats['total_words'] > 0 else 0
        
        logger.info(f"Refinamiento completado. Palabras corregidas: {stats['words_corrected']}/{stats['total_words']}")
        
        return refined_text, stats
