import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
import shutil
import logging
import tempfile
import uuid
import base64
import json
import requests
from typing import Optional, Literal
from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from s3_handler import S3Handler
from ocr_pipeline_djvu import OCRPipelineDJVU
from pdf_processor import pdf_to_images, build_searchable_pdf
import zipfile
from fastapi.responses import FileResponse

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCR-Service")

app = FastAPI(title="OCR Service DJVU")

# Modelo de request para el trabajo por lotes (imágenes)
class FolderJobRequest(BaseModel):
    bucket_name: str
    input_folder: str  # Prefijo de entrada en R2/S3
    output_folder: str # Prefijo de salida en R2/S3
    webhook_url: Optional[str] = None
    djvu_quality: int = 85
    merge_djvus: bool = True

# Modelo de request para procesamiento de PDF desde S3
class PdfJobRequest(BaseModel):
    bucket_name: str
    s3_key: str                             # clave del PDF de entrada en S3
    output_s3_prefix: str                   # carpeta destino en S3
    output_format: Literal["djvu", "pdf"] = "djvu"
    webhook_url: Optional[str] = None
    djvu_quality: int = 85

# Endpoint de prueba
@app.get("/health")
def health_check():
    return {"status": "🚀 ok!!"}

# Inicializar pipeline (se puede hacer global o por request, global ahorra carga de modelos)
pipeline = OCRPipelineDJVU(use_gpu=True) # Configurar según disponibilidad

def process_folder_task(job: FolderJobRequest,job_id: str):
    #job_id = str(uuid.uuid4())
    logger.info(f"Iniciando trabajo {job_id} para carpeta {job.input_folder}")
    
    # Directorios temporales
    temp_dir = tempfile.mkdtemp(prefix=f"ocr_job_{job_id}_")
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    s3 = S3Handler(bucket_name=job.bucket_name)
    
    try:
        # 1. Descargar archivos
        logger.info(f"Descargando archivos de {job.input_folder}...")
        downloaded = s3.download_folder(job.input_folder, input_dir)
        
        results = []
        
        generated_djvus = [] 

        # 2. Procesar cada archivo
        for file_path in downloaded:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            
            # Verificar extensión de imagen
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                logger.info(f"Procesando {filename}...")
                
                try:
                    # Ejecutar pipeline
                    text, text_lines, metadata = pipeline.extract_text(
                        file_path, preprocess=True, refine_text=False
                    )
                    
                    # Guardar resultados (DJVU y TXT)
                    csv, txt, img, djvu = pipeline.save_results_with_djvu(
                        text, text_lines, metadata, output_dir, base_name, 
                        original_image_path=file_path, 
                        djvu_quality=job.djvu_quality
                    )
                    
                    if djvu and os.path.exists(djvu):
                        generated_djvus.append(djvu)

                    results.append({
                        "file": filename,
                        "status": "success",
                        "djvu": os.path.basename(djvu) if djvu else None,
                        "txt": os.path.basename(txt) if txt else None
                    })
                    
                except Exception as e:
                    logger.error(f"Error procesando {filename}: {e}")
                    results.append({"file": filename, "status": "error", "error": str(e)})
        
        # 2.5 Unir DJVUs si se requiere
        if job.merge_djvus and generated_djvus:
            logger.info(f"Uniendo {len(generated_djvus)} archivos DJVU...")
            bundle_name = "bundle.djvu"
            bundle_path = os.path.join(output_dir, bundle_name)
            
            # Ordenar por nombre para asegurar un orden coherente
            generated_djvus.sort()
            
            success = pipeline.merge_djvu_files(bundle_path, generated_djvus)
            if success:
                logger.info(f"Bundle creado en {bundle_path}")
                # Si se fusionaron, NO subir los djvus individuales
                # Segun requerimiento "procesamiento de la carpeta va a unir todos... pero se puede mandar un parametro para que no una los archivos"
                # Asumo: Si une, sube el unido.
                
                # Eliminamos los individuales del disco para que el walker de abajo no los suba
                for djvu_path in generated_djvus:
                    try:
                       os.remove(djvu_path)
                    except: pass
            else:
                logger.error("Fallo al crear el bundle DJVU")

        # 3. Subir resultados
        logger.info(f"Subiendo resultados a {job.output_folder}...")
        # Subir solo los archivos generados en output_dir
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('_djvu_text.txt'):
                    logger.info(f"Omitiendo {file}")
                    continue
                if file.endswith('.djvu') or file.endswith('.txt') or file.endswith('text.txt'):
                    local_path = os.path.join(root, file)
                    # Construir key de S3 preservando estructura o plana
                    # "en esa misma r2 solo que en una carpeta que se establezaca"
                    s3_dest_key = os.path.join(job.output_folder, file).replace("\\", "/")
                    s3.upload_file(local_path, s3_dest_key)
        
        # 4. Notificar webhook
        if job.webhook_url:
            payload = {
                "job_id": job_id,
                "status": "completed",
                "input": job.input_folder,
                "output": job.output_folder,
                "merged": job.merge_djvus,
                #"results_count": len(results),
                #"results": results
            }
            try:
                logger.info(f"Enviando notificación de webhook a {job.webhook_url}")
                requests.post(job.webhook_url, json=payload)
            except Exception as e:
                logger.error(f"Error enviando webhook: {e}")

    except Exception as e:
        logger.error(f"Error fatal en el trabajo {job_id}: {e}")
        if job.webhook_url:
             try:
                requests.post(job.webhook_url, json={"job_id": job_id, "status": "failed", "error": str(e)})
             except: pass
    finally:
        # 5. Limpieza
        logger.info(f"Limpiando temporales de {job_id}")
        shutil.rmtree(temp_dir)

@app.post("/process-folder")
async def process_folder_endpoint(job: FolderJobRequest, background_tasks: BackgroundTasks):
    """
    Endpoint para procesar un folder completo de imágenes desde S3/R2.
    
    Args:
        job (FolderJobRequest): Configuración del trabajo.
            - bucket_name: Nombre del bucket en R2/S3.
            - input_folder: Carpeta origen con las imágenes.
            - output_folder: Carpeta destino para los resultados.
            - webhook_url: URL para notificar finalización.
            - djvu_quality: Calidad de compresión (default 85).
            - merge_djvus: Si es True, une todos los DJVUs en uno solo (default True).
            
    Returns:
        JSON con confirmación de inicio del trabajo.
    """
    job_id = str(uuid.uuid4())
    background_tasks.add_task(process_folder_task, job, job_id)
    return {"message": "Job started", "job_id": job_id, "input_folder": job.input_folder, "output_folder": job.output_folder}

@app.post("/process-image")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint para procesar una sola imagen subida directamente.
    
    Genera un archivo ZIP conteniendo:
    - Archivo .djvu con texto OCR incrustado
    - Archivo .txt con el texto extraído
    - Archivo metadata.json con métricas y detalles
    
    Args:
        file (UploadFile): Archivo de imagen (jpg, png, tif, etc.)
        
    Returns:
        FileResponse: Archivo ZIP descargable.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        base_name = os.path.splitext(file.filename)[0]

        text, text_lines, metadata = pipeline.extract_text(
            file_path, preprocess=True, refine_text=False
        )

        csv, txt, img, djvu = pipeline.save_results_with_djvu(
            text, text_lines, metadata, temp_dir, base_name,
            original_image_path=file_path
        )

        if not djvu or not os.path.exists(djvu):
            raise HTTPException(status_code=500, detail="DJVU no generado")

        # return FileResponse(
        #     path=djvu,
        #     media_type="image/vnd.djvu",
        #     filename=os.path.basename(djvu),
        #     headers={
        #         "X-Lines-Count": str(len(text_lines)),
        #         "X-OCR-Confidence": str(metadata.get("confidence", "")),
        #     }
        # )
        #### o enviar zip con todo
        zip_path = os.path.join(temp_dir, f"{base_name}.zip")

        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(djvu, arcname=f"{base_name}.djvu")
            zipf.write(txt, arcname=f"{base_name}.txt")
            zipf.write(img,arcname=f"{base_name}.jpg")
            zipf.write(csv,arcname=f"{base_name}.csv")

            metadata_serializable = pipeline._make_json_serializable(metadata)
            zipf.writestr(
                "metadata.json",
                json.dumps({
                    "message": "OCR completed",
                    "lines_count": len(text_lines),
                    "metadata": metadata_serializable
                }, ensure_ascii=False)
            )

        return FileResponse(
            zip_path,
            filename=f"{base_name}.zip",
            media_type="application/zip"
        )

    finally:
        pass


# ---------------------------------------------------------------------------
# Helpers PDF
# ---------------------------------------------------------------------------

def _process_pdf_pages(
    pdf_path: str,
    output_dir: str,
    output_format: str,
    base_name: str,
    djvu_quality: int = 85,
) -> tuple:
    """
    Procesa un PDF completo página a página y devuelve las rutas del
    archivo resultante (djvu o pdf) y del TXT combinado.
    """
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    # 1. PDF → imágenes
    page_images = pdf_to_images(pdf_path, pages_dir, dpi=300)
    if not page_images:
        raise RuntimeError("No se pudo extraer ninguna página del PDF")

    all_text = []
    djvu_pages = []      # rutas a los djvu individuales
    pages_data = []      # para build_searchable_pdf

    # 2. OCR página a página
    for i, page_img in enumerate(page_images):
        page_base = f"{base_name}_page_{i+1:04d}"
        logger.info(f"Procesando página {i+1}/{len(page_images)}: {page_img}")
        try:
            text, text_lines, metadata = pipeline.extract_text(
                page_img, preprocess=True, refine_text=False
            )
            all_text.append(f"--- Página {i+1} ---\n{text}\n")

            if output_format == "djvu":
                _, txt, _, djvu = pipeline.save_results_with_djvu(
                    text, text_lines, metadata, pages_dir, page_base,
                    original_image_path=page_img,
                    djvu_quality=djvu_quality,
                )
                if djvu and os.path.exists(djvu):
                    djvu_pages.append(djvu)
            else:  # pdf
                pages_data.append({
                    "image_path": page_img,
                    "text_lines": text_lines,
                })
        except Exception as e:
            logger.error(f"Error procesando página {i+1}: {e}")
            all_text.append(f"--- Página {i+1} --- [ERROR: {e}]\n")

    # 3. Combinar texto
    combined_txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(combined_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    # 4. Generar archivo de salida unificado
    if output_format == "djvu":
        output_file = os.path.join(output_dir, f"{base_name}.djvu")
        if not djvu_pages:
            raise RuntimeError("No se generó ningún DJVU por página")
        success = pipeline.merge_djvu_files(output_file, sorted(djvu_pages))
        if not success:
            raise RuntimeError("Fallo al unir los DJVUs de páginas")
    else:
        output_file = os.path.join(output_dir, f"{base_name}.pdf")
        if not pages_data:
            raise RuntimeError("No se generó data de páginas para el PDF")
        success = build_searchable_pdf(pages_data, output_file)
        if not success:
            raise RuntimeError("Fallo al generar el PDF searchable")

    return output_file, combined_txt_path


# ---------------------------------------------------------------------------
# POST /process-pdf/upload   — PDF subido en el request, devuelve ZIP
# ---------------------------------------------------------------------------

@app.post("/process-pdf/upload")
async def process_pdf_upload(
    file: UploadFile = File(...),
    output_format: str = Form(default="djvu"),
    djvu_quality: int = Form(default=85),
):
    """
    Recibe un PDF, lo procesa página a página con OCR y devuelve un ZIP con:
    - El archivo resultante: <nombre>.djvu  o  <nombre>.pdf (searchable)
    - El texto completo extraído: <nombre>.txt

    Args:
        file:          Archivo PDF a procesar.
        output_format: Formato de salida — 'djvu' (default) o 'pdf'.
        djvu_quality:  Calidad DJVU 0-100 (solo aplica si output_format='djvu').
    """
    if output_format not in ("djvu", "pdf"):
        raise HTTPException(status_code=400, detail="output_format debe ser 'djvu' o 'pdf'")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    temp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
    try:
        # Guardar PDF subido
        pdf_path = os.path.join(temp_dir, file.filename)
        with open(pdf_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        base_name = os.path.splitext(file.filename)[0]
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        output_file, txt_path = _process_pdf_pages(
            pdf_path, output_dir, output_format, base_name, djvu_quality
        )

        # Empaquetar en ZIP
        zip_path = os.path.join(temp_dir, f"{base_name}.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_file, arcname=os.path.basename(output_file))
            zf.write(txt_path, arcname=os.path.basename(txt_path))

        return FileResponse(
            zip_path,
            filename=f"{base_name}.zip",
            media_type="application/zip",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass


# ---------------------------------------------------------------------------
# POST /process-pdf/s3  — PDF en S3, resultado sube a S3 y notifica webhook
# ---------------------------------------------------------------------------

def _process_pdf_s3_task(job: PdfJobRequest, job_id: str):
    logger.info(f"[{job_id}] Iniciando procesamiento de PDF desde S3: {job.s3_key}")
    temp_dir = tempfile.mkdtemp(prefix=f"ocr_pdf_{job_id}_")

    try:
        s3 = S3Handler(bucket_name=job.bucket_name)

        # 1. Descargar PDF
        pdf_filename = os.path.basename(job.s3_key)
        pdf_path = os.path.join(temp_dir, pdf_filename)
        logger.info(f"[{job_id}] Descargando {job.s3_key} ...")
        s3.s3.download_file(job.bucket_name, job.s3_key, pdf_path)

        base_name = os.path.splitext(pdf_filename)[0]
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 2. Procesar
        output_file, txt_path = _process_pdf_pages(
            pdf_path, output_dir, job.output_format, base_name, job.djvu_quality
        )

        # 3. Subir archivos individuales a S3 (djvu/pdf + txt)
        output_file_name = os.path.basename(output_file)
        txt_file_name = os.path.basename(txt_path)

        output_s3_key = os.path.join(job.output_s3_prefix, output_file_name).replace("\\", "/")
        txt_s3_key = os.path.join(job.output_s3_prefix, txt_file_name).replace("\\", "/")

        logger.info(f"[{job_id}] Subiendo {output_file_name} a S3: {output_s3_key}")
        s3.upload_file(output_file, output_s3_key)

        logger.info(f"[{job_id}] Subiendo {txt_file_name} a S3: {txt_s3_key}")
        s3.upload_file(txt_path, txt_s3_key)

        # 4. Notificar webhook
        if job.webhook_url:
            payload = {
                "job_id": job_id,
                "status": "completed",
                "input_s3_key": job.s3_key,
                "output_s3_key": output_s3_key,
                "txt_s3_key": txt_s3_key,
                "output_format": job.output_format,
            }
            try:
                requests.post(job.webhook_url, json=payload, timeout=10)
                logger.info(f"[{job_id}] Webhook enviado a {job.webhook_url}")
            except Exception as e:
                logger.error(f"[{job_id}] Error enviando webhook: {e}")

    except Exception as e:
        logger.error(f"[{job_id}] Error fatal: {e}", exc_info=True)
        if job.webhook_url:
            try:
                requests.post(
                    job.webhook_url,
                    json={"job_id": job_id, "status": "failed", "error": str(e)},
                    timeout=10,
                )
            except Exception:
                pass
    finally:
        logger.info(f"[{job_id}] Limpiando temporales")
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/process-pdf/s3")
async def process_pdf_s3(
    job: PdfJobRequest, background_tasks: BackgroundTasks
):
    """
    Descarga un PDF desde S3/R2, lo procesa con OCR y sube los archivos
    resultantes individualmente a S3 (djvu/pdf + txt). Notifica finalización
    via webhook con las keys de destino.

    Args:
        job.bucket_name:      Nombre del bucket.
        job.s3_key:           Clave del PDF de entrada en S3.
        job.output_s3_prefix: Prefijo/carpeta de destino en S3.
        job.output_format:    'djvu' (default) o 'pdf'.
        job.webhook_url:      URL para notificar cuando termine.
        job.djvu_quality:     Calidad DJVU 0-100 (default 85).
    """
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_process_pdf_s3_task, job, job_id)
    return {
        "message": "Job started",
        "job_id": job_id,
        "input_s3_key": job.s3_key,
        "output_s3_prefix": job.output_s3_prefix,
        "output_format": job.output_format,
    }
