from dotenv import load_dotenv
load_dotenv()
import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self, endpoint_url=None, access_key=None, secret_key=None, bucket_name=None):
        self.endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT_URL")
        self.access_key = access_key or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME")
        
        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError("❌ Faltan variables de entorno para R2")
        
        self.s3 = boto3.client(
            's3',
            region_name='auto',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

    def download_folder(self, prefix: str, local_dir: str):
        """
        Descarga todos los archivos con un prefijo específico a un directorio local.
        """
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        downloaded_files = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'): # Skip directories
                        continue
                        
                    # Calculate local path
                    relative_path = os.path.relpath(key, prefix)
                    local_file_path = os.path.join(local_dir, relative_path)
                    
                    # Ensure directory exists
                    local_file_dir = os.path.dirname(local_file_path)
                    if not os.path.exists(local_file_dir):
                        os.makedirs(local_file_dir)
                    
                    try:
                        print(f"Descargando {key} a {local_file_path}")
                        self.s3.download_file(self.bucket_name, key, local_file_path)
                        downloaded_files.append(local_file_path)
                    except Exception as e:
                        logger.error(f"Error descargando {key}: {e}")
        
        return downloaded_files

    def upload_file(self, file_path: str, s3_key: str):
        """Sube un archivo a S3/R2"""
        try:
            self.s3.upload_file(file_path, self.bucket_name, s3_key)
            print(f"Subido {file_path} a {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error subiendo {file_path}: {e}")
            return False

    def upload_folder(self, local_dir: str, s3_prefix: str):
        """Sube todo el contenido de una carpeta local a un prefijo S3"""
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
                
                self.upload_file(local_path, s3_key)

