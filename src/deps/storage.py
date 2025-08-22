# src/deps/storage.py
import boto3
from fastapi import Depends
import os
from typing import Optional
from src.storage.base import BaseStorage
from src.storage.local import LocalStorage
from src.storage.s3 import S3Storage  # optional

_storage_instance: Optional[BaseStorage] = None

def get_storage() -> BaseStorage:
    global _storage_instance
    if _storage_instance is None:
        backend = os.getenv("STORAGE_BACKEND", "local").lower()
        if backend == "local":
            base_dir = os.getenv("LOCAL_STORAGE_DIR", "data/sets")
            _storage_instance = LocalStorage(base_dir=base_dir)
        elif backend == "s3":
            bucket = os.getenv("S3_BUCKET", "my-bucket")
            s3_client = boto3.client(
                "s3",
                endpoint_url=os.getenv("AWS_ENDPOINT"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
            _storage_instance = S3Storage(s3_client=s3_client, bucket_name=bucket)
        else:
            raise RuntimeError(f"Unknown storage backend: {backend}")
    return _storage_instance
