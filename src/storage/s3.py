# src/storage/s3.py
from .base import BaseStorage
from typing import List
import logging
from botocore.exceptions import ClientError

class S3Storage(BaseStorage):
    def __init__(self, s3_client, bucket_name: str):
        self.s3 = s3_client
        self.bucket_name = bucket_name

    def save_file(self, path: str, data: bytes) -> str:
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=path, Body=data)
            return path
        except ClientError as e:
            logging.error(e)
            raise

    def get_file(self, path: str) -> bytes:
        try:
            return self.s3.get_object(Bucket=self.bucket_name, Key=path)["Body"].read()
        except ClientError as e:
            logging.error(e)
            raise

    def delete_file(self, path: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=path)
            return True
        except ClientError as e:
            logging.error(e)
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [o["Key"] for o in response.get("Contents", [])]
        except ClientError as e:
            logging.error(e)
            return []

    def file_exists(self, path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=path)
            return True
        except ClientError:
            return False

    def get_file_size(self, path: str) -> int:
        try:
            return self.s3.head_object(Bucket=self.bucket_name, Key=path)["ContentLength"]
        except ClientError:
            return 0

    def ensure_dir(self, path: str):
        """
        Ensure a "directory" exists in S3.
        S3 doesn't have real directories, so this is optional.
        """
        if not path.endswith("/"):
            path += "/"
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=path, Body=b"")
        except ClientError as e:
            logging.error(f"Failed to ensure S3 directory {path}: {e}")
