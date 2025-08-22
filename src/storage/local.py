# src/storage/local.py
import os
from .base import BaseStorage

class LocalStorage(BaseStorage):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _full_path(self, path: str) -> str:
        return os.path.join(self.base_dir, path)

    def save_file(self, path: str, data: bytes) -> str:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(data)
        return full_path

    def get_file(self, path: str) -> bytes:
        with open(self._full_path(path), "rb") as f:
            return f.read()

    def delete_file(self, path: str) -> bool:
        full_path = self._full_path(path)
        if os.path.exists(full_path):
            os.remove(full_path)
            return True
        return False

    def list_files(self, prefix: str = "") -> list[str]:
        full_prefix = self._full_path(prefix)
        return [os.path.relpath(os.path.join(dp, f), self.base_dir)
                for dp, dn, filenames in os.walk(full_prefix)
                for f in filenames]

    def file_exists(self, path: str) -> bool:
        return os.path.exists(self._full_path(path))

    def get_file_size(self, path: str) -> int:
        return os.path.getsize(self._full_path(path))
    
    def ensure_dir(self, path: str):
        os.makedirs(self._full_path(path), exist_ok=True)
