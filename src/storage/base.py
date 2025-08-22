# src/storage/base.py
from typing import List, Optional

class BaseStorage:
    def save_file(self, path: str, data: bytes) -> str:
        """Save bytes to storage. Returns full storage path."""
        raise NotImplementedError

    def get_file(self, path: str) -> bytes:
        raise NotImplementedError

    def delete_file(self, path: str) -> bool:
        raise NotImplementedError

    def list_files(self, prefix: str = "") -> List[str]:
        raise NotImplementedError

    def file_exists(self, path: str) -> bool:
        raise NotImplementedError

    def get_file_size(self, path: str) -> int:
        raise NotImplementedError
    
    def ensure_dir(self, path: str):
        raise NotImplementedError
