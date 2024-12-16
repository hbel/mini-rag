from typing import Dict
from pathlib import Path
import json
import hashlib
from datetime import datetime

class DocumentCache:
    """
    Manages the cache for processed documents.
    Documents are hashed by their name and also checked for their modification date.
    """
    
    def __init__(self, cache_file: str = "./document_cache.json"):
        """Create new cache. Defaults to a local json file"""
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Read cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Fehler beim Laden des Cache: {e}")
                return {}
        return {}

    def save(self) -> None:
        """Save cache"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get_file_hash(self, file_path: Path) -> str:
        """Return hash value for a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_document_processed(self, file_path: Path) -> bool:
        """Check whether a document is already processed"""
        if str(file_path) not in self.cache:
            return False

        cached_info = self.cache[str(file_path)]
        current_hash = self.get_file_hash(file_path)
        current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

        return (cached_info['hash'] == current_hash and 
                cached_info['mtime'] == current_mtime)
    
    def update_document(self, file_path: Path) -> None:
        """Add a new document to the hash"""
        self.cache[str(file_path)] = {
            'hash': self.get_file_hash(file_path),
            'mtime': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'processed_at': datetime.now().isoformat()
        }
        
    def remove_document(self, file_path: str) -> None:
        """Remove document from the cache"""
        if file_path in self.cache:
            del self.cache[file_path]
            
    def clear(self) -> None:
        """Clear the whole cache"""
        self.cache = {}