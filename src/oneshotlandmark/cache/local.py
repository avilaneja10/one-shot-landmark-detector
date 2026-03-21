"""
Local disk-based caching with gzip compression.

Stores data as compressed .pt.gz files. Supports any combination of:
  - torch.Tensor
  - numpy.ndarray
  - Python dicts, lists, ints, floats, strings
  - Any picklable object

Typical compression ratios for embedding/score data: 30-60% size reduction.
"""

import os
import gzip
import time
import logging

import torch

from oneshotlandmark.cache.base import BaseCache

logger = logging.getLogger(__name__)

class LocalCache(BaseCache):
    """
    Disk-based cache with gzip compression.

    Each key maps to a single file: {cache_dir}/{key}.pt.gz
    Data is serialized with torch.save (pickle-based) and compressed with gzip.

    Args:
        cache_dir: Directory to store cache files. Created if it doesn't exist.
        compress_level: gzip compression level (1-9). Higher = smaller files
            but slower. Default 4 is a good balance for large float arrays.

    Example:
        >>> cache = LocalCache("./cache")
        >>> cache.save("embeddings", {"tensors": [tensor1, tensor2], "maps": [dict1, dict2]})
        >>> data = cache.load("embeddings")
        >>> cache.exists("embeddings")
        True
    """

    def __init__(self, cache_dir: str, compress: bool=False, compress_level: int = 4):
        self.cache_dir = cache_dir
        self.compress = compress
        self.compress_level = compress_level
        os.makedirs(cache_dir, exist_ok=True)
        mode = f"compressed (level={compress_level})" if compress else "uncompressed"
        logger.info(f"LocalCache initialized at: {cache_dir} ({mode})")

    def _key_to_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_").replace(" ", "_")
        ext = ".pt.gz" if self.compress else ".pt"
        return os.path.join(self.cache_dir, f"{safe_key}{ext}")

    def save(self, key: str, data) -> None:
        path = self._key_to_path(key)
        start = time.perf_counter()
    
        if self.compress:
            with gzip.open(path, "wb", compresslevel=self.compress_level) as f:
                torch.save(data, f)
        else:
            torch.save(data, path)
    
        elapsed = time.perf_counter() - start
        disk_mb = os.path.getsize(path) / 1e6
        logger.info(f"Cache saved: '{key}' ({disk_mb:.1f} MB, {elapsed:.2f}s)")

    def load(self, key: str):
        path = self._key_to_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache key '{key}' not found at {path}")

        start = time.perf_counter()

        if self.compress:
            with gzip.open(path, "rb") as f:
                data = torch.load(f, weights_only=False)
        else:
            data = torch.load(path, weights_only=False)

        elapsed = time.perf_counter() - start
        compressed_bytes = os.path.getsize(path)

        logger.info(
            f"Cache loaded: '{key}' "
            f"({compressed_bytes / 1e6:.1f} MB on disk, {elapsed:.2f}s)"
        )
        return data

    def exists(self, key: str) -> bool:
        return os.path.exists(self._key_to_path(key))

    def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Cache deleted: '{key}'")

    def list_keys(self) -> list[str]:
        """List all keys currently in the cache."""
        keys = []
        suffix = ".pt.gz"
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(suffix):
                keys.append(filename[: -len(suffix)])
        return sorted(keys)

    def total_size_mb(self) -> float:
        """Total disk usage of all cache files in MB."""
        total = 0
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                total += os.path.getsize(filepath)
        return total / 1e6

    def __repr__(self):
        n_keys = len(self.list_keys())
        size = self.total_size_mb()
        return (
            f"LocalCache(cache_dir='{self.cache_dir}', "
            f"entries={n_keys}, size={size:.1f} MB)"
        )