"""
Abstract base class for the caching layer.

Any caching backend (local disk, cloud storage, etc.) should extend
BaseCache and implement save, load, exists, and delete.

The cache is a simple key-value store where:
  - Keys are strings chosen by the caller (e.g., "calib_embeddings_patch")
  - Values are arbitrary dicts containing tensors, arrays, and metadata
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """
    Abstract caching interface.

    Subclasses implement the storage mechanism. The cache is agnostic to
    what is being stored — it just persists and retrieves named data blobs.

    The caller is responsible for choosing meaningful, unique keys.
    """

    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """
        Persist data under the given key.

        Args:
            key: Unique identifier for this cache entry.
            data: Data to store (typically a dict of tensors/arrays/metadata).
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """
        Retrieve data for the given key.

        Args:
            key: Identifier of the cache entry.

        Returns:
            The stored data.

        Raises:
            FileNotFoundError: If the key does not exist in the cache.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Identifier to check.

        Returns:
            True if the key has been previously saved.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Remove a cache entry.

        Args:
            key: Identifier of the entry to remove.
        """
        pass

    def list_keys(self) -> list[str]:
        """
        List all keys in the cache. Optional — not all backends support this.

        Returns:
            Sorted list of key strings.

        Raises:
            NotImplementedError: If the backend doesn't support listing.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support listing keys")