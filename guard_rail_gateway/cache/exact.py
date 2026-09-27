"""Exact cryptographic hash matching cache (Tier 1)."""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict

from guard_rail_gateway.models import CacheEntry


def normalize_prompt_for_hashing(prompt: str, model: str = "default") -> str:
    """Normalizes prompt text for deterministic hash generation.
    
    Lowercases, collapses whitespace, strips leading/trailing punctuation,
    and binds with the target model name to avoid cross-model cache pollution.
    """
    cleaned = prompt.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Remove extraneous punctuation at start/end
    cleaned = cleaned.strip(" .,!?:;\"'")
    model_tag = (model or "default").lower().strip()
    return f"{model_tag}::{cleaned}"


def compute_prompt_hash(prompt: str, model: str = "default") -> str:
    """Computes SHA-256 hex digest of normalized prompt."""
    norm = normalize_prompt_for_hashing(prompt, model)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class ExactHashCache:
    """High-speed O(1) in-memory LRU cache based on cryptographic prompt hashes."""

    def __init__(self, max_entries: int = 1000, default_ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.default_ttl = default_ttl_seconds
        # OrderedDict maintains insertion/access order for LRU eviction
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, prompt: str, model: str = "default") -> CacheEntry | None:
        """Looks up exact hash match for a given prompt and model."""
        h = compute_prompt_hash(prompt, model)
        if h not in self._store:
            return None

        entry = self._store[h]
        # Check TTL
        if entry.is_expired():
            del self._store[h]
            return None

        # Move to end for LRU recency
        self._store.move_to_end(h)
        entry.access_count += 1
        entry.last_accessed = time.time()
        return entry

    def put(
        self,
        prompt: str,
        response_content: str,
        model: str = "default",
        ttl_seconds: int | None = None,
        tokens_saved: int = 0,
        cost_saved_usd: float = 0.0,
    ) -> CacheEntry:
        """Saves a prompt-response pair with automatic LRU eviction."""
        h = compute_prompt_hash(prompt, model)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()

        # Evict oldest entry if at capacity and key is new
        if len(self._store) >= self.max_entries and h not in self._store:
            self._store.popitem(last=False)

        entry = CacheEntry(
            key=h,
            prompt=prompt,
            prompt_hash=h,
            response_content=response_content,
            model=model,
            created_at=now,
            expires_at=now + ttl,
            access_count=1,
            last_accessed=now,
            tokens_saved=tokens_saved,
            estimated_cost_saved_usd=cost_saved_usd,
        )
        self._store[h] = entry
        self._store.move_to_end(h)
        return entry

    def delete(self, prompt: str, model: str = "default") -> bool:
        """Removes a specific entry from exact cache."""
        h = compute_prompt_hash(prompt, model)
        if h in self._store:
            del self._store[h]
            return True
        return False

    def clear(self) -> None:
        """Clears all entries in the cache."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
