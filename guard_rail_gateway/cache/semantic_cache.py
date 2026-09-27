"""Unified two-tier semantic cache orchestrator for OmniGuard."""

from __future__ import annotations

import time
from typing import Any

from guard_rail_gateway.cache.exact import ExactHashCache
from guard_rail_gateway.cache.vector import VectorSemanticCache
from guard_rail_gateway.models import CacheEntry, CacheMatchResult


class SemanticCache:
    """Two-tier semantic caching engine combining exact cryptographic hashing with vector similarity.
    
    Tier 1: O(1) SHA-256 exact hash match (< 0.5ms).
    Tier 2: Pure-Python TF-IDF vector cosine similarity match (< 2ms).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        max_entries: int = 1000,
        default_ttl_seconds: int = 3600,
        cost_per_1k_tokens_usd: float = 0.005,
    ):
        self.exact_cache = ExactHashCache(
            max_entries=max_entries,
            default_ttl_seconds=default_ttl_seconds,
        )
        self.vector_cache = VectorSemanticCache(
            similarity_threshold=similarity_threshold,
        )
        self.cost_per_1k = cost_per_1k_tokens_usd

        # Real-time Telemetry Counters
        self.total_requests: int = 0
        self.exact_hits: int = 0
        self.vector_hits: int = 0
        self.total_misses: int = 0
        self.total_latency_saved_ms: float = 0.0
        self.total_cost_saved_usd: float = 0.0

    def lookup(self, prompt: str, model: str = "default") -> CacheMatchResult:
        """Performs two-tier hierarchical cache lookup."""
        start_time = time.perf_counter()
        self.total_requests += 1

        # 1. Tier 1: Check Exact Hash
        exact_entry = self.exact_cache.get(prompt, model)
        if exact_entry is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.exact_hits += 1
            # Estimate latency saved vs upstream LLM call (~1,200ms avg)
            saved_ms = max(0.0, 1200.0 - elapsed_ms)
            self.total_latency_saved_ms += saved_ms
            self.total_cost_saved_usd += exact_entry.estimated_cost_saved_usd

            return CacheMatchResult(
                is_hit=True,
                match_type="exact_hash",
                similarity_score=1.0,
                entry=exact_entry,
                lookup_latency_ms=round(elapsed_ms, 3),
            )

        # 2. Tier 2: Check Vector Semantic Similarity
        vector_entry, similarity = self.vector_cache.search_similar(prompt, model)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if vector_entry is not None:
            self.vector_hits += 1
            vector_entry.access_count += 1
            vector_entry.last_accessed = time.time()
            saved_ms = max(0.0, 1200.0 - elapsed_ms)
            self.total_latency_saved_ms += saved_ms
            self.total_cost_saved_usd += vector_entry.estimated_cost_saved_usd

            return CacheMatchResult(
                is_hit=True,
                match_type="vector_semantic",
                similarity_score=round(similarity, 4),
                entry=vector_entry,
                lookup_latency_ms=round(elapsed_ms, 3),
            )

        # 3. Cache Miss
        self.total_misses += 1
        return CacheMatchResult(
            is_hit=False,
            match_type="none",
            similarity_score=round(similarity, 4),
            entry=None,
            lookup_latency_ms=round(elapsed_ms, 3),
        )

    def store(
        self,
        prompt: str,
        response_content: str,
        model: str = "default",
        tokens_saved: int = 0,
        cost_saved_usd: float | None = None,
        ttl_seconds: int | None = None,
    ) -> CacheEntry:
        """Stores a prompt-response completion into both exact and vector cache tiers."""
        if cost_saved_usd is None:
            cost_saved_usd = (tokens_saved / 1000.0) * self.cost_per_1k

        # Put into exact cache
        entry = self.exact_cache.put(
            prompt=prompt,
            response_content=response_content,
            model=model,
            ttl_seconds=ttl_seconds,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved_usd,
        )

        # Index into vector cache
        self.vector_cache.add_entry(entry)
        return entry

    def delete(self, prompt: str, model: str = "default") -> bool:
        """Deletes an entry from both cache tiers."""
        from guard_rail_gateway.cache.exact import compute_prompt_hash
        h = compute_prompt_hash(prompt, model)
        r1 = self.exact_cache.delete(prompt, model)
        r2 = self.vector_cache.remove_entry(h)
        return r1 or r2

    def clear(self) -> None:
        """Clears all entries across both cache tiers."""
        self.exact_cache.clear()
        self.vector_cache.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Returns comprehensive operational metrics for monitoring & dashboards."""
        total = max(1, self.total_requests)
        total_hits = self.exact_hits + self.vector_hits
        hit_rate = (total_hits / total) if self.total_requests > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "total_hits": total_hits,
            "exact_hits": self.exact_hits,
            "vector_hits": self.vector_hits,
            "total_misses": self.total_misses,
            "hit_rate_percent": round(hit_rate * 100.0, 1),
            "exact_hit_ratio_percent": round((self.exact_hits / total) * 100.0, 1),
            "vector_hit_ratio_percent": round((self.vector_hits / total) * 100.0, 1),
            "total_latency_saved_seconds": round(self.total_latency_saved_ms / 1000.0, 2),
            "total_cost_saved_usd": round(self.total_cost_saved_usd, 4),
            "cached_entries_count": len(self.exact_cache),
        }
