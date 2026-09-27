"""Unit and integration tests for Exact and Vector Semantic Caching subsystems."""

import time

from guard_rail_gateway.cache.exact import (
    ExactHashCache,
    compute_prompt_hash,
)
from guard_rail_gateway.cache.semantic_cache import SemanticCache
from guard_rail_gateway.cache.vector import VectorSemanticCache, tokenize
from guard_rail_gateway.models import CacheEntry

# =============================================================================
# Exact Hash Cache Tests
# =============================================================================

def test_exact_hash_normalization_and_match():
    prompt1 = "  How do I   reset my password?  "
    prompt2 = "how do i reset my password"
    h1 = compute_prompt_hash(prompt1, model="gpt-4o")
    h2 = compute_prompt_hash(prompt2, model="gpt-4o")
    assert h1 == h2

    cache = ExactHashCache(max_entries=10)
    cache.put(prompt1, "Click Settings -> Password Reset.", model="gpt-4o")

    # Match with prompt2
    entry = cache.get(prompt2, model="gpt-4o")
    assert entry is not None
    assert entry.response_content == "Click Settings -> Password Reset."
    assert entry.access_count == 2


def test_exact_hash_model_isolation():
    cache = ExactHashCache()
    prompt = "Explain quantum computing"

    cache.put(prompt, "Response from GPT-4o", model="gpt-4o")
    cache.put(prompt, "Response from Claude", model="claude-3-5-sonnet")

    entry_gpt = cache.get(prompt, model="gpt-4o")
    entry_claude = cache.get(prompt, model="claude-3-5-sonnet")

    assert entry_gpt is not None
    assert entry_claude is not None
    assert entry_gpt.response_content == "Response from GPT-4o"
    assert entry_claude.response_content == "Response from Claude"


def test_exact_hash_ttl_expiration():
    cache = ExactHashCache(default_ttl_seconds=1)
    cache.put("expiring query", "Expiring content", ttl_seconds=1)

    # Immediately should exist
    assert cache.get("expiring query") is not None

    # Wait for TTL expiry
    time.sleep(1.05)
    assert cache.get("expiring query") is None
    assert len(cache) == 0


def test_exact_hash_lru_eviction():
    cache = ExactHashCache(max_entries=3)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.put("q3", "r3")
    assert len(cache) == 3

    # Access q1 to make it most recently used
    cache.get("q1")

    # Insert q4 -> should evict q2 (oldest unaccessed)
    cache.put("q4", "r4")
    assert len(cache) == 3
    assert cache.get("q1") is not None
    assert cache.get("q2") is None
    assert cache.get("q3") is not None
    assert cache.get("q4") is not None


def test_exact_hash_delete_and_clear():
    cache = ExactHashCache()
    cache.put("q1", "r1")
    cache.put("q2", "r2")

    assert cache.delete("q1") is True
    assert cache.get("q1") is None
    assert cache.delete("nonexistent") is False

    cache.clear()
    assert len(cache) == 0


# =============================================================================
# Vector Semantic Cache Tests
# =============================================================================

def test_vector_tokenize_and_stopwords():
    text = "How do I reset my password and account details?"
    tokens = tokenize(text)
    assert "reset" in tokens
    assert "password" in tokens
    assert "account" in tokens
    assert "detail" in tokens
    assert "how" not in tokens  # Stopword
    assert "and" not in tokens  # Stopword


def test_vector_tfidf_cosine_similarity():
    vcache = VectorSemanticCache(similarity_threshold=0.80)

    entry1 = CacheEntry(
        key="k1",
        prompt="How do I reset my user account password?",
        prompt_hash="h1",
        response_content="Instructions for password reset.",
        model="gpt-4o",
        created_at=time.time(),
        expires_at=time.time() + 3600,
    )
    vcache.add_entry(entry1)

    # Similar query (paraphrase)
    similar_prompt = "What are the steps to change my account password?"
    matched, score = vcache.search_similar(similar_prompt, model="gpt-4o")

    assert matched is not None
    assert score >= 0.70
    assert matched.key == "k1"

    # Unrelated query
    unrelated_prompt = "Recipe for authentic Italian margherita pizza"
    matched_unrelated, score_unrelated = vcache.search_similar(unrelated_prompt, model="gpt-4o")
    assert matched_unrelated is None
    assert score_unrelated < 0.30


def test_vector_threshold_matching():
    vcache = VectorSemanticCache(similarity_threshold=0.95)
    entry = CacheEntry(
        key="k1",
        prompt="Explain photosynthesis process in plant biology",
        prompt_hash="h1",
        response_content="Plants convert sunlight to chemical energy.",
        model="gpt-4o",
        created_at=time.time(),
        expires_at=time.time() + 3600,
    )
    vcache.add_entry(entry)

    # Identical keywords should pass high threshold
    matched, score = vcache.search_similar("Explain photosynthesis process in plant biology", model="gpt-4o")
    assert matched is not None
    assert score > 0.95

    # Slight variation with high threshold should not match
    matched_var, _ = vcache.search_similar("Describe plant sunlight chemical reactions", model="gpt-4o")
    assert matched_var is None


def test_vector_remove_entry():
    vcache = VectorSemanticCache()
    entry = CacheEntry(
        key="k1",
        prompt="Autonomous multi-agent system architecture",
        prompt_hash="h1",
        response_content="Multi-agent details",
        model="gpt-4o",
        created_at=time.time(),
        expires_at=time.time() + 3600,
    )
    vcache.add_entry(entry)
    assert len(vcache) == 1

    assert vcache.remove_entry("k1") is True
    assert len(vcache) == 0
    assert vcache.remove_entry("nonexistent") is False

    vcache.clear()
    assert len(vcache) == 0


# =============================================================================
# Unified SemanticCache Orchestrator Tests
# =============================================================================

def test_semantic_cache_lookup_flow():
    cache = SemanticCache(similarity_threshold=0.70)

    # Initial Lookup: Miss
    res1 = cache.lookup("How to configure FastAPI router?")
    assert res1.is_hit is False
    assert res1.match_type == "none"
    assert res1.entry is None

    # Store entry
    cache.store(
        prompt="How to configure FastAPI router?",
        response_content="Use APIRouter from fastapi and include_router in app.",
        model="gpt-4o",
        tokens_saved=35,
    )

    # Lookup 2: Exact Match
    res2 = cache.lookup("how to configure fastapi router?", model="gpt-4o")
    assert res2.is_hit is True
    assert res2.match_type == "exact_hash"
    assert res2.similarity_score == 1.0
    assert res2.lookup_latency_ms >= 0.0

    # Lookup 3: Vector Semantic Match (paraphrased)
    res3 = cache.lookup("How do I setup APIRouter in FastAPI app?", model="gpt-4o")
    assert res3.is_hit is True
    assert res3.match_type == "vector_semantic"
    assert res3.similarity_score >= 0.70


def test_semantic_cache_metrics_and_savings():
    cache = SemanticCache()
    cache.store("Question A", "Answer A", tokens_saved=100)

    # Trigger exact hit
    cache.lookup("Question A")
    # Trigger miss
    cache.lookup("Completely novel question")

    metrics = cache.get_metrics()
    assert metrics["total_requests"] == 2
    assert metrics["exact_hits"] == 1
    assert metrics["total_misses"] == 1
    assert metrics["hit_rate_percent"] == 50.0
    assert metrics["total_latency_saved_seconds"] > 0
    assert metrics["cached_entries_count"] == 1


def test_semantic_cache_delete_and_clear():
    cache = SemanticCache()
    cache.store("test prompt", "test reply")
    assert len(cache.exact_cache) == 1
    assert len(cache.vector_cache) == 1

    cache.delete("test prompt")
    assert len(cache.exact_cache) == 0
    assert len(cache.vector_cache) == 0

    cache.store("p1", "r1")
    cache.clear()
    assert len(cache.exact_cache) == 0
