"""Pure-Python TF-IDF vector space semantic matching engine (Tier 2).

Provides sub-millisecond semantic search across cached prompts using
term frequency-inverse document frequency (TF-IDF) and cosine similarity,
without requiring heavy neural network or C++ vector database dependencies.
"""

from __future__ import annotations

import math
import re

from guard_rail_gateway.models import CacheEntry

# Standard English stopwords to prevent vector drift
STOP_WORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


SYNONYMS: dict[str, str] = {
    "change": "reset",
    "modify": "reset",
    "update": "reset",
    "setup": "configure",
    "setting": "configure",
    "settings": "configure",
    "install": "configure",
    "apirouter": "router",
    "steps": "how",
    "instructions": "how",
    "guide": "how",
    "procedure": "how",
    "create": "build",
    "generate": "build",
    "make": "build",
}


def stem(word: str) -> str:
    """Lightweight suffix stemmer for English query tokens."""
    for suffix in ("ing", "tions", "tion", "ments", "ment", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Tokenizes text into clean lowercase stemmed words without stopwords."""
    lowered = text.lower()
    raw_tokens = re.findall(r"\b[a-z0-9]{2,}\b", lowered)
    tokens: list[str] = []
    for t in raw_tokens:
        if t in STOP_WORDS:
            continue
        canon = SYNONYMS.get(t, t)
        if canon in STOP_WORDS:
            continue
        stemmed = stem(canon)
        tokens.append(stemmed)
    return tokens


class VectorSemanticCache:
    """In-memory vector space cache computing TF-IDF Cosine Similarity."""

    def __init__(self, similarity_threshold: float = 0.88):
        self.threshold = similarity_threshold
        # key -> CacheEntry
        self._entries: dict[str, CacheEntry] = {}
        # key -> {term: normalized_tfidf_weight}
        self._vectors: dict[str, dict[str, float]] = {}
        # term -> count of documents containing term
        self._doc_freq: dict[str, int] = {}
        # Total cached documents
        self._num_docs: int = 0

    def _compute_vector(self, text: str) -> dict[str, float]:
        """Computes a unit-normalized TF-IDF vector for a string."""
        tokens = tokenize(text)
        if not tokens:
            return {}

        # 1. Term Frequencies (TF)
        tf: dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0

        num_tokens = len(tokens)
        for t, count in tf.items():
            tf[t] = count / num_tokens

        # 2. Multiply by IDF
        vector: dict[str, float] = {}
        n_docs = max(1, self._num_docs)
        for token, term_tf in tf.items():
            df = self._doc_freq.get(token, 0)
            idf = math.log(1.0 + (n_docs / (1.0 + df)))
            vector[token] = term_tf * idf

        # 3. L2 Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            for token, val in vector.items():
                vector[token] = val / norm

        return vector

    def add_entry(self, entry: CacheEntry) -> None:
        """Indexes a cache entry into the vector space."""
        tokens = set(tokenize(entry.prompt))
        for token in tokens:
            self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

        self._entries[entry.key] = entry
        self._num_docs = len(self._entries)

        # Compute and store vector
        vec = self._compute_vector(entry.prompt)
        self._vectors[entry.key] = vec

    def search_similar(
        self,
        prompt: str,
        model: str = "",
        min_threshold: float | None = None,
    ) -> tuple[CacheEntry | None, float]:
        """Finds the most semantically similar cached entry matching or exceeding threshold."""
        threshold = min_threshold if min_threshold is not None else self.threshold
        if not self._entries:
            return None, 0.0

        query_vec = self._compute_vector(prompt)
        if not query_vec:
            return None, 0.0

        best_score = 0.0
        best_entry: CacheEntry | None = None

        for key, candidate_vec in self._vectors.items():
            entry = self._entries.get(key)
            if not entry or entry.is_expired():
                continue

            # Ensure model affinity if model is specified
            if model and entry.model and entry.model != "default" and model != "default" and entry.model.lower() != model.lower():
                continue

            # Compute Dot Product of L2-normalized vectors (Cosine Similarity)
            dot_product = sum(query_vec[t] * candidate_vec[t] for t in query_vec if t in candidate_vec)

            if dot_product > best_score:
                best_score = dot_product
                best_entry = entry

        if best_entry and best_score >= threshold:
            return best_entry, best_score

        return None, best_score

    def remove_entry(self, key: str) -> bool:
        """Removes an entry from the vector index."""
        if key not in self._entries:
            return False

        entry = self._entries.pop(key)
        self._vectors.pop(key, None)

        tokens = set(tokenize(entry.prompt))
        for token in tokens:
            if token in self._doc_freq:
                self._doc_freq[token] = max(0, self._doc_freq[token] - 1)
                if self._doc_freq[token] == 0:
                    del self._doc_freq[token]

        self._num_docs = len(self._entries)
        return True

    def clear(self) -> None:
        """Clears all vectors and indices."""
        self._entries.clear()
        self._vectors.clear()
        self._doc_freq.clear()
        self._num_docs = 0

    def __len__(self) -> int:
        return len(self._entries)
