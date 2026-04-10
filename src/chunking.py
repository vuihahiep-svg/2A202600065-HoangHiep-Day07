from __future__ import annotations
import math
import re
from typing import Callable

class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.
    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks

class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.
    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Test expects splitting on ". ", "! ", "? " or ".\n"
        # We use a lookbehind to keep the punctuation
        sentences = re.split(r'(?<=[.!?])(?:\s|\n)', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks

class RecursiveChunker:
    """
    Recursively split text using separators in priority order.
    Default separator priority: ["\n\n", "\n", ". ", " ", ""]
    """
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        splits = text.split(sep) if sep != "" else list(text)
        
        final_chunks = []
        current_buffer = []
        
        for segment in splits:
            if len(segment) > self.chunk_size:
                if current_buffer:
                    final_chunks.append(sep.join(current_buffer))
                    current_buffer = []
                final_chunks.extend(self._split(segment, separators[1:]))
            elif current_buffer and len(sep.join(current_buffer + [segment])) > self.chunk_size:
                final_chunks.append(sep.join(current_buffer))
                current_buffer = [segment]
            else:
                current_buffer.append(segment)
        
        if current_buffer:
            final_chunks.append(sep.join(current_buffer))
        return final_chunks

class RegexChunker:
    """
    Split text based on custom regex patterns.
    Ideal for maintaining the structural integrity of SOPs and policy documents.
    Default pattern targets Markdown headers or common line separators.
    """
    # Đã bỏ (?m) ở đầu chuỗi r"..."
    def __init__(self, pattern: str = r"^(?:#{1,6}\s|={10,}|-{3,})", chunk_size: int = 500) -> None:
        self.pattern = pattern
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Đã thêm tham số flags=re.MULTILINE vào đây
        parts = re.split(f"({self.pattern})", text, flags=re.MULTILINE)
        
        final_chunks = []
        current_chunk = ""
        
        for part in parts:
            if not part: 
                continue
            if len(current_chunk) + len(part) <= self.chunk_size:
                current_chunk += part
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                current_chunk = part
                
        if current_chunk:
            final_chunks.append(current_chunk.strip())
        return final_chunks

class SemanticSimilarityChunker:
    """
    Groups sentences based on their semantic similarity using embeddings.
    New chunks are created when the topic shifts significantly (similarity drops below threshold).
    """
    def __init__(
        self, 
        embedding_fn: Callable[[str], list[float]], 
        threshold: float = 0.7, 
        chunk_size: int = 500
    ) -> None:
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])(?:\s|\n)', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences: 
            return []

        chunks = []
        current_group = [sentences[0]]
        
        for i in range(1, len(sentences)):
            vec_prev = self.embedding_fn(sentences[i-1])
            vec_curr = self.embedding_fn(sentences[i])
            sim = compute_similarity(vec_prev, vec_curr)
            
            combined_len = len(" ".join(current_group + [sentences[i]]))
            if sim < self.threshold or combined_len > self.chunk_size:
                chunks.append(" ".join(current_group))
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])
        
        if current_group:
            chunks.append(" ".join(current_group))
        return chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_prod = _dot(vec_a, vec_b)
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_prod / (mag_a * mag_b)

class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""
    def compare(self, text: str, chunk_size: int = 200, embedding_fn: Callable = None) -> dict:
        # THE TEST EXPECTS THESE EXACT KEYS: 'fixed_size', 'by_sentences', 'recursive'
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
            "regex_pattern": RegexChunker(chunk_size=chunk_size)
        }
        
        # Add semantic chunker only if an embedding function is provided
        if embedding_fn:
            strategies["semantic"] = SemanticSimilarityChunker(
                embedding_fn=embedding_fn, 
                chunk_size=chunk_size
            )
        
        results = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            results[name] = {
                "count": len(chunks), # TEST EXPECTS 'count' NOT 'num_chunks'
                "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks
            }
        return results