"""Embedding-based aggregation pipeline for RDF chunk graphs."""

from .aggregate import (
    EmbeddingBasedAggregator,
    aggregate_chunk_graphs,
)

__all__ = [
    "EmbeddingBasedAggregator",
    "aggregate_chunk_graphs",
]
