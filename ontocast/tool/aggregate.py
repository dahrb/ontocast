"""Graph aggregation for OntoCast.

This module re-exports the embedding-based aggregator as the main aggregation
implementation. Use EmbeddingBasedAggregator for aggregating and disambiguating
RDF graphs from multiple chunks.
"""

from ontocast.tool.agg.aggregate import (
    EmbeddingBasedAggregator,
    aggregate_chunk_graphs,
)

__all__ = [
    "EmbeddingBasedAggregator",
    "aggregate_chunk_graphs",
]
