"""Embedding-based RDF graph aggregator.

This module provides the main aggregator class that orchestrates entity
disambiguation using embedding-based clustering.

Pipeline:
1. Normalize entities: e -> r(e) (string representation with semantic context)
2. Embed in parallel: r(e) -> v(e) (embedding vectors)
3. Cluster by similarity: v(e) -> g(e) (groups of similar entities)
4. Select representatives: g(e) -> e_rep (best entity per group)
5. Promote to doc namespace: e_rep -> e' (document-level URIs)
6. Rewrite graphs: apply mapping e -> e' to all triples
"""

import logging
from typing import Optional

from rdflib import URIRef

from ontocast.onto.chunk import Chunk
from ontocast.onto.rdfgraph import RDFGraph

from .clustering import ClusterRepresentativeSelector, EntityClusterer
from .normalizer import EntityNormalizer
from .promoter import URIPromoter
from .rewriter import GraphRewriter

logger = logging.getLogger(__name__)


class EmbeddingBasedAggregator:
    """Main aggregator using embedding-based entity disambiguation.

    This aggregator uses a clean pipeline:
    1. Entity normalization (with semantic context)
    2. Parallel embedding
    3. Similarity-based clustering
    4. Representative selection (prefer ontology, then simplicity)
    5. URI promotion (chunk -> document namespace)
    6. Graph rewriting

    Attributes:
        normalizer: Entity normalizer for creating representations
        clusterer: Entity clusterer for grouping similar entities
        selector: Representative selector for choosing best entity per group
        promoter: URI promoter for chunk -> document namespace conversion
        rewriter: Graph rewriter for applying entity mappings
    """

    def __init__(
        self,
        ontology_namespaces: Optional[set[str]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        add_sameas_links: bool = True,
    ):
        """Initialize the embedding-based aggregator.

        Args:
            ontology_namespaces: Set of namespace URIs for ontology entities
            embedding_model: Name of sentence transformer model
            similarity_threshold: Cosine similarity threshold for clustering (0-1)
            add_sameas_links: Whether to add owl:sameAs for merged entities
        """
        self.ontology_namespaces = ontology_namespaces or set()

        # Initialize pipeline components
        self.normalizer = EntityNormalizer(self.ontology_namespaces)
        self.clusterer = EntityClusterer(
            embedding_model=embedding_model, similarity_threshold=similarity_threshold
        )
        self.selector = ClusterRepresentativeSelector()
        self.rewriter = GraphRewriter(add_sameas_links=add_sameas_links)

        # Promoter is created per aggregation (needs doc namespace)
        self.promoter: Optional[URIPromoter] = None

    def _collect_all_entities(
        self, chunks: list[Chunk]
    ) -> tuple[list[URIRef], dict[URIRef, RDFGraph]]:
        """Collect all entities from all chunk graphs.

        Args:
            chunks: List of chunks to aggregate

        Returns:
            Tuple of (entities, entity_to_graph_mapping)
        """
        entities = set()
        entity_graphs = {}

        for chunk in chunks:
            if chunk.graph is None:
                continue

            # Collect all entities (subjects and objects)
            for s, p, o in chunk.graph:
                if isinstance(s, URIRef):
                    entities.add(s)
                    entity_graphs[s] = chunk.graph
                if isinstance(o, URIRef):
                    entities.add(o)
                    entity_graphs[o] = chunk.graph
                # Note: predicates are also URIs, but we might handle them separately
                # For now, we include them implicitly through subjects/objects

        return list(entities), entity_graphs

    def _collect_chunk_namespaces(self, chunks: list[Chunk]) -> set[str]:
        """Collect all chunk namespaces.

        Args:
            chunks: List of chunks

        Returns:
            Set of chunk namespace URIs
        """
        return {chunk.namespace for chunk in chunks if chunk.namespace}

    def aggregate_graphs(self, chunks: list[Chunk], doc_namespace: str) -> RDFGraph:
        """Aggregate multiple chunk graphs with embedding-based disambiguation.

        This is the main entry point that orchestrates the entire pipeline:
        1. Collect entities from all chunks
        2. Create normalized representations r(e) with semantic context
        3. Embed all representations in parallel: r(e) -> v(e)
        4. Cluster by similarity: v(e) -> g(e)
        5. Select best representative per group: g(e) -> e_rep
        6. Promote to document namespace: e_rep -> e'
        7. Compose final mapping: e -> e'
        8. Rewrite and merge all graphs

        Args:
            chunks: List of chunks to aggregate
            doc_namespace: Document namespace for the aggregated graph

        Returns:
            Aggregated RDF graph with disambiguated entities
        """
        logger.info(f"Starting aggregation of {len(chunks)} chunks")

        if not chunks:
            return RDFGraph()

        # Step 0: Setup
        chunk_namespaces = self._collect_chunk_namespaces(chunks)
        self.promoter = URIPromoter(
            doc_namespace=doc_namespace,
            chunk_namespaces=chunk_namespaces,
            ontology_namespaces=self.ontology_namespaces,
        )

        # Step 1: Collect all entities
        logger.info("Step 1: Collecting entities from chunks...")
        entities, entity_graphs = self._collect_all_entities(chunks)
        logger.info(f"Collected {len(entities)} unique entities")

        # Step 2: Create normalized representations r(e)
        logger.info("Step 2: Creating entity representations...")
        representations = self.normalizer.create_representations_batch(
            entities, entity_graphs
        )
        logger.info(f"Created representations for {len(representations)} entities")

        # Step 3: Embed and cluster
        logger.info("Step 3: Embedding and clustering entities...")
        clusters, embeddings = self.clusterer.cluster_entities(representations)
        logger.info(f"Formed {len(clusters)} clusters")

        # Step 4: Select representatives
        logger.info("Step 4: Selecting cluster representatives...")
        clustering_mapping = self.selector.create_mapping(clusters, representations)
        logger.info(f"Selected {len(set(clustering_mapping.values()))} representatives")

        # Step 5: Promote representatives to document namespace
        logger.info("Step 5: Promoting URIs to document namespace...")
        # Get unique representatives
        representatives = list(set(clustering_mapping.values()))
        promotion_mapping = self.promoter.create_promotion_mapping(
            representatives, representations
        )

        # Step 6: Compose mappings
        logger.info("Step 6: Composing final entity mapping...")
        final_mapping = self.promoter.compose_mappings(
            clustering_mapping, promotion_mapping
        )

        # Log mapping statistics
        unchanged = sum(1 for e, mapped in final_mapping.items() if e == mapped)
        changed = len(final_mapping) - unchanged
        unique_targets = len(set(final_mapping.values()))
        logger.info(
            f"Final mapping: {len(final_mapping)} entities -> {unique_targets} unique URIs "
            f"({changed} changed, {unchanged} unchanged)"
        )

        # Step 7: Rewrite and merge graphs
        logger.info("Step 7: Rewriting and merging graphs...")
        chunk_graphs = [chunk.graph for chunk in chunks if chunk.graph is not None]
        merged_graph = self.rewriter.merge_graphs(
            chunk_graphs, final_mapping, doc_namespace
        )

        logger.info(f"Aggregation complete: {len(merged_graph)} triples in final graph")

        return merged_graph

    def aggregate_graphs_with_metadata(
        self, chunks: list[Chunk], doc_namespace: str
    ) -> tuple[RDFGraph, dict]:
        """Aggregate graphs and return additional metadata about the process.

        Args:
            chunks: List of chunks to aggregate
            doc_namespace: Document namespace

        Returns:
            Tuple of (aggregated_graph, metadata_dict)
            metadata_dict contains:
                - entity_mapping: Final e -> e' mapping
                - clusters: List of entity clusters
                - representations: Entity representations
                - embeddings: Entity embeddings
        """
        logger.info(f"Starting aggregation with metadata for {len(chunks)} chunks")

        if not chunks:
            return RDFGraph(), {}

        # Step 0: Setup
        chunk_namespaces = self._collect_chunk_namespaces(chunks)
        self.promoter = URIPromoter(
            doc_namespace=doc_namespace,
            chunk_namespaces=chunk_namespaces,
            ontology_namespaces=self.ontology_namespaces,
        )

        # Steps 1-3: Collect, normalize, embed, cluster
        entities, entity_graphs = self._collect_all_entities(chunks)
        representations = self.normalizer.create_representations_batch(
            entities, entity_graphs
        )
        clusters, embeddings = self.clusterer.cluster_entities(representations)

        # Steps 4-6: Select, promote, compose
        clustering_mapping = self.selector.create_mapping(clusters, representations)
        representatives = list(set(clustering_mapping.values()))
        promotion_mapping = self.promoter.create_promotion_mapping(
            representatives, representations
        )
        final_mapping = self.promoter.compose_mappings(
            clustering_mapping, promotion_mapping
        )

        # Step 7: Rewrite and merge
        chunk_graphs = [chunk.graph for chunk in chunks if chunk.graph is not None]
        merged_graph = self.rewriter.merge_graphs(
            chunk_graphs, final_mapping, doc_namespace
        )

        # Prepare metadata
        metadata = {
            "entity_mapping": final_mapping,
            "clusters": clusters,
            "representations": representations,
            "embeddings": embeddings,
            "num_entities": len(entities),
            "num_clusters": len(clusters),
            "num_unique_targets": len(set(final_mapping.values())),
        }

        logger.info("Aggregation with metadata complete")

        return merged_graph, metadata


# Convenience function for backward compatibility
def aggregate_chunk_graphs(
    chunks: list[Chunk],
    doc_namespace: str,
    ontology_namespaces: Optional[set[str]] = None,
    similarity_threshold: float = 0.85,
) -> RDFGraph:
    """Convenience function to aggregate chunk graphs.

    Args:
        chunks: List of chunks to aggregate
        doc_namespace: Document namespace
        ontology_namespaces: Optional set of ontology namespaces
        similarity_threshold: Cosine similarity threshold for clustering

    Returns:
        Aggregated RDF graph
    """
    aggregator = EmbeddingBasedAggregator(
        ontology_namespaces=ontology_namespaces,
        similarity_threshold=similarity_threshold,
    )
    return aggregator.aggregate_graphs(chunks, doc_namespace)
