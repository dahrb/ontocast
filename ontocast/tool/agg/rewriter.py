"""Graph rewriting for entity disambiguation.

This module handles the robust application of entity mappings to RDF graphs,
replacing all occurrences of entities according to the mapping.
"""

import logging
from collections import defaultdict

from rdflib import Literal, Node, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)


class GraphRewriter:
    """Rewrites RDF graphs by applying entity mappings.

    This class handles the robust replacement of entity URIs in RDF graphs
    according to a mapping, while preserving graph structure and metadata.
    """

    def __init__(self, add_sameas_links: bool = True):
        """Initialize the graph rewriter.

        Args:
            add_sameas_links: Whether to add owl:sameAs links for merged entities
        """
        self.add_sameas_links = add_sameas_links

    def should_map_triple_component(
        self, component: URIRef | Literal, mapping: dict[URIRef, URIRef], position: str
    ) -> bool:
        """Determine if a triple component should be mapped.

        Args:
            component: Subject, predicate, or object of a triple
            mapping: Entity mapping
            position: One of 'subject', 'predicate', 'object'

        Returns:
            True if component should be mapped
        """
        # Only map URIRefs, not Literals
        if not isinstance(component, URIRef):
            return False

        # Check if component is in mapping
        if component not in mapping:
            return False

        # Special case: don't map rdf:type objects (preserve ontology class URIs)
        # This is handled separately in apply_mapping_to_triple

        return True

    def apply_mapping_to_triple(
        self,
        subject: Node,
        predicate: Node,
        obj: Node,
        mapping: dict[URIRef, URIRef],
    ) -> tuple[Node, Node, Node]:
        """Apply entity mapping to a single triple.

        Args:
            subject: Triple subject
            predicate: Triple predicate
            obj: Triple object
            mapping: Entity mapping

        Returns:
            Mapped triple (subject, predicate, object)
        """
        # Map subject if needed
        new_subject = (
            mapping.get(subject, subject) if isinstance(subject, URIRef) else subject
        )

        # Map predicate if needed
        new_predicate = (
            mapping.get(predicate, predicate)
            if isinstance(predicate, URIRef)
            else predicate
        )

        # Map object if needed
        # Special case: preserve ontology class URIs in rdf:type triples
        if new_predicate == RDF.type and isinstance(obj, URIRef):
            # Check if object is an ontology class (not in mapping, or maps to itself)
            if obj not in mapping or mapping[obj] == obj:
                new_obj = obj  # Keep ontology class unchanged
            else:
                new_obj = mapping[
                    obj
                ]  # Map if it's a chunk entity being used as a type
        else:
            new_obj = mapping.get(obj, obj) if isinstance(obj, URIRef) else obj

        return new_subject, new_predicate, new_obj

    def collect_metadata_for_entity(
        self, entity: URIRef, graph: RDFGraph
    ) -> list[tuple[URIRef, URIRef | Literal]]:
        """Collect metadata triples for an entity (labels, comments, etc.).

        Args:
            entity: Entity URI
            graph: Source graph

        Returns:
            List of (predicate, object) pairs representing metadata
        """
        metadata = []

        metadata_predicates = {RDFS.label, RDFS.comment, RDF.type}

        for s, p, o in graph:
            if s == entity and p in metadata_predicates:
                metadata.append((p, o))

        return metadata

    def rewrite_graph(self, graph: RDFGraph, mapping: dict[URIRef, URIRef]) -> RDFGraph:
        """Rewrite a graph by applying entity mapping.

        Args:
            graph: Original RDF graph
            mapping: Entity mapping (e -> e')

        Returns:
            New RDF graph with entities replaced according to mapping
        """
        rewritten = RDFGraph()

        # Copy namespace bindings
        for prefix, namespace in graph.namespaces():
            rewritten.bind(prefix, namespace)

        # Track which entities were merged for owl:sameAs links
        merged_entities: dict[URIRef, set[URIRef]] = defaultdict(set)
        for original, mapped in mapping.items():
            if original != mapped:
                merged_entities[mapped].add(original)

        # Rewrite all triples
        processed_triples = set()

        for s, p, o in graph:
            # Apply mapping
            new_s, new_p, new_o = self.apply_mapping_to_triple(s, p, o, mapping)

            # Create triple signature to avoid duplicates
            triple_sig = (new_s, new_p, new_o)

            # Skip if we've already added this triple
            if triple_sig in processed_triples:
                continue

            # Add rewritten triple
            rewritten.add(triple_sig)
            processed_triples.add(triple_sig)

        # Add owl:sameAs links for merged entities
        if self.add_sameas_links:
            for canonical, originals in merged_entities.items():
                for original in originals:
                    # Only add sameAs if original and canonical are different
                    if original != canonical:
                        rewritten.add((canonical, OWL.sameAs, original))

        logger.info(
            f"Rewrote graph: {len(graph)} -> {len(rewritten)} triples "
            f"({len(merged_entities)} entities merged)"
        )

        return rewritten

    def rewrite_graphs(
        self, graphs: list[RDFGraph], mapping: dict[URIRef, URIRef]
    ) -> list[RDFGraph]:
        """Rewrite multiple graphs using the same mapping.

        Args:
            graphs: List of RDF graphs to rewrite
            mapping: Entity mapping

        Returns:
            List of rewritten graphs
        """
        return [self.rewrite_graph(graph, mapping) for graph in graphs]

    def merge_graphs(
        self, graphs: list[RDFGraph], mapping: dict[URIRef, URIRef], doc_namespace: str
    ) -> RDFGraph:
        """Merge multiple graphs into one, applying entity mapping.

        Args:
            graphs: List of RDF graphs to merge
            mapping: Entity mapping
            doc_namespace: Document namespace for the merged graph

        Returns:
            Single merged and rewritten graph
        """
        merged = RDFGraph()

        # Bind document namespace
        merged.bind("doc", doc_namespace)

        # Collect all namespaces from all graphs
        all_namespaces = {}
        for graph in graphs:
            for prefix, namespace in graph.namespaces():
                if prefix not in all_namespaces:
                    all_namespaces[prefix] = namespace
                elif all_namespaces[prefix] != namespace:
                    # Handle prefix conflicts
                    new_prefix = f"{prefix}_{len(all_namespaces)}"
                    all_namespaces[new_prefix] = namespace

        # Bind all namespaces
        for prefix, namespace in all_namespaces.items():
            merged.bind(prefix, namespace)

        # Track processed triples to avoid duplicates
        processed_triples = set()

        # Track merged entities for owl:sameAs
        merged_entities: dict[URIRef, set[URIRef]] = defaultdict(set)
        for original, mapped in mapping.items():
            if original != mapped:
                merged_entities[mapped].add(original)

        # Merge all graphs
        for graph in graphs:
            for s, p, o in graph:
                # Apply mapping
                new_s, new_p, new_o = self.apply_mapping_to_triple(s, p, o, mapping)

                triple_sig = (new_s, new_p, new_o)

                if triple_sig not in processed_triples:
                    merged.add(triple_sig)
                    processed_triples.add(triple_sig)

        # Add owl:sameAs links
        if self.add_sameas_links:
            for canonical, originals in merged_entities.items():
                for original in originals:
                    if original != canonical:
                        merged.add((canonical, OWL.sameAs, original))

        total_original_triples = sum(len(g) for g in graphs)
        logger.info(
            f"Merged {len(graphs)} graphs: "
            f"{total_original_triples} -> {len(merged)} triples "
            f"({len(merged_entities)} entities merged)"
        )

        return merged
