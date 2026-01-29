"""Tests for the embedding-based aggregator.

This module contains tests for all components of the embedding-based
disambiguation pipeline.
"""

from typing import cast
from unittest.mock import Mock

from rdflib import RDF, RDFS, Literal, Namespace, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.clustering import ClusterRepresentativeSelector
from ontocast.tool.agg.normalizer import EntityNormalizer, EntityRepresentation
from ontocast.tool.agg.promoter import URIPromoter
from ontocast.tool.agg.rewriter import GraphRewriter


class TestEntityNormalizer:
    """Test entity normalization."""

    def test_normalize_string(self, normalizer: EntityNormalizer) -> None:
        """Test string normalization."""
        assert normalizer.normalize_string("PLRedShift") == "pl red shift"
        assert normalizer.normalize_string("PL_red_shift_value") == "pl red shift value"
        assert normalizer.normalize_string("Café") == "cafe"

    def test_normalize_uri(self, normalizer: EntityNormalizer) -> None:
        """Test URI normalization."""
        uri1 = URIRef("http://example.org/PLRedShift")
        normalized1 = normalizer.normalize_uri(uri1)
        assert normalized1 == "pl red shift"

        uri2 = URIRef("http://example.org/PL_red_shift_value")
        normalized2 = normalizer.normalize_uri(uri2)
        assert normalized2 == "pl red shift value"

        assert normalized1.replace(" ", "") == "plredshift"
        assert normalized2.replace(" ", "").replace("value", "").strip() == "plredshift"

    def test_is_ontology_entity(self, normalizer: EntityNormalizer) -> None:
        """Test ontology entity detection."""
        ontology_entity = URIRef("http://ontology.org/Thing")
        chunk_entity = URIRef("http://chunk1.org/entity")
        assert normalizer.is_ontology_entity(ontology_entity) is True
        assert normalizer.is_ontology_entity(chunk_entity) is False

    def test_create_representation(self, normalizer: EntityNormalizer) -> None:
        """Test entity representation creation."""
        g = RDFGraph()
        EX = Namespace("http://example.org/")
        ONT = Namespace("http://ontology.org/")

        entity = EX.TestEntity
        g.add((entity, RDF.type, ONT.Thing))
        g.add((entity, RDFS.label, Literal("Test Entity")))
        g.add((entity, EX.hasValue, Literal("123")))

        rep = normalizer.create_representation(entity, g)

        assert rep.entity == entity
        assert "test entity" in rep.normal_form
        assert len(rep.types) == 1
        assert rep.types[0] == ONT.Thing
        assert "Test Entity" in rep.labels
        assert EX.hasValue in rep.properties
        assert "type" in rep.representation


class TestClusterRepresentativeSelector:
    """Test cluster representative selection."""

    def test_simplicity_score(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Test simplicity scoring."""
        simple_uri = URIRef("http://ex.org/Thing")
        complex_uri = URIRef("http://example.org/deeply/nested/path/ComplexEntity_123")

        simple_score = cluster_representative_selector.compute_simplicity_score(
            simple_uri
        )
        complex_score = cluster_representative_selector.compute_simplicity_score(
            complex_uri
        )
        assert simple_score < complex_score

    def test_select_representative_prefers_ontology(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Test that ontology entities are preferred."""
        ontology_entity = URIRef("http://ontology.org/Thing")
        chunk_entity = URIRef("http://chunk1.org/entity_with_long_complex_name")

        ont_rep = Mock()
        ont_rep.is_ontology_entity = True

        chunk_rep = Mock()
        chunk_rep.is_ontology_entity = False

        representations = cast(
            dict[URIRef, EntityRepresentation],
            {ontology_entity: ont_rep, chunk_entity: chunk_rep},
        )
        cluster = [ontology_entity, chunk_entity]

        representative = cluster_representative_selector.select_representative(
            cluster, representations
        )
        assert representative == ontology_entity

    def test_select_representative_prefers_simpler_when_no_ontology(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Test that simpler URIs are preferred when no ontology entities."""
        simple_entity = URIRef("http://chunk1.org/Thing")
        complex_entity = URIRef("http://chunk2.org/very_long_complex_entity_name_123")

        simple_rep = Mock()
        simple_rep.is_ontology_entity = False

        complex_rep = Mock()
        complex_rep.is_ontology_entity = False

        representations = cast(
            dict[URIRef, EntityRepresentation],
            {simple_entity: simple_rep, complex_entity: complex_rep},
        )
        cluster = [simple_entity, complex_entity]

        representative = cluster_representative_selector.select_representative(
            cluster, representations
        )
        assert representative == simple_entity


class TestURIPromoter:
    """Test URI promotion."""

    def test_should_promote_chunk_entity(
        self, promoter: URIPromoter, doc_namespace: str
    ) -> None:
        """Test that chunk entities should be promoted."""
        chunk_entity = URIRef("http://chunk1.org/entity")
        assert promoter.should_promote(chunk_entity) is True

    def test_should_not_promote_ontology_entity(self, promoter: URIPromoter) -> None:
        """Test that ontology entities should not be promoted."""
        ontology_entity = URIRef("http://ontology.org/Thing")
        assert promoter.should_promote(ontology_entity) is False

    def test_promote_entity(self, promoter: URIPromoter, doc_namespace: str) -> None:
        """Test entity promotion."""
        chunk_entity = URIRef("http://chunk1.org/TestEntity")

        rep = Mock()
        rep.normal_form = "test entity"
        rep.is_ontology_entity = False

        promoted = promoter.promote_entity(chunk_entity, rep)
        promoted_str = str(promoted)
        assert promoted_str.startswith(doc_namespace)
        assert "test_entity" in promoted_str

    def test_preserve_ontology_entity(self, promoter: URIPromoter) -> None:
        """Test that ontology entities are preserved."""
        ontology_entity = URIRef("http://ontology.org/Thing")

        rep = Mock()
        rep.is_ontology_entity = True

        promoted = promoter.promote_entity(ontology_entity, rep)
        assert promoted == ontology_entity

    def test_compose_mappings(self, promoter: URIPromoter) -> None:
        """Test mapping composition."""
        rep = URIRef("http://chunk1.org/representative")
        e1 = URIRef("http://chunk1.org/entity1")
        e2 = URIRef("http://chunk2.org/entity2")

        clustering_mapping = {e1: rep, e2: rep}
        promoted = URIRef("http://doc.org/promoted_entity")
        promotion_mapping = {rep: promoted}

        composed = promoter.compose_mappings(clustering_mapping, promotion_mapping)
        assert composed[e1] == promoted
        assert composed[e2] == promoted


class TestGraphRewriter:
    """Test graph rewriting."""

    def test_apply_mapping_to_triple(self, graph_rewriter: GraphRewriter) -> None:
        """Test mapping application to a single triple."""
        e1 = URIRef("http://chunk1.org/e1")
        p1 = URIRef("http://chunk1.org/p1")
        e2 = URIRef("http://chunk1.org/e2")

        e1_prime = URIRef("http://doc.org/entity1")
        p1_prime = URIRef("http://doc.org/property1")
        e2_prime = URIRef("http://doc.org/entity2")

        mapping = {e1: e1_prime, p1: p1_prime, e2: e2_prime}

        new_s, new_p, new_o = graph_rewriter.apply_mapping_to_triple(
            e1, p1, e2, mapping
        )
        assert new_s == e1_prime
        assert new_p == p1_prime
        assert new_o == e2_prime

    def test_preserve_ontology_types(self, graph_rewriter: GraphRewriter) -> None:
        """Test that ontology types are preserved in rdf:type triples."""
        entity = URIRef("http://chunk1.org/entity")
        ontology_type = URIRef("http://ontology.org/Thing")
        entity_prime = URIRef("http://doc.org/entity")

        mapping = {entity: entity_prime}

        new_s, new_p, new_o = graph_rewriter.apply_mapping_to_triple(
            entity, RDF.type, ontology_type, mapping
        )
        assert new_s == entity_prime
        assert new_p == RDF.type
        assert new_o == ontology_type

    def test_rewrite_graph(self, graph_rewriter: GraphRewriter) -> None:
        """Test complete graph rewriting."""
        g = RDFGraph()
        e1 = URIRef("http://chunk1.org/e1")
        e2 = URIRef("http://chunk1.org/e2")
        p = URIRef("http://chunk1.org/p")

        g.add((e1, p, e2))
        g.add((e1, RDF.type, URIRef("http://ontology.org/Thing")))

        e1_prime = URIRef("http://doc.org/entity1")
        e2_prime = URIRef("http://doc.org/entity2")
        p_prime = URIRef("http://doc.org/property")

        mapping = {e1: e1_prime, e2: e2_prime, p: p_prime}

        rewritten = graph_rewriter.rewrite_graph(g, mapping)

        assert (e1_prime, p_prime, e2_prime) in rewritten
        assert (
            e1_prime,
            RDF.type,
            URIRef("http://ontology.org/Thing"),
        ) in rewritten
