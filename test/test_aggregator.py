"""Tests for the embedding-based aggregator.

This module contains tests for all components of the embedding-based
disambiguation pipeline.
"""

import unittest
from unittest.mock import Mock

from rdflib import RDF, RDFS, Literal, Namespace, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.clustering import ClusterRepresentativeSelector
from ontocast.tool.agg.normalizer import EntityNormalizer
from ontocast.tool.agg.promoter import URIPromoter
from ontocast.tool.agg.rewriter import GraphRewriter


class TestEntityNormalizer(unittest.TestCase):
    """Test entity normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.ontology_ns = {"http://ontology.org/"}
        self.normalizer = EntityNormalizer(self.ontology_ns)

    def test_normalize_string(self):
        """Test string normalization."""
        # Test lowercase
        self.assertEqual(self.normalizer.normalize_string("PLRedShift"), "plredshift")

        # Test underscore to space
        self.assertEqual(
            self.normalizer.normalize_string("PL_red_shift_value"), "pl red shift value"
        )

        # Test diacritics removal
        self.assertEqual(self.normalizer.normalize_string("Café"), "cafe")

    def test_normalize_uri(self):
        """Test URI normalization."""
        # Test camelCase splitting
        uri1 = URIRef("http://example.org/PLRedShift")
        normalized1 = self.normalizer.normalize_uri(uri1)
        self.assertEqual(normalized1, "pl red shift")

        # Test snake_case
        uri2 = URIRef("http://example.org/PL_red_shift_value")
        normalized2 = self.normalizer.normalize_uri(uri2)
        self.assertEqual(normalized2, "pl red shift value")

        # Should produce same normalized form
        self.assertEqual(normalized1.replace(" ", ""), "plredshift")
        self.assertEqual(
            normalized2.replace(" ", "").replace("value", "").strip(), "plredshift"
        )

    def test_is_ontology_entity(self):
        """Test ontology entity detection."""
        ontology_entity = URIRef("http://ontology.org/Thing")
        chunk_entity = URIRef("http://chunk1.org/entity")

        self.assertTrue(self.normalizer.is_ontology_entity(ontology_entity))
        self.assertFalse(self.normalizer.is_ontology_entity(chunk_entity))

    def test_create_representation(self):
        """Test entity representation creation."""
        # Create test RDFGraph
        g = RDFGraph()
        EX = Namespace("http://example.org/")
        ONT = Namespace("http://ontology.org/")

        entity = EX.TestEntity
        g.add((entity, RDF.type, ONT.Thing))
        g.add((entity, RDFS.label, Literal("Test Entity")))
        g.add((entity, EX.hasValue, Literal("123")))

        # Create representation
        rep = self.normalizer.create_representation(entity, g)

        # Check fields
        self.assertEqual(rep.entity, entity)
        self.assertIn("test entity", rep.normal_form)
        self.assertEqual(len(rep.types), 1)
        self.assertEqual(rep.types[0], ONT.Thing)
        self.assertIn("Test Entity", rep.labels)
        self.assertIn(EX.hasValue, rep.properties)

        # Representation should include type info
        self.assertIn("type", rep.representation)


class TestClusterRepresentativeSelector(unittest.TestCase):
    """Test cluster representative selection."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = ClusterRepresentativeSelector()

    def test_simplicity_score(self):
        """Test simplicity scoring."""
        simple_uri = URIRef("http://ex.org/Thing")
        complex_uri = URIRef("http://example.org/deeply/nested/path/ComplexEntity_123")

        simple_score = self.selector.compute_simplicity_score(simple_uri)
        complex_score = self.selector.compute_simplicity_score(complex_uri)

        # Simple URI should have lower score
        self.assertLess(simple_score, complex_score)

    def test_select_representative_prefers_ontology(self):
        """Test that ontology entities are preferred."""
        # Create mock representations
        ontology_entity = URIRef("http://ontology.org/Thing")
        chunk_entity = URIRef("http://chunk1.org/entity_with_long_complex_name")

        ont_rep = Mock()
        ont_rep.is_ontology_entity = True

        chunk_rep = Mock()
        chunk_rep.is_ontology_entity = False

        representations = {ontology_entity: ont_rep, chunk_entity: chunk_rep}

        cluster = [ontology_entity, chunk_entity]

        # Should select ontology entity despite chunk entity being "simpler" by name
        representative = self.selector.select_representative(cluster, representations)
        self.assertEqual(representative, ontology_entity)

    def test_select_representative_prefers_simpler_when_no_ontology(self):
        """Test that simpler URIs are preferred when no ontology entities."""
        simple_entity = URIRef("http://chunk1.org/Thing")
        complex_entity = URIRef("http://chunk2.org/very_long_complex_entity_name_123")

        simple_rep = Mock()
        simple_rep.is_ontology_entity = False

        complex_rep = Mock()
        complex_rep.is_ontology_entity = False

        representations = {simple_entity: simple_rep, complex_entity: complex_rep}

        cluster = [simple_entity, complex_entity]

        representative = self.selector.select_representative(cluster, representations)
        self.assertEqual(representative, simple_entity)


class TestURIPromoter(unittest.TestCase):
    """Test URI promotion."""

    def setUp(self):
        """Set up test fixtures."""
        self.doc_namespace = "http://doc.org/"
        self.chunk_namespaces = {"http://chunk1.org/", "http://chunk2.org/"}
        self.ontology_namespaces = {"http://ontology.org/"}

        self.promoter = URIPromoter(
            self.doc_namespace, self.chunk_namespaces, self.ontology_namespaces
        )

    def test_should_promote_chunk_entity(self):
        """Test that chunk entities should be promoted."""
        chunk_entity = URIRef("http://chunk1.org/entity")
        self.assertTrue(self.promoter.should_promote(chunk_entity))

    def test_should_not_promote_ontology_entity(self):
        """Test that ontology entities should not be promoted."""
        ontology_entity = URIRef("http://ontology.org/Thing")
        self.assertFalse(self.promoter.should_promote(ontology_entity))

    def test_promote_entity(self):
        """Test entity promotion."""
        chunk_entity = URIRef("http://chunk1.org/TestEntity")

        # Create mock representation
        rep = Mock()
        rep.normal_form = "test entity"
        rep.is_ontology_entity = False

        promoted = self.promoter.promote_entity(chunk_entity, rep)

        # Should be in document namespace
        promoted_str = str(promoted)
        self.assertTrue(promoted_str.startswith(self.doc_namespace))
        self.assertIn("test_entity", promoted_str)

    def test_preserve_ontology_entity(self):
        """Test that ontology entities are preserved."""
        ontology_entity = URIRef("http://ontology.org/Thing")

        rep = Mock()
        rep.is_ontology_entity = True

        promoted = self.promoter.promote_entity(ontology_entity, rep)

        # Should be unchanged
        self.assertEqual(promoted, ontology_entity)

    def test_compose_mappings(self):
        """Test mapping composition."""
        # Clustering mapping: e1, e2 -> rep
        rep = URIRef("http://chunk1.org/representative")
        e1 = URIRef("http://chunk1.org/entity1")
        e2 = URIRef("http://chunk2.org/entity2")

        clustering_mapping = {e1: rep, e2: rep}

        # Promotion mapping: rep -> promoted
        promoted = URIRef("http://doc.org/promoted_entity")
        promotion_mapping = {rep: promoted}

        # Compose
        composed = self.promoter.compose_mappings(clustering_mapping, promotion_mapping)

        # Both e1 and e2 should map to promoted
        self.assertEqual(composed[e1], promoted)
        self.assertEqual(composed[e2], promoted)


class TestGraphRewriter(unittest.TestCase):
    """Test graph rewriting."""

    def setUp(self):
        """Set up test fixtures."""
        self.rewriter = GraphRewriter(add_sameas_links=True)

    def test_apply_mapping_to_triple(self):
        """Test mapping application to a single triple."""
        # Original triple
        e1 = URIRef("http://chunk1.org/e1")
        p1 = URIRef("http://chunk1.org/p1")
        e2 = URIRef("http://chunk1.org/e2")

        # Mapping
        e1_prime = URIRef("http://doc.org/entity1")
        p1_prime = URIRef("http://doc.org/property1")
        e2_prime = URIRef("http://doc.org/entity2")

        mapping = {e1: e1_prime, p1: p1_prime, e2: e2_prime}

        # Apply mapping
        new_s, new_p, new_o = self.rewriter.apply_mapping_to_triple(e1, p1, e2, mapping)

        self.assertEqual(new_s, e1_prime)
        self.assertEqual(new_p, p1_prime)
        self.assertEqual(new_o, e2_prime)

    def test_preserve_ontology_types(self):
        """Test that ontology types are preserved in rdf:type triples."""
        entity = URIRef("http://chunk1.org/entity")
        ontology_type = URIRef("http://ontology.org/Thing")

        entity_prime = URIRef("http://doc.org/entity")

        mapping = {
            entity: entity_prime
            # Note: ontology_type not in mapping (should be preserved)
        }

        # Apply mapping to type triple
        new_s, new_p, new_o = self.rewriter.apply_mapping_to_triple(
            entity, RDF.type, ontology_type, mapping
        )

        self.assertEqual(new_s, entity_prime)  # Subject mapped
        self.assertEqual(new_p, RDF.type)  # Predicate unchanged
        self.assertEqual(new_o, ontology_type)  # Type preserved!

    def test_rewrite_graph(self):
        """Test complete graph rewriting."""
        # Create original graph
        g = RDFGraph()
        e1 = URIRef("http://chunk1.org/e1")
        e2 = URIRef("http://chunk1.org/e2")
        p = URIRef("http://chunk1.org/p")

        g.add((e1, p, e2))
        g.add((e1, RDF.type, URIRef("http://ontology.org/Thing")))

        # Create mapping
        e1_prime = URIRef("http://doc.org/entity1")
        e2_prime = URIRef("http://doc.org/entity2")
        p_prime = URIRef("http://doc.org/property")

        mapping = {e1: e1_prime, e2: e2_prime, p: p_prime}

        # Rewrite
        rewritten = self.rewriter.rewrite_graph(g, mapping)

        # Check rewritten triples exist
        self.assertIn((e1_prime, p_prime, e2_prime), rewritten)
        self.assertIn(
            (e1_prime, RDF.type, URIRef("http://ontology.org/Thing")), rewritten
        )


def run_tests():
    """Run all tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
