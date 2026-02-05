"""
Build Concept Graph from K-12 Curriculum Data

This script:
1. Loads collected K-12 data
2. Groups by subject and grade level
3. Uses clustering to discover concept hierarchy
4. Builds graph with prerequisite relationships
5. Saves graph structure for analysis
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from config import *
from utils import save_json


class ConceptGraph:
    """
    Hierarchical concept graph structure
    
    Nodes = Concepts (discovered through clustering)
    Edges = Relationships (prerequisites, co-occurrence)
    """
    
    def __init__(self):
        self.nodes = {}  # concept_id -> node_data
        self.edges = []  # (source_id, target_id, relationship_type)
        self.node_counter = 0
        
    def add_node(self, name, grade_level, subject, documents, parent=None):
        """Add a concept node to the graph"""
        node_id = f"concept_{self.node_counter}"
        self.node_counter += 1
        
        self.nodes[node_id] = {
            "id": node_id,
            "name": name,
            "grade_level": grade_level,
            "subject": subject,
            "document_count": len(documents),
            "document_ids": [doc.get('id', i) for i, doc in enumerate(documents)],
            "parent": parent,
            "children": []
        }
        
        # Link to parent
        if parent and parent in self.nodes:
            self.nodes[parent]["children"].append(node_id)
        
        return node_id
    
    def add_edge(self, source_id, target_id, relationship="prerequisite", weight=1.0):
        """Add relationship between concepts"""
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "weight": weight
        })
    
    def get_level(self, level_num):
        """Get all nodes at a specific depth level"""
        # Level 0 = top-level subjects
        # Level 1 = grade groupings
        # Level 2+ = discovered concepts
        return [node_id for node_id, node in self.nodes.items() 
                if self._get_depth(node_id) == level_num]
    
    def _get_depth(self, node_id):
        """Calculate depth of a node"""
        depth = 0
        current = node_id
        while self.nodes[current].get("parent"):
            depth += 1
            current = self.nodes[current]["parent"]
        return depth
    
    def max_depth(self):
        """Maximum depth in the graph"""
        if not self.nodes:
            return 0
        return max(self._get_depth(node_id) for node_id in self.nodes)
    
    def save(self, filepath):
        """Save graph to JSON"""
        graph_data = {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "max_depth": self.max_depth()
            }
        }
        save_json(graph_data, filepath)
        print(f"💾 Graph saved to {filepath}")


class ConceptDiscovery:
    """
    Discover concepts from documents using clustering
    """
    
    def __init__(self, min_cluster_size=MIN_CLUSTER_SIZE):
        self.min_cluster_size = min_cluster_size
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def embed_documents(self, documents):
        """Create embeddings for documents"""
        texts = [doc.get('text', doc.get('content', '')) for doc in documents]
        print(f"  Embedding {len(texts)} documents...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
    
    def discover_concepts(self, documents, min_cluster_size=None):
        """
        Cluster documents to discover concepts
        
        Returns:
            List of clusters, each containing document indices
        """
        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size
            
        if len(documents) < min_cluster_size:
            # Too few documents to cluster
            return [list(range(len(documents)))]
        
        # Embed
        embeddings = self.embed_documents(documents)
        
        # Cluster
        print(f"  Clustering with min_size={min_cluster_size}...")
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                           min_samples=5)
        labels = clusterer.fit_predict(embeddings)
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # Ignore noise
                clusters[label].append(idx)
        
        print(f"  Discovered {len(clusters)} concept clusters")
        return list(clusters.values())
    
    def label_cluster(self, documents, cluster_indices):
        """
        Generate a label for a cluster
        
        For now, uses simple heuristics.
        Could use LLM for better labeling.
        """
        # Get representative documents
        sample_size = min(5, len(cluster_indices))
        sample_docs = [documents[i] for i in cluster_indices[:sample_size]]
        
        # Simple labeling: use common words in titles/topics
        titles = [doc.get('title', doc.get('topic', '')) for doc in sample_docs]
        
        # Placeholder: just return a generic label
        # In production, would use LLM or more sophisticated method
        return f"Concept_cluster_{len(cluster_indices)}_docs"


def build_k12_graph(subject_name, subject_docs):
    """
    Build concept graph for a subject
    
    Args:
        subject_name: Name of subject (e.g., "Mathematics")
        subject_docs: List of documents for this subject
    """
    print(f"\n{'='*60}")
    print(f"Building graph for: {subject_name}")
    print(f"Documents: {len(subject_docs)}")
    print('='*60)
    
    graph = ConceptGraph()
    discovery = ConceptDiscovery()
    
    # Add subject as root node
    subject_id = graph.add_node(
        name=subject_name,
        grade_level=None,
        subject=subject_name,
        documents=subject_docs,
        parent=None
    )
    
    # Group by grade level
    docs_by_grade = defaultdict(list)
    for doc in subject_docs:
        grade = doc.get('grade_level', 'unknown')
        docs_by_grade[grade].append(doc)
    
    # Process each grade
    for grade, grade_docs in sorted(docs_by_grade.items()):
        print(f"\nProcessing Grade {grade} ({len(grade_docs)} docs)")
        
        # Add grade node
        grade_id = graph.add_node(
            name=f"Grade {grade}",
            grade_level=grade,
            subject=subject_name,
            documents=grade_docs,
            parent=subject_id
        )
        
        # Discover concepts within this grade
        clusters = discovery.discover_concepts(grade_docs)
        
        for cluster_indices in clusters:
            cluster_docs = [grade_docs[i] for i in cluster_indices]
            
            # Label the concept
            concept_label = discovery.label_cluster(grade_docs, cluster_indices)
            
            # Add concept node
            concept_id = graph.add_node(
                name=concept_label,
                grade_level=grade,
                subject=subject_name,
                documents=cluster_docs,
                parent=grade_id
            )
            
            print(f"  Concept: {concept_label} ({len(cluster_docs)} docs)")
    
    return graph


def main():
    """
    Main graph construction pipeline
    """
    print("="*60)
    print("K-12 CONCEPT GRAPH CONSTRUCTION")
    print("="*60)
    
    # Check if data exists
    raw_dir = Path(K12_RAW_DIR)
    if not raw_dir.exists() or not list(raw_dir.iterdir()):
        print("❌ No K-12 data found!")
        print(f"   Please run collect_k12.py first to gather data")
        print(f"   Expected data in: {K12_RAW_DIR}")
        return
    
    print(f"\n📂 Loading data from {K12_RAW_DIR}")
    
    # For now, this is a placeholder
    # In production, would load actual collected data
    print("⚠️  Placeholder: Actual data loading to be implemented")
    print("   This requires completed data collection from collect_k12.py")
    
    # Example structure if data were available:
    """
    all_documents = []
    
    # Load Khan Academy
    khan_dir = raw_dir / "khan_academy"
    if khan_dir.exists():
        for file in khan_dir.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                all_documents.extend(data.get('documents', []))
    
    # Load OpenStax
    openstax_dir = raw_dir / "openstax"
    if openstax_dir.exists():
        for file in openstax_dir.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                # Parse chapters and sections into documents
                all_documents.extend(parse_openstax(data))
    
    # Group by subject
    docs_by_subject = defaultdict(list)
    for doc in all_documents:
        subject = doc.get('subject', 'unknown')
        docs_by_subject[subject].append(doc)
    
    # Build graph for each subject
    for subject, docs in docs_by_subject.items():
        graph = build_k12_graph(subject, docs)
        
        # Save
        output_file = Path(K12_GRAPHS_DIR) / f"{subject.lower()}_graph.json"
        graph.save(str(output_file))
    """
    
    print("\n💡 Next steps:")
    print("   1. Complete data collection (collect_k12.py)")
    print("   2. Implement data loading logic")
    print("   3. Run graph construction")
    print("   4. Analyze and visualize results")


if __name__ == "__main__":
    main()
