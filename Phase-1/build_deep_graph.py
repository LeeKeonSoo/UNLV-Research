"""
Build Deep Hierarchical Graph from Tiny-Textbooks
Creates 8-level hierarchy (Broad Domain → Fine Detail) with ~270K nodes
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import warnings
import re

from config import *
from utils import save_json

warnings.filterwarnings('ignore')  # Suppress sklearn warnings


class DeepHierarchicalGraph:
    """
    Build deep 8-level hierarchical graph
    """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_counter = 0
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder.to(DEVICE)
        print(f"🧠 Loaded embedding model on {DEVICE}")

    def add_node(self, name, level, doc_indices, parent_id=None):
        """Add node to graph"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        node = {
            'id': node_id,
            'name': name,
            'level': level,
            'document_count': len(doc_indices),
            'document_indices': doc_indices,
            'parent': parent_id,
            'children': []
        }

        if parent_id:
            # Find parent and add this as child
            for n in self.nodes:
                if n['id'] == parent_id:
                    n['children'].append(node_id)
                    break

        self.nodes.append(node)
        return node_id

    def label_cluster(self, documents, doc_indices):
        """Generate meaningful label from document content"""
        # Stopwords to filter
        stopwords = {'the', 'and', 'for', 'with', 'from', 'into', 'introduction',
                     'chapter', 'unit', 'part', 'section', 'intro', 'basics',
                     'overview', 'review', 'study', 'learning', 'understanding',
                     'this', 'that', 'these', 'those', 'what', 'when', 'where',
                     'which', 'will', 'would', 'could', 'should', 'about'}

        # Sample documents for labeling
        sample_size = min(20, len(doc_indices))
        sample_indices = doc_indices[:sample_size]

        # Extract words from text
        word_counts = Counter()
        for idx in sample_indices:
            text = documents[idx]['text'][:1000]  # First 1000 chars
            words = re.findall(r'\b[a-z]+\b', text.lower())
            meaningful = [w for w in words if len(w) > 4 and w not in stopwords]
            word_counts.update(meaningful)

        # Get top words
        if word_counts:
            top_words = [word for word, count in word_counts.most_common(3) if count >= 3]
            if not top_words:
                top_words = [word for word, count in word_counts.most_common(2)]

            if top_words:
                return ' '.join(top_words).title()

        return "General Topics"

    def cluster_documents(self, documents, doc_indices, level):
        """Cluster documents at given level"""
        if level not in LEVEL_PARAMS:
            return None

        params = LEVEL_PARAMS[level]
        n_clusters_min, n_clusters_max = params['n_clusters_range']
        min_size = params['min_size']

        # Check if enough documents
        if len(doc_indices) < min_size:
            return None

        print(f"  Level {level}: {len(doc_indices):,} docs → ", end='')

        # Embed documents
        texts = [documents[idx]['text'] for idx in doc_indices]
        embeddings = self.encoder.encode(texts, show_progress_bar=False,
                                          batch_size=256, device=DEVICE)

        # Determine optimal number of clusters
        n_clusters = min(n_clusters_max,
                         max(n_clusters_min,
                             len(doc_indices) // (min_size * 2)))

        print(f"{n_clusters} clusters... ", end='', flush=True)

        # Perform clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=METRIC,
            linkage=LINKAGE_METHOD
        )
        labels = clusterer.fit_predict(embeddings)

        # Group by cluster
        clusters = defaultdict(list)
        for local_idx, label in enumerate(labels):
            global_idx = doc_indices[local_idx]
            clusters[label].append(global_idx)

        # Sort clusters by size
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

        print(f"✓")
        return sorted_clusters

    def build_recursive(self, documents, doc_indices, level, parent_id):
        """Recursively build hierarchy"""
        if level > MAX_HIERARCHY_DEPTH:
            return

        # Cluster at this level
        clusters = self.cluster_documents(documents, doc_indices, level)

        if not clusters:
            return

        # Create node for each cluster
        for cluster_indices in clusters:
            # Generate label
            label = self.label_cluster(documents, cluster_indices)
            full_label = f"{label} ({len(cluster_indices)})"

            # Add node
            node_id = self.add_node(full_label, level, cluster_indices, parent_id)

            # Recurse deeper
            self.build_recursive(documents, cluster_indices, level + 1, node_id)

    def build(self, documents):
        """Build complete deep hierarchy"""
        print("\n" + "=" * 70)
        print("BUILDING DEEP HIERARCHICAL GRAPH")
        print("=" * 70)

        total_docs = len(documents)
        all_indices = list(range(total_docs))

        # Level 1: Root
        print(f"\nLevel 1: Root ({total_docs:,} documents)")
        root_id = self.add_node(f"All Textbooks ({total_docs:,})", 1, all_indices)

        # Build hierarchy recursively from level 2
        print(f"\nBuilding levels 2-{MAX_HIERARCHY_DEPTH}...")
        self.build_recursive(documents, all_indices, 2, root_id)

        # Create edges
        print("\n📊 Creating edges...")
        for node in self.nodes:
            if node['parent']:
                self.edges.append({
                    'source': node['parent'],
                    'target': node['id'],
                    'relationship': 'contains'
                })

        # Statistics
        level_counts = defaultdict(int)
        for node in self.nodes:
            level_counts[node['level']] += 1

        print("\n" + "=" * 70)
        print("✅ GRAPH CONSTRUCTION COMPLETE")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Total Nodes: {len(self.nodes):,}")
        print(f"   Total Edges: {len(self.edges):,}")
        print(f"   Max Depth: {max(level_counts.keys())}")
        print(f"\n   Nodes per Level:")
        for level in sorted(level_counts.keys()):
            print(f"      Level {level}: {level_counts[level]:,} nodes")

        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'statistics': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'max_depth': max(level_counts.keys()),
                'total_documents': total_docs,
                'level_counts': dict(level_counts)
            }
        }


def load_documents():
    """Load all documents from batch files"""
    print("📂 Loading documents...")
    raw_dir = Path(RAW_DATA_DIR)

    documents = []
    batch_files = sorted(raw_dir.glob("batch_*.json"))

    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {RAW_DATA_DIR}/")

    for batch_file in tqdm(batch_files, desc="Loading batches"):
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch = json.load(f)
            documents.extend(batch)

    print(f"✅ Loaded {len(documents):,} documents from {len(batch_files)} batches")
    return documents


def main():
    """Main graph building process"""
    print("=" * 70)
    print("TINY-TEXTBOOKS DEEP HIERARCHICAL ANALYSIS")
    print("=" * 70)

    try:
        # Load documents
        documents = load_documents()

        # Build graph
        graph_builder = DeepHierarchicalGraph()
        graph_data = graph_builder.build(documents)

        # Save graph
        output_file = Path(GRAPHS_DIR) / "deep_hierarchy.json"
        save_json(graph_data, str(output_file))
        print(f"\n💾 Saved graph: {output_file}")

        print("\n💡 Next steps:")
        print("   python visualize_3d.py")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
