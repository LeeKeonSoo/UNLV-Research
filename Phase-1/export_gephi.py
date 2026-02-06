"""
Export deep hierarchy graph to Gephi-compatible GEXF format
Run this after build_deep_graph.py to create .gexf file for Gephi
"""

import json
from pathlib import Path

from config import GRAPHS_DIR


def export_to_gephi(json_path, gexf_path):
    """Convert JSON graph to GEXF format for Gephi"""
    try:
        import networkx as nx
    except ImportError:
        print("\n❌ NetworkX not installed!")
        print("   Install with: pip install networkx")
        return False

    print("\n📦 Loading graph data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    print(f"   Nodes: {len(graph_data['nodes']):,}")
    print(f"   Edges: {len(graph_data['edges']):,}")

    print("\n🔄 Converting to NetworkX format...")
    
    # Create NetworkX directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in graph_data['nodes']:
        G.add_node(
            node['id'],
            label=node['name'],
            level=node['level'],
            doc_count=node['document_count'],
            parent=node.get('parent', '')
        )
    
    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'])
    
    print(f"\n💾 Saving GEXF file...")
    nx.write_gexf(G, gexf_path)
    
    print(f"\n✅ Successfully exported to Gephi format!")
    print(f"   File: {gexf_path}")
    print(f"\n📖 Next steps:")
    print(f"   1. Download Gephi: https://gephi.org/")
    print(f"   2. Open Gephi")
    print(f"   3. File → Open → {gexf_path}")
    print(f"   4. Layout → Force Atlas 2 → Run")
    print(f"   5. Statistics → Modularity → Run")
    print(f"   6. Appearance → Nodes → Color → Partition → Modularity Class")
    print(f"   7. Appearance → Nodes → Size → Ranking → doc_count")
    
    return True


def main():
    """Export existing JSON graph to GEXF"""
    print("=" * 70)
    print("EXPORT TO GEPHI FORMAT")
    print("=" * 70)
    
    json_file = Path(GRAPHS_DIR) / "deep_hierarchy.json"
    gexf_file = Path(GRAPHS_DIR) / "deep_hierarchy.gexf"
    
    if not json_file.exists():
        print(f"\n❌ Graph file not found: {json_file}")
        print("   Run: python build_deep_graph.py")
        return 1
    
    success = export_to_gephi(str(json_file), str(gexf_file))
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
