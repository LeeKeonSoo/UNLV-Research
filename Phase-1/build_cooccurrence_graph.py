"""
Build Co-occurrence Graph from Document-level Domain Scores

This script:
1. Loads per-document classification scores
2. Builds a co-occurrence matrix (edges between domains)
3. Computes node sizes (domain coverage)
4. Saves graph data for visualization
"""

import os
import json
import numpy as np
from collections import defaultdict
from config import *
from utils import save_json


def load_document_scores(source_name: str) -> list:
    """Load document-level scores for a source"""
    filepath = os.path.join(RESULTS_DIR, f"{source_name}_document_scores.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Document scores not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def build_cooccurrence_matrix(document_scores: list, threshold: float = 0.7) -> dict:
    """
    Build co-occurrence matrix from document scores
    
    Args:
        document_scores: List of {doc_id, domains: {domain: score}}
        threshold: Minimum score to consider a domain "present" in a document
    
    Returns:
        Dictionary with:
        - cooccurrence: dict of {(domain1, domain2): count}
        - domain_coverage: dict of {domain: count}
    """
    # Initialize
    cooccurrence = defaultdict(int)
    domain_coverage = defaultdict(int)
    
    total_docs = len(document_scores)
    print(f"\nBuilding co-occurrence matrix from {total_docs:,} documents...")
    print(f"Threshold: {threshold}")
    
    # Process each document
    for idx, doc in enumerate(document_scores):
        # Find high-confidence domains for this document
        high_conf_domains = [
            domain for domain, score in doc['domains'].items()
            if score >= threshold
        ]
        
        # Count individual domain coverage
        for domain in high_conf_domains:
            domain_coverage[domain] += 1
        
        # Count co-occurrences (all pairs)
        for i, domain1 in enumerate(high_conf_domains):
            for domain2 in high_conf_domains[i+1:]:
                # Store as sorted tuple to avoid duplicates
                pair = tuple(sorted([domain1, domain2]))
                cooccurrence[pair] += 1
        
        # Progress
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,}/{total_docs:,} documents", end='\r')
    
    print(f"  Processed {total_docs:,}/{total_docs:,} documents ✅")
    
    return {
        'cooccurrence': {f"{k[0]}|||{k[1]}": v for k, v in cooccurrence.items()},
        'domain_coverage': dict(domain_coverage)
    }


def compute_graph_statistics(graph_data: dict) -> dict:
    """Compute useful statistics about the graph"""
    cooccurrence = graph_data['cooccurrence']
    coverage = graph_data['domain_coverage']
    
    # Convert keys back to tuples for analysis
    edges = {tuple(k.split('|||')): v for k, v in cooccurrence.items()}
    
    stats = {
        'num_nodes': len(coverage),
        'num_edges': len(edges),
        'total_documents': sum(coverage.values()) / len(coverage) if coverage else 0,
        'top_domains': sorted(coverage.items(), key=lambda x: x[1], reverse=True)[:10],
        'top_edges': sorted(
            [(f"{k[0]} <-> {k[1]}", v) for k, v in edges.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10],
        'domain_degree': defaultdict(int)  # How many connections each domain has
    }
    
    # Compute degree (number of edges per domain)
    for (d1, d2), count in edges.items():
        stats['domain_degree'][d1] += 1
        stats['domain_degree'][d2] += 1
    
    stats['domain_degree'] = dict(sorted(
        stats['domain_degree'].items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    return stats


def main():
    print("=" * 60)
    print("DOMAIN CO-OCCURRENCE GRAPH BUILDER")
    print("=" * 60)
    print(f"\nThreshold: {CONFIDENCE_THRESHOLD}")
    print(f"Sources: {', '.join(PILE_SUBSETS)}")
    
    # Process each source
    all_graphs = {}
    
    for idx, source_name in enumerate(PILE_SUBSETS, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(PILE_SUBSETS)}] Processing {source_name}")
        print('='*60)
        
        try:
            # Load document scores
            doc_scores = load_document_scores(source_name)
            print(f"✅ Loaded {len(doc_scores):,} document scores")
            
            # Build graph
            graph_data = build_cooccurrence_matrix(doc_scores, CONFIDENCE_THRESHOLD)
            
            # Compute statistics
            stats = compute_graph_statistics(graph_data)
            
            # Combine
            graph_output = {
                'source': source_name,
                'threshold': CONFIDENCE_THRESHOLD,
                'num_documents': len(doc_scores),
                'graph': graph_data,
                'statistics': stats
            }
            
            all_graphs[source_name] = graph_output
            
            # Save individual graph
            output_file = os.path.join(RESULTS_DIR, f"{source_name}_graph.json")
            save_json(graph_output, output_file)
            
            # Print summary
            print(f"\n📊 Graph Summary:")
            print(f"  Nodes (domains): {stats['num_nodes']}")
            print(f"  Edges (co-occurrences): {stats['num_edges']}")
            print(f"\n  Top 5 domains by coverage:")
            for domain, count in stats['top_domains'][:5]:
                print(f"    {domain:40s}: {count:,} documents")
            
            print(f"\n  Top 5 co-occurrence pairs:")
            for pair, count in stats['top_edges'][:5]:
                print(f"    {pair:60s}: {count:,} documents")
            
        except Exception as e:
            print(f"❌ Error processing {source_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined graphs
    combined_file = os.path.join(RESULTS_DIR, "all_sources_graphs.json")
    save_json(all_graphs, combined_file)
    
    print("\n" + "="*60)
    print("GRAPH BUILDING COMPLETE")
    print("="*60)
    print(f"✅ Built graphs for {len(all_graphs)} sources")
    print(f"✅ Results saved to {RESULTS_DIR}/")
    print(f"  - Individual files: *_graph.json")
    print(f"  - Combined file: all_sources_graphs.json")
    print("\n🎯 Next step: Create visualizations of the domain graphs")


if __name__ == "__main__":
    main()
