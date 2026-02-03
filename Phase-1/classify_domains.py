"""
Domain Classification Script

Runs BART zero-shot classification on sampled data and saves results.
This separates the expensive classification step from visualization.
"""

import os
import json
from collections import defaultdict
from transformers import pipeline
import numpy as np

from config import *
from utils import load_jsonl_sample, save_json, print_progress


def classify_documents(texts: list, classifier) -> dict:
    """
    Classify a batch of documents
    
    Args:
        texts: List of text strings
        classifier: HuggingFace pipeline classifier
    
    Returns:
        Dictionary with domain statistics
    """
    domain_scores = defaultdict(list)
    
    print(f"Classifying {len(texts)} documents...")
    
    for i, text in enumerate(texts):
        # Truncate text for classification
        text_sample = text[:TEXT_MAX_LENGTH]
        
        # Classify
        result = classifier(text_sample, ALL_DOMAINS, multi_label=True)
        
        # Store scores for each domain
        for domain, score in zip(result['labels'], result['scores']):
            domain_scores[domain].append(score)
        
        # Progress
        if (i + 1) % 100 == 0:
            print_progress(i + 1, len(texts), prefix="Classifying")
    
    # Aggregate statistics
    stats = {}
    for domain, scores in domain_scores.items():
        stats[domain] = {
            'mean_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'std_score': float(np.std(scores)),
            'doc_count': len(scores),
            'high_confidence_count': int(sum(1 for s in scores if s > CONFIDENCE_THRESHOLD)),
            'high_confidence_ratio': float(sum(1 for s in scores if s > CONFIDENCE_THRESHOLD) / len(scores))
        }
    
    return stats


def main():
    print("=" * 60)
    print("DOMAIN CLASSIFICATION")
    print("=" * 60)
    print(f"\nSources: {', '.join(PILE_SUBSETS)}")
    print(f"Domains: {len(ALL_DOMAINS)} domains across {len(DOMAINS)} categories")
    print(f"Sample size per source: {SAMPLE_SIZE_PER_SOURCE:,} documents")
    print(f"Total documents to classify: {SAMPLE_SIZE_PER_SOURCE * len(PILE_SUBSETS):,}\n")
    
    # Load classifier
    print("Loading BART classifier...")
    classifier = pipeline(
        "zero-shot-classification",
        model=CLASSIFIER_MODEL,
        device=CLASSIFIER_DEVICE
    )
    print("✅ Classifier loaded\n")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Process each source
    all_results = {}
    
    for source_name in PILE_SUBSETS:
        print(f"\n{'='*60}")
        print(f"Processing {source_name}")
        print('='*60)
        
        # Load sample
        filepath = os.path.join(DATA_DIR, f"{source_name}.jsonl")
        print(f"Loading from {filepath}...")
        
        try:
            documents = load_jsonl_sample(filepath, sample_size=SAMPLE_SIZE_PER_SOURCE)
            texts = [doc['text'] for doc in documents]
            print(f"✅ Loaded {len(texts):,} documents\n")
            
            # Classify
            stats = classify_documents(texts, classifier)
            
            # Add metadata
            results = {
                'source': source_name,
                'num_documents': len(texts),
                'sample_size': SAMPLE_SIZE_PER_SOURCE,
                'domains': stats
            }
            
            all_results[source_name] = results
            
            # Save individual source results
            output_file = os.path.join(RESULTS_DIR, f"{source_name}_classification.json")
            save_json(results, output_file)
            
            # Print summary
            print("\nTop 5 domains by mean confidence:")
            sorted_domains = sorted(stats.items(), key=lambda x: x[1]['mean_score'], reverse=True)
            for domain, domain_stats in sorted_domains[:5]:
                print(f"  {domain:40s}: {domain_stats['mean_score']:.3f} "
                      f"(high conf: {domain_stats['high_confidence_ratio']:.1%})")
            
        except Exception as e:
            print(f"❌ Error processing {source_name}: {e}")
            continue
    
    # Save combined results
    combined_file = os.path.join(RESULTS_DIR, "all_sources_classification.json")
    save_json(all_results, combined_file)
    
    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE")
    print("="*60)
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"- Individual files: {len(all_results)} source files")
    print(f"- Combined file: all_sources_classification.json")
    print("\nNext step: Run visualize_sources.py or visualize_global.py")


if __name__ == "__main__":
    main()
