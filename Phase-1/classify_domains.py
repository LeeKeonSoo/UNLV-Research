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


def classify_documents(texts: list, classifier, source_name: str) -> dict:
    """
    Classify a batch of documents with detailed progress tracking
    
    Args:
        texts: List of text strings
        classifier: HuggingFace pipeline classifier
        source_name: Name of the source being processed
    
    Returns:
        Dictionary with domain statistics
    """
    domain_scores = defaultdict(list)
    total_docs = len(texts)
    
    print(f"\n{'='*60}")
    print(f"Classifying {total_docs:,} documents from {source_name}")
    print('='*60)
    
    # Truncate all texts first
    truncated_texts = [text[:TEXT_MAX_LENGTH] for text in texts]
    
    # Process in batches for GPU efficiency
    batch_size = BATCH_SIZE
    num_batches = (total_docs + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_docs)
        batch_texts = truncated_texts[start_idx:end_idx]
        
        try:
            # Batch classification (much faster on GPU)
            results = classifier(batch_texts, ALL_DOMAINS, multi_label=True, batch_size=batch_size)
            
            # Handle single result vs list of results
            if not isinstance(results, list):
                results = [results]
            
            # Store scores
            for i, result in enumerate(results):
                for domain, score in zip(result['labels'], result['scores']):
                    domain_scores[domain].append(score)
                
                # Progress update
                doc_idx = start_idx + i
                if (doc_idx + 1) % 10 == 0 or doc_idx == total_docs - 1:
                    progress = (doc_idx + 1) / total_docs
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    # Show top domain for current doc
                    top_domain = result['labels'][0] if result['labels'] else "unknown"
                    top_score = result['scores'][0] if result['scores'] else 0.0
                    
                    print(f'\r  Progress: |{bar}| {progress*100:.1f}% '
                          f'({doc_idx + 1:,}/{total_docs:,}) | '
                          f'Current: {top_domain[:30]:30s} ({top_score:.2f})',
                          end='', flush=True)
        
        except Exception as e:
            print(f"\n⚠️  Error in batch {batch_idx}: {e}")
            continue
    
    print()  # New line after progress
    
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
    print(f"\nModel: {CLASSIFIER_MODEL}")
    print(f"Sources: {', '.join(PILE_SUBSETS)}")
    print(f"Domains: {len(ALL_DOMAINS)} domains across {len(DOMAINS)} categories")
    print(f"Mode: {'All documents' if USE_ALL_DOCUMENTS else f'{SAMPLE_SIZE_PER_SOURCE:,} samples per source'}")
    
    # Load classifier
    print("\n" + "="*60)
    print("Loading classifier...")
    print("="*60)
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
    total_docs_processed = 0
    
    for idx, source_name in enumerate(PILE_SUBSETS, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(PILE_SUBSETS)}] Processing {source_name}")
        print('='*60)
        
        # Load documents
        filepath = os.path.join(DATA_DIR, f"{source_name}.jsonl")
        print(f"Loading from {filepath}...")
        
        try:
            # Load all or sample
            if USE_ALL_DOCUMENTS:
                documents = load_jsonl_sample(filepath, sample_size=None)
            else:
                documents = load_jsonl_sample(filepath, sample_size=SAMPLE_SIZE_PER_SOURCE)
            
            texts = [doc['text'] for doc in documents]
            num_docs = len(texts)
            total_docs_processed += num_docs
            
            print(f"✅ Loaded {num_docs:,} documents")
            
            # Classify
            stats = classify_documents(texts, classifier, source_name)
            
            # Add metadata
            results = {
                'source': source_name,
                'num_documents': num_docs,
                'sample_size': SAMPLE_SIZE_PER_SOURCE if not USE_ALL_DOCUMENTS else num_docs,
                'domains': stats
            }
            
            all_results[source_name] = results
            
            # Save individual source results
            output_file = os.path.join(RESULTS_DIR, f"{source_name}_classification.json")
            save_json(results, output_file)
            
            # Print summary
            print("\n📊 Top 5 domains by mean confidence:")
            sorted_domains = sorted(stats.items(), key=lambda x: x[1]['mean_score'], reverse=True)
            for domain, domain_stats in sorted_domains[:5]:
                print(f"  {domain:40s}: {domain_stats['mean_score']:.3f} "
                      f"(high conf: {domain_stats['high_confidence_ratio']:.1%}, "
                      f"docs: {domain_stats['doc_count']:,})")
            
        except Exception as e:
            print(f"❌ Error processing {source_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    combined_file = os.path.join(RESULTS_DIR, "all_sources_classification.json")
    save_json(all_results, combined_file)
    
    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE")
    print("="*60)
    print(f"✅ Processed {total_docs_processed:,} total documents")
    print(f"✅ Results saved to {RESULTS_DIR}/")
    print(f"  - Individual files: {len(all_results)} source files")
    print(f"  - Combined file: all_sources_classification.json")
    print("\n🎯 Next step: Run visualize_sources.py to create visualizations")


if __name__ == "__main__":
    main()