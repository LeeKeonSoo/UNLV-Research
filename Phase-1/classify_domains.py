"""
Domain Classification Script - OPTIMIZED VERSION

Uses HuggingFace Dataset for efficient GPU batch processing.
"""

import os
import json
from collections import defaultdict
from transformers import pipeline
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from config import *
from utils import load_jsonl_sample, save_json


def classify_documents(texts: list, classifier, source_name: str) -> tuple:
    """
    Classify documents using Dataset for efficient GPU batching
    
    Args:
        texts: List of text strings
        classifier: HuggingFace pipeline classifier
        source_name: Name of the source being processed
    
    Returns:
        Tuple of (stats dict, document_scores list)
    """
    total_docs = len(texts)
    print(f"\n{'='*60}")
    print(f"Classifying {total_docs:,} documents from {source_name}")
    print('='*60)
    
    # Truncate texts
    truncated_texts = [text[:TEXT_MAX_LENGTH] for text in texts]
    
    # Create HuggingFace Dataset
    print("Creating dataset...")
    dataset = Dataset.from_dict({"text": truncated_texts})
    
    # Initialize storage
    document_scores = []
    domain_scores = defaultdict(list)
    
    print(f"Processing with batch_size={BATCH_SIZE}...")
    
    # Process with dataset batching (MUCH faster on GPU)
    for idx, batch in enumerate(tqdm(
        dataset.iter(batch_size=BATCH_SIZE),
        total=(total_docs + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Classifying"
    )):
        try:
            # Batch classification
            results = classifier(
                batch['text'],
                ALL_DOMAINS,
                multi_label=True
            )
            
            # Handle single result vs list
            if not isinstance(results, list):
                results = [results]
            
            # Store per-document scores
            for i, result in enumerate(results):
                doc_idx = idx * BATCH_SIZE + i
                
                # Create score dict for this document
                doc_scores = {}
                for domain, score in zip(result['labels'], result['scores']):
                    doc_scores[domain] = float(score)
                    domain_scores[domain].append(score)
                
                document_scores.append({
                    'doc_id': doc_idx,
                    'domains': doc_scores
                })
        
        except Exception as e:
            print(f"\n⚠️  Error in batch {idx}: {e}")
            continue
    
    print("\n✅ Classification complete")
    
    # Aggregate statistics
    print("Computing statistics...")
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
    
    return stats, document_scores


def main():
    print("=" * 60)
    print("DOMAIN CLASSIFICATION (OPTIMIZED)")
    print("=" * 60)
    print(f"\nModel: {CLASSIFIER_MODEL}")
    print(f"Sources: {', '.join(PILE_SUBSETS)}")
    print(f"Domains: {len(ALL_DOMAINS)} domains across {len(DOMAINS)} categories")
    print(f"Mode: {'All documents' if USE_ALL_DOCUMENTS else f'{SAMPLE_SIZE_PER_SOURCE:,} samples per source'}")
    print(f"Batch size: {BATCH_SIZE}")
    
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
            stats, doc_scores = classify_documents(texts, classifier, source_name)
            
            # Add metadata
            results = {
                'source': source_name,
                'num_documents': num_docs,
                'sample_size': SAMPLE_SIZE_PER_SOURCE if not USE_ALL_DOCUMENTS else num_docs,
                'domains': stats
            }
            
            all_results[source_name] = results
            
            # Save individual source results (aggregated stats)
            output_file = os.path.join(RESULTS_DIR, f"{source_name}_classification.json")
            save_json(results, output_file)
            
            # Save per-document scores separately (for graph construction)
            doc_scores_file = os.path.join(RESULTS_DIR, f"{source_name}_document_scores.json")
            save_json(doc_scores, doc_scores_file)
            print(f"💾 Saved document-level scores to {doc_scores_file}")
            
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
    print("\n🎯 Next step: Run build_cooccurrence_graph.py to build domain graphs")


if __name__ == "__main__":
    main()
