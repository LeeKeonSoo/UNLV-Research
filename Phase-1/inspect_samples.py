"""
Inspect sample documents from Tiny-Textbooks dataset
Understand why clustering produced unexpected results
"""

import json
import random
from pathlib import Path
from collections import defaultdict


def load_sample_documents(raw_dir, num_samples=50):
    """Load sample documents from batch files"""
    raw_path = Path(raw_dir)
    batch_files = sorted(raw_path.glob("batch_*.json"))

    if not batch_files:
        print(f"❌ No batch files found in {raw_dir}")
        return []

    print(f"📂 Found {len(batch_files)} batch files")

    # Sample from random batches
    sample_batches = random.sample(batch_files, min(5, len(batch_files)))

    all_docs = []
    for batch_file in sample_batches:
        with open(batch_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
            all_docs.extend(docs)

    # Random sample
    samples = random.sample(all_docs, min(num_samples, len(all_docs)))

    return samples


def inspect_document(doc, doc_idx):
    """Pretty print a document"""
    print("\n" + "=" * 80)
    print(f"DOCUMENT #{doc_idx}")
    print("=" * 80)

    print(f"\n📄 ID: {doc['id']}")
    print(f"📏 Length: {doc['char_count']:,} chars, {doc['word_count']:,} words")

    # Show first 1000 characters
    text_preview = doc['text'][:1000]
    print(f"\n📝 Content Preview (first 1000 chars):")
    print("-" * 80)
    print(text_preview)
    print("-" * 80)

    # Extract key characteristics
    lines = doc['text'].split('\n')
    first_lines = [line.strip() for line in lines[:10] if line.strip()]

    print(f"\n🔍 First {min(10, len(first_lines))} non-empty lines:")
    for i, line in enumerate(first_lines, 1):
        preview = line[:100] + "..." if len(line) > 100 else line
        print(f"   {i}. {preview}")


def analyze_corpus_characteristics(docs):
    """Analyze overall corpus characteristics"""
    print("\n" + "=" * 80)
    print("CORPUS CHARACTERISTICS ANALYSIS")
    print("=" * 80)

    # Length distribution
    lengths = [doc['char_count'] for doc in docs]
    print(f"\n📏 Document Length Statistics:")
    print(f"   Min: {min(lengths):,} chars")
    print(f"   Max: {max(lengths):,} chars")
    print(f"   Mean: {sum(lengths)/len(lengths):,.0f} chars")
    print(f"   Median: {sorted(lengths)[len(lengths)//2]:,} chars")

    # Common patterns
    print(f"\n🔍 Common Patterns:")

    # Check for "Lesson" keyword
    lesson_count = sum(1 for doc in docs if 'lesson' in doc['text'].lower()[:500])
    print(f"   Contains 'lesson' in first 500 chars: {lesson_count}/{len(docs)} ({lesson_count/len(docs)*100:.1f}%)")

    # Check for educational markers
    education_markers = {
        'chapter': sum(1 for doc in docs if 'chapter' in doc['text'].lower()[:500]),
        'section': sum(1 for doc in docs if 'section' in doc['text'].lower()[:500]),
        'exercise': sum(1 for doc in docs if 'exercise' in doc['text'].lower()[:500]),
        'example': sum(1 for doc in docs if 'example' in doc['text'].lower()[:500]),
        'tutorial': sum(1 for doc in docs if 'tutorial' in doc['text'].lower()[:500]),
    }

    print(f"\n📚 Educational Markers (in first 500 chars):")
    for marker, count in sorted(education_markers.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(docs) * 100
        print(f"   '{marker}': {count}/{len(docs)} ({pct:.1f}%)")

    # Check for code/technical content
    code_markers = {
        'function': sum(1 for doc in docs if 'function' in doc['text'].lower()[:500]),
        'class': sum(1 for doc in docs if 'class ' in doc['text'].lower()[:500]),
        'import': sum(1 for doc in docs if 'import' in doc['text'].lower()[:500]),
        'array': sum(1 for doc in docs if 'array' in doc['text'].lower()[:500]),
    }

    print(f"\n💻 Code Markers:")
    for marker, count in sorted(code_markers.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(docs) * 100
        print(f"   '{marker}': {count}/{len(docs)} ({pct:.1f}%)")


def check_graph_domain_samples(graph_file, raw_dir):
    """Sample documents from specific graph clusters"""
    print("\n" + "=" * 80)
    print("CHECKING ACTUAL CLUSTER CONTENTS")
    print("=" * 80)

    # Load graph
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph = json.load(f)

    nodes = graph['nodes']

    # Get Level 2 domains
    level2_nodes = [n for n in nodes if n['level'] == 2]

    # Load all documents
    raw_path = Path(raw_dir)
    batch_files = sorted(raw_path.glob("batch_*.json"))

    all_docs = []
    for batch_file in batch_files[:5]:  # Only first 5 batches for speed
        with open(batch_file, 'r', encoding='utf-8') as f:
            all_docs.extend(json.load(f))

    print(f"\n📂 Loaded {len(all_docs):,} documents from first 5 batches")

    # Check top 3 domains
    top_domains = sorted(level2_nodes, key=lambda x: x['document_count'], reverse=True)[:3]

    for domain_node in top_domains:
        print(f"\n" + "=" * 80)
        print(f"DOMAIN: {domain_node['name']}")
        print("=" * 80)
        print(f"Total documents: {domain_node['document_count']:,}")

        # Get sample document indices
        sample_indices = domain_node['document_indices'][:5]  # First 5

        print(f"\nSampling first 5 documents from this domain:")

        for i, doc_idx in enumerate(sample_indices, 1):
            if doc_idx >= len(all_docs):
                print(f"\n   Document #{doc_idx} - OUT OF RANGE")
                continue

            doc = all_docs[doc_idx]
            text_preview = doc['text'][:300].replace('\n', ' ')

            print(f"\n   {i}. Doc #{doc_idx} ({doc['word_count']} words)")
            print(f"      Preview: {text_preview}...")


def main():
    """Main inspection routine"""
    print("=" * 80)
    print("TINY-TEXTBOOKS DATASET INSPECTION")
    print("=" * 80)

    raw_dir = "tiny_textbooks_raw"
    graph_file = "graphs/deep_hierarchy.json"

    # Check if files exist
    if not Path(raw_dir).exists():
        print(f"❌ Raw data directory not found: {raw_dir}")
        return 1

    if not Path(graph_file).exists():
        print(f"❌ Graph file not found: {graph_file}")
        return 1

    # Part 1: Random sampling
    print("\n" + "🎲 " * 40)
    print("PART 1: RANDOM DOCUMENT SAMPLING")
    print("🎲 " * 40)

    samples = load_sample_documents(raw_dir, num_samples=5)

    if not samples:
        return 1

    for i, doc in enumerate(samples, 1):
        inspect_document(doc, i)

    # Part 2: Corpus analysis
    print("\n" + "📊 " * 40)
    print("PART 2: CORPUS CHARACTERISTICS")
    print("📊 " * 40)

    more_samples = load_sample_documents(raw_dir, num_samples=100)
    analyze_corpus_characteristics(more_samples)

    # Part 3: Graph cluster sampling
    print("\n" + "🔍 " * 40)
    print("PART 3: CLUSTER CONTENT INSPECTION")
    print("🔍 " * 40)

    check_graph_domain_samples(graph_file, raw_dir)

    print("\n" + "=" * 80)
    print("✅ INSPECTION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
