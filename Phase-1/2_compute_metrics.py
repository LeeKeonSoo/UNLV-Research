"""
Step 2: Compute Domain Metrics (Simple Version - No Perplexity)

Simplified version that:
1. Uses TF-IDF prototypes from Step 1
2. Computes domain classification only (no perplexity)
3. Detects educational markers
4. Works offline without large models

Input:
  - outputs/concept_prototypes_tfidf.pkl
  - khan_k12_concepts/all_k12_concepts.json
  - tiny_textbooks_raw/*.json

Output:
  - outputs/khan_analysis.jsonl
  - outputs/tiny_textbooks_analysis.jsonl
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import jsonlines


# ==============================================================================
# Configuration
# ==============================================================================

# Paths
PROTOTYPES_PATH = "outputs/concept_prototypes_tfidf.pkl"
TAXONOMY_PATH = "outputs/khan_taxonomy.json"
KHAN_DATA_PATH = "khan_k12_concepts/all_k12_concepts.json"
TINY_TEXTBOOKS_DIR = "tiny_textbooks_raw"

# Output paths
OUTPUT_DIR = Path("outputs")
KHAN_OUTPUT = OUTPUT_DIR / "khan_analysis.jsonl"
TINY_OUTPUT = OUTPUT_DIR / "tiny_textbooks_analysis.jsonl"

# Parameters
TOP_K_DOMAINS = 5  # Number of top domain labels per paragraph
MIN_SIMILARITY = 0.1  # Lower threshold for TF-IDF (less strict than embeddings)
CHUNK_SIZE = 200  # Words per paragraph chunk


# ==============================================================================
# Load Models and Data
# ==============================================================================

def load_prototypes_and_vectorizer():
    """Load concept prototypes and TF-IDF vectorizer."""
    print("Loading concept prototypes (TF-IDF)...")
    with open(PROTOTYPES_PATH, 'rb') as f:
        data = pickle.load(f)

    prototypes = data['prototypes']
    vectorizer = data['vectorizer']

    print(f"✓ Loaded {len(prototypes)} concept prototypes")
    print(f"  Vector dimension: {len(list(prototypes.values())[0])}")

    return prototypes, vectorizer


# ==============================================================================
# Domain Classification
# ==============================================================================

def classify_domain(
    text: str,
    prototypes: Dict,
    vectorizer,
    top_k: int = 5,
    min_similarity: float = 0.1
) -> Dict[str, float]:
    """
    Classify text into domains using TF-IDF similarity to concept prototypes.

    Returns: {domain_id: similarity_score, ...} (top-k, soft labels)
    """
    # Vectorize query text
    try:
        query_vector = vectorizer.transform([text]).toarray()[0]
    except:
        # If text has no matching features, return empty
        return {}

    # Compute similarities to all prototypes
    similarities = {}
    for concept_id, prototype_vector in prototypes.items():
        # Reshape for sklearn
        query = query_vector.reshape(1, -1)
        prototype = prototype_vector.reshape(1, -1)

        similarity = cosine_similarity(query, prototype)[0][0]

        if similarity >= min_similarity:
            similarities[concept_id] = float(similarity)

    # Get top-k
    top_domains = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k])

    # Normalize to sum to 1 (soft probabilities)
    total = sum(top_domains.values())
    if total > 0:
        top_domains = {k: v / total for k, v in top_domains.items()}

    return top_domains


# ==============================================================================
# Quality Metrics (Simplified - No Perplexity)
# ==============================================================================

def detect_educational_markers(text: str) -> Dict[str, bool]:
    """
    Detect educational structure markers.

    Returns:
    {
        "has_examples": bool,
        "has_explanation": bool,
        "has_structure": bool
    }
    """
    text_lower = text.lower()

    example_markers = ["for example", "such as", "consider", "let's look at", "instance"]
    explanation_markers = ["because", "therefore", "this means", "as a result", "consequently"]
    structure_markers = ["first", "second", "third", "finally", "in summary", "in conclusion"]

    return {
        "has_examples": any(marker in text_lower for marker in example_markers),
        "has_explanation": any(marker in text_lower for marker in explanation_markers),
        "has_structure": any(marker in text_lower for marker in structure_markers)
    }


# ==============================================================================
# Text Chunking
# ==============================================================================

def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """
    Split text into paragraphs of ~chunk_size words.

    Strategy: Split by double newlines first, then by sentence if too long.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        words = para.split()
        if len(words) <= chunk_size:
            chunks.append(para)
        else:
            # Split long paragraphs by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_words = len(sentence.split())
                if current_length + sentence_words > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_words
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_words

            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return chunks


# ==============================================================================
# Dataset Processing
# ==============================================================================

def process_khan_academy(prototypes: Dict, vectorizer):
    """Process Khan Academy dataset and save analysis."""
    print("\n" + "="*60)
    print("Processing Khan Academy Dataset")
    print("="*60)

    with open(KHAN_DATA_PATH, 'r') as f:
        khan_data = json.load(f)

    results = []

    for doc in tqdm(khan_data, desc="Analyzing Khan concepts"):
        text = doc.get("content", "")
        if len(text.strip()) < 50:
            continue

        # Chunk into paragraphs
        chunks = chunk_text(text, CHUNK_SIZE)

        for chunk_id, chunk in enumerate(chunks):
            if len(chunk.split()) < 20:  # Skip very short chunks
                continue

            # Domain classification
            domain_labels = classify_domain(
                chunk, prototypes, vectorizer,
                top_k=TOP_K_DOMAINS, min_similarity=MIN_SIMILARITY
            )

            # Educational markers (no perplexity)
            markers = detect_educational_markers(chunk)

            # Save result
            results.append({
                "source": "khan_academy",
                "doc_id": doc.get("url", "unknown"),
                "subject": doc.get("subject", "Unknown"),
                "grade": doc.get("grade", "Unknown"),
                "title": doc.get("title", "Untitled"),
                "chunk_id": chunk_id,
                "text": chunk,
                "word_count": len(chunk.split()),
                "domain_labels": domain_labels,
                "educational_markers": markers
            })

    # Save to JSONL
    with jsonlines.open(KHAN_OUTPUT, 'w') as writer:
        writer.write_all(results)

    print(f"\n✓ Processed {len(results)} chunks from {len(khan_data)} Khan concepts")
    print(f"✓ Saved to {KHAN_OUTPUT}")


def process_tiny_textbooks(
    prototypes: Dict,
    vectorizer,
    max_batches: int = 5  # Default to 5 for testing
):
    """Process Tiny-Textbooks dataset and save analysis."""
    print("\n" + "="*60)
    print("Processing Tiny-Textbooks Dataset")
    print("="*60)

    batch_files = sorted(Path(TINY_TEXTBOOKS_DIR).glob("batch_*.json"))

    if max_batches:
        batch_files = batch_files[:max_batches]
        print(f"Processing first {max_batches} batches for testing")

    results = []
    total_docs = 0

    for batch_file in tqdm(batch_files, desc="Processing batches"):
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)

        for doc in batch_data:
            total_docs += 1
            text = doc.get("text", "")

            if len(text.strip()) < 100:
                continue

            # Chunk into paragraphs
            chunks = chunk_text(text, CHUNK_SIZE)

            for chunk_id, chunk in enumerate(chunks):
                if len(chunk.split()) < 20:
                    continue

                # Domain classification
                domain_labels = classify_domain(
                    chunk, prototypes, vectorizer,
                    top_k=TOP_K_DOMAINS, min_similarity=MIN_SIMILARITY
                )

                # Educational markers
                markers = detect_educational_markers(chunk)

                # Save result
                results.append({
                    "source": "tiny_textbooks",
                    "doc_id": doc.get("id", "unknown"),
                    "batch_file": batch_file.name,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "word_count": len(chunk.split()),
                    "domain_labels": domain_labels,
                    "educational_markers": markers
                })

    # Save to JSONL
    with jsonlines.open(TINY_OUTPUT, 'w') as writer:
        writer.write_all(results)

    print(f"\n✓ Processed {len(results)} chunks from {total_docs} documents")
    print(f"✓ Saved to {TINY_OUTPUT}")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DATASET ANALYSIS (Simplified - Domain Classification Only)")
    print("="*60)

    # Load models and data
    prototypes, vectorizer = load_prototypes_and_vectorizer()

    # Process Khan Academy
    process_khan_academy(prototypes, vectorizer)

    # Process Tiny-Textbooks (5 batches for testing)
    # Set max_batches=None for full run
    process_tiny_textbooks(
        prototypes, vectorizer,
        max_batches=5  # Quick test - change to None for full run
    )

    print("\n" + "="*60)
    print("✓ Metrics computation complete!")
    print("="*60)
    print(f"\nNext step: Run 3_build_dashboard.py to visualize results")
    print("\nNote: Perplexity not computed in this simplified version")
    print("      Focus is on domain classification and educational markers")


if __name__ == "__main__":
    main()
