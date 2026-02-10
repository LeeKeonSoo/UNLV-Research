"""
Step 2: Compute Domain and Quality Metrics

This script:
1. Loads concept prototypes from Step 1
2. Processes Khan Academy and Tiny-Textbooks datasets
3. For each paragraph, computes:
   - Domain coverage (multi-label classification via embedding similarity)
   - Quality metrics (perplexity, educational markers)
4. Saves annotated datasets for visualization

Input:
  - outputs/concept_prototypes.pkl
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
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import textstat
from tqdm import tqdm
import jsonlines


# ==============================================================================
# Configuration
# ==============================================================================

# Paths
PROTOTYPES_PATH = "outputs/concept_prototypes.pkl"
TAXONOMY_PATH = "outputs/khan_taxonomy.json"
KHAN_DATA_PATH = "khan_k12_concepts/all_k12_concepts.json"
TINY_TEXTBOOKS_DIR = "tiny_textbooks_raw"

# Output paths
OUTPUT_DIR = Path("outputs")
KHAN_OUTPUT = OUTPUT_DIR / "khan_analysis.jsonl"
TINY_OUTPUT = OUTPUT_DIR / "tiny_textbooks_analysis.jsonl"

# Parameters
TOP_K_DOMAINS = 5  # Number of top domain labels per paragraph
MIN_SIMILARITY = 0.3  # Minimum cosine similarity threshold
CHUNK_SIZE = 200  # Words per paragraph chunk


# ==============================================================================
# Load Models and Data
# ==============================================================================

def load_prototypes_and_models():
    """Load concept prototypes and embedding model."""
    print("Loading concept prototypes...")
    with open(PROTOTYPES_PATH, 'rb') as f:
        prototypes = pickle.load(f)

    print(f"✓ Loaded {len(prototypes)} concept prototypes")

    print("\nLoading embedding model (SentenceTransformer)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Embedding model loaded")

    print("\nLoading language model for perplexity (GPT-2)...")
    lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
    lm_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    lm_model.eval()
    if torch.cuda.is_available():
        lm_model = lm_model.cuda()
    print("✓ Language model loaded")

    return prototypes, embedding_model, lm_model, lm_tokenizer


# ==============================================================================
# Domain Classification
# ==============================================================================

def classify_domain(
    text: str,
    prototypes: Dict,
    embedding_model: SentenceTransformer,
    top_k: int = 5,
    min_similarity: float = 0.3
) -> Dict[str, float]:
    """
    Classify text into domains using concept prototype similarity.

    Returns: {domain_id: similarity_score, ...} (top-k, soft labels)
    """
    # Embed query text
    query_embedding = embedding_model.encode(text, convert_to_tensor=True)

    # Compute similarities to all prototypes
    similarities = {}
    for concept_id, prototype_embedding in prototypes.items():
        prototype_tensor = torch.tensor(prototype_embedding)
        if torch.cuda.is_available():
            prototype_tensor = prototype_tensor.cuda()

        similarity = util.cos_sim(query_embedding, prototype_tensor).item()

        if similarity >= min_similarity:
            similarities[concept_id] = similarity

    # Get top-k
    top_domains = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k])

    # Normalize to sum to 1 (soft probabilities)
    total = sum(top_domains.values())
    if total > 0:
        top_domains = {k: v / total for k, v in top_domains.items()}

    return top_domains


# ==============================================================================
# Quality Metrics
# ==============================================================================

def compute_perplexity(
    text: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    max_length: int = 512
) -> float:
    """
    Compute perplexity using GPT-2.
    Lower perplexity = more natural text.
    """
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = torch.exp(loss).item()
        return perplexity

    except Exception as e:
        print(f"Warning: Perplexity computation failed: {e}")
        return float('inf')


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


def compute_quality_metrics(
    text: str,
    lm_model: GPT2LMHeadModel,
    lm_tokenizer: GPT2TokenizerFast
) -> Dict:
    """Compute all quality metrics for a text."""
    return {
        "perplexity": compute_perplexity(text, lm_model, lm_tokenizer),
        **detect_educational_markers(text)
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

def process_khan_academy(
    prototypes: Dict,
    embedding_model: SentenceTransformer,
    lm_model: GPT2LMHeadModel,
    lm_tokenizer: GPT2TokenizerFast
):
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
                chunk, prototypes, embedding_model,
                top_k=TOP_K_DOMAINS, min_similarity=MIN_SIMILARITY
            )

            # Quality metrics
            quality = compute_quality_metrics(chunk, lm_model, lm_tokenizer)

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
                "quality": quality
            })

    # Save to JSONL
    with jsonlines.open(KHAN_OUTPUT, 'w') as writer:
        writer.write_all(results)

    print(f"\n✓ Processed {len(results)} chunks from {len(khan_data)} Khan concepts")
    print(f"✓ Saved to {KHAN_OUTPUT}")


def process_tiny_textbooks(
    prototypes: Dict,
    embedding_model: SentenceTransformer,
    lm_model: GPT2LMHeadModel,
    lm_tokenizer: GPT2TokenizerFast,
    max_batches: int = None  # Limit batches for testing
):
    """Process Tiny-Textbooks dataset and save analysis."""
    print("\n" + "="*60)
    print("Processing Tiny-Textbooks Dataset")
    print("="*60)

    batch_files = sorted(Path(TINY_TEXTBOOKS_DIR).glob("batch_*.json"))

    if max_batches:
        batch_files = batch_files[:max_batches]
        print(f"Processing first {max_batches} batches only (for testing)")

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
                    chunk, prototypes, embedding_model,
                    top_k=TOP_K_DOMAINS, min_similarity=MIN_SIMILARITY
                )

                # Quality metrics
                quality = compute_quality_metrics(chunk, lm_model, lm_tokenizer)

                # Save result
                results.append({
                    "source": "tiny_textbooks",
                    "doc_id": doc.get("id", "unknown"),
                    "batch_file": batch_file.name,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "word_count": len(chunk.split()),
                    "domain_labels": domain_labels,
                    "quality": quality
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

    # Load models and data
    prototypes, embedding_model, lm_model, lm_tokenizer = load_prototypes_and_models()

    # Process Khan Academy
    process_khan_academy(prototypes, embedding_model, lm_model, lm_tokenizer)

    # Process Tiny-Textbooks (all batches - will take ~1-2 hours)
    # For testing, set max_batches=5
    process_tiny_textbooks(
        prototypes, embedding_model, lm_model, lm_tokenizer,
        max_batches=None  # Set to 5 for quick test
    )

    print("\n" + "="*60)
    print("✓ Metrics computation complete!")
    print("="*60)
    print(f"\nNext step: Run 3_build_dashboard.py to visualize results")


if __name__ == "__main__":
    main()
