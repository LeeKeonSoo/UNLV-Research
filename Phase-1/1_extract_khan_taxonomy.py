"""
Step 1: Extract Khan Academy Taxonomy (Simple Version - No External Models)

This version uses TF-IDF instead of SentenceTransformers to avoid network issues.
Suitable for offline environments or when HuggingFace models can't be downloaded.

Input: khan_k12_concepts/all_k12_concepts.json
Output:
  - outputs/khan_taxonomy.json (hierarchical structure)
  - outputs/concept_prototypes_tfidf.pkl (TF-IDF vectors)
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def load_khan_data(data_path: str) -> List[Dict]:
    """Load Khan Academy concepts from JSON."""
    print(f"Loading Khan Academy data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} concepts")
    return data


def extract_taxonomy(data: List[Dict]) -> Dict:
    """
    Extract hierarchical taxonomy structure.

    Returns:
    {
        "Math - 4th Grade": {
            "grade": "4",
            "subject": "Math",
            "concepts": ["Place value", "Addition", ...],
            "concept_map": {"Place value": {title, url, ...}, ...}
        },
        ...
    }
    """
    print("\nExtracting taxonomy structure...")
    taxonomy = defaultdict(lambda: {
        "grade": None,
        "subject": None,
        "concepts": [],
        "concept_map": {}
    })

    for doc in tqdm(data, desc="Processing concepts"):
        subject = doc.get("subject", "Unknown")
        grade = doc.get("grade", "Unknown")
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")

        key = subject

        if taxonomy[key]["grade"] is None:
            taxonomy[key]["grade"] = grade
            taxonomy[key]["subject"] = subject.split(" - ")[0] if " - " in subject else subject

        taxonomy[key]["concepts"].append(title)
        taxonomy[key]["concept_map"][title] = {
            "title": title,
            "content": content,
            "url": doc.get("url", ""),
            "word_count": doc.get("word_count", 0)
        }

    # Convert defaultdict to regular dict
    taxonomy = dict(taxonomy)

    print(f"✓ Extracted {len(taxonomy)} subject-grade categories")
    for subject, info in list(taxonomy.items())[:5]:
        print(f"  - {subject}: {len(info['concepts'])} concepts")

    return taxonomy


def build_concept_prototypes_tfidf(taxonomy: Dict) -> tuple:
    """
    Create concept prototype vectors using TF-IDF.

    This is simpler than SentenceTransformers but works offline.
    Returns mapping: concept_id → TF-IDF vector
    """
    print(f"\nBuilding concept prototypes with TF-IDF...")

    # Collect all texts and IDs
    concept_texts = []
    concept_ids = []

    for subject, info in taxonomy.items():
        for concept_title, concept_data in info["concept_map"].items():
            concept_id = f"{subject}::{concept_title}"
            content = concept_data["content"]

            # Skip empty content
            if not content or len(content.strip()) < 50:
                continue

            concept_texts.append(content)
            concept_ids.append(concept_id)

    print(f"Processing {len(concept_texts)} concepts...")

    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=300,  # Keep vectors reasonably sized
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2  # Ignore very rare terms
    )

    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(concept_texts)

    # Create concept → vector mapping
    prototypes = {}
    for concept_id, vector in zip(concept_ids, tfidf_matrix):
        prototypes[concept_id] = vector.toarray()[0]  # Convert sparse to dense

    print(f"✓ Created {len(prototypes)} concept prototypes")
    print(f"  Vector dimension: {len(prototypes[concept_ids[0]])}")

    return prototypes, vectorizer


def save_outputs(taxonomy: Dict, prototypes: Dict, vectorizer, output_dir: str):
    """Save taxonomy and prototypes to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save taxonomy as JSON
    taxonomy_path = output_path / "khan_taxonomy.json"
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved taxonomy to {taxonomy_path}")

    # Save prototypes as pickle
    prototypes_path = output_path / "concept_prototypes_tfidf.pkl"
    with open(prototypes_path, 'wb') as f:
        pickle.dump({'prototypes': prototypes, 'vectorizer': vectorizer}, f)
    print(f"✓ Saved concept prototypes to {prototypes_path}")

    # Save metadata
    metadata = {
        "num_subjects": len(taxonomy),
        "num_concepts": len(prototypes),
        "vector_dim": len(list(prototypes.values())[0]) if prototypes else 0,
        "method": "TF-IDF (sklearn)",
        "subjects": list(taxonomy.keys())
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")


def main():
    # Configuration
    DATA_PATH = "khan_k12_concepts/all_k12_concepts.json"
    OUTPUT_DIR = "outputs"

    print("="*60)
    print("KHAN ACADEMY TAXONOMY EXTRACTION (TF-IDF Version)")
    print("="*60)

    # Step 1: Load data
    data = load_khan_data(DATA_PATH)

    # Step 2: Extract taxonomy
    taxonomy = extract_taxonomy(data)

    # Step 3: Build concept prototypes (TF-IDF)
    prototypes, vectorizer = build_concept_prototypes_tfidf(taxonomy)

    # Step 4: Save outputs
    save_outputs(taxonomy, prototypes, vectorizer, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✓ Khan Academy taxonomy extraction complete!")
    print("="*60)
    print(f"\nNext step: Run 2_compute_metrics_simple.py to analyze datasets")
    print("\nNote: Using TF-IDF vectors instead of SentenceTransformers")
    print("      This works offline but may be less accurate than embeddings.")


if __name__ == "__main__":
    main()
