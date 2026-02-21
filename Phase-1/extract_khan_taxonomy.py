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
    Extract course-level taxonomy structure.

    Groups lessons by (subject, course) so each prototype represents
    one course (e.g. "Math::Algebra 1"), not one lesson.

    Returns:
    {
        "Math::Algebra 1": {
            "subject": "Math",
            "course": "Algebra 1",
            "grade": "8-9",
            "texts": ["lesson content 1", "lesson content 2", ...],
            "lesson_count": 42
        },
        ...
    }
    """
    print("\nExtracting taxonomy structure (course-level)...")
    def _new_entry():
        return {"subject": "", "course": "", "grade": "", "texts": []}

    taxonomy = defaultdict(_new_entry)

    for doc in tqdm(data, desc="Processing concepts"):
        subject = doc.get("subject", "Unknown")
        course = doc.get("course", doc.get("subject", "Unknown"))
        grade = doc.get("grade", "Unknown")
        content = doc.get("content", "")

        key = f"{subject}::{course}"

        if taxonomy[key]["subject"] == "":
            taxonomy[key]["subject"] = subject
            taxonomy[key]["course"] = course
            taxonomy[key]["grade"] = grade

        if content and len(content.strip()) >= 50:
            taxonomy[key]["texts"].append(content)

    # Convert defaultdict to regular dict and add lesson_count
    taxonomy = dict(taxonomy)
    for info in taxonomy.values():
        info["lesson_count"] = len(info["texts"])

    print(f"✓ Extracted {len(taxonomy)} course-level categories")
    for key, info in list(taxonomy.items())[:5]:
        print(f"  - {key}: {info['lesson_count']} lessons")

    return taxonomy


def build_concept_prototypes_tfidf(taxonomy: Dict) -> tuple:
    """
    Create one TF-IDF prototype vector per course.

    All lesson texts within the same course are concatenated into
    a single document before vectorization, so each concept_id like
    "Math::Algebra 1" gets one representative vector (~30-50 total).
    """
    print(f"\nBuilding concept prototypes with TF-IDF (course-level)...")

    concept_texts = []
    concept_ids = []

    for concept_id, info in taxonomy.items():
        texts = info.get("texts", [])
        if not texts:
            continue
        # Concatenate all lesson texts for this course into one document
        combined = " ".join(texts)
        concept_texts.append(combined)
        concept_ids.append(concept_id)

    print(f"Processing {len(concept_texts)} course-level prototypes...")

    vectorizer = TfidfVectorizer(
        max_features=300,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,  # Each course document is unique; don't throw away terms
    )

    tfidf_matrix = vectorizer.fit_transform(concept_texts)

    dense_matrix = np.asarray(tfidf_matrix.todense())
    prototypes = {concept_id: dense_matrix[i] for i, concept_id in enumerate(concept_ids)}

    print(f"✓ Created {len(prototypes)} concept prototypes")
    print(f"  Vector dimension: {len(prototypes[concept_ids[0]])}")
    print(f"  Sample IDs: {concept_ids[:5]}")

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
    from collections import Counter
    subject_counts = Counter(info["subject"] for info in taxonomy.values())
    metadata = {
        "num_courses": len(taxonomy),
        "num_prototypes": len(prototypes),
        "vector_dim": len(list(prototypes.values())[0]) if prototypes else 0,
        "method": "TF-IDF (sklearn) - course-level aggregation",
        "courses": list(taxonomy.keys()),
        "subject_breakdown": dict(subject_counts),
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
    print(f"\nNext step: Run compute_metrics.py to analyze datasets")
    print("\nNote: Using TF-IDF vectors instead of SentenceTransformers")
    print("      This works offline but may be less accurate than embeddings.")


if __name__ == "__main__":
    main()
