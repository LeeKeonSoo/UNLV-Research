"""
Step 1: Extract Khan Academy Taxonomy and Build Concept Prototypes

This script:
1. Loads Khan Academy K-12 concepts from JSON
2. Extracts hierarchical structure (Subject → Grade → Concept)
3. Creates concept prototypes by embedding article content
4. Saves taxonomy and embeddings for downstream classification

Input: khan_k12_concepts/all_k12_concepts.json
Output:
  - outputs/khan_taxonomy.json (hierarchical structure)
  - outputs/concept_prototypes.pkl (embeddings)
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Alternative embedding models (commented out, can swap later):
# from InstructorEmbedding import INSTRUCTOR  # instructor-base
# from sentence_transformers import SentenceTransformer  # e5-large-v2


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


def build_concept_prototypes(
    taxonomy: Dict,
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[Dict, SentenceTransformer]:
    """
    Create concept prototype embeddings.

    For each concept, concatenate all article content and embed.
    Returns mapping: concept_id → embedding vector
    """
    print(f"\nBuilding concept prototypes with {model_name}...")

    # Load embedding model
    print(f"Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)

    # Alternative models (commented):
    # model = INSTRUCTOR('hkunlp/instructor-base')  # Instructor
    # model = SentenceTransformer('intfloat/e5-large-v2')  # E5

    prototypes = {}

    # Collect all texts to embed
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

    print(f"Embedding {len(concept_texts)} concepts...")

    # Batch embed for efficiency
    embeddings = model.encode(
        concept_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Create concept → embedding mapping
    for concept_id, embedding in zip(concept_ids, embeddings):
        prototypes[concept_id] = embedding

    print(f"✓ Created {len(prototypes)} concept prototypes")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

    return prototypes, model


def save_outputs(taxonomy: Dict, prototypes: Dict, output_dir: str):
    """Save taxonomy and prototypes to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save taxonomy as JSON
    taxonomy_path = output_path / "khan_taxonomy.json"
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved taxonomy to {taxonomy_path}")

    # Save prototypes as pickle (embeddings are numpy arrays)
    prototypes_path = output_path / "concept_prototypes.pkl"
    with open(prototypes_path, 'wb') as f:
        pickle.dump(prototypes, f)
    print(f"✓ Saved concept prototypes to {prototypes_path}")

    # Save metadata
    metadata = {
        "num_subjects": len(taxonomy),
        "num_concepts": len(prototypes),
        "embedding_dim": list(prototypes.values())[0].shape[0] if prototypes else 0,
        "model": "all-MiniLM-L6-v2",
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

    # Step 1: Load data
    data = load_khan_data(DATA_PATH)

    # Step 2: Extract taxonomy
    taxonomy = extract_taxonomy(data)

    # Step 3: Build concept prototypes
    prototypes, model = build_concept_prototypes(taxonomy)

    # Step 4: Save outputs
    save_outputs(taxonomy, prototypes, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✓ Khan Academy taxonomy extraction complete!")
    print("="*60)
    print(f"\nNext step: Run 2_compute_metrics.py to analyze datasets")


if __name__ == "__main__":
    main()
