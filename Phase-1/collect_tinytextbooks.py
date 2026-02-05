"""
Collect FULL Tiny-Textbooks Dataset from Hugging Face
Downloads all 420,000 documents for comprehensive hierarchical analysis
"""

from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm


class TinyTextbooksCollector:
    """
    Download complete tiny-textbooks dataset (420K documents)
    No sampling - we need ALL data for deep hierarchical analysis
    """

    def __init__(self, output_dir="tiny_textbooks_raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(self):
        """Download FULL dataset from Hugging Face"""
        print("=" * 70)
        print("TINY-TEXTBOOKS FULL DATASET COLLECTION")
        print("=" * 70)
        print("\n🔄 Loading from Hugging Face...")
        print("   Dataset: nampdn-ai/tiny-textbooks")
        print("   Expected: ~420,000 documents")
        print("   Time estimate: 30-60 minutes")

        # Load full dataset
        dataset = load_dataset("nampdn-ai/tiny-textbooks", split="train")

        total_docs = len(dataset)
        print(f"\n✅ Loaded dataset: {total_docs:,} documents")
        print(f"   Estimated size: ~10 GB")

        # Process all documents
        print("\n📝 Processing documents...")
        documents = []

        for idx, item in enumerate(tqdm(dataset, desc="Processing", unit=" docs")):
            doc = {
                'id': idx,
                'text': item['textbook'][:5000],  # First 5K chars for embedding
                'full_text': item['textbook'],     # Complete text
                'source': 'tiny-textbooks',
                'char_count': len(item['textbook']),
                'word_count': len(item['textbook'].split())
            }
            documents.append(doc)

        # Save in batches (10K per file for memory efficiency)
        print(f"\n💾 Saving {len(documents):,} documents in batches...")
        batch_size = 10000
        num_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_num = i // batch_size
            output_file = self.output_dir / f"batch_{batch_num:03d}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)

            print(f"   Batch {batch_num + 1}/{num_batches}: {len(batch):,} docs → {output_file.name}")

        # Statistics
        total_chars = sum(doc['char_count'] for doc in documents)
        total_words = sum(doc['word_count'] for doc in documents)
        avg_chars = total_chars // len(documents)
        avg_words = total_words // len(documents)

        print("\n" + "=" * 70)
        print("✅ COLLECTION COMPLETE")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Total Documents: {len(documents):,}")
        print(f"   Total Characters: {total_chars:,}")
        print(f"   Total Words: {total_words:,}")
        print(f"   Average per Document:")
        print(f"      - {avg_chars:,} characters")
        print(f"      - {avg_words:,} words")
        print(f"\n💾 Saved to: {self.output_dir}/")
        print(f"   Files: batch_000.json through batch_{num_batches-1:03d}.json")

        print("\n💡 Next steps:")
        print("   python build_deep_graph.py")

        return len(documents)


def main():
    """Main collection process"""
    try:
        collector = TinyTextbooksCollector()
        total = collector.collect()

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install datasets tqdm")
        print("2. Login to Hugging Face: huggingface-cli login")
        print("3. Check internet connection")
        return 1


if __name__ == "__main__":
    exit(main())
