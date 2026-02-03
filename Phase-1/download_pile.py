from datasets import load_dataset
import json
import os

PILE_SUBSETS = [
    'ArXiv',
    'StackExchange',
    'Wikipedia (en)',
    'Github',
    'PubMed Abstracts',
    'FreeLaw',
]

# 각 subset당 목표 용량 (bytes)
TARGET_SIZE_MB = 500  # per subset
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

os.makedirs('pile', exist_ok=True)

print(f"📥 Downloading The Pile subsets (equal {TARGET_SIZE_MB}MB each)")
print(f"Target: {len(PILE_SUBSETS)} subsets × {TARGET_SIZE_MB}MB = {TARGET_SIZE_MB * len(PILE_SUBSETS)}MB total\n")

try:
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True
    )
    
    # Track size for each subset
    subset_sizes = {name: 0 for name in PILE_SUBSETS}
    subset_counts = {name: 0 for name in PILE_SUBSETS}
    subset_files = {}
    
    # Open all output files
    for subset_name in PILE_SUBSETS:
        filename = subset_name.replace(' ', '_').replace('(', '').replace(')', '')
        subset_files[subset_name] = open(f'pile/{filename}.jsonl', 'w')
    
    print("Processing stream...")
    total_processed = 0
    
    for item in dataset:
        pile_set = item['meta']['pile_set_name']
        
        if pile_set not in PILE_SUBSETS:
            continue
        
        # Check if this subset has reached target size
        if subset_sizes[pile_set] >= TARGET_SIZE_BYTES:
            # Check if all subsets are done
            if all(size >= TARGET_SIZE_BYTES for size in subset_sizes.values()):
                print("\n✅ All subsets complete!")
                break
            continue
        
        # Calculate size of this document
        text = item['text']
        text_size = len(text.encode('utf-8'))
        
        # Write to file
        json.dump({'text': text}, subset_files[pile_set])
        subset_files[pile_set].write('\n')
        
        subset_sizes[pile_set] += text_size
        subset_counts[pile_set] += 1
        total_processed += 1
        
        # Progress update every 100 docs
        if total_processed % 100 == 0:
            print(f"\rProcessed: {total_processed:,} docs | ", end="")
            status = " | ".join([
                f"{name.split()[0]}: {subset_sizes[name]/(1024**2):.1f}/{TARGET_SIZE_MB}MB"
                for name in PILE_SUBSETS
            ])
            print(status, end="", flush=True)
    
    # Close all files
    for f in subset_files.values():
        f.close()
    
    print("\n\n📊 Download Summary:")
    print(f"{'Subset':<20} {'Docs':>8} {'Size (MB)':>12}")
    print("-" * 45)
    for subset_name in PILE_SUBSETS:
        filename = subset_name.replace(' ', '_').replace('(', '').replace(')', '')
        filepath = f'pile/{filename}.jsonl'
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            count = subset_counts[subset_name]
            print(f"{subset_name:<20} {count:8,} {size_mb:12.1f}")
    
    print("-" * 45)
    total_size = sum(os.path.getsize(f'pile/{name.replace(" ", "_").replace("(", "").replace(")", "")}.jsonl') 
                     for name in PILE_SUBSETS if os.path.exists(f'pile/{name.replace(" ", "_").replace("(", "").replace(")", "")}.jsonl'))
    print(f"{'TOTAL':<20} {total_processed:8,} {total_size/(1024**2):12.1f}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    for f in subset_files.values():
        if not f.closed:
            f.close()

print("\n🎉 Download complete!")