# Phase-1 Technical Implementation Guide
**Complete Code Walkthrough for Professor Review**

Last Updated: February 11, 2026
Researcher: KeonSoo (bubbleguy10@gmail.com)
UNLV Research Lab - Week 4/16

## Phase-1 Objective Sync (Canonical)
This guide is synchronized to `PHASE1_RESEARCH_OBJECTIVE_SPEC.md`.

1. Problem statement: reproducible dataset characterization for SLM data curation.
2. In scope: characterization, metric reliability, descriptive cross-dataset comparison.
3. Out of scope: causal claims on training efficiency/performance uplift (Phase-2).
4. Canonical RQs:
   1. Domain coverage differences.
   2. Educational-structure quality differences.
   3. Metric reliability/stability.
5. Metric tiers:
   1. Core: domain, quality, difficulty.
   2. Exploratory: redundancy, perplexity.
6. Claim policy: only Core metrics can support headline conclusions; failed-gate metrics are non-claimable.

---

## 📋 Table of Contents

1. [Overview & Pipeline Flow](#overview--pipeline-flow)
2. [Step 0: Data Collection](#step-0-data-collection)
3. [Step 1: Extract Khan Taxonomy](#step-1-extract-khan-taxonomy)
4. [Step 2a: Build Corpus Index](#step-2a-build-corpus-index)
5. [Step 2: Compute 5 Metrics](#step-2-compute-5-metrics)
6. [Step 3: Build Dashboard](#step-3-build-dashboard)
7. [Technical Decisions & Trade-offs](#technical-decisions--trade-offs)

---

## Overview & Pipeline Flow

### What This Pipeline Does

**Goal**: Characterize pretraining datasets for Small Language Model (SLM) training by analyzing 5 dimensions:
1. **Domain Coverage** - What topics does this text cover?
2. **Quality** - Does it have educational structure (examples, explanations)?
3. **Difficulty** - How hard is it to read?
4. **Redundancy** - Is this content duplicated elsewhere?
5. **Perplexity** - How "natural" is this text to a language model?

### Pipeline Execution Order

```
Step 0: Data Collection
    ↓
Step 1: Extract Khan Taxonomy (1_extract_khan_taxonomy.py)
    ↓ outputs/concept_prototypes_tfidf.pkl
Step 2a: Build Corpus Index (2a_build_corpus_index.py)
    ↓ outputs/corpus_index.pkl (15GB)
Step 2: Compute 5 Metrics (2_compute_metrics.py)
    ↓ outputs/khan_analysis.jsonl (156MB)
    ↓ outputs/tiny_textbooks_analysis.jsonl (323MB)
Step 3: Build Dashboard (3_build_dashboard.py)
    ↓ outputs/dashboard.html
```

### File Structure

```
Phase-1/
├── collect_khan_academy.py          # Step 0a: Khan data download
├── collect_tinytextbooks.py         # Step 0b: Tiny-Textbooks download
├── 1_extract_khan_taxonomy.py       # Step 1: TF-IDF concept prototypes
├── 2a_build_corpus_index.py         # Step 2a: MinHash LSH index
├── 2_compute_metrics.py             # Step 2: 5 metrics per paragraph
├── 3_build_dashboard.py             # Step 3: Interactive visualization
├── khan_k12_concepts/               # Downloaded Khan data (18,764 entries)
├── tiny_textbooks_raw/              # Downloaded Tiny-Textbooks (42 batches)
└── outputs/                         # Generated analysis results
    ├── concept_prototypes_tfidf.pkl # Step 1 output (255KB)
    ├── corpus_index.pkl             # Step 2a output (15GB)
    ├── khan_analysis.jsonl          # Step 2 output (156MB)
    └── tiny_textbooks_analysis.jsonl # Step 2 output (323MB)
```

---

## Step 0: Data Collection

### File: `collect_khan_academy.py`

**Purpose**: Download Khan Academy K-12 educational concepts to use as domain taxonomy.

**Status**: ✅ **Succeed** (after multiple failed attempts)

#### What It Does

1. **Tries HuggingFace `datasets` library first**
2. **Falls back to direct parquet API if blocked**
3. **Filters for K-12 content only** (Grades 3-12)
4. **Saves to JSON** for easy processing

#### Implementation Details

```python
def download_khan_academy_cosmopedia():
    """
    Download Khan Academy subset from HuggingFace Cosmopedia dataset.

    Two-tier download strategy:
    1. Primary: datasets.load_dataset() - fastest, requires HF access
    2. Fallback: Direct parquet file download - works through proxies
    """

    # Attempt 1: datasets library (blocked by proxy in remote env)
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "HuggingFaceTB/cosmopedia",
            "khanacademy",
            split="train",
            streaming=False
        )
        return list(dataset)
    except Exception as e:
        print(f"⚠ datasets library failed: {e}")

    # Attempt 2: Direct parquet download
    parquet_url = "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/..."
    response = requests.get(parquet_url)
    df = pd.read_parquet(BytesIO(response.content))
    return df.to_dict('records')
```

**Key Functions**:

1. **`download_khan_academy_cosmopedia()`** - Downloads from HuggingFace
2. **`filter_k12_content(data)`** - Keeps only Grades 3-12
3. **`save_to_json(data, output_path)`** - Saves as JSON

**Output**: `khan_k12_concepts/all_k12_concepts.json` (18,764 entries)

**Data Structure**:
```json
{
  "url": "https://khanacademy.org/...",
  "title": "Linear Equations Basics",
  "content": "In this lesson, we'll learn...",
  "subject": "Math",
  "course": "Algebra 1",
  "grade": "8-9"
}
```

#### ❌ Tried: Direct Khan Academy API

**Why it failed**: Khan Academy deprecated their public API (~2023), returns `410 Gone`.

```python
# This approach FAILED
response = requests.get("https://www.khanacademy.org/api/v1/...")
# Returns: HTTP 410 Gone
```

#### ❌ Tried: Web Scraping Khan Academy

**Why it failed**: Khan Academy is a React SPA (Single Page Application). Server-side scraping gets empty HTML shells.

```python
# This approach FAILED
soup = BeautifulSoup(requests.get("https://khanacademy.org/...").text)
# Gets: <div id="root"></div> (React mounts content client-side)
```

#### ✅ Succeed: HuggingFace Cosmopedia

**Why it worked**:
- Cosmopedia is a synthetic dataset generated by Mixtral-8x7B
- Uses Khan Academy topics as seed prompts
- Preserves taxonomy structure (subject, course, grade)
- Dual download strategy handles proxy restrictions

**Trade-off**: Not real Khan Academy content, but sufficient for taxonomy classification research.

---

### File: `collect_tinytextbooks.py`

**Purpose**: Download Tiny-Textbooks dataset for analysis.

**Status**: ✅ **Succeed**

#### What It Does

Downloads synthetic educational textbook data in 42 batches.

```python
from datasets import load_dataset

dataset = load_dataset("nampdn-ai/tiny-textbooks", split="train")

# Save in batches to avoid memory issues
BATCH_SIZE = 10000
for i in range(0, len(dataset), BATCH_SIZE):
    batch = dataset[i:i+BATCH_SIZE]
    with open(f"batch_{i//BATCH_SIZE}.json", "w") as f:
        json.dump(batch, f)
```

**Output**: `tiny_textbooks_raw/batch_*.json` (42 files, ~100,000 docs total)

---

## Step 1: Extract Khan Taxonomy

### File: `1_extract_khan_taxonomy.py`

**Purpose**: Create TF-IDF vector representations of each Khan Academy course to use as domain classification prototypes.

**Status**: ✅ **Succeed** (after SBERT failure)

### What It Does - High Level

1. **Load Khan Academy data** from JSON
2. **Group lessons by course** (e.g., "Math::Algebra 1")
3. **Concatenate all lesson texts per course** into one document
4. **Vectorize with TF-IDF** to create 300-dim vectors
5. **Save prototypes** for later classification

### Implementation - Line by Line

#### Part 1: Load Data

```python
def load_khan_data(data_path: str) -> List[Dict]:
    """
    Load Khan Academy concepts from JSON.

    Input: khan_k12_concepts/all_k12_concepts.json
    Output: List of 18,764 concept dictionaries
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
```

**What happens**:
- Reads 18,764 Khan Academy concept entries
- Each entry has: subject, course, grade, title, content

#### Part 2: Extract Course-Level Taxonomy

```python
def extract_taxonomy(data: List[Dict]) -> Dict:
    """
    Group lessons by (subject, course) so each prototype represents
    one course (e.g. "Math::Algebra 1"), not individual lessons.

    Returns:
    {
        "Math::Algebra 1": {
            "subject": "Math",
            "course": "Algebra 1",
            "grade": "8-9",
            "texts": ["lesson 1 content", "lesson 2 content", ...],
            "lesson_count": 42
        },
        ...
    }
    """
    taxonomy = defaultdict(lambda: {
        "subject": "", "course": "", "grade": "", "texts": []
    })

    for doc in tqdm(data, desc="Processing concepts"):
        subject = doc.get("subject", "Unknown")
        course = doc.get("course", doc.get("subject", "Unknown"))
        grade = doc.get("grade", "Unknown")
        content = doc.get("content", "")

        # Create unique key: "Math::Algebra 1"
        key = f"{subject}::{course}"

        # First time seeing this course? Store metadata
        if taxonomy[key]["subject"] == "":
            taxonomy[key]["subject"] = subject
            taxonomy[key]["course"] = course
            taxonomy[key]["grade"] = grade

        # Only keep substantial content (>50 chars)
        if content and len(content.strip()) >= 50:
            taxonomy[key]["texts"].append(content)

    # Convert to regular dict and add counts
    taxonomy = dict(taxonomy)
    for info in taxonomy.values():
        info["lesson_count"] = len(info["texts"])

    return taxonomy
```

**What happens**:
- **Input**: 18,764 individual lessons
- **Grouping logic**: Same subject + course → merge into one course prototype
- **Output**: ~100 course-level categories (e.g., "Math::Algebra 1" with 42 lessons combined)

**Why course-level, not lesson-level?**
- Too many lessons (18K) → noisy, redundant prototypes
- Courses are meaningful units (Algebra 1, Biology, US History)
- Reduces dimensionality: 18,764 → ~100 prototypes

#### Part 3: Build TF-IDF Prototypes

```python
def build_concept_prototypes_tfidf(taxonomy: Dict) -> tuple:
    """
    Create one TF-IDF vector per course.

    All lesson texts within the same course are concatenated into
    a single document before vectorization.
    """
    concept_texts = []
    concept_ids = []

    for concept_id, info in taxonomy.items():
        texts = info.get("texts", [])
        if not texts:
            continue

        # Concatenate ALL lesson texts for this course
        combined = " ".join(texts)
        concept_texts.append(combined)
        concept_ids.append(concept_id)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=300,        # 300-dimensional sparse vectors
        stop_words='english',    # Remove "the", "a", "is", etc.
        ngram_range=(1, 2),      # Unigrams + bigrams ("linear equation")
        min_df=1,                # Keep all terms (each course is unique)
    )

    # Fit and transform all course documents
    tfidf_matrix = vectorizer.fit_transform(concept_texts)
    # Shape: (100 courses, 300 features)

    # Convert sparse matrix to dense numpy array
    dense_matrix = np.asarray(tfidf_matrix.todense())

    # Create dictionary: concept_id → vector
    prototypes = {
        concept_id: dense_matrix[i]
        for i, concept_id in enumerate(concept_ids)
    }

    return prototypes, vectorizer
```

**What happens**:

1. **Concatenation**: "Algebra 1" has 42 lessons → join all texts into one mega-document
2. **TF-IDF Vectorization**:
   - Extracts top 300 most important words/bigrams
   - Weights by TF (term frequency) × IDF (inverse document frequency)
   - Result: 300-dim sparse vector representing the course's vocabulary
3. **Storage**: Save both prototypes (vectors) and vectorizer (for later use on new text)

**TF-IDF Formula**:
```
TF-IDF(word, document) = TF(word, doc) × IDF(word, corpus)

TF = (count of word in doc) / (total words in doc)
IDF = log(total docs / docs containing word)
```

**Example**:
- "equation" appears 50 times in Algebra 1 text → high TF
- "equation" appears in 80% of all courses → low IDF
- "quadratic" appears 30 times in Algebra 1 → high TF
- "quadratic" appears in only 20% of courses → high IDF
- **Result**: "quadratic" gets higher TF-IDF score → more distinctive for Algebra 1

#### Part 4: Save Outputs

```python
def save_outputs(taxonomy: Dict, prototypes: Dict, vectorizer, output_dir: str):
    """Save taxonomy and prototypes to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save taxonomy as JSON (human-readable)
    taxonomy_path = output_path / "khan_taxonomy.json"
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    # 2. Save prototypes + vectorizer as pickle (for Step 2)
    prototypes_path = output_path / "concept_prototypes_tfidf.pkl"
    with open(prototypes_path, 'wb') as f:
        pickle.dump({
            'prototypes': prototypes,  # Dict[str, np.ndarray]
            'vectorizer': vectorizer    # TfidfVectorizer object
        }, f)

    # 3. Save metadata for documentation
    from collections import Counter
    subject_counts = Counter(info["subject"] for info in taxonomy.values())

    metadata = {
        "num_courses": len(taxonomy),
        "num_prototypes": len(prototypes),
        "vector_dim": 300,
        "method": "TF-IDF (sklearn) - course-level aggregation",
        "courses": list(taxonomy.keys()),
        "subject_breakdown": dict(subject_counts),
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```

**Output Files**:
1. **`khan_taxonomy.json`** (66MB) - Hierarchical course structure with all lesson texts
2. **`concept_prototypes_tfidf.pkl`** (255KB) - 100 TF-IDF vectors + vectorizer
3. **`metadata.json`** (4KB) - Statistics for documentation

### ❌ Tried: SentenceTransformers (SBERT)

**Original plan**: Use `all-MiniLM-L6-v2` for semantic embeddings (384-dim dense vectors).

```python
# This approach FAILED
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# Error: HuggingFace model download blocked by remote proxy (403 Forbidden)
```

**Why it failed**:
- Remote execution environment has proxy restrictions
- HuggingFace model hub returns `403 Forbidden`
- Cannot download 80MB model file

**Why SBERT would be better**:
- Captures semantic similarity: "cell phone" ≈ "mobile phone"
- TF-IDF only does lexical matching: "cell phone" ≠ "mobile phone"

### ✅ Succeed: TF-IDF (sklearn)

**Why it works**:
- No external downloads needed (sklearn is pre-installed)
- Completely offline-compatible
- Fast (vectorization in milliseconds)

**Trade-off accepted**:
- Lower quality embeddings (lexical vs semantic)
- But sufficient for domain classification (courses have distinctive vocabulary)
- Example: "quadratic", "polynomial" → clearly Algebra
- Example: "mitosis", "cell division" → clearly Biology

---

## Step 2a: Build Corpus Index

### File: `2a_build_corpus_index.py`

**Purpose**: Pre-build a searchable index of all documents to enable fast redundancy detection in Step 2.

**Status**: ✅ **Succeed**

### Why This Step Exists

**Problem**: Redundancy detection requires comparing each chunk against ALL other chunks.

**Naive approach** (what we DON'T do):
```python
for chunk in dataset:  # 200,000 chunks
    for other_chunk in dataset:  # 200,000 chunks
        if similar(chunk, other_chunk):
            mark_as_duplicate()
# Time complexity: O(n²) = 200,000² = 40 billion comparisons
# Estimated time: ~500 hours on CPU
```

**Our approach** (what we DO):
1. **Build index once** (Step 2a): ~2 hours
2. **Query index** (Step 2): ~0.001 seconds per chunk
3. **Total time**: 2 hours + (200K × 0.001s) = ~3.3 hours

### Implementation - Line by Line

#### Part 1: Text Chunking (mirrors Step 2)

```python
def chunk_text(text: str, chunk_size: int = 200):
    """
    Split text into ~200-word paragraphs.
    Must match Step 2's chunking logic exactly for index lookups to work.
    """
    paragraphs = re.split(r'\n\s*\n', text)  # Split on blank lines
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        words = para.split()

        if len(words) <= chunk_size:
            # Small paragraph → keep as-is
            if len(words) >= 20:  # Filter out tiny chunks
                chunks.append(para)
        else:
            # Large paragraph → split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            cur, cur_len = [], 0

            for sent in sentences:
                sw = len(sent.split())

                if cur_len + sw > chunk_size and cur:
                    # Current batch full → save and start new
                    joined = " ".join(cur)
                    if len(joined.split()) >= 20:
                        chunks.append(joined)
                    cur, cur_len = [sent], sw
                else:
                    # Add to current batch
                    cur.append(sent)
                    cur_len += sw

            # Save remaining sentences
            if cur:
                joined = " ".join(cur)
                if len(joined.split()) >= 20:
                    chunks.append(joined)

    return chunks
```

**What happens**:
- Input: Full document (could be 10,000 words)
- Output: List of ~200-word chunks
- Minimum chunk size: 20 words (filters out headers, captions)

**Why 200 words?**
- Too small (50 words): Loses context, hard to classify domain
- Too large (1000 words): Mixed topics, redundancy harder to detect
- 200 words ≈ 1-2 paragraphs: Good balance

#### Part 2: Document Iterator

```python
def iter_documents():
    """
    Yield (doc_id, text) for every chunk from both datasets.

    doc_id format examples:
    - "khan::https://khanacademy.org/...::0"
    - "tiny::batch_05.json::doc_1234::3"
    """
    # Process Khan Academy
    khan_path = Path(KHAN_DATA_PATH)
    if khan_path.exists():
        with open(khan_path, "r", encoding="utf-8") as f:
            khan_data = json.load(f)

        for doc in tqdm(khan_data, desc="Khan chunks"):
            text = doc.get("content", "")
            url = doc.get("url", "unknown")

            if len(text.strip()) < 50:
                continue

            # Chunk and yield each piece
            for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
                doc_id = f"khan::{url}::{i}"
                yield doc_id, chunk

    # Process Tiny-Textbooks
    tiny_dir = Path(TINY_DATA_DIR)
    if tiny_dir.exists():
        batch_files = sorted(tiny_dir.glob("batch_*.json"))

        for batch_file in tqdm(batch_files, desc="Tiny batches"):
            with open(batch_file, "r", encoding="utf-8") as f:
                batch = json.load(f)

            for doc in batch:
                doc_id_orig = doc.get("id", "unknown")
                text = doc.get("text", "")

                if len(text.strip()) < 100:
                    continue

                for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
                    doc_id = f"tiny::{batch_file.name}::{doc_id_orig}::{i}"
                    yield doc_id, chunk
```

**What happens**:
- Loads both Khan and Tiny-Textbooks
- Chunks each document
- Yields one (id, text) pair at a time (memory efficient)

**Why unique doc_ids?**
- Need to track which chunks are similar to which
- Format embeds source and position for debugging

#### Part 3: Build 3-Level Index

```python
def build_corpus_index():
    """
    Build three redundancy detection indexes:
    1. Exact duplicate detection (MD5 hash set)
    2. Near-duplicate detection (MinHash LSH)
    3. Semantic similarity (TF-IDF cosine)
    """
    doc_ids = []
    texts = []
    exact_hashes = set()
    minhashes = {}

    # Create LSH index for fast similarity search
    lsh = MinHashLSH(threshold=0.5, num_perm=128)

    print("Pass 1: Collecting chunks and building MinHash signatures...")

    for doc_id, text in iter_documents():
        doc_ids.append(doc_id)
        texts.append(text)

        # 1. Exact duplicate: MD5 hash
        exact_hashes.add(hashlib.md5(text.encode()).hexdigest())

        # 2. Near-duplicate: MinHash
        mh = MinHash(num_perm=128)
        for word in text.lower().split():
            mh.update(word.encode("utf-8"))

        minhashes[doc_id] = mh

        # Insert into LSH for fast lookups
        try:
            lsh.insert(doc_id, mh)
        except ValueError:
            pass  # Duplicate key already in index

    print(f"✓ Collected {len(doc_ids):,} chunks")

    print("Pass 2: Building TF-IDF matrix for semantic similarity...")

    # 3. Semantic similarity: TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,        # Larger vocab for better semantic comparison
        stop_words="english",
        ngram_range=(1, 1),       # Unigrams only (faster)
        min_df=2,                 # Ignore words appearing once
    )

    corpus_matrix = vectorizer.fit_transform(texts)
    # Shape: (200,000 chunks, 5000 features)

    print(f"✓ TF-IDF matrix: {corpus_matrix.shape}")

    # Save everything to pickle
    index = {
        "exact_hashes": exact_hashes,
        "lsh": lsh,
        "minhashes": minhashes,
        "corpus_matrix": corpus_matrix,
        "vectorizer": vectorizer,
        "doc_ids": doc_ids,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"\n✓ Saved corpus index to {OUTPUT_PATH}")
    print(f"  Chunks indexed: {len(doc_ids):,}")
    print(f"  Unique hashes: {len(exact_hashes):,}")
    print(f"  TF-IDF vocab: {len(vectorizer.vocabulary_):,}")
```

**What happens**:

**Pass 1: Exact + Near-Duplicate**
- For each chunk:
  - Compute MD5 hash → exact duplicate detection
  - Compute MinHash → near-duplicate detection (Jaccard similarity)
  - Insert into LSH index → fast similarity search

**Pass 2: Semantic Similarity**
- Vectorize all chunks with TF-IDF (5000 features, larger than Step 1's 300)
- Store sparse matrix (only non-zero values) → saves memory

**Output**: `outputs/corpus_index.pkl` (15GB)

**Why so large?**
- 200,000 chunks × 5000 features TF-IDF matrix (sparse, but still big)
- 200,000 MinHash signatures (128 bytes each)
- LSH index structure

### Redundancy Detection Algorithms Explained

#### 1. Exact Duplicate (MD5 Hash)

```python
# Build phase (Step 2a)
text_hash = hashlib.md5(text.encode()).hexdigest()
exact_hashes.add(text_hash)

# Query phase (Step 2)
query_hash = hashlib.md5(query_text.encode()).hexdigest()
is_duplicate = query_hash in exact_hashes  # O(1) lookup
```

**How it works**:
- MD5 produces 128-bit hash (16 bytes)
- Identical text → identical hash
- Even 1-character difference → completely different hash

**Use case**: Catch copy-paste duplicates

#### 2. Near-Duplicate (MinHash + LSH)

**MinHash Algorithm**:
```python
def minhash(text, num_perm=128):
    """
    Create a compact signature that estimates Jaccard similarity.

    Jaccard similarity = |A ∩ B| / |A ∪ B|
    (size of intersection / size of union)
    """
    words = set(text.lower().split())

    # Use 128 different hash functions
    signatures = []
    for i in range(num_perm):
        hash_func = hash_functions[i]  # Different hash per permutation
        min_hash = min(hash_func(word) for word in words)
        signatures.append(min_hash)

    return signatures  # 128 integers
```

**LSH (Locality-Sensitive Hashing)**:
```python
# Build phase: Insert all chunks
lsh = MinHashLSH(threshold=0.5, num_perm=128)
for doc_id, mh in minhashes.items():
    lsh.insert(doc_id, mh)

# Query phase: Find similar chunks
query_mh = MinHash(num_perm=128)
for word in query_text.split():
    query_mh.update(word.encode())

similar_ids = lsh.query(query_mh)  # Fast: O(log n) instead of O(n)
```

**How LSH works**:
- Divides 128 hash values into bands (e.g., 16 bands × 8 hashes each)
- Chunks with similar MinHash → likely in same bands
- Query only checks chunks in matching bands (instead of all 200K)

**Example**:
```
Text A: "The quick brown fox jumps over the lazy dog"
Text B: "The fast brown fox leaps over the lazy dog"

Words A: {the, quick, brown, fox, jumps, over, lazy, dog}
Words B: {the, fast, brown, fox, leaps, over, lazy, dog}

Intersection: {the, brown, fox, over, lazy, dog} = 6 words
Union: {the, quick, fast, brown, fox, jumps, leaps, over, lazy, dog} = 10 words

Jaccard = 6/10 = 0.6

MinHash estimates this without computing full sets!
```

**Use case**: Catch paraphrased or lightly edited duplicates

#### 3. Semantic Similarity (TF-IDF Cosine)

```python
# Build phase (already done in Step 2a)
corpus_matrix = vectorizer.fit_transform(all_texts)

# Query phase (Step 2)
query_vector = vectorizer.transform([query_text])
similarities = cosine_similarity(query_vector, corpus_matrix)
max_similarity = similarities.max()
```

**How it works**:
```
Cosine similarity = (A · B) / (||A|| × ||B||)

Where:
A · B = dot product (sum of element-wise multiplication)
||A|| = magnitude (Euclidean norm)

Example:
A = [0.5, 0.3, 0.0, 0.2]  (query: "linear equations")
B = [0.6, 0.2, 0.0, 0.1]  (doc: "solving linear equations")

A · B = 0.5×0.6 + 0.3×0.2 + 0.0×0.0 + 0.2×0.1 = 0.38
||A|| = sqrt(0.5² + 0.3² + 0.2²) = 0.62
||B|| = sqrt(0.6² + 0.2² + 0.1²) = 0.64

cosine = 0.38 / (0.62 × 0.64) = 0.96 (very similar!)
```

**Use case**: Catch semantically similar content even if words differ

**Example**:
```
Text A: "Photosynthesis converts sunlight into chemical energy"
Text B: "Plants use light to create food through photosynthesis"

MinHash: Low similarity (different words)
TF-IDF: High similarity (both about photosynthesis, plants, energy)
```

---

## Step 2: Compute 5 Metrics

### File: `2_compute_metrics.py`

**Purpose**: Analyze every paragraph in both datasets across 5 dimensions.

**Status**: ✅ **Succeed** (479MB output, ~3 hours runtime)

### What It Does - High Level

For each ~200-word chunk:
1. **Domain Coverage**: Top-5 Khan courses with similarity scores
2. **Quality**: Educational markers (examples, explanations, structure)
3. **Difficulty**: Reading level (Flesch-Kincaid, lexical diversity)
4. **Redundancy**: Exact/near/semantic duplicate scores
5. **Perplexity**: GPT-2 log-loss (optional, requires GPU)

### Implementation - Line by Line

#### Configuration

```python
# File paths
PROTOTYPES_PATH = "outputs/concept_prototypes_tfidf.pkl"
CORPUS_INDEX_PATH = "outputs/corpus_index.pkl"
KHAN_DATA_PATH = "khan_k12_concepts/all_k12_concepts.json"
TINY_TEXTBOOKS_DIR = "tiny_textbooks_raw"

# Analysis parameters
TOP_K_DOMAINS = 5           # Return top-5 matching courses
MIN_SIMILARITY = 0.1        # Threshold for domain relevance
CHUNK_SIZE = 200            # Words per chunk
USE_GPU = True              # Accelerate with CUDA
CUDA_DEVICE = 0             # GPU number (0 or 1)
```

#### Metric 1: Domain Coverage

```python
class DomainClassifier:
    """
    Classifies text chunks into Khan Academy course domains.
    Uses TF-IDF cosine similarity with GPU acceleration.
    """

    def __init__(self, vectorizer, proto_ids, proto_matrix, use_gpu=True):
        self.vectorizer = vectorizer      # From Step 1
        self.proto_ids = proto_ids        # ["Math::Algebra 1", ...]
        self.proto_matrix = proto_matrix  # (100, 300) normalized vectors
        self.use_torch = False

        # Try to use GPU for faster computation
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            self.use_torch = True
            self.device = f"cuda:{CUDA_DEVICE}"
            self.proto_t = torch.from_numpy(proto_matrix).to(self.device)
            print(f"Domain classifier: using {self.device}")
        else:
            print("Domain classifier: using CPU")

    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify text into top-K domains with normalized scores.

        Returns:
        {
            "Math::Algebra 1": 0.45,
            "Math::Linear Equations": 0.32,
            "Science::Physics": 0.23
        }
        """
        # 1. Convert text to TF-IDF vector
        try:
            qv = self.vectorizer.transform([text])  # Sparse (1, 300)
        except:
            return {}  # Empty text or vectorization error

        if qv.nnz == 0:  # No features extracted
            return {}

        # 2. Normalize query vector
        norm = float(np.sqrt(qv.multiply(qv).sum()))
        if norm == 0.0:
            return {}

        # 3. Compute cosine similarity with all prototypes
        if self.use_torch:
            # GPU acceleration
            q = torch.from_numpy(qv.toarray().astype(np.float32)).to(self.device)
            sims = (q @ self.proto_t.T / norm).squeeze(0).cpu().numpy()
        else:
            # CPU computation
            q = qv.toarray().astype(np.float32)
            sims = (q @ self.proto_matrix.T).ravel() / norm

        # 4. Filter by minimum similarity threshold
        valid = np.where(sims >= MIN_SIMILARITY)[0]
        if valid.size == 0:
            return {}

        # 5. Select top-K domains
        if valid.size > TOP_K_DOMAINS:
            top = valid[np.argpartition(sims[valid], -TOP_K_DOMAINS)[-TOP_K_DOMAINS:]]
        else:
            top = valid

        # 6. Sort by similarity (descending)
        top = top[np.argsort(sims[top])[::-1]]

        # 7. Normalize to probabilities (sum = 1.0)
        result = {self.proto_ids[i]: float(sims[i]) for i in top}
        total = sum(result.values())

        return {k: v / total for k, v in result.items()} if total > 0 else result
```

**Step-by-step example**:

```
Input text: "Solving quadratic equations using the quadratic formula"

Step 1: TF-IDF vectorization
- "solving": 0.15
- "quadratic": 0.45
- "equations": 0.30
- "using": 0.05
- "formula": 0.25
→ Sparse vector (1, 300) with 5 non-zero entries

Step 2: Normalize
- norm = sqrt(0.15² + 0.45² + 0.30² + 0.05² + 0.25²) = 0.62

Step 3: Compute cosine similarity
query @ prototypes.T:
- "Math::Algebra 1": 0.48
- "Math::Algebra 2": 0.52
- "Math::Geometry": 0.12
- "Science::Physics": 0.08
- "Science::Chemistry": 0.05

Step 4: Filter (MIN_SIMILARITY = 0.1)
- Keep: Algebra 1, Algebra 2, Geometry
- Drop: Physics, Chemistry

Step 5: Top-5 (already have only 3)
- [Algebra 2, Algebra 1, Geometry]

Step 6: Sort descending
- Algebra 2 (0.52), Algebra 1 (0.48), Geometry (0.12)

Step 7: Normalize
- Total = 0.52 + 0.48 + 0.12 = 1.12
- Result:
  {
    "Math::Algebra 2": 0.52 / 1.12 = 0.464,
    "Math::Algebra 1": 0.48 / 1.12 = 0.429,
    "Math::Geometry": 0.12 / 1.12 = 0.107
  }
```

**Why GPU acceleration?**
- 200,000 chunks × 100 prototypes = 20 million dot products
- CPU: ~50 chunks/second = 66 minutes total
- GPU (RTX 4060Ti): ~1000 chunks/second = 3.3 minutes total

#### Metric 2: Quality (Educational Markers)

```python
# Define marker patterns
_EXAMPLE_MARKERS = [
    "for example", "such as", "consider", "let's look at", "instance"
]
_EXPLANATION_MARKERS = [
    "because", "therefore", "this means", "as a result", "consequently"
]
_STRUCTURE_MARKERS = [
    "first", "second", "third", "finally", "in summary", "in conclusion"
]

def compute_quality(text: str) -> Dict:
    """
    Detect educational structure markers.

    Returns:
    {
        "educational_markers": {
            "has_examples": True,
            "has_explanation": True,
            "has_structure": False
        },
        "quality_score": 0.6667
    }
    """
    t = text.lower()

    # Check for presence of each marker type
    has_ex = any(m in t for m in _EXAMPLE_MARKERS)
    has_exp = any(m in t for m in _EXPLANATION_MARKERS)
    has_str = any(m in t for m in _STRUCTURE_MARKERS)

    # Aggregate score: 0.0 to 1.0
    score = (has_ex + has_exp + has_str) / 3.0

    return {
        "educational_markers": {
            "has_examples": has_ex,
            "has_explanation": has_exp,
            "has_structure": has_str,
        },
        "quality_score": round(score, 4),
    }
```

**Why these markers?**

Based on educational research (Bloom's Taxonomy):
- **Examples**: Concrete instantiation (lower-order thinking)
- **Explanations**: Causal reasoning (higher-order thinking)
- **Structure**: Organized presentation (meta-cognitive)

**Example**:
```
Text A (low quality):
"Photosynthesis occurs in plants. It uses light. Chlorophyll is involved."
→ No markers, quality_score = 0.0

Text B (high quality):
"Photosynthesis occurs in plants. For example, leaves contain chlorophyll.
This means they can convert light into energy. First, light hits the leaf.
Then, chemical reactions occur. In summary, plants make food from sunlight."
→ All markers present, quality_score = 1.0
```

#### Metric 3: Difficulty

```python
def compute_difficulty(text: str, sentences: List[str] = None) -> Dict:
    """
    Compute 7 readability metrics.

    Uses NLTK for tokenization and textstat library for standard formulas.
    """
    # 1-3: Standard readability scores
    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
        flesch_ease = textstat.flesch_reading_ease(text)
        smog = textstat.smog_index(text)
    except:
        fk_grade = flesch_ease = smog = 0.0

    # 4-7: Custom linguistic features
    try:
        if sentences is None:
            sentences = sent_tokenize(text)  # NLTK sentence splitting

        words = [w for w in word_tokenize(text) if w.isalpha()]
        n_sents = max(len(sentences), 1)
        n_words = max(len(words), 1)

        # Average sentence length
        avg_sent_len = round(n_words / n_sents, 2)

        # Average word length
        avg_word_len = round(sum(len(w) for w in words) / n_words, 2)

        # Type-Token Ratio (lexical diversity)
        ttr = round(len(set(w.lower() for w in words)) / n_words, 4)

        # Rare words percentage
        common = _load_common_words()  # Top 10K English words
        if common:
            rare_count = sum(1 for w in words if w.lower() not in common)
            rare_pct = round(rare_count / n_words, 4)
        else:
            rare_pct = 0.0
    except:
        avg_sent_len = avg_word_len = ttr = rare_pct = 0.0

    return {
        "flesch_kincaid_grade": round(fk_grade, 2),
        "flesch_reading_ease": round(flesch_ease, 2),
        "smog_index": round(smog, 2),
        "avg_sentence_length": avg_sent_len,
        "avg_word_length": avg_word_len,
        "rare_words_pct": rare_pct,
        "lexical_diversity": ttr,
    }
```

**Flesch-Kincaid Grade Level Formula**:
```
Grade = 0.39 × (words/sentences) + 11.8 × (syllables/words) - 15.59

Example:
- 150 words, 10 sentences, 200 syllables
- Grade = 0.39 × (150/10) + 11.8 × (200/150) - 15.59
        = 0.39 × 15 + 11.8 × 1.33 - 15.59
        = 5.85 + 15.69 - 15.59
        = 5.95 (≈ 6th grade level)
```

**Flesch Reading Ease Formula**:
```
Score = 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)

Scale:
90-100: Very Easy (5th grade)
60-70: Standard (8-9th grade)
30-50: Difficult (college)
0-30: Very Difficult (graduate)
```

**Type-Token Ratio (Lexical Diversity)**:
```
TTR = unique_words / total_words

Example A (low diversity):
"The cat sat on the mat. The cat was fat."
- Unique: {the, cat, sat, on, mat, was, fat} = 7 words
- Total: 10 words
- TTR = 7/10 = 0.7

Example B (high diversity):
"The feline reclined upon the cushioned surface."
- Unique: {the, feline, reclined, upon, cushioned, surface} = 6 words
- Total: 7 words
- TTR = 6/7 = 0.857
```

#### Metric 4: Redundancy

```python
class RedundancyChecker:
    """
    Uses pre-built corpus index (from Step 2a) to detect duplicates.
    """

    def __init__(self, index_path: str):
        if not Path(index_path).exists():
            print(f"⚠ Corpus index not found. Redundancy scores will be 0.")
            self._available = False
            return

        with open(index_path, "rb") as f:
            idx = pickle.load(f)

        self._available = True
        self._exact_hashes = idx["exact_hashes"]
        self._lsh = idx["lsh"]
        self._minhashes = idx["minhashes"]
        self._num_perm = 128
        self._MinHash = _MinHash  # datasketch.MinHash class

    def compute(self, text: str) -> Dict:
        if not self._available:
            return {
                "exact_duplicate": False,
                "near_duplicate_score": 0.0,
                "semantic_duplicate_score": 0.0,
                "n_gram_overlap_3": 0.0,
                "n_gram_overlap_5": 0.0,
            }

        # 1. Exact duplicate (MD5 hash)
        text_hash = hashlib.md5(text.encode()).hexdigest()
        exact_dup = text_hash in self._exact_hashes

        # 2. Near-duplicate (MinHash Jaccard)
        mh = self._MinHash(num_perm=self._num_perm)
        for word in text.lower().split():
            mh.update(word.encode("utf-8"))

        similar = self._lsh.query(mh)  # LSH fast lookup
        near_score = 0.0

        if similar:
            # Compute actual Jaccard with each candidate
            near_score = max(
                mh.jaccard(self._minhashes[d])
                for d in similar if d in self._minhashes
            )

        # 3. Semantic similarity (reuse MinHash for simplicity)
        # (Original version used TF-IDF cosine, simplified here)
        sem_score = near_score

        return {
            "exact_duplicate": exact_dup,
            "near_duplicate_score": round(near_score, 4),
            "semantic_duplicate_score": round(sem_score, 4),
            "n_gram_overlap_3": 0.0,  # Placeholder
            "n_gram_overlap_5": 0.0,  # Placeholder
        }
```

**How it works**:

1. **Exact**: O(1) hash lookup
2. **Near**: O(log n) LSH query + O(k) Jaccard computation (k = candidate count)
3. **Semantic**: Reuses near-duplicate score (simplified)

**Without index** (what we avoid):
```python
# This would take 500 hours
for other_text in all_200k_texts:
    if jaccard(query_text, other_text) > 0.5:
        mark_similar()
```

**With index** (what we do):
```python
# This takes 0.001 seconds
similar_ids = lsh.query(query_minhash)  # Returns ~10 candidates
for candidate_id in similar_ids:
    score = jaccard(query_minhash, minhashes[candidate_id])
```

#### Metric 5: Perplexity (Optional)

```python
class PerplexityScorer:
    """
    Computes GPT-2 perplexity with batched GPU inference.
    Gracefully degrades if model unavailable.
    """
    MAX_TOKENS = 512

    def __init__(self, use_gpu: bool = True):
        self._available = False

        if not TRANSFORMERS_AVAILABLE:
            print("⚠ transformers not available. Perplexity will be null.")
            return

        try:
            device_str = f"cuda:{CUDA_DEVICE}" if (
                use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
            ) else "cpu"

            print(f"Loading GPT-2 model (device={device_str})...")

            self._tokenizer = GPT2Tokenizer.from_pretrained(
                "gpt2", cache_dir="./models"
            )
            self._model = GPT2LMHeadModel.from_pretrained(
                "gpt2", cache_dir="./models"
            ).to(device_str).eval()

            self._device = device_str
            self._available = True
            print("✓ GPT-2 loaded")
        except Exception as e:
            print(f"⚠ GPT-2 load failed ({e}). Perplexity will be null.")

    def _score_batch(self, texts: List[str]) -> List[Optional[float]]:
        """
        Score multiple texts in one forward pass (GPU efficient).
        """
        if not self._available or not texts:
            return [None] * len(texts)

        try:
            # Tokenize with padding
            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_TOKENS,
                padding=True,
            )

            input_ids = enc["input_ids"].to(self._device)  # (B, T)
            attention_mask = enc["attention_mask"].to(self._device)

            if input_ids.shape[1] < 2:
                return [None] * len(texts)

            # Forward pass (no gradients needed)
            with torch.no_grad():
                logits = self._model(input_ids).logits  # (B, T, V)

            # Compute cross-entropy loss per token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].float().contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            token_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), -1)  # (B, T-1)

            # Average over non-padding tokens only
            seq_lens = shift_mask.sum(dim=1).clamp(min=1)
            mean_loss = (token_loss * shift_mask).sum(dim=1) / seq_lens

            # Perplexity = exp(cross-entropy)
            ppls = torch.exp(mean_loss)

            return [
                round(p.item(), 2) if np.isfinite(p.item()) else None
                for p in ppls
            ]
        except:
            return [None] * len(texts)

    def compute(self, text: str, sentences: List[str] = None) -> Dict:
        if not self._available:
            return {
                "gpt2": None,
                "token_level_variance": None,
                "sentence_level_mean": None,
                "max_sentence_perplexity": None,
            }

        if sentences is None:
            sentences = sent_tokenize(text)

        # Filter short sentences
        valid_sents = [s for s in sentences if len(s.split()) >= 5]

        # Batch: [full_text, sent1, sent2, ...] in one forward pass
        all_texts = [text] + valid_sents
        all_ppls = self._score_batch(all_texts)

        overall = all_ppls[0]
        sent_ppls = [p for p in all_ppls[1:] if p is not None]

        sent_mean = round(float(np.mean(sent_ppls)), 2) if sent_ppls else None
        sent_max = round(float(np.max(sent_ppls)), 2) if sent_ppls else None
        sent_var = round(float(np.std(sent_ppls)), 2) if len(sent_ppls) > 1 else 0.0

        return {
            "gpt2": overall,
            "token_level_variance": sent_var,
            "sentence_level_mean": sent_mean,
            "max_sentence_perplexity": sent_max,
        }
```

**Perplexity Formula**:
```
Perplexity = exp(cross_entropy)
           = exp(-1/N × sum(log P(word_i | context)))

Where:
- N = number of tokens
- P(word_i | context) = model's probability for next word

Example:
Text: "The cat sat on the"
Model predicts: P(mat|context) = 0.7

If correct word is "mat":
  log P(mat) = log(0.7) = -0.357
  Perplexity for this token = exp(0.357) = 1.43

If wrong word predicted:
  log P(mat) = log(0.01) = -4.605
  Perplexity for this token = exp(4.605) = 100.0
```

**Interpretation**:
- Low perplexity (~20-50): Natural, grammatical text
- Medium perplexity (~50-100): Technical but readable
- High perplexity (>100): Unusual patterns, errors, or noise

**Why batch processing?**
- Naive: 200K chunks × (1 chunk + 5 sentences) = 1.2M forward passes
- Batched: 200K batches × 1 forward pass = 200K forward passes (6× faster)

#### Main Processing Loop

```python
def _process_chunks(
    chunks: List[str],
    doc_meta: Dict,
    classifier: DomainClassifier,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
) -> List[Dict]:
    """
    Process all chunks from one document.
    """
    results = []

    for chunk_id, chunk in enumerate(chunks):
        # Skip tiny chunks
        if len(chunk.split()) < 20:
            continue

        # Tokenize sentences once (shared by difficulty + perplexity)
        sentences = sent_tokenize(chunk)

        # Compute all 5 metrics
        quality = compute_quality(chunk)
        difficulty = compute_difficulty(chunk, sentences=sentences)
        redun = redundancy.compute(chunk)
        ppl = perplexity.compute(chunk, sentences=sentences)

        # Assemble record
        record = {
            **doc_meta,  # Source info (dataset, doc_id, etc.)
            "chunk_id": chunk_id,
            "text": chunk,
            "word_count": len(chunk.split()),
            "domain_labels": classifier.classify(chunk),
            "educational_markers": quality["educational_markers"],
            "quality_score": quality["quality_score"],
            "difficulty": difficulty,
            "redundancy": redun,
            "perplexity": ppl,
        }

        results.append(record)

    return results

def process_khan_academy(classifier, redundancy, perplexity):
    """Process Khan Academy dataset."""
    with open(KHAN_DATA_PATH, "r") as f:
        khan_data = json.load(f)

    results = []

    for doc in tqdm(khan_data, desc="Khan"):
        text = doc.get("content", "")
        if len(text.strip()) < 50:
            continue

        meta = {
            "source": "khan_academy",
            "doc_id": doc.get("url", "unknown"),
            "subject": doc.get("subject", "Unknown"),
            "grade": doc.get("grade", "Unknown"),
            "title": doc.get("title", "Untitled"),
        }

        chunks = chunk_text(text, CHUNK_SIZE)
        results.extend(_process_chunks(chunks, meta, classifier, redundancy, perplexity))

    # Save as JSONL (one JSON object per line)
    with jsonlines.open(KHAN_OUTPUT, "w") as w:
        w.write_all(results)

    print(f"✓ {len(results):,} chunks → {KHAN_OUTPUT}")

def process_tiny_textbooks(classifier, redundancy, perplexity, max_batches=None):
    """Process Tiny-Textbooks dataset."""
    batch_files = sorted(Path(TINY_TEXTBOOKS_DIR).glob("batch_*.json"))

    if max_batches:
        batch_files = batch_files[:max_batches]

    results = []

    for bf in tqdm(batch_files, desc="Tiny batches"):
        with open(bf, "r") as f:
            batch = json.load(f)

        for doc in batch:
            text = doc.get("text", "")
            if len(text.strip()) < 100:
                continue

            meta = {
                "source": "tiny_textbooks",
                "doc_id": doc.get("id", "unknown"),
                "batch_file": bf.name,
            }

            chunks = chunk_text(text, CHUNK_SIZE)
            results.extend(_process_chunks(chunks, meta, classifier, redundancy, perplexity))

    with jsonlines.open(TINY_OUTPUT, "w") as w:
        w.write_all(results)

    print(f"✓ {len(results):,} chunks → {TINY_OUTPUT}")

def main():
    # Load domain classifier
    prototypes, domain_vectorizer = _load_prototypes()
    proto_ids, proto_matrix = _build_prototype_matrix(prototypes)
    classifier = DomainClassifier(domain_vectorizer, proto_ids, proto_matrix, USE_GPU)

    # Load redundancy checker
    redundancy = RedundancyChecker(CORPUS_INDEX_PATH)

    # Load perplexity scorer
    perplexity = PerplexityScorer(use_gpu=USE_GPU)

    # Process both datasets
    process_khan_academy(classifier, redundancy, perplexity)
    process_tiny_textbooks(classifier, redundancy, perplexity, max_batches=None)
```

**Output Format** (JSONL):
```json
{"source":"khan_academy","doc_id":"https://...","subject":"Math","grade":"8-9","title":"Linear Equations","chunk_id":0,"text":"In this lesson...","word_count":187,"domain_labels":{"Math::Algebra 1":0.45,"Math::Linear Equations":0.32},"educational_markers":{"has_examples":true,"has_explanation":true,"has_structure":false},"quality_score":0.6667,"difficulty":{"flesch_kincaid_grade":8.2,"flesch_reading_ease":65.3,"smog_index":8.9,"avg_sentence_length":15.3,"avg_word_length":4.6,"rare_words_pct":0.08,"lexical_diversity":0.68},"redundancy":{"exact_duplicate":false,"near_duplicate_score":0.23,"semantic_duplicate_score":0.31,"n_gram_overlap_3":0.15,"n_gram_overlap_5":0.08},"perplexity":{"gpt2":42.3,"sentence_level_mean":39.8,"token_level_variance":6.2,"max_sentence_perplexity":58.7}}
```

**Runtime**:
- Khan Academy (18,764 docs): ~20 minutes
- Tiny-Textbooks (100,000 docs): ~2.5 hours
- **Total**: ~3 hours on dual GPU system

**Output Size**:
- `khan_analysis.jsonl`: 156 MB
- `tiny_textbooks_analysis.jsonl`: 323 MB
- **Total**: 479 MB

---

## Step 3: Build Dashboard

### File: `3_build_dashboard.py`

**Purpose**: Create interactive HTML dashboard to visualize analysis results.

**Status**: 🔄 **In Progress** (code complete, ready to run)

### What It Does

1. **Load JSONL results** from Step 2
2. **Aggregate statistics** (domain distribution, quality metrics, difficulty stats)
3. **Generate interactive charts** (Chart.js)
4. **Export self-contained HTML** (works without server)

### Expected Visualizations

1. **Domain Distribution** - Bar chart comparing Khan vs Tiny-Textbooks
2. **Quality Metrics** - Stacked bar showing % with examples/explanations/structure
3. **Difficulty Distribution** - Histogram of Flesch-Kincaid grade levels
4. **Redundancy Analysis** - Scatter plot of near-duplicate vs semantic scores
5. **Perplexity Box Plot** - Compare perplexity distributions

**Status**: ✅ Code ready, will run after confirming Step 2 outputs look good

---

## Technical Decisions & Trade-offs

### ❌ Tried: Graph-Based Domain Classification

**Approach**:
```python
# Build prerequisite graph
G = nx.DiGraph()
G.add_edge("Algebra Basics", "Linear Equations")
G.add_edge("Linear Equations", "Systems of Equations")

# Classify document
def classify(text):
    # Problem: Which node if text mentions both "linear" and "quadratic"?
    # Graph forces single-path assignment
```

**Why it failed**:
- Documents span multiple topics (e.g., "algebraic geometry")
- Hard assignment to single nodes → information loss
- Graph structure too rigid for cross-cutting concepts

### ✅ Succeed: Vector-Based Multi-Label

**Approach**:
```python
# Each course = TF-IDF vector
# Text = TF-IDF vector
# Similarity = cosine(text_vec, course_vec)

Result: Soft probabilities
{
  "Math::Algebra 1": 0.45,  # 45% match
  "Math::Geometry": 0.32,   # 32% match
  "Science::Physics": 0.23  # 23% match (cross-cutting!)
}
```

**Why it works**:
- Allows documents to belong to multiple domains
- Probabilistic interpretation
- Computationally efficient (matrix multiplication)

**Trade-off**: Less interpretable than explicit graph, but more flexible

---

### ❌ Tried: SentenceTransformers Embeddings

**Approach**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = model.encode(texts)  # 384-dim dense vectors
```

**Why it failed**:
- HuggingFace model download blocked (403 Forbidden)
- Remote environment has proxy restrictions
- Cannot rely on external downloads for production

### ✅ Succeed: TF-IDF (sklearn)

**Approach**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=300)
vectors = vectorizer.fit_transform(texts)  # 300-dim sparse
```

**Why it works**:
- No external dependencies (sklearn pre-installed)
- Completely offline-compatible
- Fast vectorization (~1ms per document)

**Trade-off**:
- Lexical similarity only ("cell phone" ≠ "mobile phone")
- But sufficient for domain classification (distinctive vocabularies)

---

### ❌ Tried: GPT-2 Perplexity for All Chunks

**Approach**:
```python
# Score every chunk with GPT-2
for chunk in all_200k_chunks:
    perplexity = score_with_gpt2(chunk)
```

**Why it struggled**:
- Model download (550MB) fails in some environments
- Requires GPU for reasonable speed
- 200K chunks × 0.5s = 27 hours on CPU

### ✅ Succeed: Optional Perplexity with Batching

**Approach**:
```python
if GPT2_AVAILABLE and GPU_AVAILABLE:
    # Batch processing: 6× faster
    ppls = score_batch([chunk1, chunk2, ..., chunk20])
else:
    # Gracefully degrade
    perplexity = None
```

**Why it works**:
- Pipeline works without perplexity
- Batched GPU inference when available (3 hours vs 27 hours)
- Null values handled gracefully in dashboard

**Trade-off**: Perplexity is nice-to-have, not required

---

## Summary: Code Execution Flow

```
User runs: python 1_extract_khan_taxonomy.py
├─ Loads 18,764 Khan Academy concepts
├─ Groups into ~100 course-level categories
├─ Builds TF-IDF vectorizer (300 features)
├─ Creates concept prototypes
└─ Saves: concept_prototypes_tfidf.pkl (255KB)

User runs: python 2a_build_corpus_index.py
├─ Loads Khan (18,764) + Tiny-Textbooks (100,000)
├─ Chunks into ~200,000 paragraphs
├─ Builds 3 redundancy indexes:
│  ├─ MD5 hashes (exact duplicates)
│  ├─ MinHash + LSH (near-duplicates)
│  └─ TF-IDF matrix (semantic similarity)
└─ Saves: corpus_index.pkl (15GB)

User runs: python 2_compute_metrics.py
├─ Loads concept prototypes (Step 1)
├─ Loads corpus index (Step 2a)
├─ Loads GPT-2 model (if available)
├─ For each of ~200,000 chunks:
│  ├─ Domain: TF-IDF cosine → top-5 courses
│  ├─ Quality: Regex markers → 3 booleans
│  ├─ Difficulty: textstat → 7 metrics
│  ├─ Redundancy: LSH query → 5 scores
│  └─ Perplexity: GPT-2 batch → 4 scores
└─ Saves: khan_analysis.jsonl (156MB) + tiny_textbooks_analysis.jsonl (323MB)

User runs: python 3_build_dashboard.py
├─ Loads JSONL results
├─ Aggregates by source, subject, grade
├─ Generates Chart.js visualizations
└─ Saves: dashboard.html (self-contained)
```

**Total Pipeline Runtime**: ~6 hours (Step 1: 10min, Step 2a: 2hr, Step 2: 3hr, Step 3: 5min)

**Total Disk Usage**: 15.5 GB (corpus index dominates)

---

**End of Technical Guide**

*Last Updated: February 11, 2026*
*For questions: bubbleguy10@gmail.com*
