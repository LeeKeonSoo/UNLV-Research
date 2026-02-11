# Complete Pipeline Requirements for Claude Code

**Date**: February 10, 2026
**Researcher**: KeonSoo (bubbleguy10@gmail.com)
**Project**: Phase-1 Dataset Characterization for SLM Pretraining

---

## 🎯 Overall Goal

Analyze educational datasets (Khan Academy + Tiny-Textbooks) through **5 comprehensive metrics** and build an interactive dashboard for detailed exploration.

---

## 📊 Required Metrics (All 5 Dimensions)

### 1. Domain Coverage (도메인 커버리지)
**Output Format**: Vector of probabilities

```python
{
  "domain_labels": {
    "algebra_basics": 0.45,
    "equation_solving": 0.32,
    "linear_functions": 0.23,
    "geometry_fundamentals": 0.12,
    "statistics_intro": 0.08
  }
}
```

**Implementation**:
- Use Khan Academy K-12 concepts as taxonomy
- TF-IDF vectorization (300 dimensions)
- Cosine similarity for top-5 domain assignment
- Soft assignment (probabilistic scores)

---

### 2. Quality (품질)
**Output Format**: Boolean flags + aggregate scores

```python
{
  "educational_markers": {
    "has_examples": true,
    "has_explanation": true,
    "has_structure": false
  },
  "quality_score": 0.67  # Percentage of markers present
}
```

**Implementation**:
- Educational markers detection (keyword-based)
- Examples: "for example", "such as", "consider"
- Explanations: "because", "therefore", "this means"
- Structure: "first", "second", "in summary"

---

### 3. Difficulty (난이도)
**Output Format**: Vector of readability metrics

```python
{
  "difficulty": {
    "flesch_kincaid_grade": 8.5,      # Grade level (0-18+)
    "flesch_reading_ease": 65.2,      # 0-100 (higher = easier)
    "smog_index": 9.1,                 # Years of education needed
    "avg_sentence_length": 18.2,       # Words per sentence
    "avg_word_length": 4.8,            # Characters per word
    "rare_words_pct": 0.12,            # Percentage of uncommon words
    "lexical_diversity": 0.65          # Type-Token Ratio
  }
}
```

**Implementation**:
```python
import textstat
from collections import Counter

def compute_difficulty(text):
    # Readability scores
    fk_grade = textstat.flesch_kincaid_grade(text)
    flesch_ease = textstat.flesch_reading_ease(text)
    smog = textstat.smog_index(text)

    # Sentence/word complexity
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sent_len = len(words) / len(sentences)
    avg_word_len = sum(len(w) for w in words) / len(words)

    # Lexical diversity
    unique_words = len(set(words))
    ttr = unique_words / len(words)

    # Rare words (not in common 3000 word list)
    common_words = load_common_words()  # Top 3000 English words
    rare_count = sum(1 for w in words if w.lower() not in common_words)
    rare_pct = rare_count / len(words)

    return {
        "flesch_kincaid_grade": fk_grade,
        "flesch_reading_ease": flesch_ease,
        "smog_index": smog,
        "avg_sentence_length": avg_sent_len,
        "avg_word_length": avg_word_len,
        "rare_words_pct": rare_pct,
        "lexical_diversity": ttr
    }
```

**Required Libraries**:
- `textstat` - Readability metrics
- `nltk` - Sentence/word tokenization

---

### 4. Redundancy (중복도)
**Output Format**: Vector of duplication metrics

```python
{
  "redundancy": {
    "exact_duplicate": false,              # Exact match exists
    "near_duplicate_score": 0.0,           # MinHash similarity (0-1)
    "semantic_duplicate_score": 0.0,       # Embedding similarity (0-1)
    "duplicate_cluster_id": null,          # Cluster ID if duplicate
    "n_gram_overlap_3": 0.15,              # 3-gram overlap ratio
    "n_gram_overlap_5": 0.08               # 5-gram overlap ratio
  }
}
```

**Implementation**:
```python
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer

def compute_redundancy(text, corpus_index):
    """
    corpus_index: Pre-built index of all documents for comparison
    """
    # 1. Exact duplicate check
    text_hash = hashlib.md5(text.encode()).hexdigest()
    exact_dup = text_hash in corpus_index['exact_hashes']

    # 2. Near-duplicate (MinHash LSH)
    mh = MinHash(num_perm=128)
    for word in text.split():
        mh.update(word.encode('utf8'))

    similar_docs = corpus_index['lsh'].query(mh)
    near_dup_score = max([
        mh.jaccard(corpus_index['minhashes'][doc_id])
        for doc_id in similar_docs
    ]) if similar_docs else 0.0

    # 3. N-gram overlap
    def ngram_overlap(text, n):
        vectorizer = CountVectorizer(ngram_range=(n, n))
        # Compare with corpus n-grams
        overlap = compute_jaccard_with_corpus(text, n)
        return overlap

    ngram_3 = ngram_overlap(text, 3)
    ngram_5 = ngram_overlap(text, 5)

    # 4. Semantic duplicate (TF-IDF cosine similarity)
    text_vec = corpus_index['vectorizer'].transform([text])
    similarities = cosine_similarity(text_vec, corpus_index['corpus_matrix'])
    semantic_dup_score = similarities.max()

    return {
        "exact_duplicate": exact_dup,
        "near_duplicate_score": near_dup_score,
        "semantic_duplicate_score": semantic_dup_score,
        "n_gram_overlap_3": ngram_3,
        "n_gram_overlap_5": ngram_5
    }
```

**Required Libraries**:
- `datasketch` - MinHash LSH
- `sklearn` - TF-IDF and cosine similarity

**Computational Strategy**:
- Build corpus index ONCE before processing
- Use LSH for O(1) similarity queries instead of O(n²) comparison
- Process in batches to manage memory

---

### 5. Perplexity (당황도)
**Output Format**: Scalar + distribution metrics

```python
{
  "perplexity": {
    "gpt2": 45.3,                    # GPT-2 perplexity
    "gpt2_small": 48.7,              # Smaller model baseline
    "token_level_variance": 12.4,    # Std dev of token perplexities
    "sentence_level_mean": 42.1,     # Avg sentence perplexity
    "max_sentence_perplexity": 89.3  # Worst sentence
  }
}
```

**Implementation**:
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def compute_perplexity(text, model_name="gpt2"):
    """
    Compute perplexity using GPT-2
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids']

    # Compute loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    # Sentence-level analysis
    sentences = sent_tokenize(text)
    sent_perplexities = [
        compute_perplexity(sent, model_name)
        for sent in sentences
    ]

    # Token-level variance
    token_probs = torch.softmax(outputs.logits, dim=-1)
    token_perplexities = [
        1.0 / prob.item()
        for prob in token_probs[0, range(len(input_ids[0])), input_ids[0]]
    ]

    return {
        "gpt2": perplexity,
        "token_level_variance": np.std(token_perplexities),
        "sentence_level_mean": np.mean(sent_perplexities),
        "max_sentence_perplexity": max(sent_perplexities)
    }
```

**Required Libraries**:
- `transformers` - GPT-2 model
- `torch` - PyTorch backend

**Note**: This may fail in remote environment due to model download restrictions. If so, use offline model loading:
```python
# Pre-download model to local cache
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./models")
```

---

## 🔧 Pipeline Structure

### Step 1: Extract Khan Taxonomy (현재 코드 유지)
**File**: `1_extract_khan_taxonomy.py`
- Uses TF-IDF for concept prototypes
- Output: `outputs/khan_taxonomy.json`, `outputs/concept_prototypes_tfidf.pkl`

### Step 2: Compute ALL Metrics (대폭 확장 필요)
**File**: `2_compute_metrics.py`

**Required Changes**:
1. Add difficulty computation
2. Add redundancy detection (build corpus index first)
3. Add perplexity calculation (with offline fallback)
4. Output ALL 5 metric vectors per paragraph

**Output Format** (JSONL):
```json
{
  "source": "khan_academy",
  "doc_id": "https://...",
  "chunk_id": 0,
  "text": "...",
  "word_count": 157,

  "domain_labels": { "algebra": 0.45, "geometry": 0.32, ... },

  "educational_markers": {
    "has_examples": true,
    "has_explanation": true,
    "has_structure": false
  },
  "quality_score": 0.67,

  "difficulty": {
    "flesch_kincaid_grade": 8.5,
    "flesch_reading_ease": 65.2,
    "smog_index": 9.1,
    "avg_sentence_length": 18.2,
    "avg_word_length": 4.8,
    "rare_words_pct": 0.12,
    "lexical_diversity": 0.65
  },

  "redundancy": {
    "exact_duplicate": false,
    "near_duplicate_score": 0.0,
    "semantic_duplicate_score": 0.0,
    "n_gram_overlap_3": 0.15,
    "n_gram_overlap_5": 0.08
  },

  "perplexity": {
    "gpt2": 45.3,
    "token_level_variance": 12.4,
    "sentence_level_mean": 42.1,
    "max_sentence_perplexity": 89.3
  }
}
```

### Step 3: Build Interactive Dashboard (완전히 새로 작성)
**File**: `3_build_dashboard.py`

**Required Features**:

#### 3.1 Multiple View Modes
User can switch between:
1. **Dataset Overview** - High-level statistics comparison
2. **Domain View** - Filter/sort by specific domains
3. **Document Explorer** - Browse individual documents with all metrics
4. **Metric Deep Dive** - Detailed analysis of each dimension

#### 3.2 Data Explorer Features

**A. Dataset Overview Tab**:
- Total documents per dataset
- Average scores across all 5 dimensions
- Distribution histograms (domain coverage, difficulty, quality)
- Comparison charts (Khan vs Tiny-Textbooks)

**B. Domain View Tab**:
- Dropdown: Select domain (algebra, geometry, etc.)
- Show all documents matching that domain
- Sort by: relevance score, quality, difficulty, perplexity
- Display metrics for each document

**C. Document Explorer Tab**:
- Table view with sortable columns:
  - Document ID
  - Source (Khan/Tiny-Textbooks)
  - Top domain
  - Quality score
  - Difficulty grade level
  - Redundancy score
  - Perplexity
- Click to expand: Show full text + all metric vectors
- Pagination (50 documents per page)

**D. Metric Deep Dive Tab**:
Each metric gets its own sub-section:

1. **Domain Coverage**:
   - Bar chart: Distribution across all domains
   - Heatmap: Domain co-occurrence matrix
   - Filter: Show documents with >0.3 score in selected domain

2. **Quality**:
   - Pie chart: Percentage with examples/explanations/structure
   - Scatter plot: Quality score vs difficulty
   - Filter: Show only high-quality (score > 0.8)

3. **Difficulty**:
   - Histogram: Grade level distribution
   - Box plot: Compare Khan vs Tiny-Textbooks
   - Filter: Select grade range (e.g., 6-8)

4. **Redundancy**:
   - Duplicate cluster visualization
   - List of near-duplicate pairs
   - N-gram overlap heatmap

5. **Perplexity**:
   - Distribution plot
   - Outlier detection (very high/low perplexity)
   - Correlation with other metrics

#### 3.3 Interactive Controls
```html
<!-- Filters -->
<div class="filters">
  <select id="datasetFilter">
    <option value="all">All Datasets</option>
    <option value="khan">Khan Academy</option>
    <option value="tiny">Tiny-Textbooks</option>
  </select>

  <select id="domainFilter">
    <option value="all">All Domains</option>
    <option value="math">Mathematics</option>
    <option value="science">Science</option>
    ...
  </select>

  <input type="range" id="qualitySlider" min="0" max="1" step="0.1">
  <label>Min Quality: <span id="qualityValue">0.5</span></label>

  <input type="range" id="difficultySlider" min="0" max="18" step="1">
  <label>Grade Level: <span id="difficultyValue">8</span></label>
</div>

<!-- Document Table -->
<table id="docTable">
  <thead>
    <tr>
      <th onclick="sortBy('doc_id')">Document ID ↕</th>
      <th onclick="sortBy('source')">Source ↕</th>
      <th onclick="sortBy('domain')">Top Domain ↕</th>
      <th onclick="sortBy('quality')">Quality ↕</th>
      <th onclick="sortBy('difficulty')">Grade Level ↕</th>
      <th onclick="sortBy('redundancy')">Redundancy ↕</th>
      <th onclick="sortBy('perplexity')">Perplexity ↕</th>
      <th>Actions</th>
    </tr>
  </thead>
  <tbody id="docTableBody">
    <!-- Populated by JavaScript -->
  </tbody>
</table>

<!-- Document Detail Modal -->
<div id="docModal" class="modal">
  <div class="modal-content">
    <h2>Document Details</h2>
    <pre id="docText"></pre>

    <h3>Domain Labels (Vector)</h3>
    <canvas id="domainChart"></canvas>

    <h3>Difficulty Metrics</h3>
    <table id="difficultyTable"></table>

    <h3>Redundancy Analysis</h3>
    <div id="redundancyInfo"></div>

    <h3>Perplexity Breakdown</h3>
    <canvas id="perplexityChart"></canvas>
  </div>
</div>
```

#### 3.4 Visualization Libraries
- **Chart.js** - Bar charts, line charts, scatter plots
- **D3.js** - Heatmaps, custom visualizations
- **DataTables.js** - Sortable/filterable tables

#### 3.5 Dashboard Structure
```javascript
// Main dashboard object
const dashboard = {
  data: null,  // All analysis results

  // Filter state
  filters: {
    dataset: 'all',
    domain: 'all',
    minQuality: 0.0,
    gradeLevel: [0, 18],
    showDuplicates: false
  },

  // Methods
  loadData: function() { /* Load JSONL */ },
  applyFilters: function() { /* Filter documents */ },
  renderTable: function() { /* Update document table */ },
  showDocument: function(docId) { /* Open modal */ },

  // Visualizations
  charts: {
    domainDistribution: null,
    qualityComparison: null,
    difficultyHistogram: null,
    perplexityScatter: null
  },

  updateCharts: function() { /* Re-render all charts */ }
};
```

---

## 📦 Required Dependencies

Add to `requirements.txt`:
```
# Existing
scikit-learn
numpy
pandas
tqdm

# NEW: Difficulty metrics
textstat
nltk

# NEW: Redundancy detection
datasketch
hashlib  # Built-in

# NEW: Perplexity (optional, may fail in remote env)
torch
transformers

# Utilities
jsonlines
```

---

## 🚀 Execution Flow

### Build Corpus Index (NEW Step 2a)
```bash
python 2a_build_corpus_index.py
```
- Reads all documents
- Builds MinHash LSH index
- Saves to `outputs/corpus_index.pkl`
- Required for redundancy detection

### Compute All Metrics (Updated Step 2)
```bash
python 2_compute_metrics.py
```
- Loads corpus index
- Computes all 5 metric vectors
- Saves to `outputs/khan_analysis.jsonl` and `outputs/tiny_textbooks_analysis.jsonl`

### Build Dashboard (Completely Rewritten Step 3)
```bash
python 3_build_dashboard.py
```
- Loads analysis results
- Generates self-contained HTML with embedded data
- Saves to `outputs/dashboard.html`

---

## 🎨 Dashboard Design Requirements

1. **Responsive**: Works on desktop and tablet
2. **Fast**: Handle 100k+ documents with virtual scrolling
3. **Self-contained**: No external dependencies (all data embedded)
4. **Professional**: Clean UI, consistent colors, tooltips

---

## ✅ Success Criteria

- [ ] All 5 metrics computed for every paragraph
- [ ] Dashboard loads in <3 seconds
- [ ] Can filter/sort by any metric
- [ ] Can view individual document details
- [ ] Can compare datasets side-by-side
- [ ] Can export filtered results as CSV

---

**이 문서를 Claude Code에 전달하면 전체 파이프라인을 구현할 수 있습니다.**
