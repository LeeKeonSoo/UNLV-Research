# Phase 1: Dataset Characterization for SLM Pretraining

**Research Goal**: Analyze domain coverage and quality characteristics of educational datasets to inform curriculum-aware pretraining strategies for Small Language Models (SLMs).

---

## 📋 Overview

This project develops a systematic methodology to characterize pretraining datasets along two critical dimensions:

1. **Domain Coverage**: What subjects, concepts, and granularities are present?
2. **Quality Metrics**: How natural, structured, and educational is the content?

**Current Status**: Week 4/16 - Foundation metrics implementation

---

## 🎯 Research Questions

1. How can we build a fine-grained domain taxonomy from educational resources?
2. What is the domain distribution of educational vs. synthetic textbook datasets?
3. How do quality metrics (perplexity, educational structure) compare across datasets?
4. What percentage of content exhibits cross-cutting concepts (multi-domain)?

---

## 📊 Datasets Analyzed

| Dataset | Size | Source | Characteristics |
|---------|------|--------|-----------------|
| **Khan Academy K-12** | ~2MB, 19 subjects | Educational content | Structured, grade-labeled, FAQ format |
| **Tiny-Textbooks** | ~10GB, 420K docs | GPT-3.5 generated | Synthetic, textbook format, unlabeled |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for readability metrics)
python -c "import nltk; nltk.download('punkt')"
```

### Three-Step Pipeline

#### Step 1: Extract Khan Academy Taxonomy (5-10 minutes)

```bash
python 1_extract_khan_taxonomy.py
```

**What it does**:
- Loads Khan Academy K-12 concepts
- Extracts Subject → Grade → Concept hierarchy
- Creates concept prototypes using SentenceTransformer embeddings
- Saves taxonomy and embeddings to `outputs/`

**Output**:
- `outputs/khan_taxonomy.json` - Hierarchical structure
- `outputs/concept_prototypes.pkl` - Concept embeddings (384-dim)
- `outputs/metadata.json` - Dataset statistics

---

#### Step 2: Compute Metrics (1-2 hours for full datasets)

```bash
python 2_compute_metrics.py
```

**What it does**:
- Loads concept prototypes from Step 1
- Processes both Khan Academy and Tiny-Textbooks
- For each text chunk (paragraph), computes:
  - **Domain Classification**: Multi-label, soft assignment via embedding similarity
  - **Quality Metrics**: Perplexity (GPT-2), educational markers (examples, explanations, structure)
- Saves annotated datasets to `outputs/`

**Output**:
- `outputs/khan_analysis.jsonl` - Khan Academy analysis results
- `outputs/tiny_textbooks_analysis.jsonl` - Tiny-Textbooks analysis results

**Configuration**:
- Edit `2_compute_metrics.py` to set `max_batches=5` for quick testing
- Full run processes all 42 Tiny-Textbooks batches (~420K documents)

---

#### Step 3: Build Dashboard (< 1 minute)

```bash
python 3_build_dashboard.py
```

**What it does**:
- Loads analysis results from Step 2
- Aggregates statistics (domain distributions, quality metrics)
- Generates interactive HTML dashboard with Chart.js visualizations

**Output**:
- `outputs/dashboard.html` - Interactive dashboard (open in browser)

**Dashboard Features**:
- Domain distribution comparison (Khan vs Tiny-Textbooks)
- Quality metrics comparison (perplexity, educational markers)
- Top 10 concepts by frequency
- Cross-cutting analysis (multi-domain percentage)
- Fully self-contained (no server required)

---

## 📁 Directory Structure

```
Phase-1/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── 1_extract_khan_taxonomy.py          # Step 1: Build taxonomy
├── 2_compute_metrics.py                # Step 2: Analyze datasets
├── 3_build_dashboard.py                # Step 3: Visualize results
│
├── khan_k12_concepts/                  # Khan Academy data (19 subjects)
│   ├── all_k12_concepts.json           # Merged dataset (982KB)
│   └── [subject]_[grade].json          # Individual subject files
│
├── tiny_textbooks_raw/                 # Tiny-Textbooks data (420K docs)
│   ├── batch_000.json
│   ├── batch_001.json
│   └── ... (42 batches total)
│
├── outputs/                            # Analysis outputs
│   ├── khan_taxonomy.json              # Hierarchical taxonomy
│   ├── concept_prototypes.pkl          # Concept embeddings
│   ├── metadata.json                   # Statistics
│   ├── khan_analysis.jsonl             # Khan analysis results
│   ├── tiny_textbooks_analysis.jsonl   # Tiny-Textbooks analysis
│   └── dashboard.html                  # Interactive dashboard
│
├── legacy/                             # Archived old implementations
│   ├── old_analysis/                   # Previous graph-based approach
│   └── ...
│
├── research_plan_refined.md            # Detailed research plan
└── subtask1_metrics_design.md          # Metrics documentation
```

---

## 🔬 Methodology

### Domain Classification

**Approach**: Embedding-based similarity to concept prototypes

1. **Concept Prototypes**: For each Khan Academy concept, embed all article content → 384-dim vector
2. **Query Embedding**: Embed each paragraph from target dataset
3. **Similarity Scoring**: Compute cosine similarity to all prototypes
4. **Multi-Label Assignment**: Top-K concepts with similarity > threshold → soft labels

**Example**:
```python
paragraph = "Fractions represent parts of a whole. For example, 1/2 means..."
domain_labels = {
    "Math - 4th Grade::Equivalent fractions": 0.78,
    "Math - 5th Grade::Add and subtract fractions": 0.52,
    "Reading - 3rd Grade": 0.15
}
# Sum = 1.0 (normalized probabilities)
```

**Advantages**:
- ✅ Handles unlabeled data
- ✅ Multi-label (cross-cutting concepts)
- ✅ Soft assignment (continuous scores, not binary)
- ✅ No manual labeling required

---

### Quality Metrics

#### 1. Perplexity (GPT-2)
- **Definition**: How "surprised" a language model is by the text
- **Lower = Better**: Natural, well-formed text
- **Typical Ranges**:
  - High-quality educational text: 30-60
  - Web-scraped text: 80-150
  - Noisy/corrupted text: >200

#### 2. Educational Markers (Binary Features)
- **Has Examples**: Contains "for example", "such as", "consider"
- **Has Explanation**: Contains "because", "therefore", "this means"
- **Has Structure**: Contains "first", "second", "finally", "in summary"

**Why these matter**:
- Educational content should teach, not just state facts
- Examples aid understanding
- Explanations build reasoning
- Structure aids retention

---

## 📈 Expected Results

### Domain Coverage

**Hypothesis**:
- Khan Academy: Well-balanced across K-12 subjects, sparse in advanced topics
- Tiny-Textbooks: More uniform distribution (GPT-generated), potential bias toward common topics

**Metrics**:
- Subject distribution (histogram)
- Multi-domain ratio (% of paragraphs with >1 domain label)
- Domain entropy (higher = more diverse)

### Quality Comparison

**Hypothesis**:
- Khan Academy: Lower perplexity (human-written, curated)
- Tiny-Textbooks: Slightly higher perplexity (GPT artifacts), but more consistent structure

**Metrics**:
- Average/median perplexity
- Educational marker prevalence (%)

---

## 🛠️ Troubleshooting

### Out of Memory (OOM) Errors

**Symptom**: `torch.cuda.OutOfMemoryError` or process killed

**Solutions**:
1. Reduce batch size in `2_compute_metrics.py`:
   ```python
   embeddings = model.encode(texts, batch_size=8)  # Default: 32
   ```

2. Process fewer batches for testing:
   ```python
   process_tiny_textbooks(..., max_batches=5)  # Test on 5 batches
   ```

3. Use CPU instead of GPU (slower, but more memory):
   ```python
   model = GPT2LMHeadModel.from_pretrained("gpt2")
   # Don't call .cuda()
   ```

### Slow Processing

**Expected Runtime**:
- Step 1 (Taxonomy extraction): 5-10 minutes
- Step 2 (Khan Academy): 10-15 minutes
- Step 2 (Tiny-Textbooks, full): 1-2 hours
- Step 3 (Dashboard): < 1 minute

**Speed Tips**:
- Test on subset first (`max_batches=5`)
- Use GPU if available (4060Ti + 3070Ti = ~24GB VRAM)
- Reduce `TOP_K_DOMAINS` in config (less similarity computations)

### Dashboard Not Showing Charts

**Issue**: `Chart.js` not loading (no internet)

**Solution**: Dashboard uses CDN for Chart.js. Requires internet connection to load.
If offline, download `chart.js` locally and update HTML `<script>` tag.

---

## 📚 Next Steps

### Short-term (Week 5-6)
1. ✅ Run pipeline on full datasets
2. ⏳ Validate domain classification (manual inspection)
3. ⏳ Add difficulty metrics (Flesch-Kincaid readability)
4. ⏳ Extend to The Pile sample (5GB stratified sample)

### Medium-term (Week 7-10)
1. Prerequisite mining (concept co-occurrence analysis)
2. Comparative analysis (Khan vs Tiny vs Pile)
3. Identify domain gaps for augmentation

### Long-term (Week 11+)
1. Train 100M-300M SLM with curriculum ordering
2. Validate that domain-balanced data → better efficiency

---

## 🎓 Research Context

**Phase 1 Goal (Current)**: Build domain characterization toolkit

**Phase 2 Goal (Planned)**: Train curriculum-aware SLM prototype

**Phase 3 Goal (Future)**: Systematic dataset refinement

**Timeline**: 16 weeks total (Week 4/16 as of Feb 2026)

**Deliverable**: Workshop/short paper at COLM 2026 or EMNLP 2026

---

## 🔗 Resources

- **Khan Academy Data**: Collected via `collect_khan_academy.py` (legacy)
- **Tiny-Textbooks**: [HuggingFace: nampdn-ai/tiny-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks)
- **SentenceTransformers**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **GPT-2**: [Hugging Face: gpt2](https://huggingface.co/gpt2)

---

## 📝 Citation

```bibtex
@misc{lee2026phase1,
  author = {Lee, Keonsoo},
  title = {Phase 1: Dataset Characterization for Small Language Model Pretraining},
  year = {2026},
  institution = {University of Nevada, Las Vegas},
  note = {Research in Progress}
}
```

---

## 💬 Contact

**Researcher**: Keonsoo Lee (bubbleguy10@gmail.com)

**Institution**: UNLV

**Last Updated**: February 10, 2026

---

## 📄 License

Research code for academic purposes. Khan Academy data subject to their [Terms of Service](https://www.khanacademy.org/about/tos). Tiny-Textbooks data under [Apache 2.0](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks).
