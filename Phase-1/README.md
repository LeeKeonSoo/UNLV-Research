# Phase 1: Dataset Characterization

Comprehensive analysis of datasets to understand their unique characteristics through graph-based representation.

## Quick Start

### Current Status
We are transitioning from analyzing research datasets (The Pile) to building foundational knowledge graphs from K-12 curriculum data.

### Phase 1a: The Pile Analysis (Completed)
```bash
# Already completed - classification results available
python classify_domains.py          # 24-domain classification
python build_cooccurrence_graph.py  # Co-occurrence analysis
python visualize_sources.py         # Generate visualizations
```

### Phase 1b: K-12 Foundational Graphs (Current Focus)
```bash
# 1. Collect K-12 curriculum data
python collect_k12.py

# 2. Build concept graphs
python build_k12_graph.py

# 3. Analyze coverage
python analyze_k12_coverage.py
```

## Project Structure

```
Phase-1/
├── README.md                      # This file
├── description.md                 # Detailed project documentation
├── config.py                      # All configuration parameters
├── requirements.txt               # Python dependencies
│
├── Legacy: The Pile Analysis (Phase 1a)
├── classify_domains.py            # 24-domain classification
├── build_cooccurrence_graph.py    # Co-occurrence graphs
├── visualize_sources.py           # Visualization generation
├── pile/                          # Raw Pile data (~3GB)
└── results/classifications/       # Classification results
│
├── New: K-12 Curriculum Graphs (Phase 1b)
├── collect_k12.py                 # Data collection scripts
├── build_k12_graph.py             # Concept graph construction
├── analyze_k12_coverage.py        # Coverage analysis
├── k12_raw/                       # Raw K-12 data
├── k12_processed/                 # Processed data
├── k12_graphs/                    # Concept graphs
└── k12_reports/                   # Analysis reports
│
└── Utilities
    ├── utils.py                   # Helper functions
    └── domain_graph.py            # Legacy testing script
```

## Installation

```bash
# Create virtual environment
conda create -n phase1 python=3.11
conda activate phase1

# Install dependencies
pip install -r requirements.txt
```

## Workflows

### Workflow 1: Analyze The Pile (Already Done)
This workflow analyzed research datasets with 24-domain classification.

**Results available:**
- `results/classifications/ArXiv_classification.json` - Domain statistics
- `results/classifications/ArXiv_document_scores.json` - Per-document scores
- Visualizations (heatmap, stacked chart, 3D)

**Key findings:**
- ArXiv: 71% Math, 53% Physics
- PubMed: 85% Medicine
- Wikipedia: Most balanced
- Each source has clear specialization

### Workflow 2: Build K-12 Knowledge Graph (Current)
This workflow builds foundational concept graphs from curriculum data.

**Steps:**
1. **Data Collection** (`collect_k12.py`)
   - Khan Academy curriculum
   - OpenStax textbooks
   - Curated content

2. **Graph Construction** (`build_k12_graph.py`)
   - Cluster documents by concept
   - Build hierarchical structure
   - Identify prerequisites

3. **Analysis** (`analyze_k12_coverage.py`)
   - Coverage metrics
   - Gap identification
   - Grade-level distribution

**Expected outputs:**
- K-12 concept graphs (JSON)
- Coverage analysis report
- Visualization of concept hierarchy

## Configuration

Edit `config.py` to customize:

### K-12 Data Sources
```python
K12_SOURCES = {
    "khan_academy": {...},
    "openstax": {...},
    "ck12": {...}
}
```

### Graph Construction
```python
MIN_CLUSTER_SIZE = 50          # Min docs per concept
MAX_CLUSTER_DEPTH = 3          # Max hierarchy depth
SIMILARITY_THRESHOLD = 0.85    # Deduplication threshold
```

### Processing
```python
BATCH_SIZE = 16
TEXT_MAX_LENGTH = 1000
CONFIDENCE_THRESHOLD = 0.7
```

## Key Concepts

### Why K-12 First?
Research datasets (ArXiv, etc.) assume foundational knowledge. We need to:
1. Build from basics (1+1=2 → addition → algebra → calculus)
2. Create verifiable foundation (textbooks are explicit)
3. Establish baseline for comparing advanced content

### Graph-Based Discovery
Instead of pre-defining concepts, we:
1. Cluster similar documents
2. Discover natural concept groupings
3. Let data reveal hierarchy
4. Build bottom-up, not top-down

### Concept-Level Deduplication
Rather than "this document = Math", we identify:
- Specific concepts taught (e.g., "linear equations", "photosynthesis")
- Redundant examples (10 docs teaching same concept)
- Coverage gaps (missing concepts)

## Troubleshooting

### "No data found"
Run data collection first:
```bash
python collect_k12.py
```

### "Out of memory"
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 8  # or 4
```

### "GPU not detected"
CPU is fine but slower. Or set GPU explicitly:
```python
GPU_INDEX = 0  # Use first GPU
```

## Next Steps

### Phase 1b Completion (Current)
- [ ] Complete K-12 data collection
- [ ] Build and validate concept graphs
- [ ] Generate coverage reports
- [ ] Identify gaps

### Phase 2 (Future)
- Compare K-12 baseline with The Pile
- Determine optimal mixing ratios
- Plan refined dataset composition

### Phase 3 (Future)
- Build refined 10-20GB dataset
- Train small model for validation
- Produce final deliverables

## Documentation

See `description.md` for:
- Detailed project evolution
- Implementation plans
- Expected outcomes
- Future phases

## License

Research project - UNLV DiSC Lab
