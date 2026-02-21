# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

Run the complete K-12 curriculum analysis pipeline:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect K-12 data (5-10 minutes, network-dependent)
python collect_k12.py

# 3. Build concept graphs (2-3 minutes, GPU-accelerated if available)
python build_k12_graph.py

# 4. Generate interactive visualizations (1 minute)
python visualize_k12_graph.py

# 5. Optional: Analyze coverage and identify gaps
python analyze_k12_coverage.py
```

View results by opening HTML files in `k12_reports/` in a web browser.

## Project Overview

This is **Phase 1 of a dataset characterization research project** at UNLV. The goal is to build a foundational knowledge baseline from K-12 curriculum data to understand what prerequisite knowledge advanced datasets (like The Pile) assume.

**Key Research Insight:** Research papers teach "convergence proofs" but assume you know "convergence". We need to build from basics (1+1=2 → addition → algebra → calculus → research) to understand what a good training dataset should include.

**Approach:**
- Bottom-up concept discovery using clustering (data-driven, not pre-defined taxonomies)
- Hierarchical knowledge graphs showing prerequisite relationships
- Curriculum-based dataset characterization starting with K-12 foundations

**Current Phase:** Collecting and analyzing K-12 curriculum to establish foundational baseline.

**Future Phases:**
- Phase 2: Compare The Pile against K-12 baseline, identify what advanced content adds
- Phase 3: Build refined 10-20GB dataset with optimal foundation/advanced mixing

## Pipeline Architecture

The codebase implements a 3-stage pipeline. Understanding the flow between these stages is critical:

```
┌─────────────────────────────────────────────────┐
│ STAGE 1: Data Collection                        │
│ File: collect_k12.py                            │
├─────────────────────────────────────────────────┤
│ • Scrapes OpenStax textbooks via HTTP/BeautifulSoup
│ • Includes pre-made curated K-12 samples        │
│ • Covers 9 books: Algebra, Biology, Chemistry,  │
│   Physics, US History, American Government, etc.│
│ • Output: k12_raw/openstax/ + k12_raw/curated/  │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ STAGE 2: Graph Construction                     │
│ File: build_k12_graph.py                        │
├─────────────────────────────────────────────────┤
│ 1. Load documents from k12_raw/                 │
│ 2. Group by subject (math, science, soc.stud.)  │
│ 3. For each grade within subject:               │
│    - Generate sentence embeddings               │
│    - Apply hierarchical clustering              │
│    - Label clusters as "concepts"               │
│ 4. Build hierarchical graph structure           │
│ • Output: k12_graphs/*.json (one per subject)   │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ STAGE 3: Visualization                          │
│ File: visualize_k12_graph.py                    │
├─────────────────────────────────────────────────┤
│ • Loads graphs from k12_graphs/                 │
│ • Uses NetworkX for layout algorithms           │
│ • Renders with Plotly for interactivity         │
│ • Creates 4 complementary visualizations:       │
│   1. 3D network (rotate/zoom)                   │
│   2. Sunburst hierarchy (drill-down)            │
│   3. Coverage heatmap (grade × subject)         │
│   4. Distribution charts (bar charts)           │
│ • Output: k12_reports/*.html                    │
└─────────────────────────────────────────────────┘
```

All configuration is centralized in [config.py](config.py).

## Key Configuration (config.py)

### Subject Taxonomy

The project uses a K-12-focused taxonomy defined in `K12_SUBJECTS`:
- **Mathematics**: counting, arithmetic, fractions, algebra, geometry, trigonometry, statistics
- **Science**: life science, physical science, earth science, biology, chemistry, physics
- **Language Arts**: phonics, grammar, writing, literature, composition
- **Social Studies**: geography, history, civics, government, economics

### Clustering Parameters

Concept discovery is controlled by:
- `MIN_CLUSTER_SIZE = 50` - Minimum documents to form a concept cluster
- `SIMILARITY_THRESHOLD = 0.85` - Threshold for deduplication
- `MIN_EXAMPLES_PER_CONCEPT = 10` - Minimum docs per concept
- `MAX_EXAMPLES_PER_CONCEPT = 100` - Maximum docs to keep per concept

### Directory Structure

Paths are defined as constants:
- `K12_RAW_DIR = "k12_raw"` - Raw collected data
- `K12_PROCESSED_DIR = "k12_processed"` - Intermediate processed data
- `K12_GRAPHS_DIR = "k12_graphs"` - Generated concept graphs (JSON)
- `K12_REPORTS_DIR = "k12_reports"` - Visualization outputs (HTML)

### Device Configuration

The system auto-detects available hardware:
- CUDA GPUs: Automatically selects least-used GPU
- Apple MPS: Uses Metal Performance Shaders on M1/M2 Macs
- CPU fallback: Works fine but slower (5 min vs 2 min for graph construction)

Check device status in terminal output: `🎮 Device: ...`

## Data Flow & Directory Structure

```
k12_raw/
├── openstax/                    # Scraped OpenStax textbooks (~880 KB)
│   ├── elementary-algebra-2e.json
│   ├── biology-2e.json
│   ├── chemistry-2e.json
│   └── ... (9 books total)
│
└── curated/                     # Pre-made K-12 samples
    ├── mathematics/
    │   ├── counting/
    │   ├── arithmetic/
    │   ├── algebra_1/
    │   └── ... (25+ domains)
    ├── science/
    ├── english/
    └── social_studies/

k12_graphs/                      # Generated concept graphs (~4 KB total)
├── mathematics_graph.json       # ~1.3 KB
├── science_graph.json           # ~1.3 KB
└── social_studies_graph.json   # ~1.3 KB

k12_reports/                     # Interactive visualizations (~18 MB total)
├── 3d_network.html             # Rotatable 3D force-directed network
├── sunburst.html               # Hierarchical drill-down view
├── coverage_heatmap.html       # Grade × Subject coverage matrix
└── distribution.html           # Bar charts for concept/doc distribution
```

**Data Statistics:**
- Raw data: ~880 KB (9 OpenStax books + 39 curated samples)
- Processed graphs: ~4 KB total (3 subject graphs)
- Visualizations: ~18 MB (4 HTML files with embedded Plotly.js)
- Typical document count: 40-60 total
- Typical concept count: ~15 total (5 per subject after clustering)

## Key Algorithms

### 1. Document Embedding
**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight (22M params), fast on CPU/GPU
- Converts text to 384-dimensional vectors
- Good for semantic similarity tasks

### 2. Concept Clustering
**Method:** `sklearn.cluster.AgglomerativeClustering`
- Hierarchical agglomerative clustering
- Metric: Euclidean distance on embeddings
- Linkage: Ward (minimizes within-cluster variance)
- Default: n_clusters=3 per grade level (discovers 3 concept groups)

### 3. Concept Labeling
**Heuristic-based approach:**
- Extract document titles from each cluster
- Identify most common words (length > 3 characters)
- Generate label: `"{Common Words} ({document_count} docs)"`
- Example: "Equations Foundations (13 docs)"

Simple but effective for K-12 content with descriptive titles.

### 4. Graph Layout (3D Visualization)
**Method:** `NetworkX spring_layout(dim=3, k=2.5, iterations=50)`
- Algorithm: Fruchterman-Reingold force-directed layout
- k=2.5: optimal distance between nodes
- iterations=50: convergence quality
- Produces natural-looking 3D spatial arrangement

## Graph Structure

Graphs are stored as JSON with this hierarchical structure:

```python
{
  "nodes": {
    "concept_0": {
      "id": "concept_0",
      "name": "Mathematics",           # Subject (root)
      "subject": "mathematics",
      "grade_level": None,
      "document_count": 0,
      "parent": None,
      "children": ["concept_1", "concept_2"]
    },
    "concept_1": {
      "id": "concept_1",
      "name": "Grade 3",               # Grade level
      "subject": "mathematics",
      "grade_level": 3,
      "document_count": 0,
      "parent": "concept_0",
      "children": ["concept_3", "concept_4"]
    },
    "concept_3": {
      "id": "concept_3",
      "name": "Multiplication (13 docs)", # Discovered concept
      "subject": "mathematics",
      "grade_level": 3,
      "document_count": 13,
      "parent": "concept_1",
      "children": []
    }
  },
  "edges": [
    {
      "source": "concept_0",
      "target": "concept_1",
      "relationship": "parent-child",
      "weight": 1.0
    }
  ]
}
```

**Hierarchy:**
```
Subject (root node)
└── Grade Level (1-12)
    └── Concept Cluster (discovered via clustering)
        └── Documents (stored in document_count, not as separate nodes)
```

**Node Properties:**
- `id`: Unique identifier ("concept_N")
- `name`: Human-readable label
- `subject`: Which subject category
- `grade_level`: Target grade (None for subject/grade nodes)
- `document_count`: Number of backing documents
- `parent`: Parent concept ID (None for root)
- `children`: List of child concept IDs

**Edge Properties:**
- `source`, `target`: Connected concept IDs
- `relationship`: Type (parent-child, prerequisite, related)
- `weight`: Relationship strength (currently uniform 1.0)

## Data Collection Implementation

[collect_k12.py](collect_k12.py) contains two main collectors:

### FullTextbookCollector
- **Method:** HTTP scraping with BeautifulSoup
- **Sources:** OpenStax textbooks (9 books defined in `get_complete_catalog()`)
- **Process:** Fetches chapter HTML → parses content → saves to JSON
- **Rate limiting:** 0.5s delay between requests
- **Output format:** One JSON file per book with chapters array

### ComprehensiveSampleCollector
- **Method:** Pre-made K-12 content samples
- **Coverage:** 39 manually curated documents across grades K-12
- **Organization:** Hierarchical by subject/domain/concept
- **Purpose:** Ensures baseline coverage when OpenStax scraping fails

## Working with Modified Files

Current git status shows uncommitted changes in:
- [collect_k12.py](collect_k12.py) - Data collection (actively being developed)
- [visualize_k12_graph.py](visualize_k12_graph.py) - Visualization (actively being developed)

These are the most actively developed files. When working on them:
- Recent commits include bug fixes and graph configuration updates
- Changes may involve improving scraping reliability or visualization quality
- Test with: Run the full pipeline and verify outputs in `k12_reports/`

## Research Context

This project evolved from initial analysis of The Pile subsets:

**Phase 1a (Completed):** Analyzed The Pile using 24-domain classification
- Downloaded ArXiv, StackExchange, Wikipedia, GitHub, PubMed, FreeLaw (~3GB)
- Used DeBERTa for domain classification
- **Finding:** Research datasets are too advanced, assume foundational knowledge

**Phase 1b (Current):** K-12 Foundational Dataset
- Collecting K-12 curriculum from OpenStax and curated sources
- Building concept graphs with clustering
- Creating interactive visualizations

See [description.md](description.md) for detailed project evolution and research rationale.

## Expected Outputs

After running the full pipeline:
- ✅ ~40-60 documents collected (OpenStax chapters + samples)
- ✅ 3-4 subject graphs (mathematics, science, social_studies)
- ✅ 10-20 concept clusters discovered via clustering
- ✅ 4 interactive HTML visualizations in `k12_reports/`

**Validation:** Open HTML files in Chrome/Firefox. The 3D network should be rotatable/zoomable, sunburst should allow drill-down, heatmap should show grade×subject coverage.
