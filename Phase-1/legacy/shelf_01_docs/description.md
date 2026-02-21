# Phase 1: Dataset Characterization

## Goal
Conduct comprehensive analysis of existing datasets to understand their unique characteristics and strengths by building a hierarchical knowledge graph from foundational (K-12) curriculum data.

---

## Evolution of Approach

### Initial Approach (Completed)
**Goal:** Analyze The Pile subsets with domain classification

**What we did:**
1. **Downloaded The Pile subsets** (~3GB)
   - ArXiv, StackExchange, Wikipedia, Github, PubMed, FreeLaw
   
2. **24-domain classification** using DeBERTa
   - Classified documents across 6 categories, 24 domains
   - Generated per-document scores and aggregated statistics
   
3. **Visualization**
   - Heatmap: Source × Domain confidence
   - Stacked chart: Domain distribution by source
   - 3D visualization: Coverage relationships

**Key findings:**
- ArXiv: Strong in advanced mathematics (71%), physics (53%)
- PubMed: Dominant in health/medicine (85%), biology
- Wikipedia: Most balanced distribution across domains
- Each source has clear domain specialization

**Limitations identified:**
- Too coarse-grained (24 domains insufficient)
- Cannot identify concept-level redundancy
- Missing hierarchical structure (what's foundational vs advanced?)
- Research papers assume prerequisite knowledge

---

### Revised Approach (Current)
**Insight:** Need foundational knowledge first, then advanced content

**Key realization:**
- Research papers (ArXiv) teach "convergence proofs" but assume you know "convergence"
- Need to build from basics: 1+1=2 → addition → algebra → calculus → research
- Graph-based discovery: Let data reveal concept hierarchy, not pre-define it

**New strategy: K-12 Foundational Dataset First**

#### Why K-12?
1. **Foundational:** All advanced knowledge builds on K-12 curriculum
2. **Structured:** Already organized by grade, prerequisites clear
3. **Verifiable:** Textbooks explicitly state learning objectives
4. **Complete:** Covers basics that research datasets assume

#### Scope
- ✅ K-12 Mathematics (grades 1-12)
- ✅ K-12 Science (elementary through high school)
- ✅ K-12 Language Arts (reading, writing, grammar)
- ✅ K-12 Social Studies (history, civics, geography)
- ❌ College-level content (Phase 2)
- ❌ Research papers (Phase 2)

---

## Implementation Plan

### Step 1: K-12 Data Collection (Weeks 1-2)

**Primary sources:**
1. **Khan Academy**
   - Complete K-12 curriculum
   - Structured by grade/subject/topic
   - Articles, exercises, videos (transcripts)
   
2. **OpenStax Textbooks**
   - High-quality, peer-reviewed
   - Free, open-license
   - Books: Prealgebra, Algebra, Biology, Chemistry, Physics, etc.
   
3. **Common Core Standards + Problems**
   - Grade-by-grade learning objectives
   - Example problems and solutions
   
4. **CK-12 Foundation**
   - K-12 STEM curriculum
   - Structured content

**Output:** `k12_raw/` directory with collected data

### Step 2: Concept Graph Construction (Week 3)

**Approach: Bottom-up discovery**
1. Group documents by subject and grade level
2. Use clustering to discover natural concept groupings
3. Label clusters using LLM or manual inspection
4. Build hierarchical graph with prerequisites
5. Identify relationships (co-occurrence, prerequisites, related)

**Graph structure:**
```
Subject (Math)
└─ Grade Level (Grade 3)
   └─ Concept Cluster (Multiplication)
      └─ Sub-concepts (Times tables, Multi-digit, Word problems)
         └─ Documents (Teaching materials)
```

**Output:** 
- `k12_graphs/mathematics_graph.json`
- `k12_graphs/science_graph.json`
- `k12_graphs/full_k12_graph.json`

### Step 3: Characterization & Analysis (Week 4)

**Metrics to compute:**
1. **Coverage:** Which concepts are well-covered? Which have gaps?
2. **Depth:** How many levels deep does each concept go?
3. **Breadth:** How many distinct concepts per grade/subject?
4. **Prerequisites:** What's the dependency structure?
5. **Balance:** Is coverage even across grades and subjects?

**Analysis:**
- Concept coverage heatmap (grade × subject)
- Prerequisite chain visualization
- Gap identification (under-covered concepts)
- Document diversity within concepts

**Output:**
- `k12_characterization_report.md`
- Updated visualizations showing hierarchical structure

### Step 4: Deduplication Strategy (Week 5)

**Goal:** Remove redundant examples while preserving concept coverage

**Method:**
1. Within each concept cluster, compute document embeddings
2. Use semantic similarity to identify near-duplicates
3. Keep diverse representatives (maximal marginal relevance)
4. Ensure each concept has sufficient examples (min 10, max 100)

**Output:**
- `k12_deduplicated/` directory
- Deduplication statistics report

---

## Current Status

### Completed
- [x] The Pile subset collection and classification
- [x] 24-domain taxonomy analysis
- [x] Initial visualizations (heatmap, stacked, 3D)
- [x] Document-level score storage for co-occurrence analysis

### In Progress
- [ ] K-12 data source identification and access setup
- [ ] Data collection scripts (Khan Academy, OpenStax)
- [ ] Concept graph construction pipeline

### Next Steps
1. Set up data collection for Khan Academy
2. Parse OpenStax textbooks
3. Build initial concept graph from collected data
4. Validate graph structure with manual inspection

---

## File Structure

```
Phase-1/
├── description.md              # This file - project overview
├── config.py                   # Configuration for all pipelines
├── requirements.txt            # Python dependencies
│
├── pile/                       # Original Pile data (Phase 1a - completed)
├── results/classifications/    # Classification results from Pile
│
├── k12_sources.py             # NEW: K-12 data source definitions
├── collect_k12.py             # NEW: Data collection scripts
├── build_concept_graph.py     # NEW: Graph construction from K-12 data
├── analyze_k12_coverage.py    # NEW: Coverage analysis and reporting
│
├── k12_raw/                   # NEW: Raw K-12 collected data
├── k12_processed/             # NEW: Processed K-12 data
├── k12_graphs/                # NEW: Concept graphs
└── k12_reports/               # NEW: Analysis reports
```

---

## Key Insights

### From The Pile Analysis
1. Research datasets are **too advanced** - assume foundational knowledge
2. Simple domain classification (24 labels) is **too coarse**
3. Need **hierarchical structure** to understand concept relationships
4. **Redundancy is high** - many documents teach same concepts

### New Direction
1. **Start with foundations** - K-12 curriculum before research papers
2. **Build bottom-up** - discover concept hierarchy from data
3. **Graph-based representation** - captures prerequisites and relationships
4. **Concept-level deduplication** - remove redundant examples, keep diversity

---

## Expected Outcomes (Phase 1)

1. **K-12 Foundational Dataset** (5-10GB)
   - Complete coverage of grades 1-12
   - All major subjects (Math, Science, Language, Social Studies)
   - High quality, curriculum-aligned content

2. **Hierarchical Knowledge Graph**
   - Clear concept hierarchy (subject → grade → topic → concept)
   - Prerequisite chains (A → B → C)
   - Relationship mapping (co-occurrence, related concepts)

3. **Characterization Report**
   - What concepts are covered and to what depth
   - Where gaps exist
   - Baseline for comparing advanced datasets (Phase 2)

4. **Validation**
   - Small model (100M params) trained on foundational data
   - Performance on basic benchmarks (elementary math, simple reasoning)
   - Proof that "foundations matter" for SLM training

---

## Future Phases

### Phase 2: Advanced Dataset Integration
- Compare The Pile sources against K-12 baseline
- Identify what advanced content adds (beyond foundations)
- Determine optimal mixing ratios

### Phase 3: Refined Dataset Production
- Combine foundational + advanced content
- Apply concept-aware deduplication
- Produce 10-20GB refined dataset optimized for SLM training
