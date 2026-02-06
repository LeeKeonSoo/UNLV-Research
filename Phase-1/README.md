# Tiny-Textbooks Deep Hierarchical Analysis

**Comprehensive analysis of 420,000 synthetic textbook documents with 8-level deep domain hierarchy and interactive visualizations.**

---

## 📋 Overview

This project performs deep hierarchical analysis on the complete **Hugging Face `nampdn-ai/tiny-textbooks`** dataset:

- **Dataset Size:** 420,000 synthetic textbook documents
- **Hierarchy Depth:** 8 levels (Root → Broad Domain → Subject → Topic → Subtopic → Concept → Detail → Fine Detail)
- **Expected Nodes:** ~270,000 hierarchical nodes
- **Visualizations:** Interactive 3D HTML + Gephi analysis

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Three-Step Pipeline

#### 1. Collect Full Dataset (30-60 minutes)
```bash
python collect_tinytextbooks.py
```

#### 2. Build Deep Hierarchy (2-4 hours)
```bash
python build_deep_graph.py
```

**Output:**
- `graphs/deep_hierarchy.json` - For HTML visualization
- `graphs/deep_hierarchy.gexf` - For Gephi analysis

#### 3. Visualize

**Option A: HTML Visualization (lightweight, browser-based)**
```bash
python visualize_3d.py
open visualizations/3d_interactive.html
```

**Option B: Gephi Analysis (recommended for deep exploration)**
```bash
# If .gexf wasn't created automatically:
python export_gephi.py

# Then:
# 1. Download Gephi: https://gephi.org/
# 2. Open Gephi
# 3. File → Open → graphs/deep_hierarchy.gexf
# 4. Layout → Force Atlas 2 → Run
# 5. Statistics → Modularity → Run
# 6. Appearance → Nodes → Color by Modularity
# 7. Appearance → Nodes → Size by doc_count
```

---

## 📊 Visualization Comparison

| Feature | HTML (visualize_3d.py) | Gephi (export_gephi.py) |
|---------|------------------------|-------------------------|
| **Performance** | Lags at Level 8 (~270K nodes) | Smooth with 270K+ nodes |
| **Analysis Tools** | None | Statistics, Filters, Community Detection |
| **Layout** | Basic force-directed | Advanced Force Atlas 2 |
| **Interactivity** | Domain toggles | Full graph manipulation |
| **Best For** | Quick preview, presentations | Deep analysis, research |
| **Export** | Screenshot only | PDF, SVG, Statistics CSV |

**Recommendation:** Use Gephi for serious analysis. The HTML version is great for quick demos but becomes slow with all levels visible.

---

## 🔧 Files

### Core Pipeline
- `collect_tinytextbooks.py` - Download 420K docs from Hugging Face
- `build_deep_graph.py` - Build 8-level hierarchy (creates both .json and .gexf)
- `visualize_3d.py` - Generate interactive HTML
- `export_gephi.py` - Convert existing .json to .gexf (if needed)

### Configuration
- `config.py` - Clustering parameters, paths
- `utils.py` - Helper functions

---

## 📁 Directory Structure

```
Phase-1/
├── collect_tinytextbooks.py
├── build_deep_graph.py
├── visualize_3d.py
├── export_gephi.py          # NEW: Gephi export
├── config.py
├── utils.py
├── requirements.txt
├── README.md
│
├── tiny_textbooks_raw/      # 420K documents (42 batch files, ~10 GB)
├── graphs/                  
│   ├── deep_hierarchy.json  # For HTML visualization
│   └── deep_hierarchy.gexf  # For Gephi (NEW)
├── visualizations/
│   └── 3d_interactive.html
└── legacy/                  # Archived K-12 implementation
```

---

## 🎯 Gephi Workflow

### Quick Start
```bash
# After running build_deep_graph.py:
open graphs/deep_hierarchy.gexf  # Opens in Gephi (if installed)
```

### Analysis Steps

**1. Layout (2-3 minutes)**
- Layout → Force Atlas 2
- Click "Run"
- Wait until stabilizes
- Click "Stop"

**2. Community Detection**
- Statistics → Modularity
- Click "Run"
- View report

**3. Visual Styling**
- Appearance → Nodes → Color → Partition → Modularity Class
- Appearance → Nodes → Size → Ranking → doc_count (min: 1, max: 50)
- Appearance → Edges → Color → Unique (gray, opacity 0.2)

**4. Filtering**
- Filters → Topology → Degree Range
- Filter → Attributes → level (show specific levels)

**5. Export Results**
- File → Export → PDF/SVG (for papers/presentations)
- Statistics → Export table (CSV)
- Screenshot tool (high-quality images)

---

## 💡 Tips

### Performance
- **HTML lags at Level 8?** → Use Gephi instead
- **Gephi too slow?** → Filter by level or degree to show subgraph
- **Out of memory?** → Reduce batch size in collect script

### Analysis
- **Find important concepts:** Sort by degree centrality
- **Discover clusters:** Use Modularity detection
- **Compare domains:** Filter by level 2, color by modularity
- **Export for paper:** Use PDF export with anti-aliasing

### Workflow
1. Build graph once with `build_deep_graph.py`
2. Use HTML for quick demos to others
3. Use Gephi for your own analysis
4. Export pretty images from Gephi for presentations

---

## 🛠️ Troubleshooting

**Q: Gephi file not created?**
```bash
python export_gephi.py
```

**Q: NetworkX not installed?**
```bash
pip install networkx
```

**Q: HTML visualization slow?**
- This is expected with 270K nodes
- Use Gephi instead for deep exploration
- Or filter to show only Levels 1-5 in HTML

**Q: Gephi crashes on open?**
- Increase Gephi memory: Edit `gephi.conf`, set `-Xmx8g` (8GB RAM)
- Or filter graph before export (modify `export_gephi.py`)

---

## 📚 Next Steps

1. **Explore in Gephi:** Find natural domain clusters
2. **Extract statistics:** Degree distribution, modularity scores
3. **Compare with other datasets:** Extend to The Pile, C4
4. **Generate reports:** Export analysis for research documentation

---

## 🎓 Research Context

**Phase 1 (Current):** Build deep hierarchical knowledge graph from synthetic textbooks
- Establish baseline understanding of educational content structure
- Develop methodology for graph-based dataset characterization

**Phase 2 (Planned):** Extend analysis to large-scale datasets
- Apply same methodology to The Pile components, RedPajama, C4
- Compare "graph shapes" across different data sources

**Phase 3 (Planned):** Create refined SLM training dataset
- Use graph insights to balance domain coverage
- Develop principled dataset composition strategies

---

**Last Updated:** 2026-02-06
**Version:** 1.1 (Added Gephi support)
