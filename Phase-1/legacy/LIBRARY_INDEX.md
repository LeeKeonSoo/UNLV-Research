# Legacy Library Index

This folder is an archive of historical experiments and deprecated pipelines.
These files are preserved for traceability, not for active Phase-1 execution.

## Shelf Layout

### shelf_01_docs
Purpose: historical notes and narrative context.

Files:
- `shelf_01_docs/CLAUDE_old.md`
- `shelf_01_docs/description.md`

### shelf_02_collection
Purpose: early data collection scripts.

Files:
- `shelf_02_collection/collect_k12.py`
- `shelf_02_collection/download_pile.py`
- `shelf_02_collection/config.py`
- `shelf_02_collection/utils.py`

### shelf_03_graph_pipeline
Purpose: legacy domain classification and graph construction pipeline.

Files:
- `shelf_03_graph_pipeline/classify_domains.py`
- `shelf_03_graph_pipeline/build_k12_graph.py`
- `shelf_03_graph_pipeline/build_cooccurrence_graph.py`
- `shelf_03_graph_pipeline/analyze_k12_coverage.py`
- `shelf_03_graph_pipeline/domain_graph.py`
- `shelf_03_graph_pipeline/config.py`
- `shelf_03_graph_pipeline/utils.py`

### shelf_04_visualization
Purpose: legacy visualization scripts (Plotly / web-network variants).

Files:
- `shelf_04_visualization/visualize_k12.py`
- `shelf_04_visualization/visualize_k12_graph.py`
- `shelf_04_visualization/visualize_hierarchical.py`
- `shelf_04_visualization/visualize_3d_force.py`
- `shelf_04_visualization/visualize_sources.py`
- `shelf_04_visualization/visualizer.py`
- `shelf_04_visualization/config.py`
- `shelf_04_visualization/utils.py`

### shelf_05_old_analysis_bundle
Purpose: older Tiny-Textbooks deep-graph experiments.

Files:
- `shelf_05_old_analysis_bundle/old_analysis/build_deep_graph.py`
- `shelf_05_old_analysis_bundle/old_analysis/analyze_dataset.py`
- `shelf_05_old_analysis_bundle/old_analysis/inspect_samples.py`
- `shelf_05_old_analysis_bundle/old_analysis/collect_khan_articles.py`
- `shelf_05_old_analysis_bundle/old_analysis/debug_khan_structure.py`
- `shelf_05_old_analysis_bundle/old_analysis/export_gephi.py`
- `shelf_05_old_analysis_bundle/old_analysis/visualize_2d.py`
- `shelf_05_old_analysis_bundle/old_analysis/visualize_3d.py`
- `shelf_05_old_analysis_bundle/old_analysis/config.py`
- `shelf_05_old_analysis_bundle/old_analysis/utils.py`

## Archive Policy

- Active Phase-1 pipeline is in project root scripts:
  - `collect_khan_academy.py`
  - `collect_tiny_textbooks.py`
  - `extract_khan_taxonomy.py`
  - `build_corpus_index.py`
  - `compute_metrics.py`
  - `build_dashboard.py`
- Do not add new production logic under `legacy/`.
- If a legacy file is referenced in current docs, migrate the reference to root pipeline files.
