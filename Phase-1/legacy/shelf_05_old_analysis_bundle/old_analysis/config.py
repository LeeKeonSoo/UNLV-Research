"""
Configuration for Tiny-Textbooks Deep Hierarchical Analysis
"""

import torch
import os

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICE_NAME = "Apple MPS (Metal Performance Shaders)"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

print(f"🎮 Device: {DEVICE_NAME}")

# =============================================================================
# HIERARCHICAL CLUSTERING PARAMETERS
# =============================================================================

# Deep hierarchy parameters (8 levels)
MAX_HIERARCHY_DEPTH = 8

# Clustering parameters per level (decreasing minimums for deeper levels)
LEVEL_PARAMS = {
    2: {'n_clusters_range': (15, 25), 'min_size': 5000},   # Broad domains
    3: {'n_clusters_range': (8, 15),  'min_size': 1000},   # Subjects
    4: {'n_clusters_range': (5, 10),  'min_size': 500},    # Topics
    5: {'n_clusters_range': (3, 8),   'min_size': 200},    # Subtopics
    6: {'n_clusters_range': (2, 5),   'min_size': 80},     # Concepts
    7: {'n_clusters_range': (2, 4),   'min_size': 30},     # Details
    8: {'n_clusters_range': (2, 3),   'min_size': 10},     # Fine details
}

# Clustering algorithm parameters
SIMILARITY_THRESHOLD = 0.85
LINKAGE_METHOD = 'ward'
METRIC = 'euclidean'

# =============================================================================
# PATHS
# =============================================================================

RAW_DATA_DIR = "tiny_textbooks_raw"
GRAPHS_DIR = "graphs"
VISUALIZATIONS_DIR = "visualizations"

# Ensure directories exist
for directory in [RAW_DATA_DIR, GRAPHS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Color palette for broad domains (will be generated dynamically)
DOMAIN_COLORS = [
    '#FF6B6B',  # Red
    '#4ECDC4',  # Teal
    '#45B7D1',  # Blue
    '#FFA07A',  # Light Salmon
    '#98D8C8',  # Mint
    '#F7DC6F',  # Yellow
    '#BB8FCE',  # Purple
    '#85C1E2',  # Sky Blue
    '#F8B88B',  # Peach
    '#AAB7B8',  # Gray
    '#FAD7A0',  # Beige
    '#D5F4E6',  # Light Green
    '#FADBD8',  # Pink
    '#D6EAF8',  # Light Blue
    '#FCF3CF',  # Light Yellow
    '#E8DAEF',  # Lavender
    '#D5D8DC',  # Silver
    '#EAEDED',  # White Gray
    '#F9E79F',  # Golden
    '#A9DFBF',  # Sage
]

print(f"📁 Paths:")
print(f"   Raw Data: {RAW_DATA_DIR}/")
print(f"   Graphs: {GRAPHS_DIR}/")
print(f"   Visualizations: {VISUALIZATIONS_DIR}/")
