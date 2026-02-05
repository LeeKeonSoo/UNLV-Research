"""
Configuration file for K-12 curriculum analysis and dataset characterization
"""

import torch

# =============================================================================
# K-12 DATA SOURCES
# =============================================================================

# Primary K-12 curriculum sources
K12_SOURCES = {
    "khan_academy": {
        "name": "Khan Academy",
        "url": "https://www.khanacademy.org/",
        "subjects": ["math", "science", "computing", "arts-humanities"],
        "grade_range": "K-12",
        "format": "articles, videos, exercises",
        "license": "CC BY-NC-SA",
        "priority": 1
    },
    "openstax": {
        "name": "OpenStax",
        "url": "https://openstax.org/",
        "subjects": ["math", "science", "social-sciences"],
        "grade_range": "6-12",
        "format": "textbooks (HTML/PDF)",
        "license": "CC BY",
        "priority": 1
    },
    "ck12": {
        "name": "CK-12 Foundation",
        "url": "https://www.ck12.org/",
        "subjects": ["math", "science", "engineering"],
        "grade_range": "K-12",
        "format": "flexbooks, simulations",
        "license": "CC BY-NC",
        "priority": 2
    }
}

# OpenStax K-12 relevant books
OPENSTAX_K12_BOOKS = [
    # Mathematics
    "prealgebra-2e",
    "elementary-algebra-2e",
    "intermediate-algebra-2e",
    "algebra-and-trigonometry-2e",
    
    # Science
    "biology-2e",
    "chemistry-2e", 
    "physics",
    
    # Social Studies
    "us-history",
    "american-government-3e"
]

# Khan Academy subject structure
KHAN_SUBJECTS = {
    "math": [
        "early-math",           # K-2
        "arithmetic",           # 3-5
        "pre-algebra",          # 6-8
        "algebra-basics",       # 6-8
        "algebra1",             # 9
        "geometry",             # 9-10
        "algebra2",             # 10-11
        "trigonometry",         # 11
        "precalculus"           # 11-12
    ],
    "science": [
        "biology",
        "chemistry",
        "physics",
        "organic-chemistry",
        "ap-biology",
        "ap-chemistry",
        "ap-physics-1"
    ],
    "computing": [
        "computer-programming",
        "computer-science",
        "algorithms"
    ]
}

# =============================================================================
# K-12 CURRICULUM STRUCTURE
# =============================================================================

# Grade level mapping
GRADE_LEVELS = {
    "elementary": {
        "range": (1, 5),
        "description": "Elementary school",
        "focus": ["basic arithmetic", "reading", "basic science"]
    },
    "middle": {
        "range": (6, 8),
        "description": "Middle school",
        "focus": ["pre-algebra", "intermediate science", "writing"]
    },
    "high": {
        "range": (9, 12),
        "description": "High school", 
        "focus": ["algebra", "advanced science", "analysis"]
    }
}

# Subject taxonomy (simplified from 24 domains to K-12 focus)
K12_SUBJECTS = {
    "Mathematics": [
        "counting and number sense",
        "basic arithmetic",
        "fractions and decimals",
        "ratios and proportions",
        "pre-algebra",
        "algebra",
        "geometry",
        "trigonometry",
        "statistics and probability"
    ],
    "Science": [
        "life science",
        "physical science",
        "earth science",
        "biology",
        "chemistry",
        "physics"
    ],
    "Language Arts": [
        "phonics and reading",
        "grammar and writing",
        "literature",
        "composition"
    ],
    "Social Studies": [
        "geography",
        "us history",
        "world history",
        "civics and government",
        "economics"
    ]
}

# Flatten for classification (backward compatibility)
ALL_K12_SUBJECTS = []
SUBJECT_TO_CATEGORY = {}
for category, subjects in K12_SUBJECTS.items():
    for subject in subjects:
        ALL_K12_SUBJECTS.append(subject)
        SUBJECT_TO_CATEGORY[subject] = category

# =============================================================================
# LEGACY: THE PILE CONFIGURATION (for comparison later)
# =============================================================================

# Original domain taxonomy (24 domains from Phase 1a)
PILE_DOMAINS = {
    "Mathematics": [
        "arithmetic and basic math",
        "algebra and equations", 
        "geometry and spatial reasoning",
        "statistics and probability",
        "advanced mathematics"
    ],
    "Natural Sciences": [
        "physics and mechanics",
        "chemistry and materials",
        "biology and life sciences",
        "earth and environmental science"
    ],
    "Computer Science": [
        "programming and algorithms",
        "data structures and systems",
        "AI and machine learning",
        "web development and databases"
    ],
    "Social Sciences": [
        "psychology and behavior",
        "economics and business",
        "sociology and anthropology",
        "political science and law"
    ],
    "Humanities": [
        "history and historical events",
        "philosophy and ethics",
        "literature and literary analysis",
        "languages and linguistics"
    ],
    "Practical Knowledge": [
        "everyday life and advice",
        "health and medicine",
        "engineering and technology",
        "arts and creativity"
    ]
}

# Flatten Pile domains
ALL_PILE_DOMAINS = []
PILE_DOMAIN_TO_CATEGORY = {}
for category, subdomains in PILE_DOMAINS.items():
    for domain in subdomains:
        ALL_PILE_DOMAINS.append(domain)
        PILE_DOMAIN_TO_CATEGORY[domain] = category

# Pile data sources
PILE_SUBSETS = [
    'ArXiv',
    'StackExchange', 
    'Wikipedia_en',
    'Github',
    'PubMed_Abstracts',
    'FreeLaw'
]

# =============================================================================
# CLASSIFICATION & PROCESSING PARAMETERS
# =============================================================================

# Model selection
CLASSIFIER_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# Alternative faster model: "facebook/bart-large-mnli"

# GPU configuration
GPU_INDEX = -1  # -1 for auto-select, 0/1 for specific GPU

# Auto-detect device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    
    if GPU_INDEX == -1:
        # Auto-select least used GPU
        gpu_memory_free = []
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            free_mem = torch.cuda.mem_get_info()[0]
            gpu_memory_free.append((i, free_mem))
        best_gpu = max(gpu_memory_free, key=lambda x: x[1])[0]
        CLASSIFIER_DEVICE = best_gpu
    else:
        CLASSIFIER_DEVICE = GPU_INDEX
    
    DEVICE_NAME = f"CUDA GPU {CLASSIFIER_DEVICE}: {torch.cuda.get_device_name(CLASSIFIER_DEVICE)} ({num_gpus} GPUs available)"
    
elif torch.backends.mps.is_available():
    CLASSIFIER_DEVICE = "mps"
    DEVICE_NAME = "Apple MPS (Metal Performance Shaders)"
else:
    CLASSIFIER_DEVICE = -1
    DEVICE_NAME = "CPU (No GPU detected)"

print(f"🎮 Device: {DEVICE_NAME}")

# Processing parameters
BATCH_SIZE = 16
TEXT_MAX_LENGTH = 1000
CONFIDENCE_THRESHOLD = 0.7

# Sampling parameters
USE_ALL_DOCUMENTS = True
SAMPLE_SIZE_PER_SOURCE = 5000

# =============================================================================
# GRAPH CONSTRUCTION PARAMETERS
# =============================================================================

# Clustering parameters for concept discovery
MIN_CLUSTER_SIZE = 50          # Minimum documents to form a concept
MAX_CLUSTER_DEPTH = 3          # Maximum hierarchy depth
SIMILARITY_THRESHOLD = 0.85    # For deduplication

# Concept coverage requirements
MIN_EXAMPLES_PER_CONCEPT = 10  # Minimum docs per concept
MAX_EXAMPLES_PER_CONCEPT = 100 # Maximum docs to keep per concept

# =============================================================================
# PATHS
# =============================================================================

# Legacy Pile data
DATA_DIR = "pile"
PILE_RESULTS_DIR = "results/classifications"
PLOTS_DIR = "results/plots"

# New K-12 data structure
K12_RAW_DIR = "k12_raw"
K12_PROCESSED_DIR = "k12_processed"
K12_GRAPHS_DIR = "k12_graphs"
K12_REPORTS_DIR = "k12_reports"

# Ensure directories exist
import os
for directory in [K12_RAW_DIR, K12_PROCESSED_DIR, K12_GRAPHS_DIR, K12_REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)
