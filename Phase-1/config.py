"""
Configuration file for domain classification and visualization
"""

# Domain taxonomy
DOMAINS = {
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

# Flatten domains for classification
ALL_DOMAINS = []
DOMAIN_TO_CATEGORY = {}
for category, subdomains in DOMAINS.items():
    for domain in subdomains:
        ALL_DOMAINS.append(domain)
        DOMAIN_TO_CATEGORY[domain] = category

# Data sources
PILE_SUBSETS = [
    'ArXiv',
    'StackExchange', 
    'Wikipedia_en',
    'Github',
    'PubMed_Abstracts',
    'FreeLaw'
]

# Classification parameters
CLASSIFIER_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
#"facebook/bart-large-mnli"
# GPU Selection (for dual GPU systems)
GPU_INDEX = -1  # Change to 1 for second GPU, -1 to auto-select least used

# Auto-detect device (CUDA GPU / MPS / CPU)
import torch
if torch.cuda.is_available():
    # CUDA GPU (Windows/Linux)
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
    # Apple Silicon GPU (Mac)
    CLASSIFIER_DEVICE = "mps"
    DEVICE_NAME = "Apple MPS (Metal Performance Shaders)"
else:
    # CPU fallback
    CLASSIFIER_DEVICE = -1
    DEVICE_NAME = "CPU (No GPU detected)"

print(f"🎮 Device: {DEVICE_NAME}")

BATCH_SIZE = 16
TEXT_MAX_LENGTH = 1000  # Characters to use for classification

# Sampling parameters
SAMPLE_SIZE_PER_SOURCE = 5000  # Number of documents to sample per source
CONFIDENCE_THRESHOLD = 0.7  # Threshold for "high confidence" classification

# Paths
DATA_DIR = "pile"
RESULTS_DIR = "results/classifications"
PLOTS_DIR = "results/plots"
