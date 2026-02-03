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
CLASSIFIER_DEVICE = 0  # GPU device
BATCH_SIZE = 32
TEXT_MAX_LENGTH = 1000  # Characters to use for classification

# Sampling parameters
SAMPLE_SIZE_PER_SOURCE = 5000  # Number of documents to sample per source
CONFIDENCE_THRESHOLD = 0.7  # Threshold for "high confidence" classification

# Paths
DATA_DIR = "pile"
RESULTS_DIR = "results/classifications"
PLOTS_DIR = "results/plots"
