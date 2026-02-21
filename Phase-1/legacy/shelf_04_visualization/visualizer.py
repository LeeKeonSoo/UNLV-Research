import json
import random
from transformers import pipeline
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np

"""
Parameters & Domains
"""

PILE_SUBSETS = {
    'ArXiv': 2000,              
    'StackExchange': 2000,      
    'Wikipedia_en': 2000,       
    'Github': 2000,             
    'PubMed_Abstracts': 2000,   
    'FreeLaw': 2000,            
}

DOMAINS = {
    # Level 1: Broad categories
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

# Flatten for classification
ALL_DOMAINS = []
for category, subdomains in DOMAINS.items():
    ALL_DOMAINS.extend(subdomains)

SUBSETS = ['ArXiv', 'StackExchange', 'Wikipedia_en', 'Github', 'PubMed_Abstracts', 'FreeLaw']

# 1. Load classifier
print("Loading classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

# 2. Sample documents
def load_pile_sample(subset_name, num_docs=1000):
    """Load random sample from The Pile subset"""
    docs = []
    # The Pile은 jsonl 형식
    with open(f'pile/{subset_name}.jsonl', 'r') as f:
        all_lines = f.readlines()
        sample_lines = random.sample(all_lines, min(num_docs, len(all_lines)))
        for line in sample_lines:
            data = json.loads(line)
            docs.append(data['text'][:1000])  # 첫 1000 chars만
    return docs

# 3. Classify documents
def classify_batch(texts, batch_size=32):
    """Classify texts and aggregate results"""
    domain_scores = defaultdict(list)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing {i}/{len(texts)}...")
        
        for text in batch:
            result = classifier(text, ALL_DOMAINS, multi_label=True)
            for domain, score in zip(result['labels'], result['scores']):
                domain_scores[domain].append(score)
    
    # Aggregate
    domain_stats = {}
    for domain, scores in domain_scores.items():
        domain_stats[domain] = {
            'mean_score': np.mean(scores),
            'max_score': np.max(scores),
            'doc_count': len(scores),
            'high_confidence_count': sum(1 for s in scores if s > 0.7)
        }
    
    return domain_stats

# 4. Collect all stats
all_stats = {}
for subset in PILE_SUBSETS:
    print(f"\n=== Processing {subset} ===")
    docs = load_pile_sample(subset, num_docs=1000)
    stats = classify_batch(docs)
    all_stats[subset] = stats

# 5. Prepare 3D visualization data
domains_list = list(ALL_DOMAINS)
x_values = []  # mean confidence
y_values = []  # document count
z_values = []  # high confidence ratio
sizes = []     # total coverage
colors = []    # by category
texts = []

for domain in domains_list:
    # Aggregate across all subsets
    total_docs = sum(all_stats[s][domain]['doc_count'] 
                     for s in PILE_SUBSETS if domain in all_stats[s])
    avg_confidence = np.mean([all_stats[s][domain]['mean_score'] 
                              for s in PILE_SUBSETS if domain in all_stats[s]])
    high_conf_count = sum(all_stats[s][domain]['high_confidence_count'] 
                          for s in PILE_SUBSETS if domain in all_stats[s])
    
    x_values.append(avg_confidence)
    y_values.append(total_docs)
    z_values.append(high_conf_count / max(total_docs, 1))
    sizes.append(total_docs / 100)  # scale for visibility
    texts.append(domain)
    
    # Color by category
    for cat_idx, (category, subdomains) in enumerate(DOMAINS.items()):
        if domain in subdomains:
            colors.append(cat_idx)
            break

# 6. Create 3D plot
fig = go.Figure(data=[go.Scatter3d(
    x=x_values,
    y=y_values,
    z=z_values,
    mode='markers+text',
    marker=dict(
        size=sizes,
        color=colors,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Domain Category"),
        opacity=0.8
    ),
    text=texts,
    textposition="top center",
    textfont=dict(size=8),
    hovertemplate='<b>%{text}</b><br>' +
                  'Avg Confidence: %{x:.2f}<br>' +
                  'Doc Count: %{y}<br>' +
                  'High Conf Ratio: %{z:.2f}<br>' +
                  '<extra></extra>'
)])

fig.update_layout(
    title="Dataset Domain Coverage (3D Visualization)",
    scene=dict(
        xaxis_title='Average Confidence',
        yaxis_title='Document Count',
        zaxis_title='High Confidence Ratio',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    ),
    width=1200,
    height=800
)

fig.write_html("domain_coverage_3d.html")
print("\n✅ Visualization saved to domain_coverage_3d.html")

# 7. Print summary
print("\n=== Domain Coverage Summary ===")
for domain in sorted(domains_list, 
                     key=lambda d: sum(all_stats[s][d]['doc_count'] 
                                      for s in PILE_SUBSETS if d in all_stats[s]),
                     reverse=True)[:10]:
    total = sum(all_stats[s][domain]['doc_count'] 
                for s in PILE_SUBSETS if domain in all_stats[s])
    print(f"{domain:40s}: {total:6d} docs")

