"""
Source-Specific Domain Visualization

Creates interactive visualizations showing domain distribution across different sources.
Includes:
- Stacked bar chart (source-level domain breakdown)
- Heatmap (source × domain matrix)
- Interactive 3D scatter (with source highlighting)
"""

import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import *
from utils import load_json


def create_stacked_bar_chart(data: dict) -> go.Figure:
    """
    Create stacked bar chart showing domain distribution per source
    
    Categories are stacked, with interactive hover showing exact values
    """
    # Prepare data
    sources = list(data.keys())
    categories = list(DOMAINS.keys())
    
    # Aggregate by category
    category_data = {cat: [] for cat in categories}
    
    for source in sources:
        source_domains = data[source]['domains']
        
        # Sum up domains by category
        for category, domain_list in DOMAINS.items():
            category_total = sum(
                source_domains.get(domain, {}).get('mean_score', 0) 
                for domain in domain_list
            )
            category_data[category].append(category_total)
    
    # Normalize to percentages
    for i, source in enumerate(sources):
        total = sum(category_data[cat][i] for cat in categories)
        if total > 0:
            for cat in categories:
                category_data[cat][i] = (category_data[cat][i] / total) * 100
    
    # Create figure
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, category in enumerate(categories):
        fig.add_trace(go.Bar(
            name=category,
            x=sources,
            y=category_data[category],
            marker_color=colors[i % len(colors)],
            hovertemplate='<b>%{x}</b><br>' +
                         f'{category}: %{{y:.1f}}%<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Domain Distribution by Source (Category Level)",
        xaxis_title="Data Source",
        yaxis_title="Percentage (%)",
        barmode='stack',
        height=600,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def create_heatmap(data: dict) -> go.Figure:
    """
    Create heatmap showing source × domain confidence matrix
    """
    sources = list(data.keys())
    domains = ALL_DOMAINS
    
    # Build matrix
    matrix = []
    for domain in domains:
        row = []
        for source in sources:
            score = data[source]['domains'].get(domain, {}).get('mean_score', 0)
            row.append(score)
        matrix.append(row)
    
    # Get category for each domain (for coloring)
    domain_categories = [DOMAIN_TO_CATEGORY[d] for d in domains]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=sources,
        y=domains,
        colorscale='Viridis',
        hovertemplate='Source: %{x}<br>Domain: %{y}<br>Confidence: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Mean<br>Confidence")
    ))
    
    fig.update_layout(
        title="Source × Domain Confidence Heatmap",
        xaxis_title="Data Source",
        yaxis_title="Domain",
        height=1000,
        width=1000
    )
    
    return fig


def create_3d_interactive(data: dict) -> go.Figure:
    """
    Create 3D interactive scatter plot
    
    - Nodes = Domains
    - Size = Total coverage across sources
    - Color = Source with highest coverage
    - Hover on source name highlights relevant domains
    """
    # Prepare data points
    plot_data = []
    
    for domain in ALL_DOMAINS:
        # Aggregate stats across sources
        total_coverage = 0
        max_source = None
        max_score = 0
        source_scores = {}
        
        for source in data.keys():
            score = data[source]['domains'].get(domain, {}).get('mean_score', 0)
            high_conf_ratio = data[source]['domains'].get(domain, {}).get('high_confidence_ratio', 0)
            
            total_coverage += score
            source_scores[source] = score
            
            if score > max_score:
                max_score = score
                max_source = source
        
        # Calculate position (simplified - can be improved)
        # X = variance across sources (diversity)
        # Y = total coverage
        # Z = max confidence
        
        scores_array = list(source_scores.values())
        diversity = np.std(scores_array) if len(scores_array) > 0 else 0
        
        plot_data.append({
            'domain': domain,
            'category': DOMAIN_TO_CATEGORY[domain],
            'x': diversity,
            'y': total_coverage,
            'z': max_score,
            'size': total_coverage * 100,
            'dominant_source': max_source,
            'source_scores': source_scores
        })
    
    # Create figure
    fig = go.Figure()
    
    # Color by category
    categories = list(DOMAINS.keys())
    colors = px.colors.qualitative.Set2
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
    
    for category in categories:
        category_points = [p for p in plot_data if p['category'] == category]
        
        if not category_points:
            continue
        
        # Prepare hover text
        hover_texts = []
        for p in category_points:
            source_breakdown = "<br>".join([
                f"{src}: {score:.3f}" 
                for src, score in sorted(p['source_scores'].items(), key=lambda x: x[1], reverse=True)
            ])
            hover_text = (
                f"<b>{p['domain']}</b><br>"
                f"Category: {p['category']}<br>"
                f"Total Coverage: {p['y']:.3f}<br>"
                f"Dominant Source: {p['dominant_source']}<br>"
                f"<br><b>Source Breakdown:</b><br>{source_breakdown}"
            )
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Scatter3d(
            x=[p['x'] for p in category_points],
            y=[p['y'] for p in category_points],
            z=[p['z'] for p in category_points],
            mode='markers+text',
            name=category,
            marker=dict(
                size=[p['size'] for p in category_points],
                color=category_colors[category],
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=[p['domain'].split()[0] for p in category_points],  # First word only
            textposition='top center',
            textfont=dict(size=8),
            hovertext=hover_texts,
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title="3D Domain Space: Coverage, Diversity, and Confidence",
        scene=dict(
            xaxis_title='Source Diversity (std)',
            yaxis_title='Total Coverage',
            zaxis_title='Max Confidence',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1200,
        height=800,
        hovermode='closest'
    )
    
    return fig


def main():
    print("=" * 60)
    print("SOURCE-SPECIFIC VISUALIZATION")
    print("=" * 60)
    
    # Load classification results
    results_file = os.path.join(RESULTS_DIR, "all_sources_classification.json")
    
    if not os.path.exists(results_file):
        print(f"\n❌ Results file not found: {results_file}")
        print("Please run classify_domains.py first!")
        return
    
    print(f"\nLoading results from {results_file}...")
    data = load_json(results_file)
    print(f"✅ Loaded results for {len(data)} sources\n")
    
    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Stacked bar chart
    print("  - Stacked bar chart (category level)...")
    fig1 = create_stacked_bar_chart(data)
    output1 = os.path.join(PLOTS_DIR, "source_domain_stacked.html")
    fig1.write_html(output1)
    print(f"    ✅ Saved to {output1}")
    
    # 2. Heatmap
    print("  - Heatmap (source × domain)...")
    fig2 = create_heatmap(data)
    output2 = os.path.join(PLOTS_DIR, "source_domain_heatmap.html")
    fig2.write_html(output2)
    print(f"    ✅ Saved to {output2}")
    
    # 3. 3D interactive
    print("  - 3D interactive scatter...")
    fig3 = create_3d_interactive(data)
    output3 = os.path.join(PLOTS_DIR, "source_domain_3d.html")
    fig3.write_html(output3)
    print(f"    ✅ Saved to {output3}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nGenerated plots in {PLOTS_DIR}/:")
    print("  1. source_domain_stacked.html - Stacked bar chart")
    print("  2. source_domain_heatmap.html - Heatmap matrix")
    print("  3. source_domain_3d.html - 3D interactive scatter")
    print("\nOpen these HTML files in your browser to explore!")


if __name__ == "__main__":
    main()
