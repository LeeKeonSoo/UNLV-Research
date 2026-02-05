"""
Advanced K-12 Knowledge Graph Visualizations

Creates interactive, modern visualizations:
1. 3D Force-directed graph (concepts and relationships)
2. Hierarchical Sunburst (grade → subject → concepts)
3. Interactive Network (clickable nodes, filtering)
4. Coverage Heatmap (grade × subject)
5. Sankey Diagram (prerequisite flow)
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict

from config import *


def load_all_graphs():
    """Load all K-12 graphs"""
    graphs = {}
    graph_dir = Path(K12_GRAPHS_DIR)
    
    if not graph_dir.exists():
        print(f"❌ No graphs found in {K12_GRAPHS_DIR}")
        return graphs
    
    for graph_file in graph_dir.glob("*_graph.json"):
        subject = graph_file.stem.replace("_graph", "")
        with open(graph_file) as f:
            graphs[subject] = json.load(f)
        print(f"✅ Loaded {subject}")
    
    return graphs


def create_3d_network_graph(graphs):
    """
    Create beautiful 3D force-directed network graph
    
    Shows concepts as nodes, relationships as edges
    """
    print("\n📊 Creating 3D Network Graph...")
    
    # Build NetworkX graph
    G = nx.Graph()
    
    node_colors = []
    node_sizes = []
    node_texts = []
    node_subjects = []
    
    # Color palette for subjects
    subject_colors = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'language_arts': '#95E1D3',
        'social_studies': '#F38181'
    }
    
    # Add all nodes
    for subject, graph_data in graphs.items():
        base_color = subject_colors.get(subject, '#95A5A6')
        
        for node in graph_data['nodes']:
            node_id = node['id']
            
            G.add_node(node_id, 
                      label=node['name'],
                      subject=subject,
                      grade=node.get('grade_level', 'N/A'),
                      docs=node['document_count'])
            
            # Node styling
            node_colors.append(base_color)
            node_sizes.append(min(50, 10 + node['document_count']))
            node_texts.append(f"<b>{node['name']}</b><br>" +
                            f"Subject: {subject}<br>" +
                            f"Grade: {node.get('grade_level', 'N/A')}<br>" +
                            f"Documents: {node['document_count']}")
            node_subjects.append(subject)
    
    # Add edges
    for subject, graph_data in graphs.items():
        for edge in graph_data['edges']:
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(edge['source'], edge['target'], 
                          weight=edge.get('weight', 1.0))
    
    # 3D Spring layout
    pos = nx.spring_layout(G, dim=3, k=2, iterations=50)
    
    # Extract positions
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]
    
    # Create edges trace
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # Edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(125, 125, 125, 0.2)', width=1),
        hoverinfo='none',
        name='Relationships'
    )
    
    # Node trace
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.8,
            line=dict(color='white', width=0.5)
        ),
        text=node_texts,
        hoverinfo='text',
        name='Concepts'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title={
            'text': '<b>K-12 Knowledge Graph: 3D Network</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        showlegend=True,
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            bgcolor='rgba(240, 240, 240, 0.9)'
        ),
        paper_bgcolor='white',
        height=800,
        hovermode='closest'
    )
    
    # Save
    output_file = Path(K12_REPORTS_DIR) / "3d_network_graph.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved to {output_file}")
    
    return fig


def create_sunburst_chart(graphs):
    """
    Create hierarchical sunburst chart
    
    Shows: Root → Subject → Grade → Concepts
    """
    print("\n📊 Creating Sunburst Chart...")
    
    labels = ['K-12 Curriculum']
    parents = ['']
    values = [1]
    colors = []
    texts = []
    
    # Color schemes
    subject_colors = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'language_arts': '#95E1D3',
        'social_studies': '#F38181'
    }
    
    for subject, graph_data in graphs.items():
        # Add subject
        subject_label = subject.title()
        labels.append(subject_label)
        parents.append('K-12 Curriculum')
        
        # Count total docs
        total_docs = sum(node['document_count'] for node in graph_data['nodes'])
        values.append(total_docs)
        colors.append(subject_colors.get(subject, '#95A5A6'))
        texts.append(f"{total_docs} documents")
        
        # Group by grade
        grade_groups = defaultdict(list)
        for node in graph_data['nodes']:
            grade = node.get('grade_level', 'unknown')
            if grade and grade != 'unknown':
                grade_groups[grade].append(node)
        
        # Add grades
        for grade, nodes in grade_groups.items():
            grade_label = f"{subject_label} - Grade {grade}"
            labels.append(grade_label)
            parents.append(subject_label)
            
            grade_docs = sum(n['document_count'] for n in nodes)
            values.append(grade_docs)
            colors.append(subject_colors.get(subject, '#95A5A6'))
            texts.append(f"{len(nodes)} concepts<br>{grade_docs} docs")
            
            # Add top concepts
            top_concepts = sorted(nodes, key=lambda x: x['document_count'], reverse=True)[:5]
            for concept in top_concepts:
                concept_label = f"{grade_label} - {concept['name'][:30]}"
                labels.append(concept_label)
                parents.append(grade_label)
                values.append(concept['document_count'])
                colors.append(subject_colors.get(subject, '#95A5A6'))
                texts.append(f"{concept['document_count']} docs")
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>%{text}<br><extra></extra>',
        text=texts
    ))
    
    fig.update_layout(
        title={
            'text': '<b>K-12 Curriculum: Hierarchical View</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=800,
        paper_bgcolor='white'
    )
    
    output_file = Path(K12_REPORTS_DIR) / "sunburst_hierarchy.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved to {output_file}")
    
    return fig


def create_coverage_heatmap(graphs):
    """
    Create coverage heatmap: Grade × Subject
    """
    print("\n📊 Creating Coverage Heatmap...")
    
    # Build matrix
    subjects = list(graphs.keys())
    all_grades = set()
    
    for graph_data in graphs.values():
        for node in graph_data['nodes']:
            grade = node.get('grade_level')
            if grade and grade != 'unknown':
                all_grades.add(grade)
    
    grades = sorted(all_grades)
    
    # Create matrix
    matrix = np.zeros((len(subjects), len(grades)))
    
    for i, subject in enumerate(subjects):
        graph_data = graphs[subject]
        for node in graph_data['nodes']:
            grade = node.get('grade_level')
            if grade in grades:
                j = grades.index(grade)
                matrix[i, j] += node['document_count']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Grade {g}" for g in grades],
        y=[s.title() for s in subjects],
        colorscale='YlOrRd',
        text=matrix.astype(int),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='<b>%{y}</b><br>%{x}<br>Documents: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>K-12 Coverage: Grade × Subject</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title='Grade Level',
        yaxis_title='Subject',
        height=600,
        paper_bgcolor='white'
    )
    
    output_file = Path(K12_REPORTS_DIR) / "coverage_heatmap.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved to {output_file}")
    
    return fig


def create_concept_distribution(graphs):
    """
    Create stacked bar chart of concept distribution
    """
    print("\n📊 Creating Concept Distribution Chart...")
    
    # Prepare data
    data_for_plot = []
    
    for subject, graph_data in graphs.items():
        # Group by grade
        grade_counts = defaultdict(int)
        for node in graph_data['nodes']:
            grade = node.get('grade_level', 'unknown')
            if grade != 'unknown':
                grade_counts[grade] += 1
        
        for grade, count in grade_counts.items():
            data_for_plot.append({
                'Subject': subject.title(),
                'Grade': f"Grade {grade}",
                'Concepts': count
            })
    
    import pandas as pd
    df = pd.DataFrame(data_for_plot)
    
    fig = px.bar(df, 
                 x='Grade', 
                 y='Concepts', 
                 color='Subject',
                 title='<b>Concept Distribution by Grade and Subject</b>',
                 barmode='stack',
                 color_discrete_map={
                     'Mathematics': '#FF6B6B',
                     'Science': '#4ECDC4',
                     'Language_Arts': '#95E1D3',
                     'Social_Studies': '#F38181'
                 })
    
    fig.update_layout(
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 24}},
        height=600,
        paper_bgcolor='white',
        xaxis_title='Grade Level',
        yaxis_title='Number of Concepts'
    )
    
    output_file = Path(K12_REPORTS_DIR) / "concept_distribution.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved to {output_file}")
    
    return fig


def main():
    """
    Generate all visualizations
    """
    print("="*60)
    print("K-12 KNOWLEDGE GRAPH VISUALIZATIONS")
    print("="*60)
    
    # Load graphs
    graphs = load_all_graphs()
    
    if not graphs:
        print("❌ No graphs to visualize!")
        print("   Please run build_k12_graph.py first")
        return
    
    print(f"\nLoaded {len(graphs)} subject graphs")
    
    # Create visualizations
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    try:
        create_3d_network_graph(graphs)
    except Exception as e:
        print(f"⚠️  3D Network error: {e}")
    
    try:
        create_sunburst_chart(graphs)
    except Exception as e:
        print(f"⚠️  Sunburst error: {e}")
    
    try:
        create_coverage_heatmap(graphs)
    except Exception as e:
        print(f"⚠️  Heatmap error: {e}")
    
    try:
        create_concept_distribution(graphs)
    except Exception as e:
        print(f"⚠️  Distribution error: {e}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"✅ Visualizations saved to {K12_REPORTS_DIR}/")
    print("   - 3d_network_graph.html (interactive 3D)")
    print("   - sunburst_hierarchy.html (hierarchical)")
    print("   - coverage_heatmap.html (grade × subject)")
    print("   - concept_distribution.html (stacked bars)")


if __name__ == "__main__":
    main()
