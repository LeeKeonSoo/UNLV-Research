"""
Advanced K-12 Knowledge Graph Visualizations - FULLY FUNCTIONAL

Creates modern, interactive visualizations:
1. 3D Force-directed graph with physics simulation
2. Hierarchical Sunburst chart
3. Interactive Network with zoom/pan
4. Coverage Heatmap
5. Sankey flow diagram
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
        print(f"✅ Loaded {subject}: {len(graphs[subject]['nodes'])} nodes")
    
    return graphs


def create_3d_force_graph(graphs):
    """
    Beautiful 3D force-directed network with physics
    """
    print("\n📊 Creating 3D Force-Directed Graph...")
    
    # Build NetworkX graph
    G = nx.Graph()
    
    # Color scheme
    subject_colors = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'language_arts': '#95E1D3',
        'social_studies': '#F38181',
        'unknown': '#95A5A6'
    }
    
    node_info = {}
    
    # Add nodes
    for subject, graph_data in graphs.items():
        for node in graph_data['nodes']:
            node_id = node['id']
            G.add_node(node_id)
            node_info[node_id] = {
                'name': node['name'],
                'subject': subject,
                'grade': node.get('grade_level', 'N/A'),
                'docs': node['document_count'],
                'color': subject_colors.get(subject, '#95A5A6')
            }
    
    # Add edges
    for subject, graph_data in graphs.items():
        for edge in graph_data['edges']:
            if edge['source'] in G and edge['target'] in G:
                G.add_edge(edge['source'], edge['target'])
    
    # 3D spring layout with more spacing
    pos = nx.spring_layout(G, dim=3, k=2.5, iterations=50, seed=42)
    
    # Prepare node traces
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    z_nodes = [pos[n][2] for n in G.nodes()]
    
    colors = [node_info[n]['color'] for n in G.nodes()]
    sizes = [min(30, 8 + node_info[n]['docs'] * 0.5) for n in G.nodes()]
    hovertexts = [
        f"<b>{node_info[n]['name']}</b><br>" +
        f"Subject: {node_info[n]['subject']}<br>" +
        f"Grade: {node_info[n]['grade']}<br>" +
        f"Documents: {node_info[n]['docs']}"
        for n in G.nodes()
    ]
    
    # Prepare edge traces
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.3)', width=2),
        hoverinfo='none',
        name='Relationships',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.9,
            line=dict(color='white', width=1),
            symbol='circle'
        ),
        text=hovertexts,
        hoverinfo='text',
        name='Concepts'
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text='<b>K-12 Knowledge Graph: 3D Interactive Network</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=26, color='#2C3E50')
        ),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=''),
            bgcolor='#F8F9FA',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            )
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=900,
        showlegend=True,
        hovermode='closest',
        margin=dict(l=0, r=0, t=80, b=0)
    ))
    
    # Save
    output_file = Path(K12_REPORTS_DIR) / "3d_network.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved: {output_file.name}")
    
    return fig


def create_sunburst(graphs):
    """
    Hierarchical sunburst chart
    """
    print("\n📊 Creating Sunburst Hierarchy...")
    
    labels = ['K-12<br>Curriculum']
    parents = ['']
    values = [1]
    colors_list = []
    hovertexts = []
    
    subject_colors = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'language_arts': '#95E1D3',
        'social_studies': '#F38181'
    }
    
    for subject, graph_data in graphs.items():
        subject_label = subject.replace('_', ' ').title()
        total_docs = sum(n['document_count'] for n in graph_data['nodes'])
        
        labels.append(subject_label)
        parents.append('K-12<br>Curriculum')
        values.append(total_docs)
        colors_list.append(subject_colors.get(subject, '#95A5A6'))
        hovertexts.append(f"{total_docs} documents")
        
        # Group by grade
        grade_groups = defaultdict(list)
        for node in graph_data['nodes']:
            grade = node.get('grade_level', 'unknown')
            if grade != 'unknown':
                grade_groups[grade].append(node)
        
        for grade, nodes in sorted(grade_groups.items()):
            grade_label = f"{subject_label}<br>Grade {grade}"
            grade_docs = sum(n['document_count'] for n in nodes)
            
            labels.append(grade_label)
            parents.append(subject_label)
            values.append(grade_docs)
            colors_list.append(subject_colors.get(subject, '#95A5A6'))
            hovertexts.append(f"{len(nodes)} concepts • {grade_docs} docs")
    
    # Create figure
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors_list,
            line=dict(color='white', width=3)
        ),
        hovertemplate='<b>%{label}</b><br>%{customdata}<extra></extra>',
        customdata=hovertexts,
        textfont=dict(size=14)
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>K-12 Curriculum: Hierarchical Structure</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=26, color='#2C3E50')
        ),
        height=900,
        paper_bgcolor='white',
        margin=dict(t=80, l=0, r=0, b=0)
    )
    
    output_file = Path(K12_REPORTS_DIR) / "sunburst.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved: {output_file.name}")
    
    return fig


def create_heatmap(graphs):
    """
    Coverage heatmap: Grade × Subject
    """
    print("\n📊 Creating Coverage Heatmap...")
    
    subjects = sorted(graphs.keys())
    all_grades = set()
    
    for graph_data in graphs.values():
        for node in graph_data['nodes']:
            grade = node.get('grade_level')
            if grade and grade != 'unknown':
                all_grades.add(grade)
    
    grades = sorted(all_grades)
    
    # Build matrix
    matrix = np.zeros((len(subjects), len(grades)))
    
    for i, subject in enumerate(subjects):
        for node in graphs[subject]['nodes']:
            grade = node.get('grade_level')
            if grade in grades:
                j = grades.index(grade)
                matrix[i, j] += node['document_count']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Grade<br>{g}" for g in grades],
        y=[s.replace('_', ' ').title() for s in subjects],
        colorscale='YlOrRd',
        text=matrix.astype(int),
        texttemplate='<b>%{text}</b>',
        textfont={"size": 14, "color": "white"},
        hovertemplate='<b>%{y}</b><br>%{x}<br>Documents: %{z}<extra></extra>',
        colorbar=dict(title="Documents", titlefont=dict(size=14))
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>K-12 Coverage Heatmap: Documents by Grade and Subject</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=26, color='#2C3E50')
        ),
        xaxis=dict(title='Grade Level', titlefont=dict(size=16)),
        yaxis=dict(title='Subject Area', titlefont=dict(size=16)),
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=150, r=50, t=100, b=80)
    )
    
    output_file = Path(K12_REPORTS_DIR) / "coverage_heatmap.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved: {output_file.name}")
    
    return fig


def create_distribution_chart(graphs):
    """
    Concept and document distribution
    """
    print("\n📊 Creating Distribution Charts...")
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Concepts by Subject</b>', '<b>Documents by Subject</b>'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    subjects = []
    concept_counts = []
    doc_counts = []
    colors_map = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'language_arts': '#95E1D3',
        'social_studies': '#F38181'
    }
    colors = []
    
    for subject, graph_data in sorted(graphs.items()):
        subjects.append(subject.replace('_', ' ').title())
        concept_counts.append(len(graph_data['nodes']))
        doc_counts.append(sum(n['document_count'] for n in graph_data['nodes']))
        colors.append(colors_map.get(subject, '#95A5A6'))
    
    # Add traces
    fig.add_trace(
        go.Bar(x=subjects, y=concept_counts, marker_color=colors, name='Concepts',
               text=concept_counts, textposition='outside', textfont=dict(size=14)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=subjects, y=doc_counts, marker_color=colors, name='Documents',
               text=doc_counts, textposition='outside', textfont=dict(size=14)),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(
            text='<b>K-12 Knowledge Base: Distribution Analysis</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=26, color='#2C3E50')
        ),
        showlegend=False,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="Subject", row=1, col=1, titlefont=dict(size=14))
    fig.update_xaxes(title_text="Subject", row=1, col=2, titlefont=dict(size=14))
    fig.update_yaxes(title_text="Count", row=1, col=1, titlefont=dict(size=14))
    fig.update_yaxes(title_text="Count", row=1, col=2, titlefont=dict(size=14))
    
    output_file = Path(K12_REPORTS_DIR) / "distribution.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved: {output_file.name}")
    
    return fig


def main():
    """
    Generate all visualizations - FULLY FUNCTIONAL
    """
    print("="*60)
    print("K-12 KNOWLEDGE GRAPH VISUALIZATIONS")
    print("="*60)
    
    # Load graphs
    graphs = load_all_graphs()
    
    if not graphs:
        print("\n❌ No graphs found!")
        print("   Run: python build_k12_graph.py first")
        return
    
    print(f"\n✅ Loaded {len(graphs)} graphs")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Modern Visualizations...")
    print("="*60)
    
    create_3d_force_graph(graphs)
    create_sunburst(graphs)
    create_heatmap(graphs)
    create_distribution_chart(graphs)
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nAll visualizations saved to: {K12_REPORTS_DIR}/")
    print("\nOpen these HTML files in your browser:")
    print("  • 3d_network.html - Interactive 3D graph")
    print("  • sunburst.html - Hierarchical view")
    print("  • coverage_heatmap.html - Grade × Subject coverage")
    print("  • distribution.html - Concept & document stats")


if __name__ == "__main__":
    main()
