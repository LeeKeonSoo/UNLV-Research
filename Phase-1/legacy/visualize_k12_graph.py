"""
K-12 Knowledge Graph Visualizations - FIXED VERSION
All sorting errors resolved
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict

from config import K12_GRAPHS_DIR, K12_REPORTS_DIR


def load_all_graphs():
    """Load all K-12 graphs"""
    graphs = {}
    graph_dir = Path(K12_GRAPHS_DIR)
    
    if not graph_dir.exists():
        print(f"❌ No graphs in {K12_GRAPHS_DIR}")
        return graphs
    
    for graph_file in graph_dir.glob("*_graph.json"):
        subject = graph_file.stem.replace("_graph", "")
        with open(graph_file) as f:
            graphs[subject] = json.load(f)
        print(f"✅ {subject}: {len(graphs[subject]['nodes'])} nodes")
    
    return graphs


def create_3d_network(graphs):
    """3D Force-Directed Network"""
    print("\n📊 Creating 3D Network...")
    
    G = nx.Graph()
    
    subject_colors = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'social_studies': '#F38181',
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
    
    # Layout
    pos = nx.spring_layout(G, dim=3, k=2.5, iterations=50, seed=42)
    
    # Nodes
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    z_nodes = [pos[n][2] for n in G.nodes()]
    
    colors = [node_info[n]['color'] for n in G.nodes()]
    sizes = [min(30, 8 + node_info[n]['docs']) for n in G.nodes()]
    texts = [
        f"<b>{node_info[n]['name']}</b><br>Subject: {node_info[n]['subject']}<br>Grade: {node_info[n]['grade']}<br>Docs: {node_info[n]['docs']}"
        for n in G.nodes()
    ]
    
    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.3)', width=2),
        hoverinfo='none',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(size=sizes, color=colors, opacity=0.9, line=dict(color='white', width=1)),
        text=texts,
        hoverinfo='text',
        name='Concepts'
    ))
    
    fig.update_layout(
        title='<b>K-12 Knowledge Graph: 3D Network</b>',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            bgcolor='#F8F9FA'
        ),
        height=900,
        showlegend=True
    )
    
    output = Path(K12_REPORTS_DIR) / "3d_network.html"
    fig.write_html(str(output))
    print(f"✅ {output.name}")


def create_sunburst(graphs):
    """Sunburst Hierarchy - FIXED SORTING"""
    print("\n📊 Creating Sunburst...")
    
    labels = ['K-12']
    parents = ['']
    values = [1]
    colors = []
    
    subject_colors = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'social_studies': '#F38181'
    }
    
    for subject, graph_data in graphs.items():
        subject_label = subject.replace('_', ' ').title()
        total_docs = sum(n['document_count'] for n in graph_data['nodes'])
        
        labels.append(subject_label)
        parents.append('K-12')
        values.append(total_docs)
        colors.append(subject_colors.get(subject, '#95A5A6'))
        
        # Group by grade - FIX: Handle None grades
        grade_groups = defaultdict(list)
        for node in graph_data['nodes']:
            grade = node.get('grade_level')
            if grade:  # Only add if grade exists
                grade_groups[grade].append(node)
        
        # FIX: Sort with None-safe key
        for grade, nodes in sorted(grade_groups.items(), key=lambda x: (x[0] is None, str(x[0]))):
            grade_label = f"{subject_label} - {grade}"
            grade_docs = sum(n['document_count'] for n in nodes)
            
            labels.append(grade_label)
            parents.append(subject_label)
            values.append(grade_docs)
            colors.append(subject_colors.get(subject, '#95A5A6'))
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color='white', width=3))
    ))
    
    fig.update_layout(
        title='<b>K-12 Curriculum Hierarchy</b>',
        height=900
    )
    
    output = Path(K12_REPORTS_DIR) / "sunburst.html"
    fig.write_html(str(output))
    print(f"✅ {output.name}")


def create_heatmap(graphs):
    """Coverage Heatmap"""
    print("\n📊 Creating Heatmap...")
    
    subjects = sorted(graphs.keys())
    all_grades = set()
    
    for graph_data in graphs.values():
        for node in graph_data['nodes']:
            grade = node.get('grade_level')
            if grade:
                all_grades.add(grade)
    
    grades = sorted(all_grades, key=lambda x: (x is None, str(x)))
    
    # Matrix
    matrix = np.zeros((len(subjects), len(grades)))
    
    for i, subject in enumerate(subjects):
        for node in graphs[subject]['nodes']:
            grade = node.get('grade_level')
            if grade in grades:
                j = grades.index(grade)
                matrix[i, j] += node['document_count']
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Grade {g}" for g in grades],
        y=[s.replace('_', ' ').title() for s in subjects],
        colorscale='YlOrRd',
        text=matrix.astype(int),
        texttemplate='<b>%{text}</b>',
        textfont={"size": 14}
    ))
    
    fig.update_layout(
        title='<b>K-12 Coverage Heatmap</b>',
        xaxis_title='Grade Level',
        yaxis_title='Subject',
        height=600
    )
    
    output = Path(K12_REPORTS_DIR) / "coverage_heatmap.html"
    fig.write_html(str(output))
    print(f"✅ {output.name}")


def create_distribution(graphs):
    """Distribution Charts"""
    print("\n📊 Creating Distribution...")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Concepts by Subject</b>', '<b>Documents by Subject</b>')
    )
    
    subjects = []
    concept_counts = []
    doc_counts = []
    
    colors_map = {
        'mathematics': '#FF6B6B',
        'science': '#4ECDC4',
        'social_studies': '#F38181'
    }
    colors = []
    
    for subject, graph_data in sorted(graphs.items()):
        subjects.append(subject.replace('_', ' ').title())
        concept_counts.append(len(graph_data['nodes']))
        doc_counts.append(sum(n['document_count'] for n in graph_data['nodes']))
        colors.append(colors_map.get(subject, '#95A5A6'))
    
    fig.add_trace(
        go.Bar(x=subjects, y=concept_counts, marker_color=colors, text=concept_counts, textposition='outside'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=subjects, y=doc_counts, marker_color=colors, text=doc_counts, textposition='outside'),
        row=1, col=2
    )
    
    fig.update_layout(
        title='<b>K-12 Distribution Analysis</b>',
        showlegend=False,
        height=600
    )
    
    output = Path(K12_REPORTS_DIR) / "distribution.html"
    fig.write_html(str(output))
    print(f"✅ {output.name}")


def main():
    """Generate all visualizations"""
    print("="*60)
    print("K-12 VISUALIZATIONS")
    print("="*60)
    
    graphs = load_all_graphs()
    
    if not graphs:
        print("\n❌ No graphs found!")
        print("   Run: python build_k12_graph.py first")
        return
    
    print(f"\n✅ Loaded {len(graphs)} graphs")
    
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    try:
        create_3d_network(graphs)
    except Exception as e:
        print(f"⚠️  3D Network error: {e}")
    
    try:
        create_sunburst(graphs)
    except Exception as e:
        print(f"⚠️  Sunburst error: {e}")
    
    try:
        create_heatmap(graphs)
    except Exception as e:
        print(f"⚠️  Heatmap error: {e}")
    
    try:
        create_distribution(graphs)
    except Exception as e:
        print(f"⚠️  Distribution error: {e}")
    
    # Generate advanced 3D visualizations
    print("\n" + "="*60)
    print("Advanced 3D Visualizations...")
    print("="*60)

    try:
        import subprocess
        print("\n🚀 Generating 3D Force Graph (WebGL)...")
        result = subprocess.run(['python', 'visualize_3d_force.py'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 3D Force Graph complete")
        else:
            print(f"⚠️  3D Force Graph error: {result.stderr}")
    except Exception as e:
        print(f"⚠️  3D Force Graph error: {e}")

    try:
        print("\n🚀 Generating Hierarchical View (Cytoscape.js)...")
        result = subprocess.run(['python', 'visualize_hierarchical.py'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Hierarchical View complete")
        else:
            print(f"⚠️  Hierarchical View error: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Hierarchical View error: {e}")

    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("="*60)
    print(f"\nVisualization files in: {K12_REPORTS_DIR}/")
    print("\n📊 Standard Views (Plotly):")
    print("  • 3d_network.html          - 3D network (Plotly)")
    print("  • sunburst.html            - Hierarchical sunburst")
    print("  • coverage_heatmap.html    - Grade × Subject heatmap")
    print("  • distribution.html        - Bar charts")
    print("\n🚀 Advanced Views (WebGL/Interactive):")
    print("  • 3d_force_graph.html      - 3D Force-Directed (Three.js)")
    print("  • hierarchical_view.html   - Hierarchical Layout (Cytoscape.js)")


if __name__ == "__main__":
    main()
