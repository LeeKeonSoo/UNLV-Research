"""
Advanced 3D Network Visualization using 3d-force-graph (Three.js WebGL)

This script creates high-performance, interactive 3D visualizations of K-12 knowledge graphs.
Uses 3d-force-graph library for WebGL-accelerated rendering (handles 5000+ nodes smoothly).

Features:
- WebGL rendering for high performance
- Interactive node inspection (click to focus)
- Color-coded by subject
- Node size proportional to document count
- Edge particles showing relationships
- Smooth camera transitions
"""

import json
from pathlib import Path
import numpy as np

# Subject color scheme (vibrant, distinct colors)
SUBJECT_COLORS = {
    'mathematics': '#FF6B6B',      # Red
    'science': '#4ECDC4',           # Teal
    'social_studies': '#45B7D1',    # Blue
    'english': '#FFA07A',           # Coral
    'language_arts': '#98D8C8',     # Mint
    'root': '#95A5A6'               # Gray for root nodes
}

def load_graphs(graphs_dir='k12_graphs'):
    """Load all K-12 graphs"""
    graphs_path = Path(graphs_dir)
    graphs = {}

    for graph_file in graphs_path.glob('*_graph.json'):
        subject = graph_file.stem.replace('_graph', '')
        with open(graph_file) as f:
            graphs[subject] = json.load(f)

    return graphs

def convert_to_3d_force_graph_format(graphs):
    """
    Convert K-12 graphs to 3d-force-graph format

    Format:
    {
        "nodes": [
            {
                "id": "concept_0",
                "name": "Mathematics",
                "val": 10,  # Node size (cube root of doc count)
                "color": "#FF6B6B",
                "grade": null,
                "subject": "mathematics",
                "docs": 0,
                "description": "Root mathematics node"
            }
        ],
        "links": [
            {
                "source": "concept_0",
                "target": "concept_1",
                "value": 1  # Link strength
            }
        ]
    }
    """
    nodes_data = []
    links_data = []
    node_id_map = {}  # Map original IDs to unique IDs across all subjects

    global_node_id = 0

    for subject, graph_data in graphs.items():
        # Process nodes
        for node in graph_data.get('nodes', []):
            unique_id = f"{subject}_{node['id']}"
            node_id_map[node['id']] = unique_id

            # Determine node color
            node_subject = node.get('subject', subject)
            color = SUBJECT_COLORS.get(node_subject, SUBJECT_COLORS['root'])

            # Calculate node size (perceptual scaling - cube root)
            doc_count = node.get('document_count', 0)
            node_size = np.cbrt(max(doc_count, 1)) * 3  # Scale factor 3

            # Build description
            grade = node.get('grade_level')
            grade_str = f"Grade: {grade}" if grade else "All grades"
            desc = f"{node['name']}\n{grade_str}\nDocuments: {doc_count}"

            nodes_data.append({
                'id': unique_id,
                'name': node['name'],
                'val': node_size,
                'color': color,
                'grade': grade,
                'subject': node_subject,
                'docs': doc_count,
                'description': desc,
                'fx': None,  # Allow physics to position
                'fy': None,
                'fz': None
            })

            global_node_id += 1

        # Process edges (parent-child relationships)
        for node in graph_data.get('nodes', []):
            if node.get('parent'):
                source_id = node_id_map.get(node['parent'])
                target_id = node_id_map.get(node['id'])

                if source_id and target_id:
                    links_data.append({
                        'source': source_id,
                        'target': target_id,
                        'value': 1  # Uniform link strength
                    })

    return {
        'nodes': nodes_data,
        'links': links_data
    }

def generate_3d_force_html(data, output_path='k12_reports/3d_force_graph.html'):
    """
    Generate standalone HTML with embedded 3d-force-graph

    Uses CDN for 3d-force-graph library.
    """

    # Inline the graph data (for simplicity)
    graph_data_json = json.dumps(data, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-12 Knowledge Graph - 3D Interactive</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
        }}

        #graph-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}

        #info-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            max-width: 300px;
            z-index: 1000;
        }}

        #info-panel h2 {{
            margin: 0 0 10px 0;
            font-size: 20px;
            color: #333;
        }}

        #info-panel p {{
            margin: 5px 0;
            font-size: 14px;
            color: #666;
        }}

        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            z-index: 1000;
        }}

        .legend h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #333;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 14px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}

        .controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            z-index: 1000;
        }}

        .controls button {{
            display: block;
            margin: 5px 0;
            padding: 8px 15px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}

        .controls button:hover {{
            background: #764ba2;
        }}

        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            z-index: 2000;
        }}

        .spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    <script src="https://unpkg.com/3d-force-graph@1.73.2/dist/3d-force-graph.min.js"></script>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <p>Loading K-12 Knowledge Graph...</p>
    </div>

    <div id="graph-container"></div>

    <div id="info-panel">
        <h2>K-12 Knowledge Graph</h2>
        <p><strong>Nodes:</strong> <span id="node-count">0</span></p>
        <p><strong>Links:</strong> <span id="link-count">0</span></p>
        <p><strong>Subjects:</strong> <span id="subject-count">0</span></p>
        <p style="margin-top: 15px; font-size: 12px; color: #999;">
            💡 Click nodes to focus<br>
            💡 Drag to rotate<br>
            💡 Scroll to zoom
        </p>
    </div>

    <div class="legend">
        <h3>Subjects</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF6B6B;"></div>
            <span>Mathematics</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #4ECDC4;"></div>
            <span>Science</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #45B7D1;"></div>
            <span>Social Studies</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FFA07A;"></div>
            <span>English</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #95A5A6;"></div>
            <span>Root Nodes</span>
        </div>
    </div>

    <div class="controls">
        <button onclick="resetCamera()">Reset View</button>
        <button onclick="toggleParticles()">Toggle Particles</button>
        <button onclick="toggleLabels()">Toggle Labels</button>
    </div>

    <script>
        // Graph data
        const graphData = {graph_data_json};

        // Hide loading screen
        setTimeout(() => {{
            document.getElementById('loading').style.display = 'none';
        }}, 1000);

        // Update info panel
        document.getElementById('node-count').textContent = graphData.nodes.length;
        document.getElementById('link-count').textContent = graphData.links.length;

        const subjects = new Set(graphData.nodes.map(n => n.subject));
        document.getElementById('subject-count').textContent = subjects.size;

        // Configuration
        let showParticles = true;
        let showLabels = true;

        // Create 3D force graph
        const Graph = ForceGraph3D()
            (document.getElementById('graph-container'))
            .graphData(graphData)
            .nodeLabel(node => node.description)
            .nodeVal('val')
            .nodeColor('color')
            .nodeResolution(16)
            .linkWidth(link => 2)
            .linkOpacity(0.4)
            .linkDirectionalParticles(link => showParticles ? 2 : 0)
            .linkDirectionalParticleSpeed(0.005)
            .linkDirectionalParticleWidth(2)
            .linkColor(() => 'rgba(255,255,255,0.3)')
            .backgroundColor('#000011')
            .showNavInfo(false)
            .onNodeClick(node => {{
                // Focus camera on clicked node
                const distance = 200;
                const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);

                Graph.cameraPosition(
                    {{
                        x: node.x * distRatio,
                        y: node.y * distRatio,
                        z: node.z * distRatio
                    }},
                    node,
                    3000  // Animation duration in ms
                );

                console.log('Clicked:', node.name, '|', node.docs, 'documents');
            }})
            .onNodeHover(node => {{
                document.body.style.cursor = node ? 'pointer' : 'default';
            }})
            .d3Force('charge').strength(-120)
            .d3Force('link').distance(link => 30);

        // Camera positioning (initial view from angle)
        Graph.cameraPosition(
            {{ x: 0, y: 0, z: 400 }},
            {{ x: 0, y: 0, z: 0 }},
            1000
        );

        // Control functions
        function resetCamera() {{
            Graph.cameraPosition(
                {{ x: 0, y: 0, z: 400 }},
                {{ x: 0, y: 0, z: 0 }},
                2000
            );
        }}

        function toggleParticles() {{
            showParticles = !showParticles;
            Graph.linkDirectionalParticles(link => showParticles ? 2 : 0);
        }}

        function toggleLabels() {{
            showLabels = !showLabels;
            // Note: 3d-force-graph doesn't have built-in label toggle
            // This would require custom implementation
            alert(showLabels ? 'Labels enabled' : 'Labels disabled');
        }}

        // Animate force simulation
        Graph.d3Force('charge').strength(-120);
        Graph.d3Force('link').distance(30);

        console.log('3D Force Graph loaded successfully!');
        console.log('Nodes:', graphData.nodes.length);
        console.log('Links:', graphData.links.length);
    </script>
</body>
</html>"""

    # Save HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ 3D Force Graph saved to: {output_path}")
    return output_path

def main():
    print("=" * 70)
    print("3D FORCE GRAPH VISUALIZATION")
    print("=" * 70)

    # Load graphs
    print("\n📊 Loading K-12 graphs...")
    graphs = load_graphs()
    print(f"   Loaded {len(graphs)} subject graphs")

    # Convert to 3d-force-graph format
    print("\n🔄 Converting to 3D format...")
    graph_data = convert_to_3d_force_graph_format(graphs)
    print(f"   Nodes: {len(graph_data['nodes'])}")
    print(f"   Links: {len(graph_data['links'])}")

    # Generate HTML
    print("\n🎨 Generating HTML visualization...")
    output_path = generate_3d_force_html(graph_data)

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n📂 Open in browser: {output_path.resolve()}")
    print("\n💡 Features:")
    print("   - WebGL-accelerated 3D rendering")
    print("   - Click nodes to focus camera")
    print("   - Drag to rotate, scroll to zoom")
    print("   - Particle flow shows relationships")
    print("   - Color-coded by subject")

if __name__ == '__main__':
    main()
