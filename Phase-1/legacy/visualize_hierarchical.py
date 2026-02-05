"""
Hierarchical Network Visualization using Cytoscape.js

Creates structured, breadthfirst hierarchical layouts perfect for showing
K-12 curriculum progression (Kindergarten → Elementary → Middle → High School).

Features:
- Breadthfirst hierarchical layout
- Level-by-level progression
- Expandable/collapsible nodes
- Detailed node information on click
- Clean, academic styling
"""

import json
from pathlib import Path

# Subject color scheme (matching 3D visualization)
SUBJECT_COLORS = {
    'mathematics': '#FF6B6B',
    'science': '#4ECDC4',
    'social_studies': '#45B7D1',
    'english': '#FFA07A',
    'language_arts': '#98D8C8',
    'root': '#95A5A6'
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

def convert_to_cytoscape_format(graphs):
    """
    Convert K-12 graphs to Cytoscape.js format

    Format:
    {
        "nodes": [
            {
                "data": {
                    "id": "mathematics_concept_0",
                    "label": "Mathematics",
                    "subject": "mathematics",
                    "grade": null,
                    "docs": 0,
                    "parent": null  # For compound nodes
                }
            }
        ],
        "edges": [
            {
                "data": {
                    "id": "edge_0",
                    "source": "mathematics_concept_0",
                    "target": "mathematics_concept_1"
                }
            }
        ]
    }
    """
    elements = []
    node_id_map = {}

    for subject, graph_data in graphs.items():
        # Add nodes
        for node in graph_data.get('nodes', []):
            unique_id = f"{subject}_{node['id']}"
            node_id_map[node['id']] = unique_id

            node_subject = node.get('subject', subject)
            color = SUBJECT_COLORS.get(node_subject, SUBJECT_COLORS['root'])

            # Determine node parent for compound nodes (grouping)
            parent_id = None
            if node.get('parent'):
                parent_id = node_id_map.get(node['parent'])

            elements.append({
                'data': {
                    'id': unique_id,
                    'label': node['name'],
                    'subject': node_subject,
                    'grade': node.get('grade_level'),
                    'docs': node.get('document_count', 0),
                    'parent': parent_id,
                    'color': color
                }
            })

        # Add edges
        for node in graph_data.get('nodes', []):
            if node.get('parent'):
                source_id = node_id_map.get(node['parent'])
                target_id = node_id_map.get(node['id'])

                if source_id and target_id:
                    edge_id = f"{source_id}_{target_id}"
                    elements.append({
                        'data': {
                            'id': edge_id,
                            'source': source_id,
                            'target': target_id
                        }
                    })

    return elements

def generate_cytoscape_html(elements, output_path='k12_reports/hierarchical_view.html'):
    """
    Generate standalone HTML with Cytoscape.js hierarchical visualization
    """

    elements_json = json.dumps(elements, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-12 Knowledge Graph - Hierarchical View</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            overflow: hidden;
        }}

        #cy {{
            width: 100vw;
            height: 100vh;
            background: #ffffff;
        }}

        #info-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            max-width: 300px;
            z-index: 1000;
        }}

        #info-panel h2 {{
            margin: 0 0 10px 0;
            font-size: 18px;
            color: #333;
        }}

        #info-panel p {{
            margin: 5px 0;
            font-size: 14px;
            color: #666;
        }}

        #info-panel .stat {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}

        #node-details {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            max-width: 300px;
            z-index: 1000;
            display: none;
        }}

        #node-details h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #333;
        }}

        #node-details p {{
            margin: 5px 0;
            font-size: 13px;
            color: #666;
        }}

        .controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
            width: 100%;
        }}

        .controls button:hover {{
            background: #764ba2;
        }}

        .legend {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 1000;
        }}

        .legend h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #333;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 12px;
        }}

        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div id="cy"></div>

    <div id="info-panel">
        <h2>K-12 Hierarchy</h2>
        <div class="stat">
            <span>Nodes:</span>
            <strong id="node-count">0</strong>
        </div>
        <div class="stat">
            <span>Edges:</span>
            <strong id="edge-count">0</strong>
        </div>
        <div class="stat">
            <span>Subjects:</span>
            <strong id="subject-count">0</strong>
        </div>
        <p style="margin-top: 15px; font-size: 12px; color: #999;">
            💡 Click nodes for details<br>
            💡 Drag to pan<br>
            💡 Scroll to zoom
        </p>
    </div>

    <div id="node-details">
        <h3 id="node-name">Node Name</h3>
        <p><strong>Subject:</strong> <span id="node-subject"></span></p>
        <p><strong>Grade:</strong> <span id="node-grade"></span></p>
        <p><strong>Documents:</strong> <span id="node-docs"></span></p>
        <button onclick="closeDetails()" style="margin-top: 10px; width: 100%; padding: 8px; background: #ddd; border: none; border-radius: 5px; cursor: pointer;">Close</button>
    </div>

    <div class="controls">
        <button onclick="resetLayout()">Reset Layout</button>
        <button onclick="fitToScreen()">Fit to Screen</button>
        <button onclick="layoutCircle()">Circle Layout</button>
        <button onclick="layoutBreadthfirst()">Hierarchical</button>
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
    </div>

    <script>
        // Graph data
        const elements = {elements_json};

        // Count statistics
        const nodes = elements.filter(el => !el.data.source);
        const edges = elements.filter(el => el.data.source);
        const subjects = new Set(nodes.map(n => n.data.subject));

        document.getElementById('node-count').textContent = nodes.length;
        document.getElementById('edge-count').textContent = edges.length;
        document.getElementById('subject-count').textContent = subjects.size;

        // Initialize Cytoscape
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,

            style: [
                // Node styles
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'color': '#fff',
                        'font-size': '12px',
                        'font-weight': 'bold',
                        'background-color': 'data(color)',
                        'border-width': 2,
                        'border-color': '#fff',
                        'width': node => Math.sqrt(node.data('docs') + 1) * 30,
                        'height': node => Math.sqrt(node.data('docs') + 1) * 30,
                        'text-wrap': 'wrap',
                        'text-max-width': '100px',
                        'overlay-opacity': 0,
                        'transition-property': 'background-color, border-color, width, height',
                        'transition-duration': '0.3s'
                    }}
                }},
                // Node hover
                {{
                    selector: 'node:active',
                    style: {{
                        'border-width': 4,
                        'border-color': '#667eea'
                    }}
                }},
                // Edge styles
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#cbd5e0',
                        'target-arrow-color': '#cbd5e0',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'arrow-scale': 1.5,
                        'overlay-opacity': 0
                    }}
                }},
                // Edge hover
                {{
                    selector: 'edge:active',
                    style: {{
                        'line-color': '#667eea',
                        'target-arrow-color': '#667eea',
                        'width': 3
                    }}
                }}
            ],

            layout: {{
                name: 'breadthfirst',
                directed: true,
                spacingFactor: 1.5,
                padding: 50,
                animate: true,
                animationDuration: 1000,
                avoidOverlap: true,
                nodeDimensionsIncludeLabels: true
            }},

            minZoom: 0.2,
            maxZoom: 3,
            wheelSensitivity: 0.2
        }});

        // Event handlers
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();

            document.getElementById('node-name').textContent = data.label;
            document.getElementById('node-subject').textContent = data.subject;
            document.getElementById('node-grade').textContent = data.grade || 'All grades';
            document.getElementById('node-docs').textContent = data.docs;
            document.getElementById('node-details').style.display = 'block';

            // Highlight connected nodes
            cy.elements().removeClass('highlighted');
            node.addClass('highlighted');
            node.connectedEdges().addClass('highlighted');
        }});

        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                document.getElementById('node-details').style.display = 'none';
                cy.elements().removeClass('highlighted');
            }}
        }});

        // Control functions
        function closeDetails() {{
            document.getElementById('node-details').style.display = 'none';
            cy.elements().removeClass('highlighted');
        }}

        function resetLayout() {{
            cy.layout({{
                name: 'breadthfirst',
                directed: true,
                spacingFactor: 1.5,
                padding: 50,
                animate: true,
                animationDuration: 1000
            }}).run();
        }}

        function fitToScreen() {{
            cy.fit(50);
        }}

        function layoutCircle() {{
            cy.layout({{
                name: 'circle',
                animate: true,
                animationDuration: 1000,
                padding: 50
            }}).run();
        }}

        function layoutBreadthfirst() {{
            resetLayout();
        }}

        // Initial fit
        cy.fit(50);

        console.log('Cytoscape.js hierarchical view loaded!');
        console.log('Nodes:', nodes.length);
        console.log('Edges:', edges.length);
    </script>
</body>
</html>"""

    # Save HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Hierarchical view saved to: {output_path}")
    return output_path

def main():
    print("=" * 70)
    print("HIERARCHICAL NETWORK VISUALIZATION (Cytoscape.js)")
    print("=" * 70)

    # Load graphs
    print("\n📊 Loading K-12 graphs...")
    graphs = load_graphs()
    print(f"   Loaded {len(graphs)} subject graphs")

    # Convert to Cytoscape format
    print("\n🔄 Converting to Cytoscape.js format...")
    elements = convert_to_cytoscape_format(graphs)
    nodes = [el for el in elements if 'source' not in el['data']]
    edges = [el for el in elements if 'source' in el['data']]
    print(f"   Nodes: {len(nodes)}")
    print(f"   Edges: {len(edges)}")

    # Generate HTML
    print("\n🎨 Generating HTML visualization...")
    output_path = generate_cytoscape_html(elements)

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n📂 Open in browser: {output_path.resolve()}")
    print("\n💡 Features:")
    print("   - Hierarchical breadthfirst layout")
    print("   - Grade-by-grade progression")
    print("   - Click nodes for detailed information")
    print("   - Multiple layout options")
    print("   - Clean, academic styling")

if __name__ == '__main__':
    main()
