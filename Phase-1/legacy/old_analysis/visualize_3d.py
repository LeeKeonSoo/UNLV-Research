"""
3D Interactive Force Graph with Domain Toggle
Visualizes deep hierarchical graph with on/off controls for each domain
"""

import json
from pathlib import Path
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np

from config import DOMAIN_COLORS, GRAPHS_DIR, VISUALIZATIONS_DIR


class Interactive3DVisualizer:
    """
    Create 3D force-directed graph with domain toggle controls
    """

    def __init__(self, graph_file):
        print("\n📊 Loading graph data...")
        with open(graph_file, 'r', encoding='utf-8') as f:
            self.graph_data = json.load(f)

        self.nodes = self.graph_data['nodes']
        self.edges = self.graph_data['edges']

        print(f"   Nodes: {len(self.nodes):,}")
        print(f"   Edges: {len(self.edges):,}")

    def extract_domain(self, node):
        """Extract broad domain from node name"""
        # Level 2 nodes are broad domains
        if node['level'] == 2:
            return node['name'].split('(')[0].strip()

        # Find parent at level 2
        parent_id = node['parent']
        while parent_id:
            parent = next((n for n in self.nodes if n['id'] == parent_id), None)
            if parent and parent['level'] == 2:
                return parent['name'].split('(')[0].strip()
            if parent:
                parent_id = parent['parent']
            else:
                break

        return "General"

    def assign_domain_colors(self):
        """Assign colors to each domain"""
        # Get all unique domains
        domains = set()
        for node in self.nodes:
            if node['level'] >= 2:  # Skip root
                domain = self.extract_domain(node)
                domains.add(domain)

        domains = sorted(list(domains))

        # Assign colors
        domain_color_map = {}
        for idx, domain in enumerate(domains):
            color_idx = idx % len(DOMAIN_COLORS)
            domain_color_map[domain] = DOMAIN_COLORS[color_idx]

        print(f"\n🎨 Assigned colors to {len(domains)} domains:")
        for domain in domains[:10]:  # Show first 10
            print(f"   {domain}: {domain_color_map[domain]}")
        if len(domains) > 10:
            print(f"   ... and {len(domains) - 10} more")

        return domain_color_map, domains

    def compute_layout(self):
        """Compute 3D force-directed layout"""
        print("\n🔄 Computing 3D layout...")

        # Create node position mapping
        node_positions = {}

        # Use hierarchical layout with some randomness
        for node in self.nodes:
            level = node['level']

            # Add random offset based on level
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 5)

            x = radius * np.cos(angle) + np.random.normal(0, 0.5)
            y = radius * np.sin(angle) + np.random.normal(0, 0.5)
            z = -level * 2 + np.random.normal(0, 0.3)  # Vertical separation by level

            node_positions[node['id']] = (x, y, z)

        return node_positions

    def create_html(self, output_file):
        """Create interactive HTML with domain toggles"""
        print("\n🎨 Generating HTML visualization...")

        # Assign domains and colors
        domain_color_map, domains = self.assign_domain_colors()

        # Add domain info to nodes
        for node in self.nodes:
            if node['level'] == 1:
                node['domain'] = 'Root'
            else:
                node['domain'] = self.extract_domain(node)

        # Prepare data for JavaScript
        graph_json = json.dumps({
            'nodes': self.nodes,
            'edges': self.edges,
            'domains': domains,
            'domainColors': domain_color_map
        })

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Tiny-Textbooks Deep Hierarchy - 3D Interactive</title>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }}
        #graph-container {{
            width: 100vw;
            height: 100vh;
        }}
        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            max-height: 90vh;
            overflow-y: auto;
            width: 250px;
            z-index: 1000;
        }}
        #controls h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 18px;
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
        }}
        .domain-toggle {{
            display: block;
            margin-bottom: 10px;
            cursor: pointer;
            padding: 5px;
            border-radius: 5px;
            transition: background-color 0.2s;
        }}
        .domain-toggle:hover {{
            background-color: #f0f0f0;
        }}
        .domain-toggle input {{
            margin-right: 8px;
            cursor: pointer;
        }}
        .domain-color {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-left: 5px;
        }}
        #stats {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 2px solid #ddd;
            font-size: 13px;
        }}
        .stat-item {{
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-weight: bold;
            color: #555;
        }}
        #select-all {{
            margin-bottom: 15px;
            padding: 8px 15px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            font-size: 14px;
        }}
        #select-all:hover {{
            background: #45a049;
        }}
        #deselect-all {{
            padding: 8px 15px;
            background: #f44336;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            font-size: 14px;
        }}
        #deselect-all:hover {{
            background: #da190b;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            max-width: 300px;
            z-index: 1000;
        }}
        #info h3 {{
            margin-top: 0;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div id="graph-container"></div>

    <div id="info">
        <h3>🔍 Tiny-Textbooks Hierarchy</h3>
        <div class="stat-item">
            <span class="stat-label">Total Documents:</span>
            <span id="total-docs">420,000</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Visible Nodes:</span>
            <span id="visible-nodes">0</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Max Depth:</span>
            <span id="max-depth">8</span>
        </div>
    </div>

    <div id="controls">
        <h3>🎛️ Domain Controls</h3>
        <button id="select-all">✓ Select All</button>
        <button id="deselect-all">✗ Deselect All</button>
        <div id="domain-toggles" style="margin-top: 15px;"></div>

        <div id="stats">
            <div class="stat-item">
                <span class="stat-label">Active Domains:</span>
                <span id="active-domains">0</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Visible Edges:</span>
                <span id="visible-edges">0</span>
            </div>
        </div>
    </div>

    <script>
        // Load graph data
        const graphData = {graph_json};

        // Initialize domain states (all visible)
        const domainStates = {{}};
        graphData.domains.forEach(domain => {{
            domainStates[domain] = true;
        }});

        // Get container
        const container = document.getElementById('graph-container');

        // Initialize 3D force graph
        const Graph = ForceGraph3D()(container)
            .graphData({{nodes: [], links: []}})
            .nodeLabel(node => `${{node.name}}\\nLevel: ${{node.level}}\\nDocs: ${{node.document_count.toLocaleString()}}`)
            .nodeColor(node => graphData.domainColors[node.domain] || '#999')
            .nodeVal(node => Math.sqrt(node.document_count) / 10)
            .linkColor(() => 'rgba(150, 150, 150, 0.2)')
            .linkWidth(0.5)
            .backgroundColor('#000011')
            .d3AlphaDecay(0.02)
            .d3VelocityDecay(0.3);

        // Create domain toggles
        const toggleContainer = document.getElementById('domain-toggles');
        graphData.domains.forEach(domain => {{
            const label = document.createElement('label');
            label.className = 'domain-toggle';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = true;
            checkbox.onchange = () => {{
                domainStates[domain] = checkbox.checked;
                updateGraph();
            }};

            const colorBox = document.createElement('span');
            colorBox.className = 'domain-color';
            colorBox.style.backgroundColor = graphData.domainColors[domain];

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(domain));
            label.appendChild(colorBox);
            toggleContainer.appendChild(label);
        }});

        // Select/Deselect all buttons
        document.getElementById('select-all').onclick = () => {{
            graphData.domains.forEach(domain => domainStates[domain] = true);
            document.querySelectorAll('#domain-toggles input').forEach(cb => cb.checked = true);
            updateGraph();
        }};

        document.getElementById('deselect-all').onclick = () => {{
            graphData.domains.forEach(domain => domainStates[domain] = false);
            document.querySelectorAll('#domain-toggles input').forEach(cb => cb.checked = false);
            updateGraph();
        }};

        // Update graph based on active domains
        function updateGraph() {{
            // Filter nodes by active domains
            const visibleNodes = graphData.nodes.filter(node =>
                node.level === 1 || domainStates[node.domain]
            );
            const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

            // Filter edges to only show between visible nodes
            const visibleLinks = graphData.edges
                .filter(edge => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target))
                .map(edge => ({{
                    source: edge.source,
                    target: edge.target
                }}));

            // Update graph
            Graph.graphData({{
                nodes: visibleNodes,
                links: visibleLinks
            }});

            // Update stats
            const activeDomains = graphData.domains.filter(d => domainStates[d]).length;
            document.getElementById('visible-nodes').textContent = visibleNodes.length.toLocaleString();
            document.getElementById('visible-edges').textContent = visibleLinks.length.toLocaleString();
            document.getElementById('active-domains').textContent = activeDomains;
        }}

        // Initial render
        updateGraph();

        // Set camera angle
        Graph.cameraPosition({{ z: 400 }});
    </script>
</body>
</html>"""

        # Save HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Saved: {output_file}")


def main():
    """Generate 3D interactive visualization"""
    print("=" * 70)
    print("3D INTERACTIVE VISUALIZATION WITH DOMAIN TOGGLES")
    print("=" * 70)

    try:
        # Find graph file
        graph_file = Path(GRAPHS_DIR) / "deep_hierarchy.json"

        if not graph_file.exists():
            print(f"\n❌ Graph file not found: {graph_file}")
            print("   Run: python build_deep_graph.py")
            return 1

        # Create visualizer
        visualizer = Interactive3DVisualizer(str(graph_file))

        # Generate HTML
        output_file = Path(VISUALIZATIONS_DIR) / "3d_interactive.html"
        visualizer.create_html(str(output_file))

        print("\n" + "=" * 70)
        print("✅ VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"\n📂 Open in browser:")
        print(f"   {output_file}")
        print("\n💡 Features:")
        print("   - Domain toggle checkboxes (show/hide domains)")
        print("   - Real-time filtering")
        print("   - Interactive 3D rotation/zoom")
        print("   - Color-coded by domain")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
