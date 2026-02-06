"""
2D Graph Visualization - Obsidian-style
High-performance hierarchical graph with domain toggles
"""

import json
from pathlib import Path
from collections import defaultdict

from config import DOMAIN_COLORS, GRAPHS_DIR, VISUALIZATIONS_DIR


class ObsidianStyleVisualizer:
    """
    Create 2D hierarchical graph visualization (Obsidian Graph View style)
    Optimized for performance and visibility
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
        """Extract broad domain from node"""
        if node['level'] == 2:
            return node['name'].split('(')[0].strip()

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
        domains = set()
        for node in self.nodes:
            if node['level'] >= 2:
                domain = self.extract_domain(node)
                domains.add(domain)

        domains = sorted(list(domains))

        domain_color_map = {}
        for idx, domain in enumerate(domains):
            color_idx = idx % len(DOMAIN_COLORS)
            domain_color_map[domain] = DOMAIN_COLORS[color_idx]

        print(f"\n🎨 Found {len(domains)} domains")
        return domain_color_map, domains

    def create_html(self, output_file):
        """Create high-performance 2D visualization"""
        print("\n🎨 Generating 2D Obsidian-style visualization...")

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
    <title>Tiny-Textbooks Hierarchy - 2D Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            overflow: hidden;
            background: #1a1a1a;
            color: #e0e0e0;
        }}

        #graph-container {{
            width: 100vw;
            height: 100vh;
        }}

        #controls {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(30, 30, 30, 0.95);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #444;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            max-height: 85vh;
            overflow-y: auto;
            width: 280px;
            z-index: 1000;
        }}

        #controls h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 16px;
            border-bottom: 2px solid #555;
            padding-bottom: 10px;
            color: #fff;
        }}

        .control-section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }}

        .control-section:last-child {{
            border-bottom: none;
        }}

        .section-title {{
            font-size: 13px;
            font-weight: 600;
            color: #aaa;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .domain-toggle {{
            display: block;
            margin-bottom: 8px;
            cursor: pointer;
            padding: 6px 8px;
            border-radius: 6px;
            transition: background-color 0.2s;
            font-size: 13px;
        }}

        .domain-toggle:hover {{
            background-color: rgba(255, 255, 255, 0.05);
        }}

        .domain-toggle input {{
            margin-right: 8px;
            cursor: pointer;
        }}

        .domain-color {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 2px;
            margin-left: 6px;
        }}

        .button-group {{
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }}

        .control-button {{
            flex: 1;
            padding: 8px 12px;
            background: #2a5f8f;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: background 0.2s;
        }}

        .control-button:hover {{
            background: #1e4a70;
        }}

        .control-button.secondary {{
            background: #555;
        }}

        .control-button.secondary:hover {{
            background: #444;
        }}

        .level-control {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}

        .level-control input[type="checkbox"] {{
            margin-right: 8px;
        }}

        .level-control label {{
            font-size: 13px;
            color: #ccc;
            cursor: pointer;
        }}

        #stats {{
            font-size: 12px;
            color: #999;
        }}

        .stat-item {{
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
        }}

        .stat-label {{
            color: #aaa;
        }}

        .stat-value {{
            color: #fff;
            font-weight: 500;
        }}

        #info {{
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(30, 30, 30, 0.95);
            padding: 15px 20px;
            border-radius: 12px;
            border: 1px solid #444;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            z-index: 1000;
        }}

        #info h3 {{
            margin: 0 0 10px 0;
            font-size: 15px;
            color: #fff;
        }}

        #info .info-item {{
            font-size: 12px;
            margin-bottom: 4px;
            color: #ccc;
        }}

        .node {{
            cursor: pointer;
            transition: r 0.2s;
        }}

        .node:hover {{
            stroke: #fff;
            stroke-width: 2px;
        }}

        .link {{
            stroke: #444;
            stroke-opacity: 0.3;
        }}

        .link.highlighted {{
            stroke: #aaa;
            stroke-opacity: 0.6;
        }}

        /* Custom scrollbar */
        #controls::-webkit-scrollbar {{
            width: 8px;
        }}

        #controls::-webkit-scrollbar-track {{
            background: #222;
            border-radius: 4px;
        }}

        #controls::-webkit-scrollbar-thumb {{
            background: #555;
            border-radius: 4px;
        }}

        #controls::-webkit-scrollbar-thumb:hover {{
            background: #666;
        }}
    </style>
</head>
<body>
    <div id="graph-container"></div>

    <div id="info">
        <h3>📚 Tiny-Textbooks Hierarchy</h3>
        <div class="info-item">Total Docs: <strong>420,000</strong></div>
        <div class="info-item">Visible: <strong id="visible-count">0</strong> nodes</div>
        <div class="info-item">Depth: <strong>8</strong> levels</div>
    </div>

    <div id="controls">
        <div class="control-section">
            <div class="section-title">🎛️ View Controls</div>
            <div class="button-group">
                <button class="control-button" onclick="resetZoom()">Reset Zoom</button>
                <button class="control-button secondary" onclick="centerGraph()">Center</button>
            </div>
        </div>

        <div class="control-section">
            <div class="section-title">📊 Domains</div>
            <div class="button-group">
                <button class="control-button" id="select-all">All</button>
                <button class="control-button secondary" id="deselect-all">None</button>
            </div>
            <div id="domain-toggles"></div>
        </div>

        <div class="control-section">
            <div class="section-title">🔢 Hierarchy Levels</div>
            <div id="level-toggles"></div>
        </div>

        <div class="control-section">
            <div class="section-title">📈 Statistics</div>
            <div id="stats"></div>
        </div>
    </div>

    <script>
        // Load graph data
        const graphData = {graph_json};

        // State management
        const state = {{
            domainStates: {{}},
            levelStates: {{}},
            simulation: null,
            svg: null,
            g: null,
            link: null,
            node: null,
            zoom: null
        }};

        // Initialize domain states
        graphData.domains.forEach(domain => {{
            state.domainStates[domain] = true;
        }});

        // Initialize level states (all visible)
        for (let i = 1; i <= 8; i++) {{
            state.levelStates[i] = true;
        }}

        // Setup SVG
        const width = window.innerWidth;
        const height = window.innerHeight;

        state.svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        // Add zoom behavior
        state.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                state.g.attr("transform", event.transform);
            }});

        state.svg.call(state.zoom);

        // Create container group
        state.g = state.svg.append("g");

        // Create force simulation with hierarchical layout
        state.simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(d => {{
                // Distance based on level difference
                const sourceLevel = d.source.level || 0;
                const targetLevel = d.target.level || 0;
                return 30 + (targetLevel - sourceLevel) * 15;
            }}))
            .force("charge", d3.forceManyBody()
                .strength(d => {{
                    // Stronger repulsion for higher-level nodes
                    return -50 - (10 - d.level) * 10;
                }}))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => {{
                return Math.sqrt(d.document_count) / 30 + 3;
            }}))
            .force("radial", d3.forceRadial(
                d => (d.level - 1) * 80,  // Distance from center based on level
                width / 2,
                height / 2
            ).strength(0.3));

        // Create domain toggles
        const toggleContainer = d3.select("#domain-toggles");
        graphData.domains.forEach(domain => {{
            const label = toggleContainer.append("label")
                .attr("class", "domain-toggle");

            label.append("input")
                .attr("type", "checkbox")
                .property("checked", true)
                .on("change", function() {{
                    state.domainStates[domain] = this.checked;
                    updateGraph();
                }});

            label.append("span").text(domain);

            label.append("span")
                .attr("class", "domain-color")
                .style("background-color", graphData.domainColors[domain]);
        }});

        // Create level toggles
        const levelContainer = d3.select("#level-toggles");
        for (let i = 1; i <= 8; i++) {{
            const div = levelContainer.append("div")
                .attr("class", "level-control");

            div.append("input")
                .attr("type", "checkbox")
                .attr("id", `level-${{i}}`)
                .property("checked", true)
                .on("change", function() {{
                    state.levelStates[i] = this.checked;
                    updateGraph();
                }});

            div.append("label")
                .attr("for", `level-${{i}}`)
                .text(`Level ${{i}}`);
        }}

        // Select/Deselect all buttons
        d3.select("#select-all").on("click", () => {{
            graphData.domains.forEach(d => state.domainStates[d] = true);
            d3.selectAll("#domain-toggles input").property("checked", true);
            updateGraph();
        }});

        d3.select("#deselect-all").on("click", () => {{
            graphData.domains.forEach(d => state.domainStates[d] = false);
            d3.selectAll("#domain-toggles input").property("checked", false);
            updateGraph();
        }});

        // Update graph based on filters
        function updateGraph() {{
            // Filter nodes
            const visibleNodes = graphData.nodes.filter(node => {{
                const domainVisible = node.level === 1 || state.domainStates[node.domain];
                const levelVisible = state.levelStates[node.level];
                return domainVisible && levelVisible;
            }});

            const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

            // Filter links
            const visibleLinks = graphData.edges
                .filter(edge => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target))
                .map(edge => ({{...edge}}));

            // Update simulation
            renderGraph(visibleNodes, visibleLinks);

            // Update stats
            updateStats(visibleNodes, visibleLinks);
        }}

        function renderGraph(nodes, links) {{
            // Clear existing
            state.g.selectAll("*").remove();

            // Create links
            state.link = state.g.append("g")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("class", "link")
                .attr("stroke-width", 1);

            // Create nodes
            state.node = state.g.append("g")
                .selectAll("circle")
                .data(nodes)
                .join("circle")
                .attr("class", "node")
                .attr("r", d => Math.max(2, Math.sqrt(d.document_count) / 30))
                .attr("fill", d => graphData.domainColors[d.domain] || "#999")
                .attr("opacity", d => 0.7 + (d.level / 20))
                .call(drag(state.simulation))
                .on("mouseover", function(event, d) {{
                    d3.select(this)
                        .attr("stroke", "#fff")
                        .attr("stroke-width", 2);

                    // Highlight connected links
                    state.link.classed("highlighted", l =>
                        l.source.id === d.id || l.target.id === d.id
                    );
                }})
                .on("mouseout", function() {{
                    d3.select(this)
                        .attr("stroke", null)
                        .attr("stroke-width", 0);

                    state.link.classed("highlighted", false);
                }});

            // Add tooltips
            state.node.append("title")
                .text(d => `${{d.name}}\\nLevel: ${{d.level}}\\nDocs: ${{d.document_count.toLocaleString()}}`);

            // Update simulation
            state.simulation
                .nodes(nodes)
                .on("tick", ticked);

            state.simulation.force("link")
                .links(links);

            // Restart simulation
            state.simulation.alpha(0.3).restart();
        }}

        function ticked() {{
            state.link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            state.node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
        }}

        function drag(simulation) {{
            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }}

            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}

            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }}

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }}

        function updateStats(nodes, links) {{
            const activeDomains = graphData.domains.filter(d => state.domainStates[d]).length;
            const activeLevels = Object.values(state.levelStates).filter(v => v).length;

            d3.select("#visible-count").text(nodes.length.toLocaleString());

            d3.select("#stats").html(`
                <div class="stat-item">
                    <span class="stat-label">Nodes:</span>
                    <span class="stat-value">${{nodes.length.toLocaleString()}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Edges:</span>
                    <span class="stat-value">${{links.length.toLocaleString()}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Domains:</span>
                    <span class="stat-value">${{activeDomains}} / ${{graphData.domains.length}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Levels:</span>
                    <span class="stat-value">${{activeLevels}} / 8</span>
                </div>
            `);
        }}

        function resetZoom() {{
            state.svg.transition()
                .duration(750)
                .call(state.zoom.transform, d3.zoomIdentity);
        }}

        function centerGraph() {{
            const bounds = state.g.node().getBBox();
            const fullWidth = bounds.width;
            const fullHeight = bounds.height;
            const midX = bounds.x + fullWidth / 2;
            const midY = bounds.y + fullHeight / 2;

            const scale = 0.8 / Math.max(fullWidth / width, fullHeight / height);
            const translate = [width / 2 - scale * midX, height / 2 - scale * midY];

            state.svg.transition()
                .duration(750)
                .call(state.zoom.transform, d3.zoomIdentity
                    .translate(translate[0], translate[1])
                    .scale(scale));
        }}

        // Initial render
        updateGraph();

        // Auto-center after initial layout
        setTimeout(() => {{
            centerGraph();
        }}, 2000);
    </script>
</body>
</html>"""

        # Save HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Saved: {output_file}")


def main():
    """Generate 2D Obsidian-style visualization"""
    print("=" * 70)
    print("2D HIERARCHICAL GRAPH VISUALIZATION (OBSIDIAN-STYLE)")
    print("=" * 70)

    try:
        # Find graph file
        graph_file = Path(GRAPHS_DIR) / "deep_hierarchy.json"

        if not graph_file.exists():
            print(f"\n❌ Graph file not found: {graph_file}")
            print("   Run: python build_deep_graph.py")
            return 1

        # Create visualizer
        visualizer = ObsidianStyleVisualizer(str(graph_file))

        # Generate HTML
        output_file = Path(VISUALIZATIONS_DIR) / "2d_graph.html"
        visualizer.create_html(str(output_file))

        print("\n" + "=" * 70)
        print("✅ VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"\n📂 Open in browser:")
        print(f"   {output_file}")
        print("\n💡 Features:")
        print("   - Obsidian-style 2D hierarchical layout")
        print("   - Domain & level toggle controls")
        print("   - Zoom, pan, and drag nodes")
        print("   - High performance (D3.js force-directed)")
        print("   - Centered at origin with radial spread")
        print("   - Clear node and edge visibility")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
