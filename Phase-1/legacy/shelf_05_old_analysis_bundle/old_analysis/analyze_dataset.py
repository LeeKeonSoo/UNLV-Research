"""
Dataset Analysis for Tiny-Textbooks Hierarchical Graph
Analyzes strengths, weaknesses, and coverage for SLM training optimization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
import seaborn as sns

# Configuration
GRAPHS_DIR = "graphs"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class DatasetAnalyzer:
    """
    Comprehensive analysis of hierarchical graph for SLM dataset optimization
    """

    def __init__(self, graph_file):
        print("=" * 70)
        print("DATASET ANALYSIS FOR SLM OPTIMIZATION")
        print("=" * 70)
        print(f"\n📂 Loading: {graph_file}")

        with open(graph_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.nodes = self.data['nodes']
        self.edges = self.data['edges']
        self.stats = self.data['statistics']

        print(f"✅ Loaded {len(self.nodes):,} nodes, {len(self.edges):,} edges")

        # Precompute useful structures
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build lookup tables for efficient querying"""
        print("\n🔨 Building lookup tables...")

        # Node by ID
        self.node_by_id = {node['id']: node for node in self.nodes}

        # Nodes by level
        self.nodes_by_level = defaultdict(list)
        for node in self.nodes:
            self.nodes_by_level[node['level']].append(node)

        # Leaf nodes (nodes with no children)
        self.leaf_nodes = [n for n in self.nodes if len(n['children']) == 0]

        # Extract domain for each node
        for node in self.nodes:
            node['domain'] = self._extract_domain(node)

        # Nodes by domain (Level 2 domains)
        self.level2_domains = [n for n in self.nodes if n['level'] == 2]
        self.domain_names = [n['name'].split('(')[0].strip() for n in self.level2_domains]

        print(f"   Level 2 domains: {len(self.level2_domains)}")
        print(f"   Leaf nodes: {len(self.leaf_nodes):,}")

    def _extract_domain(self, node):
        """Extract Level 2 domain for a node"""
        if node['level'] == 1:
            return "Root"
        if node['level'] == 2:
            return node['name'].split('(')[0].strip()

        # Find Level 2 ancestor
        parent_id = node['parent']
        while parent_id:
            parent = self.node_by_id.get(parent_id)
            if not parent:
                break
            if parent['level'] == 2:
                return parent['name'].split('(')[0].strip()
            parent_id = parent['parent']

        return "Unknown"

    # =========================================================================
    # ANALYSIS 1: BASIC STATISTICS
    # =========================================================================

    def analyze_basic_stats(self):
        """Compute and display basic statistics"""
        print("\n" + "=" * 70)
        print("📊 BASIC STATISTICS")
        print("=" * 70)

        total_docs = self.stats['total_documents']
        total_nodes = self.stats['total_nodes']
        max_depth = self.stats['max_depth']

        print(f"\n📚 Dataset Overview:")
        print(f"   Total Documents: {total_docs:,}")
        print(f"   Total Nodes: {total_nodes:,}")
        print(f"   Max Hierarchy Depth: {max_depth}")
        print(f"   Level 2 Domains: {len(self.level2_domains)}")
        print(f"   Leaf Nodes: {len(self.leaf_nodes):,}")

        # Average branching factor
        non_leaf = [n for n in self.nodes if len(n['children']) > 0]
        avg_children = np.mean([len(n['children']) for n in non_leaf])
        print(f"\n🌳 Tree Structure:")
        print(f"   Average Branching Factor: {avg_children:.2f}")
        print(f"   Non-leaf Nodes: {len(non_leaf):,}")

        return {
            'total_documents': total_docs,
            'total_nodes': total_nodes,
            'max_depth': max_depth,
            'num_domains': len(self.level2_domains),
            'num_leaf_nodes': len(self.leaf_nodes),
            'avg_branching': avg_children
        }

    # =========================================================================
    # ANALYSIS 2: LEVEL DISTRIBUTION
    # =========================================================================

    def analyze_level_distribution(self):
        """Analyze node and document distribution across levels"""
        print("\n" + "=" * 70)
        print("📊 LEVEL-WISE DISTRIBUTION")
        print("=" * 70)

        level_data = []

        print(f"\n{'Level':<8} {'Nodes':<12} {'Docs/Node':<15} {'Total Docs':<15} {'% Docs':<10}")
        print("-" * 70)

        for level in sorted(self.nodes_by_level.keys()):
            nodes_at_level = self.nodes_by_level[level]
            num_nodes = len(nodes_at_level)
            total_docs_in_level = sum(n['document_count'] for n in nodes_at_level)
            avg_docs_per_node = total_docs_in_level / num_nodes if num_nodes > 0 else 0
            pct = (total_docs_in_level / self.stats['total_documents']) * 100

            level_data.append({
                'level': level,
                'num_nodes': num_nodes,
                'total_docs': total_docs_in_level,
                'avg_docs_per_node': avg_docs_per_node,
                'pct_docs': pct
            })

            print(f"{level:<8} {num_nodes:<12,} {avg_docs_per_node:<15,.1f} {total_docs_in_level:<15,} {pct:<10.1f}%")

        # Leaf node distribution by level
        print("\n📍 Leaf Node Distribution:")
        leaf_by_level = defaultdict(list)
        for node in self.leaf_nodes:
            leaf_by_level[node['level']].append(node)

        print(f"{'Level':<8} {'Leaf Nodes':<15} {'Docs in Leaves':<20} {'% Total Docs':<15}")
        print("-" * 70)

        for level in sorted(leaf_by_level.keys()):
            leaves = leaf_by_level[level]
            num_leaves = len(leaves)
            docs_in_leaves = sum(n['document_count'] for n in leaves)
            pct = (docs_in_leaves / self.stats['total_documents']) * 100

            print(f"{level:<8} {num_leaves:<15,} {docs_in_leaves:<20,} {pct:<15.1f}%")

        return level_data, leaf_by_level

    # =========================================================================
    # ANALYSIS 3: DOMAIN COVERAGE
    # =========================================================================

    def analyze_domain_coverage(self):
        """Analyze coverage across Level 2 domains"""
        print("\n" + "=" * 70)
        print("📊 DOMAIN COVERAGE ANALYSIS (Level 2)")
        print("=" * 70)

        domain_stats = []

        print(f"\n{'Domain':<35} {'Docs':<12} {'% Total':<10} {'Nodes':<10}")
        print("-" * 70)

        # Sort by document count
        sorted_domains = sorted(self.level2_domains,
                                key=lambda x: x['document_count'],
                                reverse=True)

        for domain_node in sorted_domains:
            domain_name = domain_node['name'].split('(')[0].strip()
            doc_count = domain_node['document_count']
            pct = (doc_count / self.stats['total_documents']) * 100

            # Count nodes in this domain
            domain_nodes = [n for n in self.nodes if n['domain'] == domain_name]
            num_nodes = len(domain_nodes)

            domain_stats.append({
                'domain': domain_name,
                'doc_count': doc_count,
                'pct': pct,
                'num_nodes': num_nodes
            })

            print(f"{domain_name:<35} {doc_count:<12,} {pct:<10.1f}% {num_nodes:<10,}")

        # Calculate Gini coefficient (inequality measure)
        doc_counts = [d['doc_count'] for d in domain_stats]
        gini = self._calculate_gini(doc_counts)

        print(f"\n📈 Domain Balance:")
        print(f"   Gini Coefficient: {gini:.3f}")
        print(f"   (0 = perfect equality, 1 = maximum inequality)")

        if gini > 0.4:
            print(f"   ⚠️  High inequality detected - some domains are overrepresented")
        elif gini > 0.25:
            print(f"   ⚡ Moderate inequality - consider balancing")
        else:
            print(f"   ✅ Good balance across domains")

        return domain_stats, gini

    def _calculate_gini(self, values):
        """Calculate Gini coefficient"""
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * sum((i+1) * val for i, val in enumerate(sorted_values)) -
                (n + 1) * cumsum[-1]) / (n * cumsum[-1])

    # =========================================================================
    # ANALYSIS 4: DEPTH ANALYSIS
    # =========================================================================

    def analyze_depth(self):
        """Analyze depth characteristics per domain"""
        print("\n" + "=" * 70)
        print("📊 DEPTH ANALYSIS BY DOMAIN")
        print("=" * 70)

        domain_depth_stats = []

        print(f"\n{'Domain':<35} {'Avg Depth':<12} {'Max Depth':<12} {'Shallow %':<12}")
        print("-" * 70)

        for domain_node in self.level2_domains:
            domain_name = domain_node['name'].split('(')[0].strip()

            # Get all leaf nodes in this domain
            domain_leaves = [n for n in self.leaf_nodes if n['domain'] == domain_name]

            if not domain_leaves:
                continue

            # Calculate depth stats
            leaf_levels = [n['level'] for n in domain_leaves]
            avg_depth = np.mean(leaf_levels)
            max_depth = max(leaf_levels)

            # Percentage of "shallow" leaves (Level 5 or less)
            shallow_count = sum(1 for lvl in leaf_levels if lvl <= 5)
            shallow_pct = (shallow_count / len(domain_leaves)) * 100

            domain_depth_stats.append({
                'domain': domain_name,
                'avg_depth': avg_depth,
                'max_depth': max_depth,
                'shallow_pct': shallow_pct,
                'num_leaves': len(domain_leaves)
            })

            print(f"{domain_name:<35} {avg_depth:<12.2f} {max_depth:<12} {shallow_pct:<12.1f}%")

        # Overall depth insights
        print(f"\n🔍 Depth Insights:")
        shallow_domains = [d for d in domain_depth_stats if d['shallow_pct'] > 50]
        deep_domains = [d for d in domain_depth_stats if d['avg_depth'] >= 7.5]

        print(f"   Shallow domains (>50% at Level ≤5): {len(shallow_domains)}")
        if shallow_domains:
            for d in shallow_domains[:3]:
                print(f"      - {d['domain']}: {d['shallow_pct']:.1f}%")

        print(f"   Deep domains (avg depth ≥7.5): {len(deep_domains)}")
        if deep_domains:
            for d in deep_domains[:3]:
                print(f"      - {d['domain']}: avg {d['avg_depth']:.2f}")

        return domain_depth_stats

    # =========================================================================
    # ANALYSIS 5: GRANULARITY & REDUNDANCY
    # =========================================================================

    def analyze_granularity(self):
        """Analyze document granularity and potential redundancy"""
        print("\n" + "=" * 70)
        print("📊 GRANULARITY & REDUNDANCY ANALYSIS")
        print("=" * 70)

        # Leaf node size distribution
        leaf_sizes = [n['document_count'] for n in self.leaf_nodes]

        print(f"\n📏 Leaf Node Size Distribution:")
        print(f"   Total Leaf Nodes: {len(self.leaf_nodes):,}")
        print(f"   Mean docs/leaf: {np.mean(leaf_sizes):.1f}")
        print(f"   Median docs/leaf: {np.median(leaf_sizes):.1f}")
        print(f"   Std dev: {np.std(leaf_sizes):.1f}")

        # Size buckets
        size_buckets = {
            '1-10': sum(1 for s in leaf_sizes if 1 <= s <= 10),
            '11-50': sum(1 for s in leaf_sizes if 11 <= s <= 50),
            '51-100': sum(1 for s in leaf_sizes if 51 <= s <= 100),
            '101-500': sum(1 for s in leaf_sizes if 101 <= s <= 500),
            '500+': sum(1 for s in leaf_sizes if s > 500)
        }

        print(f"\n📦 Leaf Node Size Buckets:")
        for bucket, count in size_buckets.items():
            pct = (count / len(self.leaf_nodes)) * 100
            print(f"   {bucket:<10} docs: {count:>6,} nodes ({pct:>5.1f}%)")

        # Identify potentially redundant clusters
        large_leaves = sorted([n for n in self.leaf_nodes if n['document_count'] > 100],
                              key=lambda x: x['document_count'], reverse=True)

        print(f"\n⚠️  Large Leaf Nodes (potential redundancy):")
        print(f"   Found {len(large_leaves):,} leaf nodes with >100 documents")

        if large_leaves:
            print(f"\n   Top 10 largest:")
            print(f"   {'Name':<50} {'Docs':<10} {'Level':<10}")
            print("   " + "-" * 70)
            for node in large_leaves[:10]:
                name = node['name'][:47] + "..." if len(node['name']) > 50 else node['name']
                print(f"   {name:<50} {node['document_count']:<10,} {node['level']:<10}")

        # Calculate percentage in large leaves
        docs_in_large_leaves = sum(n['document_count'] for n in large_leaves)
        pct_in_large = (docs_in_large_leaves / self.stats['total_documents']) * 100

        print(f"\n   Total docs in large leaves: {docs_in_large_leaves:,} ({pct_in_large:.1f}%)")

        if pct_in_large > 20:
            print(f"   ⚠️  High redundancy risk - {pct_in_large:.1f}% in large clusters")

        return {
            'leaf_sizes': leaf_sizes,
            'size_buckets': size_buckets,
            'large_leaves': large_leaves,
            'pct_in_large_leaves': pct_in_large
        }

    # =========================================================================
    # ANALYSIS 6: SLM SUITABILITY ASSESSMENT
    # =========================================================================

    def assess_slm_suitability(self, level_data, domain_stats, depth_stats, granularity_stats):
        """Assess dataset suitability for SLM training"""
        print("\n" + "=" * 70)
        print("🎯 SLM TRAINING SUITABILITY ASSESSMENT")
        print("=" * 70)

        scores = {}
        issues = []
        recommendations = []

        # 1. Foundation Coverage (Level 3-5)
        foundation_levels = [3, 4, 5]
        foundation_docs = sum(ld['total_docs'] for ld in level_data
                              if ld['level'] in foundation_levels)
        foundation_pct = (foundation_docs / self.stats['total_documents']) * 100

        print(f"\n1️⃣ Foundation Coverage (Levels 3-5):")
        print(f"   Documents: {foundation_docs:,} ({foundation_pct:.1f}%)")

        if foundation_pct < 30:
            score = "⚠️  LOW"
            issues.append("Insufficient foundational content for SLM")
            recommendations.append("Increase Level 3-5 coverage to at least 30%")
        elif foundation_pct < 45:
            score = "⚡ MODERATE"
            recommendations.append("Consider boosting Level 3-5 content")
        else:
            score = "✅ GOOD"

        scores['foundation'] = score
        print(f"   Score: {score}")

        # 2. Domain Balance
        gini = domain_stats[1]
        print(f"\n2️⃣ Domain Balance:")
        print(f"   Gini Coefficient: {gini:.3f}")

        if gini > 0.4:
            score = "⚠️  POOR - High imbalance"
            issues.append("Severe domain imbalance detected")
            recommendations.append("Resample to balance domain representation")
        elif gini > 0.25:
            score = "⚡ FAIR - Moderate imbalance"
            recommendations.append("Consider balancing overrepresented domains")
        else:
            score = "✅ GOOD"

        scores['balance'] = score
        print(f"   Score: {score}")

        # 3. Redundancy Risk
        pct_large = granularity_stats['pct_in_large_leaves']
        print(f"\n3️⃣ Redundancy Risk:")
        print(f"   Docs in large leaves: {pct_large:.1f}%")

        if pct_large > 30:
            score = "⚠️  HIGH"
            issues.append("High redundancy - many docs in large clusters")
            recommendations.append("Apply deduplication to large leaf nodes")
        elif pct_large > 15:
            score = "⚡ MODERATE"
            recommendations.append("Consider selective deduplication")
        else:
            score = "✅ LOW"

        scores['redundancy'] = score
        print(f"   Score: {score}")

        # 4. Depth Distribution
        shallow_domains = [d for d in depth_stats if d['shallow_pct'] > 60]
        deep_domains = [d for d in depth_stats if d['avg_depth'] >= 7.5]

        print(f"\n4️⃣ Depth Distribution:")
        print(f"   Very shallow domains: {len(shallow_domains)}")
        print(f"   Very deep domains: {len(deep_domains)}")

        if len(shallow_domains) > 5:
            issues.append(f"{len(shallow_domains)} domains lack depth")
            recommendations.append("Supplement shallow domains with detailed content")

        if len(deep_domains) > 8:
            issues.append(f"{len(deep_domains)} domains may be over-detailed")
            recommendations.append("Consider pruning overly granular content")

        # Overall Summary
        print(f"\n" + "=" * 70)
        print("📋 SUMMARY")
        print("=" * 70)

        print(f"\n✅ Strengths:")
        print(f"   • {len(self.level2_domains)} diverse domains covered")
        print(f"   • Deep hierarchical structure (8 levels)")
        print(f"   • {self.stats['total_documents']:,} total documents")

        if issues:
            print(f"\n⚠️  Issues Found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")

        if recommendations:
            print(f"\n💡 Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        return {
            'scores': scores,
            'issues': issues,
            'recommendations': recommendations
        }

    # =========================================================================
    # MAIN ANALYSIS RUNNER
    # =========================================================================

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 70)
        print("🚀 STARTING FULL ANALYSIS")
        print("=" * 70)

        # Run all analyses
        basic_stats = self.analyze_basic_stats()
        level_data, leaf_by_level = self.analyze_level_distribution()
        domain_stats, gini = self.analyze_domain_coverage()
        depth_stats = self.analyze_depth()
        granularity_stats = self.analyze_granularity()
        slm_assessment = self.assess_slm_suitability(
            level_data, (domain_stats, gini), depth_stats, granularity_stats
        )

        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)

        return {
            'basic_stats': basic_stats,
            'level_data': level_data,
            'domain_stats': domain_stats,
            'depth_stats': depth_stats,
            'granularity_stats': granularity_stats,
            'slm_assessment': slm_assessment
        }


def main():
    """Main analysis entry point"""
    graph_file = Path(GRAPHS_DIR) / "deep_hierarchy.json"

    if not graph_file.exists():
        print(f"❌ Graph file not found: {graph_file}")
        print("   Run: python build_deep_graph.py")
        return 1

    # Run analysis
    analyzer = DatasetAnalyzer(str(graph_file))
    results = analyzer.run_full_analysis()

    # Save results
    output_file = Path("results") / "analysis_report.json"
    output_file.parent.mkdir(exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results_serializable = convert_types(results)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
