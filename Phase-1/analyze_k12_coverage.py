"""
K-12 Coverage Analysis and Reporting

Analyzes the constructed concept graphs to understand:
- What concepts are covered
- Coverage depth and breadth
- Gaps and insufficiencies
- Grade-level distribution
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from config import *


class CoverageAnalyzer:
    """
    Analyze concept coverage from graphs
    """
    
    def __init__(self, graph_dir=K12_GRAPHS_DIR):
        self.graph_dir = Path(graph_dir)
        self.graphs = {}
        
    def load_graphs(self):
        """Load all available graphs"""
        print("📊 Loading graphs...")
        
        for graph_file in self.graph_dir.glob("*_graph.json"):
            subject = graph_file.stem.replace("_graph", "")
            
            with open(graph_file) as f:
                self.graphs[subject] = json.load(f)
            
            print(f"  ✅ Loaded {subject}: {len(self.graphs[subject]['nodes'])} nodes")
        
        return self.graphs
    
    def analyze_subject(self, subject_name, graph_data):
        """
        Analyze coverage for a single subject
        """
        nodes = {node['id']: node for node in graph_data['nodes']}
        edges = graph_data['edges']
        
        analysis = {
            "subject": subject_name,
            "total_concepts": len(nodes),
            "total_documents": sum(node['document_count'] for node in nodes.values()),
            "depth": graph_data['statistics']['max_depth'],
            "by_grade": defaultdict(lambda: {"concepts": 0, "documents": 0}),
            "well_covered": [],
            "under_covered": [],
            "gaps": []
        }
        
        # Analyze by grade
        for node in nodes.values():
            grade = node.get('grade_level')
            if grade:
                analysis['by_grade'][grade]['concepts'] += 1
                analysis['by_grade'][grade]['documents'] += node['document_count']
                
                # Identify well-covered vs under-covered
                if node['document_count'] >= MIN_EXAMPLES_PER_CONCEPT:
                    analysis['well_covered'].append({
                        "concept": node['name'],
                        "grade": grade,
                        "docs": node['document_count']
                    })
                else:
                    analysis['under_covered'].append({
                        "concept": node['name'],
                        "grade": grade,
                        "docs": node['document_count']
                    })
        
        return analysis
    
    def generate_report(self):
        """
        Generate comprehensive coverage report
        """
        if not self.graphs:
            self.load_graphs()
        
        report = {
            "summary": {
                "subjects_analyzed": len(self.graphs),
                "total_concepts": 0,
                "total_documents": 0
            },
            "by_subject": {}
        }
        
        # Analyze each subject
        for subject, graph_data in self.graphs.items():
            analysis = self.analyze_subject(subject, graph_data)
            report["by_subject"][subject] = analysis
            
            report["summary"]["total_concepts"] += analysis["total_concepts"]
            report["summary"]["total_documents"] += analysis["total_documents"]
        
        return report
    
    def save_report(self, report, filename="k12_coverage_report.json"):
        """Save report to file"""
        output_path = Path(K12_REPORTS_DIR) / filename
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to {output_path}")
        return output_path
    
    def print_summary(self, report):
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("K-12 CURRICULUM COVERAGE ANALYSIS")
        print("="*60)
        
        summary = report["summary"]
        print(f"\nOverall Statistics:")
        print(f"  Subjects analyzed: {summary['subjects_analyzed']}")
        print(f"  Total concepts: {summary['total_concepts']}")
        print(f"  Total documents: {summary['total_documents']}")
        
        for subject, analysis in report["by_subject"].items():
            print(f"\n{subject}:")
            print(f"  Concepts: {analysis['total_concepts']}")
            print(f"  Documents: {analysis['total_documents']}")
            print(f"  Max depth: {analysis['depth']}")
            print(f"  Well-covered: {len(analysis['well_covered'])}")
            print(f"  Under-covered: {len(analysis['under_covered'])}")
            
            # Grade distribution
            print(f"  By grade:")
            for grade in sorted(analysis['by_grade'].keys()):
                grade_data = analysis['by_grade'][grade]
                print(f"    Grade {grade}: {grade_data['concepts']} concepts, "
                      f"{grade_data['documents']} docs")


def main():
    """
    Run coverage analysis
    """
    analyzer = CoverageAnalyzer()
    
    # Check if graphs exist
    if not Path(K12_GRAPHS_DIR).exists() or not list(Path(K12_GRAPHS_DIR).glob("*.json")):
        print("❌ No graphs found!")
        print(f"   Please run build_k12_graph.py first")
        print(f"   Expected graphs in: {K12_GRAPHS_DIR}")
        return
    
    # Load and analyze
    analyzer.load_graphs()
    report = analyzer.generate_report()
    
    # Print summary
    analyzer.print_summary(report)
    
    # Save detailed report
    analyzer.save_report(report)
    
    print("\n💡 Next steps:")
    print("   1. Review coverage report")
    print("   2. Identify gaps (under-covered concepts)")
    print("   3. Plan data collection for gaps")
    print("   4. Compare with Pile datasets (Phase 2)")


if __name__ == "__main__":
    main()
