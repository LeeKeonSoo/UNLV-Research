"""
Step 3: Build Interactive Dashboard for Dataset Analysis

This script:
1. Loads analysis results from Step 2
2. Aggregates statistics for visualization
3. Generates an interactive HTML dashboard with:
   - Domain distribution comparison (Khan vs Tiny-Textbooks)
   - Quality metrics comparison
   - Cross-cutting analysis (multi-domain documents)
   - Top concepts by frequency
   - Interactive filters

Output: outputs/dashboard.html
"""

import json
import jsonlines
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np


# ==============================================================================
# Configuration
# ==============================================================================

KHAN_ANALYSIS = "outputs/khan_analysis.jsonl"
TINY_ANALYSIS = "outputs/tiny_textbooks_analysis.jsonl"
OUTPUT_HTML = "outputs/dashboard.html"


# ==============================================================================
# Data Loading and Aggregation
# ==============================================================================

def load_analysis(filepath: str) -> List[Dict]:
    """Load JSONL analysis file."""
    results = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            results.append(obj)
    return results


def aggregate_domain_stats(data: List[Dict]) -> Dict:
    """
    Aggregate domain statistics.

    Returns:
    {
        "domain_counts": {domain_id: count, ...},
        "subject_counts": {subject: count, ...},
        "multi_domain_ratio": float,
        "top_concepts": [(concept_id, count), ...]
    }
    """
    domain_counter = Counter()
    subject_counter = Counter()
    multi_domain_count = 0

    for item in data:
        domains = item.get("domain_labels", {})

        # Count domains
        for domain_id in domains:
            domain_counter[domain_id] += 1

            # Extract subject (before ::)
            subject = domain_id.split("::")[0] if "::" in domain_id else "Unknown"
            subject_counter[subject] += 1

        # Multi-domain detection
        if len(domains) > 1:
            multi_domain_count += 1

    multi_domain_ratio = multi_domain_count / len(data) if data else 0

    return {
        "domain_counts": dict(domain_counter),
        "subject_counts": dict(subject_counter),
        "multi_domain_ratio": multi_domain_ratio,
        "top_concepts": domain_counter.most_common(20)
    }


def aggregate_quality_stats(data: List[Dict]) -> Dict:
    """
    Aggregate quality statistics.

    Returns:
    {
        "has_examples_ratio": float,
        "has_explanation_ratio": float,
        "has_structure_ratio": float
    }
    """
    has_examples = 0
    has_explanation = 0
    has_structure = 0

    for item in data:
        markers = item.get("educational_markers", {})

        if markers.get("has_examples", False):
            has_examples += 1
        if markers.get("has_explanation", False):
            has_explanation += 1
        if markers.get("has_structure", False):
            has_structure += 1

    total = len(data)

    return {
        "has_examples_ratio": has_examples / total if total else 0,
        "has_explanation_ratio": has_explanation / total if total else 0,
        "has_structure_ratio": has_structure / total if total else 0
    }


# ==============================================================================
# HTML Dashboard Generation
# ==============================================================================

def generate_dashboard_html(khan_data: List[Dict], tiny_data: List[Dict]) -> str:
    """Generate interactive HTML dashboard."""

    # Aggregate statistics
    khan_domain_stats = aggregate_domain_stats(khan_data)
    tiny_domain_stats = aggregate_domain_stats(tiny_data)

    khan_quality_stats = aggregate_quality_stats(khan_data)
    tiny_quality_stats = aggregate_quality_stats(tiny_data)

    # Prepare data for charts
    # 1. Subject distribution
    khan_subjects = khan_domain_stats["subject_counts"]
    tiny_subjects = tiny_domain_stats["subject_counts"]

    all_subjects = sorted(set(khan_subjects.keys()) | set(tiny_subjects.keys()))

    khan_subject_values = [khan_subjects.get(s, 0) for s in all_subjects]
    tiny_subject_values = [tiny_subjects.get(s, 0) for s in all_subjects]

    # 2. Top concepts
    khan_top_concepts = khan_domain_stats["top_concepts"][:10]
    tiny_top_concepts = tiny_domain_stats["top_concepts"][:10]

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Analysis Dashboard - Phase 1</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        h1 {{
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: #718096;
            font-size: 1.1em;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .kpi-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.15);
        }}

        .kpi-label {{
            color: #718096;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}

        .kpi-value {{
            color: #2d3748;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .kpi-comparison {{
            color: #48bb78;
            font-size: 0.9em;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}

        .chart-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}

        .chart-title {{
            color: #2d3748;
            font-size: 1.3em;
            margin-bottom: 20px;
            font-weight: 600;
        }}

        canvas {{
            max-height: 400px !important;
        }}

        .comparison-table {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}

        th {{
            background: #f7fafc;
            color: #2d3748;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }}

        td {{
            color: #4a5568;
        }}

        .metric-good {{
            color: #48bb78;
            font-weight: bold;
        }}

        .metric-bad {{
            color: #f56565;
            font-weight: bold;
        }}

        footer {{
            text-align: center;
            color: white;
            margin-top: 50px;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Dataset Analysis Dashboard</h1>
            <p class="subtitle">Phase 1: Domain Coverage & Quality Metrics Comparison</p>
            <p class="subtitle" style="margin-top: 10px;">
                Khan Academy ({len(khan_data):,} chunks) vs Tiny-Textbooks ({len(tiny_data):,} chunks)
            </p>
        </header>

        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Khan Multi-Domain %</div>
                <div class="kpi-value">{khan_domain_stats['multi_domain_ratio']*100:.1f}%</div>
                <div class="kpi-comparison">Cross-cutting concepts</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-label">Tiny Multi-Domain %</div>
                <div class="kpi-value">{tiny_domain_stats['multi_domain_ratio']*100:.1f}%</div>
                <div class="kpi-comparison">Cross-cutting concepts</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-label">Khan Avg Perplexity</div>
                <div class="kpi-value">{khan_quality_stats['avg_perplexity']:.1f}</div>
                <div class="kpi-comparison">Lower = more natural</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-label">Tiny Avg Perplexity</div>
                <div class="kpi-value">{tiny_quality_stats['avg_perplexity']:.1f}</div>
                <div class="kpi-comparison">Lower = more natural</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="chart-grid">
            <div class="chart-card">
                <h3 class="chart-title">Subject Distribution Comparison</h3>
                <canvas id="subjectChart"></canvas>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">Educational Markers Comparison</h3>
                <canvas id="markersChart"></canvas>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-card">
                <h3 class="chart-title">Top 10 Concepts - Khan Academy</h3>
                <canvas id="khanConceptsChart"></canvas>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">Top 10 Concepts - Tiny-Textbooks</h3>
                <canvas id="tinyConceptsChart"></canvas>
            </div>
        </div>

        <!-- Comparison Table -->
        <div class="comparison-table">
            <h3 class="chart-title">Quality Metrics Detailed Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Khan Academy</th>
                        <th>Tiny-Textbooks</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Average Perplexity</td>
                        <td>{khan_quality_stats['avg_perplexity']:.2f}</td>
                        <td>{tiny_quality_stats['avg_perplexity']:.2f}</td>
                        <td class="{'metric-good' if khan_quality_stats['avg_perplexity'] < tiny_quality_stats['avg_perplexity'] else 'metric-bad'}">
                            {'Khan' if khan_quality_stats['avg_perplexity'] < tiny_quality_stats['avg_perplexity'] else 'Tiny'}
                        </td>
                    </tr>
                    <tr>
                        <td>Has Examples %</td>
                        <td>{khan_quality_stats['has_examples_ratio']*100:.1f}%</td>
                        <td>{tiny_quality_stats['has_examples_ratio']*100:.1f}%</td>
                        <td class="{'metric-good' if khan_quality_stats['has_examples_ratio'] > tiny_quality_stats['has_examples_ratio'] else 'metric-bad'}">
                            {'Khan' if khan_quality_stats['has_examples_ratio'] > tiny_quality_stats['has_examples_ratio'] else 'Tiny'}
                        </td>
                    </tr>
                    <tr>
                        <td>Has Explanation %</td>
                        <td>{khan_quality_stats['has_explanation_ratio']*100:.1f}%</td>
                        <td>{tiny_quality_stats['has_explanation_ratio']*100:.1f}%</td>
                        <td class="{'metric-good' if khan_quality_stats['has_explanation_ratio'] > tiny_quality_stats['has_explanation_ratio'] else 'metric-bad'}">
                            {'Khan' if khan_quality_stats['has_explanation_ratio'] > tiny_quality_stats['has_explanation_ratio'] else 'Tiny'}
                        </td>
                    </tr>
                    <tr>
                        <td>Has Structure %</td>
                        <td>{khan_quality_stats['has_structure_ratio']*100:.1f}%</td>
                        <td>{tiny_quality_stats['has_structure_ratio']*100:.1f}%</td>
                        <td class="{'metric-good' if khan_quality_stats['has_structure_ratio'] > tiny_quality_stats['has_structure_ratio'] else 'metric-bad'}">
                            {'Khan' if khan_quality_stats['has_structure_ratio'] > tiny_quality_stats['has_structure_ratio'] else 'Tiny'}
                        </td>
                    </tr>
                    <tr>
                        <td>Multi-Domain %</td>
                        <td>{khan_domain_stats['multi_domain_ratio']*100:.1f}%</td>
                        <td>{tiny_domain_stats['multi_domain_ratio']*100:.1f}%</td>
                        <td class="{'metric-good' if khan_domain_stats['multi_domain_ratio'] > tiny_domain_stats['multi_domain_ratio'] else 'metric-bad'}">
                            {'Khan' if khan_domain_stats['multi_domain_ratio'] > tiny_domain_stats['multi_domain_ratio'] else 'Tiny'}
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>

        <footer>
            <p>Generated by Phase 1 Analysis Pipeline - February 2026</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Khan Academy: {len(khan_data):,} chunks | Tiny-Textbooks: {len(tiny_data):,} chunks
            </p>
        </footer>
    </div>

    <script>
        // Chart.js configuration
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        Chart.defaults.font.size = 13;

        // Subject Distribution Chart
        new Chart(document.getElementById('subjectChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(all_subjects)},
                datasets: [
                    {{
                        label: 'Khan Academy',
                        data: {json.dumps(khan_subject_values)},
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Tiny-Textbooks',
                        data: {json.dumps(tiny_subject_values)},
                        backgroundColor: 'rgba(237, 100, 166, 0.8)',
                        borderColor: 'rgba(237, 100, 166, 1)',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Number of Chunks'
                        }}
                    }}
                }}
            }}
        }});

        // Educational Markers Chart
        new Chart(document.getElementById('markersChart'), {{
            type: 'bar',
            data: {{
                labels: ['Has Examples', 'Has Explanation', 'Has Structure'],
                datasets: [
                    {{
                        label: 'Khan Academy',
                        data: [
                            {khan_quality_stats['has_examples_ratio']*100:.1f},
                            {khan_quality_stats['has_explanation_ratio']*100:.1f},
                            {khan_quality_stats['has_structure_ratio']*100:.1f}
                        ],
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Tiny-Textbooks',
                        data: [
                            {tiny_quality_stats['has_examples_ratio']*100:.1f},
                            {tiny_quality_stats['has_explanation_ratio']*100:.1f},
                            {tiny_quality_stats['has_structure_ratio']*100:.1f}
                        ],
                        backgroundColor: 'rgba(237, 100, 166, 0.8)',
                        borderColor: 'rgba(237, 100, 166, 1)',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        title: {{
                            display: true,
                            text: 'Percentage (%)'
                        }}
                    }}
                }}
            }}
        }});

        // Khan Top Concepts Chart
        new Chart(document.getElementById('khanConceptsChart'), {{
            type: 'horizontalBar',
            data: {{
                labels: {json.dumps([c[0].split("::")[-1][:40] for c in khan_top_concepts])},
                datasets: [{{
                    label: 'Frequency',
                    data: {json.dumps([c[1] for c in khan_top_concepts])},
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        // Tiny Top Concepts Chart
        new Chart(document.getElementById('tinyConceptsChart'), {{
            type: 'horizontalBar',
            data: {{
                labels: {json.dumps([c[0].split("::")[-1][:40] for c in tiny_top_concepts])},
                datasets: [{{
                    label: 'Frequency',
                    data: {json.dumps([c[1] for c in tiny_top_concepts])},
                    backgroundColor: 'rgba(237, 100, 166, 0.8)',
                    borderColor: 'rgba(237, 100, 166, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    return html


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    print("="*60)
    print("Building Interactive Dashboard")
    print("="*60)

    # Load analysis data
    print("\nLoading analysis results...")
    khan_data = load_analysis(KHAN_ANALYSIS)
    tiny_data = load_analysis(TINY_ANALYSIS)

    print(f"✓ Loaded Khan Academy: {len(khan_data):,} chunks")
    print(f"✓ Loaded Tiny-Textbooks: {len(tiny_data):,} chunks")

    # Generate dashboard
    print("\nGenerating dashboard HTML...")
    html_content = generate_dashboard_html(khan_data, tiny_data)

    # Save to file
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Dashboard saved to {OUTPUT_HTML}")

    print("\n" + "="*60)
    print("✓ Dashboard generation complete!")
    print("="*60)
    print(f"\nOpen {OUTPUT_HTML} in your browser to view the dashboard.")


if __name__ == "__main__":
    main()
