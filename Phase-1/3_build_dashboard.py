"""
Step 3: Build Interactive Dashboard (4-Tab Rewrite)

Loads all 5 metric vectors from Step 2 output and generates a
self-contained interactive HTML dashboard with:
  Tab 1 - Dataset Overview
  Tab 2 - Domain View
  Tab 3 - Document Explorer (DataTables + modal)
  Tab 4 - Metric Deep Dive

Output: outputs/dashboard.html
"""

import json
import jsonlines
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ==============================================================================
# Config
# ==============================================================================

KHAN_ANALYSIS = "outputs/khan_analysis.jsonl"
TINY_ANALYSIS = "outputs/tiny_textbooks_analysis.jsonl"
OUTPUT_HTML   = "outputs/dashboard.html"
MAX_EXPLORER  = 2000   # documents sampled per dataset for the Explorer tab


# ==============================================================================
# Data loading — single-pass streaming
# ==============================================================================

def process_dataset(filepath: str, n_explorer: int = 2000):
    """
    Stream through a JSONL file exactly once, simultaneously computing:
      - aggregate statistics (stats dict)
      - a reservoir sample for the Explorer tab (list of compact rows)

    Memory usage: O(n_explorer) — never loads the full dataset.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"  WARNING: {filepath} not found.")
        return {"total": 0}, []

    # --- aggregate accumulators ---
    n = 0
    domain_ctr:  Counter = Counter()
    subject_ctr: Counter = Counter()
    multi_domain = 0
    quality_scores: List[float] = []
    has_ex = has_expl = has_struct = 0
    fk_grades: List[float] = []
    fk_ease_v: List[float] = []
    smog_v:    List[float] = []
    lex_v:     List[float] = []
    exact_dup = 0
    near_dup_v: List[float] = []
    sem_dup_v:  List[float] = []
    ppl_v:      List[float] = []

    # --- reservoir sampling (Algorithm R, seed=42) ---
    rng       = random.Random(42)
    reservoir: List[dict] = []

    def to_row(item):
        domains   = item.get("domain_labels") or {}
        top_dom   = max(domains.items(), key=lambda x: x[1])[0] if domains else "N/A"
        top_score = round(float(domains.get(top_dom, 0.0)), 3) if domains else 0.0
        d = item.get("difficulty")  or {}
        r = item.get("redundancy")  or {}
        p = item.get("perplexity")  or {}
        m = item.get("educational_markers") or {}

        def rv(x, dec=2):
            v = _flt(x)
            return round(v, dec) if v is not None else None

        text = item.get("text", "")
        return {
            "doc_id":           str(item.get("doc_id") or "")[:60],
            "source":           item.get("source", ""),
            "subject":          item.get("subject", ""),
            "grade":            item.get("grade", ""),
            "title":            (item.get("title") or "")[:60],
            "chunk_id":         item.get("chunk_id", 0),
            "word_count":       item.get("word_count", 0),
            "top_domain":       top_dom.split("::")[-1][:40],
            "top_domain_full":  top_dom,
            "top_domain_score": top_score,
            "quality_score":    rv(item.get("quality_score"), 2),
            "has_examples":     bool(m.get("has_examples")),
            "has_explanation":  bool(m.get("has_explanation")),
            "has_structure":    bool(m.get("has_structure")),
            "fk_grade":         rv(d.get("flesch_kincaid_grade"), 1),
            "fk_ease":          rv(d.get("flesch_reading_ease"), 1),
            "lex_div":          rv(d.get("lexical_diversity"), 3),
            "rare_words":       rv(d.get("rare_words_pct"), 3),
            "exact_dup":        bool(r.get("exact_duplicate")),
            "near_dup":         rv(r.get("near_duplicate_score"), 3),
            "semantic_dup":     rv(r.get("semantic_duplicate_score"), 3),
            "ngram3":           rv(r.get("n_gram_overlap_3"), 3),
            "perplexity":       rv(p.get("gpt2"), 1),
            "domain_labels":    {k: round(float(v), 3) for k, v in list(domains.items())[:8]},
            "text_preview":     text[:300] + ("..." if len(text) > 300 else ""),
        }

    with jsonlines.open(filepath) as reader:
        for item in reader:
            n += 1

            # --- aggregate ---
            domains = item.get("domain_labels") or {}
            for did in domains:
                domain_ctr[did] += 1
                subject_ctr[did.split("::")[0] if "::" in did else "Unknown"] += 1
            if len(domains) > 1:
                multi_domain += 1

            qs = _flt(item.get("quality_score"))
            if qs is not None:
                quality_scores.append(qs)
            mk = item.get("educational_markers") or {}
            has_ex    += int(bool(mk.get("has_examples")))
            has_expl  += int(bool(mk.get("has_explanation")))
            has_struct += int(bool(mk.get("has_structure")))

            d = item.get("difficulty") or {}
            for key, lst in [
                ("flesch_kincaid_grade", fk_grades),
                ("flesch_reading_ease",  fk_ease_v),
                ("smog_index",           smog_v),
                ("lexical_diversity",    lex_v),
            ]:
                v = _flt(d.get(key))
                if v is not None:
                    lst.append(v)

            r = item.get("redundancy") or {}
            exact_dup += int(bool(r.get("exact_duplicate")))
            for key, lst in [
                ("near_duplicate_score",     near_dup_v),
                ("semantic_duplicate_score", sem_dup_v),
            ]:
                v = _flt(r.get(key))
                if v is not None:
                    lst.append(v)

            p = item.get("perplexity") or {}
            pv = _flt(p.get("gpt2"))
            if pv is not None and pv < 2000:
                ppl_v.append(pv)

            # --- reservoir sample (Algorithm R) ---
            row = to_row(item)
            if len(reservoir) < n_explorer:
                reservoir.append(row)
            else:
                j = rng.randint(0, n - 1)
                if j < n_explorer:
                    reservoir[j] = row

    if n == 0:
        return {"total": 0}, []

    stats = {
        "total": n,
        "subject_counts":      dict(subject_ctr),
        "top_domains":         [[k, v] for k, v in domain_ctr.most_common(20)],
        "multi_domain_ratio":  multi_domain / n,
        "avg_quality":         _mean(quality_scores),
        "quality_hist":        _hist(quality_scores, 10, 0.0, 1.0),
        "has_examples_pct":    has_ex    / n * 100,
        "has_explanation_pct": has_expl  / n * 100,
        "has_structure_pct":   has_struct / n * 100,
        "avg_fk_grade":        _mean(fk_grades),
        "avg_fk_ease":         _mean(fk_ease_v),
        "fk_grade_hist":       _hist(fk_grades, 15, 0, 18),
        "fk_ease_hist":        _hist(fk_ease_v, 15, 0, 100),
        "smog_hist":           _hist(smog_v, 15, 0, 18),
        "lex_div_hist":        _hist(lex_v, 10, 0, 1),
        "exact_dup_pct":       exact_dup / n * 100,
        "avg_near_dup":        _mean(near_dup_v),
        "near_dup_hist":       _hist(near_dup_v, 20, 0, 1),
        "semantic_dup_hist":   _hist(sem_dup_v, 20, 0, 1),
        "perplexity_available": len(ppl_v) > 0,
        "avg_perplexity":      _mean(ppl_v),
        "ppl_hist":            _hist(ppl_v, 20),
    }

    rng.shuffle(reservoir)
    return stats, reservoir


# ==============================================================================
# Aggregation helpers
# ==============================================================================

def _flt(v, default=None):
    """Safe float conversion."""
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _mean(lst: list):
    return float(np.mean(lst)) if lst else None


def _hist(values: list, bins: int = 20, vmin=None, vmax=None) -> dict:
    if not values:
        return {"counts": [], "edges": []}
    arr = np.array(values, dtype=float)
    if vmin is not None and vmax is not None:
        arr = np.clip(arr, vmin, vmax)
        counts, edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
    else:
        counts, edges = np.histogram(arr, bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": [round(float(e), 3) for e in edges.tolist()],
    }




# ==============================================================================
# Dashboard HTML
# ==============================================================================

def _j(obj) -> str:
    """Serialize Python object to JS-embeddable JSON string."""
    return json.dumps(obj, ensure_ascii=False)


def _fmt(v, fmt=".1f", fallback="N/A") -> str:
    if v is None:
        return fallback
    return format(v, fmt)


def generate_dashboard_html(ks: dict, khan_exp: List[dict],
                            ts: dict, tiny_exp: List[dict]) -> str:

    # Subject chart data
    all_subj = sorted(set(ks["subject_counts"]) | set(ts["subject_counts"]))
    k_subj_vals = [ks["subject_counts"].get(s, 0) for s in all_subj]
    t_subj_vals = [ts["subject_counts"].get(s, 0) for s in all_subj]

    # Top domains for overview (short names)
    k_top = [[d[0].split("::")[-1][:35], d[1]] for d in ks["top_domains"][:15]]
    t_top = [[d[0].split("::")[-1][:35], d[1]] for d in ts["top_domains"][:15]]

    embedded = {"khan": {"stats": ks, "explorer": khan_exp},
                "tiny": {"stats": ts, "explorer": tiny_exp}}

    # -------- Comparison table rows (Python side, no JS braces needed) --------
    def winner(k_val, t_val, higher_is_better=True):
        if k_val is None or t_val is None:
            return "", ""
        if higher_is_better:
            kc = ' class="green"' if k_val >= t_val else ""
            tc = ' class="green"' if t_val > k_val else ""
        else:
            kc = ' class="green"' if k_val <= t_val else ""
            tc = ' class="green"' if t_val < k_val else ""
        return kc, tc

    kq = (ks.get("avg_quality") or 0) * 100
    tq = (ts.get("avg_quality") or 0) * 100
    kqc, tqc = winner(kq, tq)
    kec, tec = winner(ks["has_examples_pct"], ts["has_examples_pct"])
    kxc, txc = winner(ks["has_explanation_pct"], ts["has_explanation_pct"])
    ksc, tsc = winner(ks["has_structure_pct"], ts["has_structure_pct"])
    kdc, tdc = winner(ks["exact_dup_pct"], ts["exact_dup_pct"], higher_is_better=False)

    cmp_rows = f"""
<tr><td>Total Chunks</td><td>{ks["total"]:,}</td><td>{ts["total"]:,}</td><td>Text segments analyzed</td></tr>
<tr><td>Avg Quality Score</td><td{kqc}>{kq:.1f}%</td><td{tqc}>{tq:.1f}%</td><td>Avg of 3 educational markers</td></tr>
<tr><td>Has Examples</td><td{kec}>{ks['has_examples_pct']:.1f}%</td><td{tec}>{ts['has_examples_pct']:.1f}%</td><td>"for example", "such as"</td></tr>
<tr><td>Has Explanation</td><td{kxc}>{ks['has_explanation_pct']:.1f}%</td><td{txc}>{ts['has_explanation_pct']:.1f}%</td><td>"because", "therefore"</td></tr>
<tr><td>Has Structure</td><td{ksc}>{ks['has_structure_pct']:.1f}%</td><td{tsc}>{ts['has_structure_pct']:.1f}%</td><td>"first", "second", "in summary"</td></tr>
<tr><td>Avg FK Grade Level</td><td>{_fmt(ks['avg_fk_grade'])}</td><td>{_fmt(ts['avg_fk_grade'])}</td><td>Flesch-Kincaid reading grade</td></tr>
<tr><td>Avg Reading Ease</td><td>{_fmt(ks['avg_fk_ease'])}</td><td>{_fmt(ts['avg_fk_ease'])}</td><td>Flesch Reading Ease (0-100, higher=easier)</td></tr>
<tr><td>Exact Duplicates</td><td{kdc}>{ks['exact_dup_pct']:.2f}%</td><td{tdc}>{ts['exact_dup_pct']:.2f}%</td><td>MD5 exact match rate</td></tr>
<tr><td>Multi-Domain %</td><td>{ks['multi_domain_ratio']*100:.1f}%</td><td>{ts['multi_domain_ratio']*100:.1f}%</td><td>Documents spanning 2+ domains</td></tr>
<tr><td>Avg Perplexity (GPT-2)</td><td>{_fmt(ks.get('avg_perplexity'), '.1f')}</td><td>{_fmt(ts.get('avg_perplexity'), '.1f')}</td><td>Lower = more predictable text</td></tr>
"""

    diff_cmp_rows = f"""
<tr><td>Avg FK Grade Level</td><td>{_fmt(ks['avg_fk_grade'])}</td><td>{_fmt(ts['avg_fk_grade'])}</td></tr>
<tr><td>Avg Reading Ease</td><td>{_fmt(ks['avg_fk_ease'])}</td><td>{_fmt(ts['avg_fk_ease'])}</td></tr>
"""

    red_cmp_rows = f"""
<tr><td>Exact Duplicates</td><td>{ks['exact_dup_pct']:.2f}%</td><td>{ts['exact_dup_pct']:.2f}%</td></tr>
<tr><td>Avg Near-Dup Score</td><td>{_fmt(ks['avg_near_dup'], '.3f')}</td><td>{_fmt(ts['avg_near_dup'], '.3f')}</td></tr>
"""

    ppl_note = ""
    if not ks.get("perplexity_available") and not ts.get("perplexity_available"):
        ppl_note = "Perplexity data not available (GPT-2 model was not loaded during Step 2)."

    # KPI values
    kpi_html = f"""
<div class="kpi"><div class="kpi-label">Khan Chunks</div><div class="kpi-value">{ks["total"]:,}</div><div class="kpi-sub">text segments</div></div>
<div class="kpi"><div class="kpi-label">Tiny Chunks</div><div class="kpi-value">{ts["total"]:,}</div><div class="kpi-sub">text segments</div></div>
<div class="kpi"><div class="kpi-label">Khan Avg Quality</div><div class="kpi-value">{kq:.0f}%</div><div class="kpi-sub">educational markers</div></div>
<div class="kpi"><div class="kpi-label">Tiny Avg Quality</div><div class="kpi-value">{tq:.0f}%</div><div class="kpi-sub">educational markers</div></div>
<div class="kpi"><div class="kpi-label">Khan Avg FK Grade</div><div class="kpi-value">{_fmt(ks['avg_fk_grade'])}</div><div class="kpi-sub">Flesch-Kincaid</div></div>
<div class="kpi"><div class="kpi-label">Tiny Avg FK Grade</div><div class="kpi-value">{_fmt(ts['avg_fk_grade'])}</div><div class="kpi-sub">Flesch-Kincaid</div></div>
<div class="kpi"><div class="kpi-label">Khan Exact Dups</div><div class="kpi-value">{ks['exact_dup_pct']:.1f}%</div><div class="kpi-sub">duplicate rate</div></div>
<div class="kpi"><div class="kpi-label">Tiny Exact Dups</div><div class="kpi-value">{ts['exact_dup_pct']:.1f}%</div><div class="kpi-sub">duplicate rate</div></div>
"""

    # ---- Assemble the HTML ----
    # All JavaScript code uses {{ }} for literal braces
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Analysis Dashboard – Phase 1</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#2d3748}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:24px 32px}}
.header h1{{font-size:1.8em;margin-bottom:4px}}
.header p{{opacity:.85;font-size:.95em}}
.tab-nav{{background:white;border-bottom:2px solid #e2e8f0;padding:0 32px;display:flex}}
.tab-btn{{padding:16px 24px;cursor:pointer;border:none;background:none;font-size:.95em;font-weight:500;color:#718096;border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .2s}}
.tab-btn:hover{{color:#667eea}}
.tab-btn.active{{color:#667eea;border-bottom-color:#667eea}}
.tab-panel{{display:none;padding:24px 32px;max-width:1400px;margin:0 auto}}
.tab-panel.active{{display:block}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}}
.kpi{{background:white;padding:20px 24px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
.kpi-label{{font-size:.78em;text-transform:uppercase;letter-spacing:1px;color:#718096;margin-bottom:8px}}
.kpi-value{{font-size:2em;font-weight:700;color:#2d3748}}
.kpi-sub{{font-size:.82em;color:#a0aec0;margin-top:4px}}
.chart-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px;margin-bottom:20px}}
.card{{background:white;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.07);padding:24px}}
.card h3{{font-size:1em;font-weight:600;color:#4a5568;margin-bottom:16px}}
canvas{{max-height:340px!important}}
.cmp-table{{width:100%;border-collapse:collapse;font-size:.9em}}
.cmp-table th{{background:#f7fafc;padding:11px 14px;text-align:left;color:#4a5568;font-size:.78em;text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid #e2e8f0}}
.cmp-table td{{padding:11px 14px;border-bottom:1px solid #f0f2f5;color:#4a5568}}
.cmp-table tr:last-child td{{border-bottom:none}}
.green{{color:#38a169;font-weight:600}}
.red{{color:#e53e3e;font-weight:600}}
.filters{{background:white;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.07);padding:18px 24px;margin-bottom:20px;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end}}
.fg{{display:flex;flex-direction:column;gap:4px}}
.fg label{{font-size:.78em;font-weight:600;color:#718096;text-transform:uppercase;letter-spacing:.5px}}
.fg select,.fg input[type=range]{{padding:6px 10px;border:1px solid #e2e8f0;border-radius:8px;font-size:.88em;color:#2d3748;background:white;min-width:150px}}
.fv{{font-size:.88em;color:#667eea;font-weight:600}}
.btn{{padding:7px 16px;background:#667eea;color:white;border:none;border-radius:8px;cursor:pointer;font-size:.88em;font-weight:500}}
.btn:hover{{background:#5a67d8}}
.btn-sm{{padding:3px 10px;font-size:.78em}}
.btn-outline{{background:white;color:#667eea;border:1px solid #667eea}}
.btn-outline:hover{{background:#f0f2ff}}
.table-wrap{{background:white;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.07);padding:24px}}
#docTable{{width:100%;font-size:.85em}}
#docTable thead th{{background:#f7fafc;color:#4a5568;font-weight:600;font-size:.77em;text-transform:uppercase;letter-spacing:.5px}}
.badge{{display:inline-block;padding:2px 7px;border-radius:20px;font-size:.76em;font-weight:600}}
.bk{{background:#ebf4ff;color:#3182ce}}
.bt{{background:#faf0ff;color:#805ad5}}
.bd{{background:#fff5f5;color:#e53e3e}}
.modal-overlay{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.5);z-index:1000;overflow:auto}}
.modal-overlay.open{{display:flex;align-items:flex-start;justify-content:center;padding:40px 16px}}
.modal{{background:white;border-radius:16px;max-width:860px;width:100%;padding:32px;position:relative}}
.modal h2{{font-size:1.2em;margin-bottom:16px;color:#2d3748;padding-right:40px}}
.modal h4{{font-size:.85em;font-weight:600;color:#4a5568;margin:16px 0 8px;text-transform:uppercase;letter-spacing:.5px}}
.modal-text{{background:#f7fafc;border-radius:8px;padding:14px;font-size:.86em;line-height:1.6;max-height:180px;overflow-y:auto;color:#4a5568;white-space:pre-wrap}}
.m-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px}}
.mi{{background:#f7fafc;border-radius:8px;padding:9px 12px}}
.mi-lbl{{font-size:.72em;color:#718096;text-transform:uppercase}}
.mi-val{{font-size:1em;font-weight:600;color:#2d3748;margin-top:2px}}
.close-btn{{position:absolute;top:20px;right:24px;font-size:1.6em;cursor:pointer;color:#a0aec0;background:none;border:none;line-height:1}}
.close-btn:hover{{color:#2d3748}}
.metric-tabs{{display:flex;gap:6px;margin-bottom:20px;flex-wrap:wrap}}
.mt-btn{{padding:8px 16px;cursor:pointer;border:1px solid #e2e8f0;background:white;border-radius:8px;font-size:.88em;font-weight:500;color:#718096;transition:all .2s}}
.mt-btn:hover{{border-color:#667eea;color:#667eea}}
.mt-btn.active{{background:#667eea;color:white;border-color:#667eea}}
.mp{{display:none}}
.mp.active{{display:block}}
.dom-ctrl{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
.dom-ctrl select{{padding:7px 12px;border:1px solid #e2e8f0;border-radius:8px;font-size:.9em;min-width:180px}}
.footer{{text-align:center;padding:28px;color:#a0aec0;font-size:.87em}}
</style>
</head>
<body>

<div class="header">
  <h1>📊 Dataset Analysis Dashboard</h1>
  <p>Phase 1 · SLM Pretraining Dataset Characterization · Khan Academy ({ks["total"]:,} chunks) vs Tiny-Textbooks ({ts["total"]:,} chunks)</p>
</div>

<div class="tab-nav">
  <button class="tab-btn active"  onclick="switchTab('overview',this)">📈 Dataset Overview</button>
  <button class="tab-btn"         onclick="switchTab('domain',this)">🗂 Domain View</button>
  <button class="tab-btn"         onclick="switchTab('explorer',this)">🔍 Document Explorer</button>
  <button class="tab-btn"         onclick="switchTab('deepdive',this)">🔬 Metric Deep Dive</button>
</div>

<!-- ======================================================================
     TAB 1: OVERVIEW
     ====================================================================== -->
<div id="tab-overview" class="tab-panel active">
  <div class="kpi-grid">{kpi_html}</div>

  <div class="chart-row">
    <div class="card"><h3>Subject Distribution</h3><canvas id="subjectChart"></canvas></div>
    <div class="card"><h3>Educational Markers (%)</h3><canvas id="markersChart"></canvas></div>
  </div>
  <div class="chart-row">
    <div class="card"><h3>Top 15 Domains — Khan Academy</h3><canvas id="kDomChart"></canvas></div>
    <div class="card"><h3>Top 15 Domains — Tiny-Textbooks</h3><canvas id="tDomChart"></canvas></div>
  </div>
  <div class="card" style="margin-bottom:20px">
    <h3>Metrics Comparison</h3>
    <table class="cmp-table">
      <thead><tr><th>Metric</th><th>Khan Academy</th><th>Tiny-Textbooks</th><th>Notes</th></tr></thead>
      <tbody>{cmp_rows}</tbody>
    </table>
  </div>
</div>

<!-- ======================================================================
     TAB 2: DOMAIN VIEW
     ====================================================================== -->
<div id="tab-domain" class="tab-panel">
  <div class="card" style="margin-bottom:20px">
    <h3>Domain Coverage Comparison</h3>
    <div class="dom-ctrl" style="margin-top:12px">
      <label style="font-size:.88em;font-weight:600;color:#718096">Dataset:</label>
      <select id="domDS" onchange="updateDomainView()">
        <option value="both">Both</option>
        <option value="khan">Khan Academy</option>
        <option value="tiny">Tiny-Textbooks</option>
      </select>
      <label style="font-size:.88em;font-weight:600;color:#718096">Show top:</label>
      <select id="domN" onchange="updateDomainView()">
        <option value="10">10</option>
        <option value="15" selected>15</option>
        <option value="20">20</option>
      </select>
    </div>
    <canvas id="domCompareChart" style="max-height:520px!important"></canvas>
  </div>
  <div class="chart-row">
    <div class="card"><h3>Subject Coverage — Khan Academy</h3><canvas id="kSubjPie"></canvas></div>
    <div class="card"><h3>Subject Coverage — Tiny-Textbooks</h3><canvas id="tSubjPie"></canvas></div>
  </div>
</div>

<!-- ======================================================================
     TAB 3: DOCUMENT EXPLORER
     ====================================================================== -->
<div id="tab-explorer" class="tab-panel">
  <div class="filters">
    <div class="fg">
      <label>Dataset</label>
      <select id="fDS" onchange="applyFilters()">
        <option value="all">All</option>
        <option value="khan_academy">Khan Academy</option>
        <option value="tiny_textbooks">Tiny-Textbooks</option>
      </select>
    </div>
    <div class="fg">
      <label>Subject</label>
      <select id="fSubj" onchange="applyFilters()"><option value="all">All Subjects</option></select>
    </div>
    <div class="fg">
      <label>Min Quality: <span class="fv" id="fQv">0.0</span></label>
      <input type="range" id="fQ" min="0" max="1" step="0.1" value="0"
             oninput="document.getElementById('fQv').textContent=this.value;applyFilters()">
    </div>
    <div class="fg">
      <label>Max FK Grade: <span class="fv" id="fGv">18</span></label>
      <input type="range" id="fG" min="0" max="18" step="1" value="18"
             oninput="document.getElementById('fGv').textContent=this.value;applyFilters()">
    </div>
    <div class="fg">
      <label>Duplicates</label>
      <select id="fDup" onchange="applyFilters()">
        <option value="all">Show All</option>
        <option value="no_dup">Exclude Exact Dups</option>
      </select>
    </div>
    <button class="btn btn-outline" onclick="exportCSV()">⬇ Export CSV</button>
    <button class="btn btn-outline" onclick="resetFilters()">Reset</button>
    <span id="fCount" style="font-size:.88em;color:#718096;align-self:center"></span>
  </div>
  <div class="table-wrap">
    <table id="docTable" class="display" style="width:100%">
      <thead>
        <tr>
          <th>Source</th><th>Subject</th><th>Grade</th>
          <th>Top Domain</th><th>Dom Score</th>
          <th>Quality</th><th>FK Grade</th>
          <th>Near Dup</th><th>Perplexity</th><th>Words</th>
          <th>View</th>
        </tr>
      </thead>
      <tbody id="explorerBody"></tbody>
    </table>
  </div>
</div>

<!-- ======================================================================
     TAB 4: METRIC DEEP DIVE
     ====================================================================== -->
<div id="tab-deepdive" class="tab-panel">
  <div class="metric-tabs">
    <button class="mt-btn active" onclick="switchMetricTab('difficulty',this)">📚 Difficulty</button>
    <button class="mt-btn"        onclick="switchMetricTab('quality',this)">⭐ Quality</button>
    <button class="mt-btn"        onclick="switchMetricTab('redundancy',this)">🔁 Redundancy</button>
    <button class="mt-btn"        onclick="switchMetricTab('perplexity',this)">🤔 Perplexity</button>
  </div>

  <div id="mp-difficulty" class="mp active">
    <div class="chart-row">
      <div class="card"><h3>FK Grade Level Distribution</h3><canvas id="fkGradeHist"></canvas></div>
      <div class="card"><h3>Reading Ease Distribution (higher = easier)</h3><canvas id="fkEaseHist"></canvas></div>
    </div>
    <div class="card" style="margin-bottom:20px">
      <h3>Difficulty Comparison</h3>
      <table class="cmp-table">
        <thead><tr><th>Metric</th><th>Khan Academy</th><th>Tiny-Textbooks</th></tr></thead>
        <tbody>{diff_cmp_rows}</tbody>
      </table>
    </div>
  </div>

  <div id="mp-quality" class="mp">
    <div class="chart-row">
      <div class="card"><h3>Quality Score Distribution</h3><canvas id="qualHist"></canvas></div>
      <div class="card"><h3>Educational Markers Breakdown (%)</h3><canvas id="markersDeep"></canvas></div>
    </div>
  </div>

  <div id="mp-redundancy" class="mp">
    <div class="chart-row">
      <div class="card"><h3>Near-Duplicate Score Distribution</h3><canvas id="nearDupHist"></canvas></div>
      <div class="card"><h3>Semantic Duplicate Score Distribution</h3><canvas id="semDupHist"></canvas></div>
    </div>
    <div class="card" style="margin-bottom:20px">
      <h3>Redundancy Summary</h3>
      <table class="cmp-table">
        <thead><tr><th>Metric</th><th>Khan Academy</th><th>Tiny-Textbooks</th></tr></thead>
        <tbody>{red_cmp_rows}</tbody>
      </table>
    </div>
  </div>

  <div id="mp-perplexity" class="mp">
    <div class="chart-row">
      <div class="card">
        <h3>GPT-2 Perplexity Distribution</h3>
        <canvas id="pplHist"></canvas>
        <p id="pplNote" style="margin-top:12px;font-size:.88em;color:#718096">{ppl_note}</p>
      </div>
      <div class="card">
        <h3>Perplexity Summary</h3>
        <table class="cmp-table" style="margin-top:8px">
          <thead><tr><th>Metric</th><th>Khan Academy</th><th>Tiny-Textbooks</th></tr></thead>
          <tbody>
            <tr><td>Avg Perplexity</td>
                <td>{_fmt(ks.get('avg_perplexity'), '.1f')}</td>
                <td>{_fmt(ts.get('avg_perplexity'), '.1f')}</td></tr>
            <tr><td>Data Available</td>
                <td>{'Yes' if ks.get('perplexity_available') else 'No'}</td>
                <td>{'Yes' if ts.get('perplexity_available') else 'No'}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- ======================================================================
     DOCUMENT DETAIL MODAL
     ====================================================================== -->
<div id="docModal" class="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <button class="close-btn" onclick="closeModal()">×</button>
    <h2 id="mTitle">Document Details</h2>
    <h4>Text Preview</h4>
    <div class="modal-text" id="mText"></div>
    <h4>Domain Labels</h4>
    <canvas id="mDomChart" style="max-height:180px!important"></canvas>
    <h4>All Metrics</h4>
    <div class="m-grid" id="mMetrics"></div>
  </div>
</div>

<div class="footer">Phase 1 Dataset Analysis Pipeline · Generated February 2026 · UNLV Lab Research</div>

<script>
// ============================================================
// EMBEDDED DATA
// ============================================================
const EMB = {_j(embedded)};
const KS = EMB.khan.stats;
const TS = EMB.tiny.stats;
const ALL_DOCS = [...EMB.khan.explorer, ...EMB.tiny.explorer];

// ============================================================
// TAB MANAGEMENT
// ============================================================
function switchTab(name, btn) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'explorer' && !window._expInit) initExplorer();
  if (name === 'deepdive' && !window._ddInit) initDeepDive();
  if (name === 'domain'   && !window._domInit) initDomainView();
}}

function switchMetricTab(name, btn) {{
  document.querySelectorAll('.mt-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.mp').forEach(p => p.classList.remove('active'));
  document.getElementById('mp-' + name).classList.add('active');
  btn.classList.add('active');
}}

// ============================================================
// CHART HELPERS
// ============================================================
const CK = 'rgba(102,126,234,0.8)', CKB = 'rgba(102,126,234,1)';
const CT = 'rgba(237,100,166,0.8)', CTB = 'rgba(237,100,166,1)';
const PIE_COLORS = ['#667eea','#ed64a6','#48bb78','#ed8936','#9f7aea','#38b2ac','#f56565','#ecc94b','#fc8181','#68d391'];

function barChart(id, labels, kVals, tVals, opts) {{
  const xLabel = opts && opts.xLabel || 'Count';
  const indexAxis = opts && opts.horiz ? 'y' : 'x';
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label:'Khan Academy', data:kVals, backgroundColor:CK, borderColor:CKB, borderWidth:1 }},
        {{ label:'Tiny-Textbooks', data:tVals, backgroundColor:CT, borderColor:CTB, borderWidth:1 }}
      ]
    }},
    options: {{
      indexAxis, responsive:true, maintainAspectRatio:true,
      plugins: {{ legend: {{ position:'top' }} }},
      scales: {{
        x: {{ beginAtZero:true, title: {{ display:!!xLabel, text:xLabel }} }},
        y: {{ beginAtZero:true }}
      }}
    }}
  }});
}}

function histChart(id, hK, hT, xLabel) {{
  if (!hK.edges.length && !hT.edges.length) return;
  const edges = (hK.edges.length ? hK : hT).edges;
  const labels = edges.slice(0,-1).map((e,i) => e.toFixed(1)+'-'+edges[i+1].toFixed(1));
  const kCounts = hK.counts.length ? hK.counts : new Array(labels.length).fill(0);
  const tCounts = hT.counts.length ? hT.counts : new Array(labels.length).fill(0);
  new Chart(document.getElementById(id), {{
    type:'bar',
    data: {{
      labels,
      datasets: [
        {{ label:'Khan Academy',   data:kCounts, backgroundColor:CK, borderColor:CKB, borderWidth:1 }},
        {{ label:'Tiny-Textbooks', data:tCounts, backgroundColor:CT, borderColor:CTB, borderWidth:1 }}
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:true,
      plugins: {{ legend: {{ position:'top' }} }},
      scales: {{
        x: {{ title: {{ display:!!xLabel, text:xLabel }} }},
        y: {{ beginAtZero:true, title: {{ display:true, text:'Count' }} }}
      }}
    }}
  }});
}}

function pieChart(id, subjectCounts) {{
  const labels = Object.keys(subjectCounts);
  const values = Object.values(subjectCounts);
  new Chart(document.getElementById(id), {{
    type:'doughnut',
    data: {{ labels, datasets:[{{ data:values, backgroundColor:PIE_COLORS }}] }},
    options: {{ responsive:true, plugins: {{ legend: {{ position:'right' }} }} }}
  }});
}}

// ============================================================
// OVERVIEW CHARTS (init on page load)
// ============================================================
const ALL_SUBJ = {_j(all_subj)};
const K_SUBJ  = {_j(k_subj_vals)};
const T_SUBJ  = {_j(t_subj_vals)};
barChart('subjectChart', ALL_SUBJ, K_SUBJ, T_SUBJ, {{horiz:true, xLabel:'Chunk Count'}});

barChart('markersChart',
  ['Has Examples','Has Explanation','Has Structure'],
  [KS.has_examples_pct, KS.has_explanation_pct, KS.has_structure_pct],
  [TS.has_examples_pct, TS.has_explanation_pct, TS.has_structure_pct],
  {{xLabel:'%'}}
);

const K_TOP = {_j(k_top)};
const T_TOP = {_j(t_top)};
new Chart(document.getElementById('kDomChart'), {{
  type:'bar',
  data: {{ labels:K_TOP.map(x=>x[0]), datasets:[{{ label:'Chunk Count', data:K_TOP.map(x=>x[1]), backgroundColor:CK, borderColor:CKB, borderWidth:1 }}] }},
  options: {{ indexAxis:'y', responsive:true, plugins:{{legend:{{display:false}}}}, scales:{{x:{{beginAtZero:true}}}} }}
}});
new Chart(document.getElementById('tDomChart'), {{
  type:'bar',
  data: {{ labels:T_TOP.map(x=>x[0]), datasets:[{{ label:'Chunk Count', data:T_TOP.map(x=>x[1]), backgroundColor:CT, borderColor:CTB, borderWidth:1 }}] }},
  options: {{ indexAxis:'y', responsive:true, plugins:{{legend:{{display:false}}}}, scales:{{x:{{beginAtZero:true}}}} }}
}});

// ============================================================
// DOMAIN VIEW
// ============================================================
let domChart = null;
function initDomainView() {{
  window._domInit = true;
  updateDomainView();
  pieChart('kSubjPie', KS.subject_counts);
  pieChart('tSubjPie', TS.subject_counts);
}}

function updateDomainView() {{
  const ds = document.getElementById('domDS').value;
  const N  = parseInt(document.getElementById('domN').value);
  const kMap = Object.fromEntries(KS.top_domains.map(d => [d[0].split('::').pop(), d[1]]));
  const tMap = Object.fromEntries(TS.top_domains.map(d => [d[0].split('::').pop(), d[1]]));
  let items;
  if (ds === 'khan') {{
    items = KS.top_domains.slice(0,N).map(d => [d[0].split('::').pop().substring(0,35), d[1], tMap[d[0].split('::').pop()]||0]);
  }} else if (ds === 'tiny') {{
    items = TS.top_domains.slice(0,N).map(d => [d[0].split('::').pop().substring(0,35), kMap[d[0].split('::').pop()]||0, d[1]]);
  }} else {{
    const all = {{}};
    KS.top_domains.forEach(d => {{ const k=d[0].split('::').pop(); all[k]=(all[k]||[0,0]); all[k][0]=d[1]; }});
    TS.top_domains.forEach(d => {{ const k=d[0].split('::').pop(); all[k]=(all[k]||[0,0]); all[k][1]=d[1]; }});
    items = Object.entries(all).map(([k,v]) => [k.substring(0,35), v[0], v[1]]).sort((a,b)=>(b[1]+b[2])-(a[1]+a[2])).slice(0,N);
  }}
  if (domChart) domChart.destroy();
  domChart = new Chart(document.getElementById('domCompareChart'), {{
    type:'bar',
    data: {{
      labels: items.map(x=>x[0]),
      datasets: [
        {{ label:'Khan Academy',   data:items.map(x=>x[1]), backgroundColor:CK, borderColor:CKB, borderWidth:1 }},
        {{ label:'Tiny-Textbooks', data:items.map(x=>x[2]), backgroundColor:CT, borderColor:CTB, borderWidth:1 }}
      ]
    }},
    options: {{
      indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins: {{ legend: {{ position:'top' }} }},
      scales: {{ x: {{ beginAtZero:true, title: {{ display:true, text:'Chunk Count' }} }} }}
    }}
  }});
}}

// ============================================================
// DOCUMENT EXPLORER
// ============================================================
let currDocs = [];
let dt = null;

function initExplorer() {{
  window._expInit = true;
  const subjs = [...new Set(ALL_DOCS.map(d=>d.subject).filter(Boolean))].sort();
  const sel = document.getElementById('fSubj');
  subjs.forEach(s => {{ const o=document.createElement('option'); o.value=s; o.textContent=s; sel.appendChild(o); }});
  applyFilters();
}}

function applyFilters() {{
  const ds   = document.getElementById('fDS').value;
  const subj = document.getElementById('fSubj').value;
  const minQ = parseFloat(document.getElementById('fQ').value);
  const maxG = parseFloat(document.getElementById('fG').value);
  const dup  = document.getElementById('fDup').value;
  currDocs = ALL_DOCS.filter(d => {{
    if (ds !== 'all' && d.source !== ds) return false;
    if (subj !== 'all' && d.subject !== subj) return false;
    if (d.quality_score !== null && d.quality_score < minQ) return false;
    if (d.fk_grade !== null && d.fk_grade > maxG) return false;
    if (dup === 'no_dup' && d.exact_dup) return false;
    return true;
  }});
  document.getElementById('fCount').textContent = currDocs.length.toLocaleString() + ' documents';
  renderTable();
}}

function renderTable() {{
  const tbody = document.getElementById('explorerBody');
  tbody.innerHTML = '';
  currDocs.forEach((doc, i) => {{
    const srcBadge = doc.source === 'khan_academy'
      ? '<span class="badge bk">Khan</span>'
      : '<span class="badge bt">Tiny</span>';
    const dupBadge = doc.exact_dup ? ' <span class="badge bd">DUP</span>' : '';
    const ppl = doc.perplexity !== null && doc.perplexity !== undefined ? doc.perplexity.toFixed(1) : '—';
    const nd  = doc.near_dup  !== null && doc.near_dup  !== undefined ? doc.near_dup.toFixed(3)  : '—';
    const fk  = doc.fk_grade  !== null && doc.fk_grade  !== undefined ? doc.fk_grade.toFixed(1)  : '—';
    const qpct = doc.quality_score !== null ? (doc.quality_score*100).toFixed(0)+'%' : '—';
    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    tr.innerHTML = `
      <td>${{srcBadge}}${{dupBadge}}</td>
      <td>${{doc.subject}}</td>
      <td>${{doc.grade||'—'}}</td>
      <td title="${{doc.top_domain_full}}">${{doc.top_domain}}</td>
      <td>${{doc.top_domain_score.toFixed(2)}}</td>
      <td>${{qpct}}</td>
      <td>${{fk}}</td>
      <td>${{nd}}</td>
      <td>${{ppl}}</td>
      <td>${{doc.word_count}}</td>
      <td><button class="btn btn-sm" onclick="openModal(parseInt(this.closest('tr').dataset.idx))">View</button></td>
    `;
    tbody.appendChild(tr);
  }});
  if (dt) {{ dt.destroy(); dt = null; }}
  dt = $('#docTable').DataTable({{
    pageLength:50,
    lengthMenu:[25,50,100],
    order:[],
    columnDefs:[{{orderable:false,targets:-1}}],
    dom:'lrtip'
  }});
}}

function resetFilters() {{
  document.getElementById('fDS').value   = 'all';
  document.getElementById('fSubj').value = 'all';
  document.getElementById('fQ').value    = 0;
  document.getElementById('fG').value    = 18;
  document.getElementById('fDup').value  = 'all';
  document.getElementById('fQv').textContent = '0.0';
  document.getElementById('fGv').textContent = '18';
  applyFilters();
}}

function exportCSV() {{
  const cols = ['source','subject','grade','top_domain','top_domain_score','quality_score','fk_grade','fk_ease','near_dup','semantic_dup','ngram3','perplexity','word_count','exact_dup'];
  const hdr  = cols.join(',');
  const rows = currDocs.map(d => cols.map(c => {{
    const v = d[c];
    if (v === null || v === undefined) return '';
    if (typeof v === 'string' && v.includes(',')) return '"'+v+'"';
    return v;
  }}).join(','));
  const blob = new Blob([[hdr,...rows].join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'dataset_analysis.csv';
  a.click();
}}

// ============================================================
// DOCUMENT MODAL
// ============================================================
let mDomChart = null;
function openModal(idx) {{
  const doc = currDocs[idx];
  if (!doc) return;
  document.getElementById('mTitle').textContent = doc.title || doc.doc_id || 'Document';
  document.getElementById('mText').textContent  = doc.text_preview;
  const labels = Object.keys(doc.domain_labels).map(l=>l.split('::').pop());
  const values = Object.values(doc.domain_labels);
  if (mDomChart) {{ mDomChart.destroy(); mDomChart = null; }}
  mDomChart = new Chart(document.getElementById('mDomChart'), {{
    type:'bar',
    data: {{ labels, datasets:[{{ label:'Score', data:values, backgroundColor:CK }}] }},
    options: {{ indexAxis:'y', responsive:true, plugins:{{legend:{{display:false}}}}, scales:{{x:{{beginAtZero:true,max:1}}}} }}
  }});
  const mets = [
    ['Quality Score', doc.quality_score!==null ? (doc.quality_score*100).toFixed(0)+'%' : '—'],
    ['Has Examples',   doc.has_examples   ? '✓ Yes' : '✗ No'],
    ['Has Explanation',doc.has_explanation ? '✓ Yes' : '✗ No'],
    ['Has Structure',  doc.has_structure   ? '✓ Yes' : '✗ No'],
    ['FK Grade',       doc.fk_grade  !== null ? doc.fk_grade  : '—'],
    ['Reading Ease',   doc.fk_ease   !== null ? doc.fk_ease   : '—'],
    ['Lex Diversity',  doc.lex_div   !== null ? doc.lex_div   : '—'],
    ['Rare Words',     doc.rare_words!== null ? (doc.rare_words*100).toFixed(1)+'%' : '—'],
    ['Exact Dup',      doc.exact_dup ? '⚠ Yes' : '✓ No'],
    ['Near-Dup',       doc.near_dup  !== null ? doc.near_dup  : '—'],
    ['Semantic Dup',   doc.semantic_dup !== null ? doc.semantic_dup : '—'],
    ['3-gram Overlap', doc.ngram3    !== null ? doc.ngram3    : '—'],
    ['Perplexity',     doc.perplexity!== null ? doc.perplexity : '—'],
    ['Word Count',     doc.word_count],
  ];
  document.getElementById('mMetrics').innerHTML = mets.map(([l,v]) =>
    `<div class="mi"><div class="mi-lbl">${{l}}</div><div class="mi-val">${{v}}</div></div>`
  ).join('');
  document.getElementById('docModal').classList.add('open');
}}
function closeModal() {{
  document.getElementById('docModal').classList.remove('open');
}}

// ============================================================
// METRIC DEEP DIVE (lazy init)
// ============================================================
function initDeepDive() {{
  window._ddInit = true;
  histChart('fkGradeHist', KS.fk_grade_hist, TS.fk_grade_hist, 'Grade Level');
  histChart('fkEaseHist',  KS.fk_ease_hist,  TS.fk_ease_hist,  'Reading Ease');
  histChart('qualHist',    KS.quality_hist,   TS.quality_hist,   'Quality Score');
  barChart('markersDeep',
    ['Has Examples','Has Explanation','Has Structure'],
    [KS.has_examples_pct, KS.has_explanation_pct, KS.has_structure_pct],
    [TS.has_examples_pct, TS.has_explanation_pct, TS.has_structure_pct],
    {{xLabel:'%'}}
  );
  histChart('nearDupHist', KS.near_dup_hist,     TS.near_dup_hist,     'Near-Dup Score');
  histChart('semDupHist',  KS.semantic_dup_hist,  TS.semantic_dup_hist,  'Semantic Score');
  if (KS.perplexity_available || TS.perplexity_available) {{
    histChart('pplHist', KS.ppl_hist, TS.ppl_hist, 'Perplexity');
  }}
}}
</script>
</body>
</html>"""


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("BUILDING INTERACTIVE DASHBOARD")
    print("=" * 60)

    print("\nStreaming analysis results...")
    print("  Processing Khan Academy (single pass)...")
    ks, khan_exp = process_dataset(KHAN_ANALYSIS, MAX_EXPLORER)
    print(f"  Khan Academy:  {ks.get('total', 0):,} chunks → {len(khan_exp)} explorer rows")

    print("  Processing Tiny-Textbooks (single pass)...")
    ts, tiny_exp = process_dataset(TINY_ANALYSIS, MAX_EXPLORER)
    print(f"  Tiny-Textbooks: {ts.get('total', 0):,} chunks → {len(tiny_exp)} explorer rows")

    if ks.get("total", 0) == 0 and ts.get("total", 0) == 0:
        print("\nERROR: No analysis files found. Run 2_compute_metrics.py first.")
        return

    print("\nGenerating dashboard HTML...")
    html = generate_dashboard_html(ks, khan_exp, ts, tiny_exp)

    Path("outputs").mkdir(exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = Path(OUTPUT_HTML).stat().st_size / 1024
    print(f"\n✓ Dashboard saved to {OUTPUT_HTML}")
    print(f"  File size: {size_kb:.0f} KB")
    print("\n" + "=" * 60)
    print("✓ Dashboard generation complete!")
    print("=" * 60)
    print(f"\nOpen {OUTPUT_HTML} in your browser to view.")


if __name__ == "__main__":
    main()
