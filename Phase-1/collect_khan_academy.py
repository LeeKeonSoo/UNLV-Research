"""
Khan Academy K-12 Content Collector (Cosmopedia Version)

Uses HuggingFaceTB/cosmopedia 'khanacademy' subset as data source.
24,123 textbook-style educational texts generated from Khan Academy topics.

Coverage: Grades 3-12, all core subjects
  - Math: Arithmetic, Pre-algebra, Algebra 1/2, Geometry, Precalculus, Statistics
  - Science: Biology, Chemistry, Physics, Earth Science
  - Social Studies: US History, World History, Economics, Government
  - Language Arts: Grammar, Reading, Vocabulary
  - Computer Science

Output:
  khan_k12_concepts/all_k12_concepts.json
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm


# ==============================================================================
# Grade and Subject Mapping
# ==============================================================================

# Maps course-name keywords to (subject_category, grade_range)
# Order matters: more specific patterns first
COURSE_GRADE_MAP = [
    # ── Explicit grade-labeled courses ─────────────────────────────────────────
    (r'^3rd grade',                         ('Math',           '3')),
    (r'^Grade 3\b',                         ('Math',           '3')),
    (r'^4th grade',                         ('Math',           '4')),
    (r'^Grade 4\b',                         ('Math',           '4')),
    (r'^5th grade',                         ('Math',           '5')),
    (r'^Grade 5\b',                         ('Math',           '5')),
    (r'^6th grade',                         ('Math',           '6')),
    (r'^Grade 6\b',                         ('Math',           '6')),
    (r'^7th grade',                         ('Math',           '7')),
    (r'^Grade 7\b',                         ('Math',           '7')),
    (r'^8th grade',                         ('Math',           '8')),
    (r'^Grade 8\b',                         ('Math',           '8')),

    # ── "Get ready for" → one grade up ─────────────────────────────────────────
    (r'Get ready for 3rd grade',            ('Math',           '3')),
    (r'Get ready for 4th grade',            ('Math',           '4')),
    (r'Get ready for 5th grade',            ('Math',           '5')),
    (r'Get ready for 6th grade',            ('Math',           '6')),
    (r'Get ready for 7th grade',            ('Math',           '7')),
    (r'Get ready for 8th grade',            ('Math',           '8')),

    # ── High-school math ────────────────────────────────────────────────────────
    (r'Pre-algebra|Prealgebra',             ('Math',           '7-8')),
    (r'Algebra basics',                     ('Math',           '7-8')),
    (r'Algebra 1\b|Algebra I\b',            ('Math',           '8-9')),
    (r'^Algebra 2\b|Algebra II\b',          ('Math',           '10-11')),
    (r'Algebra \(all content\)',            ('Math',           '7-10')),
    (r'Integrated math 1',                  ('Math',           '9')),
    (r'Integrated math 2',                  ('Math',           '10')),
    (r'Integrated math 3',                  ('Math',           '11')),
    (r'High school geometry',               ('Math',           '9-10')),
    (r'Geometry \(all|Geometry \(FL|Geometry \(Eureka',  ('Math', '9-10')),
    (r'^Geometry$',                         ('Math',           '9-10')),
    (r'Get ready for Geometry',             ('Math',           '9')),
    (r'Trigonometry',                       ('Math',           '10-11')),
    (r'Precalculus|Pre-calculus',           ('Math',           '11-12')),
    (r'Get ready for Precalculus',          ('Math',           '11')),
    (r'Get ready for AP.*Calc',             ('Math',           '12')),
    (r'AP.*Calculus|Calculus 1\b|Calculus 2\b|Differential Calculus|Integral Calculus', ('Math', '12')),
    (r'High school statistics',             ('Math',           '11-12')),
    (r'Get ready for AP.*Statistics',       ('Math',           '11')),
    (r'AP.*Statistics|Statistics and prob', ('Math',           '11-12')),
    (r'Arithmetic \(all|^Arithmetic$',      ('Math',           '3-5')),
    (r'Basic geometry and measurement',     ('Math',           '3-6')),
    (r'MAP Recommended Practice',           ('Math',           '3-8')),

    # ── Science ─────────────────────────────────────────────────────────────────
    (r'Middle school biology',              ('Science',        '6-8')),
    (r'Middle school Earth',                ('Science',        '6-8')),
    (r'Middle school physics',              ('Science',        '6-8')),
    (r'High school biology',                ('Science',        '9-10')),
    (r'High school physics',                ('Science',        '11-12')),
    (r'AP.*Biology|Biology library',        ('Science',        '9-12')),
    (r'AP.*Chemistry|Chemistry library',    ('Science',        '10-12')),
    (r'AP.*Physics|Physics library',        ('Science',        '11-12')),
    (r'AP.*Environmental',                  ('Science',        '11-12')),
    (r'Cosmology and astronomy',            ('Science',        '9-12')),

    # ── Social Studies / History ─────────────────────────────────────────────────
    (r'AP.*US History|^US history$',        ('Social Studies', '11')),
    (r'World history|World History',        ('Social Studies', '10')),
    (r'US government|civics|Civics',        ('Social Studies', '12')),
    (r'AP.*Government|AP.*Politics',        ('Social Studies', '12')),
    (r'Macroeconomics',                     ('Social Studies', '12')),
    (r'Microeconomics',                     ('Social Studies', '12')),
    (r'Finance and capital',                ('Social Studies', '11-12')),

    # ── Language Arts / ELA ─────────────────────────────────────────────────────
    (r'3rd grade reading',                  ('Language Arts',  '3')),
    (r'4th grade reading',                  ('Language Arts',  '4')),
    (r'5th grade reading',                  ('Language Arts',  '5')),
    (r'6th grade reading',                  ('Language Arts',  '6')),
    (r'7th grade reading',                  ('Language Arts',  '7')),
    (r'8th grade reading',                  ('Language Arts',  '8')),
    (r'9th grade reading',                  ('Language Arts',  '9')),
    (r'^Grammar',                           ('Language Arts',  '3-8')),
    (r'Storytelling',                       ('Language Arts',  '3-6')),

    # ── Computer Science ─────────────────────────────────────────────────────────
    (r'Computer programming|Computer Programming',  ('Computer Science', '9-12')),
    (r'Computer science|AP.*Computer Science',       ('Computer Science', '9-12')),
    (r'Code\.org|Computers and the Internet',        ('Computer Science', '6-12')),
]

# Courses to skip (college-only or K-2)
SKIP_PATTERNS = [
    r'^Kindergarten', r'^1st grade', r'^2nd grade', r'^Early math',
    r'Organic chemistry', r'Linear algebra', r'Multivariable calculus',
    r'Differential equations', r'College Algebra', r'AP.*College Calculus',
    r'AP.*College Chemistry', r'AP.*College Biology', r'AP.*College Physics',
    r'AP.*College Statistics', r'AP.*College Micro', r'AP.*College Macro',
    r'AP.*College US', r'AP.*College Art', r'AP.*College Computer',
    r'AP.*College Environment', r'Health and medicine', r'Electrical engineering',
    r'Pixar in a Box', r'Big History Project', r'^Get ready for 3rd grade$',
]


def classify_course(course_name: str):
    """
    Returns (subject, grade) or None if course should be skipped.
    grade is a string like "3", "7-8", "9-12"
    """
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, course_name, re.IGNORECASE):
            return None

    for pattern, (subject, grade) in COURSE_GRADE_MAP:
        if re.search(pattern, course_name, re.IGNORECASE):
            return subject, grade

    return None  # Unknown course → skip


# ==============================================================================
# Prompt Parsing
# ==============================================================================

def extract_course_and_title(prompt: str):
    """
    Parse the cosmopedia prompt to get:
      - course_name: e.g. "Algebra 1 - Linear equations"
      - lesson_title: e.g. "Solving one-step equations"
    """
    # Course: textbook on "COURSE - TOPIC"
    m = re.search(r'textbook on "([^"]+)"', prompt)
    if not m:
        return None, None
    course_full = m.group(1).strip()
    course_name = course_full.split(' - ')[0].strip()

    # Lesson title: sub-unit titled "TITLE"
    m2 = re.search(r'sub-unit titled "([^"]+)"', prompt)
    if m2:
        lesson_title = m2.group(1).strip()
    else:
        # Fallback: use the chapter name
        m3 = re.search(r'chapter.*?"([^"]+)"', prompt)
        lesson_title = m3.group(1).strip() if m3 else course_full

    return course_name, lesson_title


# ==============================================================================
# Download Functions
# ==============================================================================

def download_with_datasets_library():
    """Primary method: HuggingFace datasets library."""
    print("Trying datasets library...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/cosmopedia", "khanacademy", split="train")
    return ds


def download_with_requests():
    """Fallback method: direct parquet download via HuggingFace API."""
    import requests
    import io

    print("Trying direct parquet download...")
    # Get parquet file URL from HuggingFace API
    api_url = (
        "https://huggingface.co/api/datasets/"
        "HuggingFaceTB/cosmopedia/parquet/khanacademy/train"
    )
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    parquet_urls = resp.json()

    if not parquet_urls:
        raise ValueError("No parquet URLs returned from HuggingFace API")

    print(f"  Found {len(parquet_urls)} parquet file(s). Downloading...")

    # Download parquet file(s)
    import pandas as pd
    dfs = []
    for url in parquet_urls:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content))
        dfs.append(df)
        print(f"  Downloaded {len(df)} rows from {url.split('/')[-1]}")

    import pandas as pd
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows")
    return combined.to_dict(orient='records')


def load_cosmopedia():
    """Try datasets library first, fall back to requests."""
    try:
        ds = download_with_datasets_library()
        # Convert to list of dicts for uniform handling
        return list(ds)
    except Exception as e:
        print(f"  datasets library failed: {e}")

    try:
        return download_with_requests()
    except Exception as e:
        print(f"  requests fallback failed: {e}")
        raise RuntimeError(
            "Both download methods failed.\n"
            "Install datasets library: pip install datasets\n"
            "Or check internet connection."
        )


# ==============================================================================
# Transformation
# ==============================================================================

def transform_to_k12_concepts(raw_data):
    """
    Convert cosmopedia entries to the khan_k12_concepts format:
    [
      {
        "subject": "Math",
        "grade": "8-9",
        "course": "Algebra 1",
        "title": "Solving linear equations",
        "content": "...",
        "url": "cosmopedia://khanacademy/...",
        "word_count": 450
      },
      ...
    ]
    Only keeps grades 3-12 content.
    """
    results = []
    skipped_counter = Counter()
    subject_counter = Counter()

    print(f"\nTransforming {len(raw_data)} entries...")

    for entry in tqdm(raw_data, desc="Classifying"):
        prompt = entry.get("prompt", "")
        text = entry.get("text", "")

        if not prompt or not text or len(text.strip()) < 100:
            skipped_counter["empty"] += 1
            continue

        course_name, lesson_title = extract_course_and_title(prompt)
        if not course_name:
            skipped_counter["no_course"] += 1
            continue

        classification = classify_course(course_name)
        if classification is None:
            skipped_counter["out_of_scope"] += 1
            continue

        subject, grade = classification

        results.append({
            "subject": subject,
            "grade": grade,
            "course": course_name,
            "title": lesson_title or course_name,
            "content": text.strip(),
            "url": f"cosmopedia://khanacademy/{course_name.lower().replace(' ', '_')}",
            "word_count": len(text.split()),
        })
        subject_counter[subject] += 1

    return results, skipped_counter, subject_counter


# ==============================================================================
# Main
# ==============================================================================

def check_existing_data():
    """Check if data already exists and is valid."""
    output_file = Path("khan_k12_concepts/all_k12_concepts.json")
    if output_file.exists():
        with open(output_file, "r") as f:
            data = json.load(f)
        if len(data) > 0:
            print(f"\n✅ Khan Academy data already exists!")
            print(f"   Location: {output_file}")
            print(f"   Concepts: {len(data)}")

            subjects = Counter(d["subject"] for d in data)
            print("\n   Subjects:")
            for subj, cnt in sorted(subjects.items()):
                print(f"     - {subj}: {cnt} entries")

            print("\n✅ Using existing data.")
            print("   Proceed to Step 1: python 1_extract_khan_taxonomy.py")
            return True
    return False


def main():
    print("=" * 70)
    print("KHAN ACADEMY K-12 CONTENT COLLECTOR")
    print("Source: HuggingFaceTB/cosmopedia (khanacademy subset)")
    print("=" * 70)

    # Check cache
    if check_existing_data():
        return 0

    # Download
    print("\n📥 Downloading cosmopedia khanacademy dataset...")
    try:
        raw_data = load_cosmopedia()
    except RuntimeError as e:
        print(f"\n❌ Download failed:\n{e}")
        return 1

    print(f"\n✅ Downloaded {len(raw_data)} total entries")

    # Transform
    concepts, skipped, subject_counts = transform_to_k12_concepts(raw_data)

    # Report
    print(f"\n📊 Results:")
    print(f"   Total downloaded : {len(raw_data)}")
    print(f"   Kept (grade 3-12): {len(concepts)}")
    print(f"   Skipped           : {sum(skipped.values())}")
    for reason, cnt in skipped.items():
        print(f"     - {reason}: {cnt}")

    print(f"\n   By subject:")
    for subj, cnt in sorted(subject_counts.items(), key=lambda x: -x[1]):
        print(f"     {cnt:5d}  {subj}")

    # Grade distribution
    grade_counts = Counter(c["grade"] for c in concepts)
    print(f"\n   By grade range:")
    for grade, cnt in sorted(grade_counts.items()):
        print(f"     Grade {grade:6s}: {cnt}")

    # Save
    output_dir = Path("khan_k12_concepts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "all_k12_concepts.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(concepts)} concepts to {output_file}")
    print("\n   Next step: python 1_extract_khan_taxonomy.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
