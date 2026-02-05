"""
K-12 Data Collection Script

Collects curriculum data from:
- Khan Academy
- OpenStax textbooks
- Other K-12 sources

Organizes by subject and grade level.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path

from config import *
from utils import save_json


class KhanAcademyCollector:
    """
    Collect content from Khan Academy
    
    Note: Khan Academy doesn't have official API for content.
    This is a placeholder for manual collection or web scraping.
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "khan_academy"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_subject(self, subject_name):
        """
        Collect all content for a subject
        
        For now, this is a placeholder.
        In practice, would need to:
        1. Scrape Khan Academy pages
        2. Use their internal API (if accessible)
        3. Or manually curate content
        """
        print(f"📚 Collecting Khan Academy: {subject_name}")
        print("⚠️  Note: Requires manual data collection or API access")
        print("    Khan Academy content is available but needs proper collection method")
        
        # Placeholder structure
        collected = {
            "source": "Khan Academy",
            "subject": subject_name,
            "collection_method": "manual",
            "documents": []
        }
        
        output_file = self.output_dir / f"{subject_name}.json"
        save_json(collected, str(output_file))
        
        return collected


class OpenStaxCollector:
    """
    Collect OpenStax textbooks
    
    OpenStax books are available as HTML and can be parsed.
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "openstax"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://openstax.org/books/"
        
    def collect_book(self, book_slug):
        """
        Collect a single OpenStax book
        
        Args:
            book_slug: Book identifier (e.g., 'prealgebra-2e')
        """
        print(f"📖 Collecting OpenStax book: {book_slug}")
        
        # OpenStax book structure
        book_url = f"{self.base_url}{book_slug}/pages"
        
        try:
            # This is a placeholder - actual implementation would:
            # 1. Fetch the book's table of contents
            # 2. Parse each chapter/section
            # 3. Extract text content, examples, exercises
            
            collected = {
                "source": "OpenStax",
                "book": book_slug,
                "url": book_url,
                "chapters": [],
                "collection_method": "HTML parsing"
            }
            
            # Save
            output_file = self.output_dir / f"{book_slug}.json"
            save_json(collected, str(output_file))
            
            print(f"✅ Saved to {output_file}")
            return collected
            
        except Exception as e:
            print(f"❌ Error collecting {book_slug}: {e}")
            return None
    
    def collect_all_k12_books(self):
        """Collect all OpenStax K-12 relevant books"""
        print("\n" + "="*60)
        print("COLLECTING OPENSTAX K-12 TEXTBOOKS")
        print("="*60)
        
        results = {}
        for book_slug in OPENSTAX_K12_BOOKS:
            result = self.collect_book(book_slug)
            if result:
                results[book_slug] = result
            time.sleep(1)  # Be respectful to server
        
        return results


class ManualCuratorCollector:
    """
    Manually curated K-12 content
    
    For high-quality sources that require manual selection:
    - Core civic documents (Constitution)
    - Exemplar essays and writing samples
    - Carefully selected practice problems
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "curated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_document(self, subject, grade_level, title, content, metadata=None):
        """
        Add a manually curated document
        
        Args:
            subject: Subject area (math, science, etc.)
            grade_level: Grade level (1-12)
            title: Document title
            content: Full text content
            metadata: Optional dict with additional info
        """
        doc = {
            "source": "Manual Curation",
            "subject": subject,
            "grade_level": grade_level,
            "title": title,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Organize by subject and grade
        subject_dir = self.output_dir / subject
        subject_dir.mkdir(exist_ok=True)
        
        # Generate filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"grade{grade_level}_{safe_title}.json"
        
        output_file = subject_dir / filename
        save_json(doc, str(output_file))
        
        print(f"✅ Added: {subject}/Grade {grade_level}/{title}")
        return doc


def create_collection_manifest():
    """
    Create a manifest of what needs to be collected
    """
    manifest = {
        "collection_plan": {
            "khan_academy": {
                "subjects": list(KHAN_SUBJECTS.keys()),
                "status": "pending",
                "method": "manual or scraping"
            },
            "openstax": {
                "books": OPENSTAX_K12_BOOKS,
                "status": "pending", 
                "method": "HTML parsing"
            },
            "curated": {
                "priority_items": [
                    "US Constitution (civics)",
                    "Bill of Rights (civics)",
                    "Common Core example problems (math)",
                    "Science lab procedures (science)"
                ],
                "status": "manual",
                "method": "human curation"
            }
        },
        "target_coverage": {
            "mathematics": "K-12 complete",
            "science": "Elementary through high school",
            "language_arts": "Reading and writing fundamentals",
            "social_studies": "History and civics basics"
        }
    }
    
    output_file = Path(K12_RAW_DIR) / "collection_manifest.json"
    save_json(manifest, str(output_file))
    print(f"📋 Collection manifest saved to {output_file}")
    
    return manifest


def main():
    """
    Main collection orchestrator
    """
    print("="*60)
    print("K-12 CURRICULUM DATA COLLECTION")
    print("="*60)
    
    # Create manifest
    manifest = create_collection_manifest()
    
    # Initialize collectors
    khan_collector = KhanAcademyCollector(K12_RAW_DIR)
    openstax_collector = OpenStaxCollector(K12_RAW_DIR)
    curated_collector = ManualCuratorCollector(K12_RAW_DIR)
    
    # Collect OpenStax (easiest to automate)
    print("\n" + "="*60)
    print("PHASE 1: OpenStax Collection")
    print("="*60)
    openstax_results = openstax_collector.collect_all_k12_books()
    
    # Khan Academy placeholder
    print("\n" + "="*60)
    print("PHASE 2: Khan Academy Collection")
    print("="*60)
    print("⚠️  Khan Academy requires manual collection or approved API access")
    print("    Recommended: Use Khan Academy's bulk data exports if available")
    print("    Or: Manually curate key articles and exercises")
    
    # Manual curation examples
    print("\n" + "="*60)
    print("PHASE 3: Manual Curation Setup")
    print("="*60)
    print("📝 Use curated_collector.add_document() to add high-quality content")
    print("   Example civic documents, math problems, science explanations")
    
    # Summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"OpenStax books collected: {len(openstax_results)}")
    print(f"Output directory: {K12_RAW_DIR}")
    print(f"Manifest file: {K12_RAW_DIR}/collection_manifest.json")
    print("\n💡 Next steps:")
    print("   1. Review collected OpenStax content")
    print("   2. Set up Khan Academy collection method")
    print("   3. Begin manual curation of priority content")
    print("   4. Run build_concept_graph.py once data is ready")


if __name__ == "__main__":
    main()
