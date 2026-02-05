"""
COMPREHENSIVE K-12 CURRICULUM COLLECTION
FULL TEXTBOOK CONTENT - NO SAMPLING

Collects COMPLETE chapter content from OpenStax textbooks
"""

import os
import json
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import re

from config import K12_RAW_DIR
from utils import save_json


class FullTextbookCollector:
    """
    Collect FULL textbook chapters - NO content limits
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "openstax"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research)'
        })
        self.toc_cache = {}  # Cache TOC results to avoid repeat scraping

    def get_book_toc(self, book_slug):
        """
        Scrape book's table of contents to get REAL chapter page slugs
        This fixes the 404 errors by getting actual OpenStax URLs
        """
        if book_slug in self.toc_cache:
            return self.toc_cache[book_slug]

        toc_url = f"https://openstax.org/books/{book_slug}"
        print(f"\n   📚 Scraping TOC from: {book_slug}...")

        try:
            response = self.session.get(toc_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            chapters = []

            # Method 1: Find links in navigation or TOC section
            # Look for links containing "/pages/" in href
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')

                # Check if it's a chapter/page link
                if '/pages/' in href and book_slug in href:
                    # Extract page slug
                    page_slug = href.split('/pages/')[-1]
                    # Remove any query parameters or anchors
                    page_slug = page_slug.split('?')[0].split('#')[0]

                    # Get chapter title
                    title = link.get_text(strip=True)

                    # Filter out very short titles or navigation items
                    if title and len(title) > 3 and page_slug:
                        # Avoid duplicates
                        if not any(slug == page_slug for slug, _ in chapters):
                            chapters.append((page_slug, title))

            # If we got chapters, cache and return
            if chapters:
                print(f"   ✅ Found {len(chapters)} chapters via TOC scraping")
                self.toc_cache[book_slug] = chapters
                return chapters

            # Method 2: Fallback - try common patterns
            print(f"   ⚠️  TOC scraping didn't find chapters, using fallback")
            return None

        except Exception as e:
            print(f"   ❌ TOC scraping failed: {str(e)[:50]}")
            return None

    def get_complete_catalog(self):
        """EXPANDED OpenStax K-12 Catalog - 27 Books (Target: 25-30)"""
        return {
            # ============================================================
            # MATHEMATICS (11 books)
            # ============================================================

            # Middle School Math
            'prealgebra-2e': {
                'subject': 'mathematics', 'grade': '7', 'domain': 'prealgebra',
                'chapters': []  # TOC scraping will get actual chapters
            },

            'elementary-algebra-2e': {
                'subject': 'mathematics', 'grade': '8', 'domain': 'algebra_1',
                'chapters': []
            },

            # High School Math
            'intermediate-algebra-2e': {
                'subject': 'mathematics', 'grade': '9', 'domain': 'algebra_2',
                'chapters': []
            },

            'algebra-and-trigonometry-2e': {
                'subject': 'mathematics', 'grade': '10', 'domain': 'algebra_trig',
                'chapters': []
            },

            'precalculus-2e': {
                'subject': 'mathematics', 'grade': '11', 'domain': 'precalculus',
                'chapters': []
            },

            # College Intro Math
            'college-algebra-2e': {
                'subject': 'mathematics', 'grade': '12', 'domain': 'college_algebra',
                'chapters': []
            },

            'calculus-volume-1': {
                'subject': 'mathematics', 'grade': '12', 'domain': 'calculus',
                'chapters': []
            },

            'calculus-volume-2': {
                'subject': 'mathematics', 'grade': '12', 'domain': 'calculus',
                'chapters': []
            },

            'calculus-volume-3': {
                'subject': 'mathematics', 'grade': '12', 'domain': 'calculus',
                'chapters': []
            },

            'introductory-statistics-2e': {
                'subject': 'mathematics', 'grade': '11', 'domain': 'statistics',
                'chapters': []
            },

            'statistics': {
                'subject': 'mathematics', 'grade': '11', 'domain': 'statistics',
                'chapters': []
            },

            # ============================================================
            # SCIENCE (10 books)
            # ============================================================

            # Biology
            'biology-2e': {
                'subject': 'science', 'grade': '10', 'domain': 'biology',
                'chapters': []
            },

            'biology-ap-courses-2e': {
                'subject': 'science', 'grade': '11', 'domain': 'biology_ap',
                'chapters': []
            },

            'microbiology': {
                'subject': 'science', 'grade': '12', 'domain': 'microbiology',
                'chapters': []
            },

            # Chemistry
            'chemistry-2e': {
                'subject': 'science', 'grade': '10', 'domain': 'chemistry',
                'chapters': []
            },

            'chemistry-atoms-first-2e': {
                'subject': 'science', 'grade': '11', 'domain': 'chemistry_advanced',
                'chapters': []
            },

            # Physics
            'physics': {
                'subject': 'science', 'grade': '11', 'domain': 'physics',
                'chapters': []
            },

            'college-physics-2e': {
                'subject': 'science', 'grade': '12', 'domain': 'physics_intro',
                'chapters': []
            },

            'college-physics-ap-courses-2e': {
                'subject': 'science', 'grade': '11', 'domain': 'physics_ap',
                'chapters': []
            },

            # Astronomy & Earth Science
            'astronomy-2e': {
                'subject': 'science', 'grade': '10', 'domain': 'astronomy',
                'chapters': []
            },

            # Anatomy (Advanced)
            'anatomy-and-physiology-2e': {
                'subject': 'science', 'grade': '12', 'domain': 'anatomy',
                'chapters': []
            },

            # ============================================================
            # SOCIAL STUDIES (6 books)
            # ============================================================

            # History
            'us-history': {
                'subject': 'social_studies', 'grade': '11', 'domain': 'us_history',
                'chapters': []
            },

            'world-history-volume-1': {
                'subject': 'social_studies', 'grade': '10', 'domain': 'world_history',
                'chapters': []
            },

            'world-history-volume-2': {
                'subject': 'social_studies', 'grade': '10', 'domain': 'world_history',
                'chapters': []
            },

            # Government & Economics
            'american-government-3e': {
                'subject': 'social_studies', 'grade': '12', 'domain': 'government',
                'chapters': []
            },

            'principles-microeconomics-3e': {
                'subject': 'social_studies', 'grade': '11', 'domain': 'economics',
                'chapters': []
            },

            'principles-macroeconomics-3e': {
                'subject': 'social_studies', 'grade': '12', 'domain': 'economics',
                'chapters': []
            },
        }
    
    def extract_full_content(self, book_slug, chapter_slug):
        """
        Extract COMPLETE chapter content - NO LIMITS
        Get entire textbook chapter
        """
        url = f'https://openstax.org/books/{book_slug}/pages/{chapter_slug}'
        
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find main content
            content_div = (
                soup.find('div', {'data-type': 'page'}) or
                soup.find('main') or
                soup.find('article')
            )
            
            if not content_div:
                return None
            
            # Extract ALL text elements - complete chapter
            all_elements = content_div.find_all([
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'li', 'td', 'th', 'div', 'span', 'section'
            ])
            
            full_text_parts = []
            for elem in all_elements:
                text = elem.get_text(strip=True)
                if text:  # Include everything with text
                    full_text_parts.append(text)
            
            # Join everything - COMPLETE chapter content
            complete_text = '\n\n'.join(full_text_parts)
            
            # NO LENGTH LIMIT - return full chapter
            print(f"({len(complete_text):,} chars)", end=' ')
            
            return complete_text if complete_text else None
            
        except Exception as e:
            print(f"[{str(e)[:20]}]", end=' ')
            return None
    
    def collect_book(self, book_slug, book_info):
        """Collect complete book with full chapters"""
        print(f"\n📖 {book_slug.upper()}")
        print(f"   Domain: {book_info['domain']} | Grade: {book_info['grade']}")

        # Try to get real chapter URLs from TOC
        toc_chapters = self.get_book_toc(book_slug)

        if toc_chapters:
            # Use scraped TOC (real URLs!)
            chapters_to_use = toc_chapters
            print(f"   ✅ Using {len(toc_chapters)} chapters from TOC scraping")
        else:
            # Fallback to hardcoded catalog
            chapters_to_use = book_info['chapters']
            print(f"   ⚠️  Using {len(chapters_to_use)} chapters from hardcoded catalog")

        print(f"   Chapters: {len(chapters_to_use)}")

        collected_chapters = []
        total_chars = 0

        for i, (slug, title) in enumerate(chapters_to_use, 1):
            print(f"   [{i:2d}/{len(book_info['chapters'])}] {title[:35]:35s} ", end='')
            
            full_content = self.extract_full_content(book_slug, slug)
            
            if full_content:
                collected_chapters.append({
                    'title': title,
                    'url': f'https://openstax.org/books/{book_slug}/pages/{slug}',
                    'content': full_content,
                    'domain': book_info['domain'],
                    'char_count': len(full_content)
                })
                total_chars += len(full_content)
                print("✓")
            else:
                print("✗")
            
            time.sleep(0.5)
        
        # Save
        if collected_chapters:
            data = {
                "source": "OpenStax",
                "book": book_slug,
                "subject": book_info['subject'],
                "grade_level": book_info['grade'],
                "domain": book_info['domain'],
                "chapters": collected_chapters,
                "statistics": {
                    "total_chapters": len(collected_chapters),
                    "total_characters": total_chars,
                    "avg_chars_per_chapter": total_chars // len(collected_chapters) if collected_chapters else 0
                }
            }
            
            save_json(data, str(self.output_dir / f"{book_slug}.json"))
            print(f"   ✅ {len(collected_chapters)} chapters | {total_chars:,} total chars")
            return data
        
        return None
    
    def collect_all(self):
        """Collect all books with full content"""
        catalog = self.get_complete_catalog()
        results = {}
        
        total_books = len(catalog)
        grand_total_chars = 0
        
        for idx, (slug, info) in enumerate(catalog.items(), 1):
            print(f"\n{'='*70}")
            print(f"BOOK {idx}/{total_books}")
            print('='*70)
            
            try:
                result = self.collect_book(slug, info)
                if result:
                    results[slug] = result
                    grand_total_chars += result['statistics']['total_characters']
                time.sleep(2)
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return results, grand_total_chars


class ComprehensiveSampleCollector:
    """K-12 sample data with domain classification"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "curated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_doc(self, subject, grade, domain, title, content):
        """Add document"""
        doc = {
            "source": "Curated",
            "subject": subject,
            "grade_level": grade,
            "domain": domain,
            "title": title,
            "text": content,
            "content": content,
            "char_count": len(content)
        }
        
        dir_path = self.output_dir / subject / domain
        dir_path.mkdir(parents=True, exist_ok=True)
        
        safe = re.sub(r'[^\w\s-]', '', title).replace(' ', '_')
        save_json(doc, str(dir_path / f"grade_{grade}_{safe}.json"))
    
    def create_samples(self):
        """Create comprehensive K-12 samples"""
        print("\n📝 Creating K-12 samples (K through 12)...")
        
        samples = [
            # K-1
            ("mathematics", "K", "counting", "Counting 1-10", "One two three four five six seven eight nine ten. Practice counting objects."),
            ("mathematics", "1", "arithmetic", "Addition Facts", "1+1=2, 2+1=3, 3+1=4, 4+1=5. Adding one more."),
            ("science", "K", "life_science", "Living vs Non-living", "Plants and animals are living things. They grow, eat, and reproduce. Rocks and water are not living."),
            ("english", "K", "phonics", "Letter Sounds", "A says 'ah' in apple. B says 'buh' in ball. C says 'kuh' in cat."),
            
            # 2-5 Elementary
            ("mathematics", "2", "arithmetic", "Two-Digit Addition", "24 + 35 = 59. Line up the ones place and tens place. Add ones first, then tens."),
            ("mathematics", "3", "multiplication", "Times Tables", "3 × 4 = 12 means three groups of four. 3+3+3+3 = 12."),
            ("mathematics", "4", "fractions", "Understanding Fractions", "1/2 means one part out of two equal parts. 3/4 means three parts out of four."),
            ("mathematics", "5", "decimals", "Decimal Numbers", "0.5 = 1/2. 0.25 = 1/4. 0.75 = 3/4. Decimals show parts of a whole."),
            ("science", "2", "physical_science", "States of Matter", "Matter exists in three states: solid (ice), liquid (water), and gas (steam)."),
            ("science", "3", "earth_science", "The Water Cycle", "Water evaporates from oceans, condenses into clouds, and falls as precipitation."),
            ("science", "4", "geology", "Rock Types", "Three rock types: igneous (cooled lava), sedimentary (layers), metamorphic (changed by heat/pressure)."),
            ("science", "5", "ecosystems", "Food Chains", "Sun → Plants (producers) → Herbivores (consumers) → Carnivores (predators) → Decomposers."),
            ("english", "2", "reading", "Reading Comprehension", "Read the story and answer: Who? What? When? Where? Why? How?"),
            ("english", "3", "writing", "Paragraph Structure", "A paragraph has a topic sentence, supporting details, and a concluding sentence."),
            ("english", "4", "grammar", "Parts of Speech", "Nouns name people, places, things. Verbs show action. Adjectives describe nouns."),
            ("english", "5", "literature", "Story Elements", "Every story has characters, setting, plot, conflict, and theme."),
            
            # 6-8 Middle School
            ("mathematics", "6", "pre_algebra", "Solving Equations", "x + 5 = 12. Subtract 5 from both sides: x = 7."),
            ("mathematics", "7", "algebra_1", "Linear Equations", "y = mx + b is slope-intercept form. m is slope, b is y-intercept."),
            ("mathematics", "8", "geometry", "Pythagorean Theorem", "a² + b² = c² for right triangles. Find the hypotenuse."),
            ("science", "6", "cell_biology", "Cell Structure", "Cells have nucleus (control center), mitochondria (powerhouse), cell membrane (barrier)."),
            ("science", "7", "chemistry", "Periodic Table", "Elements organized by atomic number. Rows are periods, columns are groups with similar properties."),
            ("science", "8", "physics", "Newton's Laws", "First law: objects at rest stay at rest. Second law: F=ma. Third law: action-reaction."),
            ("english", "6", "composition", "Essay Writing", "Essays have introduction (with thesis), body paragraphs (with evidence), and conclusion."),
            ("english", "7", "literary_devices", "Figurative Language", "Metaphor compares directly. Simile uses 'like' or 'as'. Personification gives human traits to non-human things."),
            ("english", "8", "research_writing", "Research Papers", "Find credible sources, take notes, cite using MLA or APA format, avoid plagiarism."),
            
            # 9-12 High School
            ("mathematics", "9", "algebra_1", "Quadratic Equations", "x² + 5x + 6 = 0. Factor: (x+2)(x+3) = 0. Solutions: x = -2 or x = -3."),
            ("mathematics", "10", "geometry", "Triangle Properties", "Sum of angles = 180°. Pythagorean theorem for right triangles. Similar triangles have proportional sides."),
            ("mathematics", "11", "algebra_2", "Exponential Growth", "y = a(b)ˣ where b > 1 is growth, 0 < b < 1 is decay. Used in population, finance, radioactivity."),
            ("mathematics", "12", "calculus", "Derivatives", "Derivative f'(x) is the rate of change. Limit definition: lim(h→0) [f(x+h) - f(x)]/h."),
            ("science", "9", "biology", "DNA and Genetics", "DNA is double helix. Genes code for proteins. Punnett squares predict offspring traits."),
            ("science", "10", "chemistry", "Chemical Reactions", "Reactants → Products. Balance equations: 2H₂ + O₂ → 2H₂O. Conservation of mass."),
            ("science", "11", "physics", "Kinematics", "v = v₀ + at (velocity). d = v₀t + ½at² (distance). Motion with constant acceleration."),
            ("science", "12", "environmental_science", "Climate Change", "Greenhouse gases (CO₂, CH₄) trap heat. Rising temperatures affect ecosystems globally."),
            ("english", "9", "literature", "Shakespeare Analysis", "Romeo and Juliet: tragedy of feuding families. Themes: love vs hate, fate vs free will."),
            ("english", "10", "rhetoric", "Persuasive Appeals", "Ethos (credibility), Pathos (emotion), Logos (logic). Use in arguments and persuasive writing."),
            ("english", "11", "research", "Academic Research", "Develop thesis, find peer-reviewed sources, analyze evidence, write with proper citations."),
            ("english", "12", "critical_analysis", "Literary Criticism", "Analyze themes, symbols, character development. Apply literary theory to interpret texts."),
            
            # Social Studies
            ("social_studies", "6", "ancient_history", "Ancient Civilizations", "Egypt: pyramids, pharaohs. Greece: democracy, philosophy. Rome: republic, empire, law."),
            ("social_studies", "7", "us_history", "American Revolution", "13 colonies rebelled against British rule. Declaration of Independence 1776. Won freedom 1783."),
            ("social_studies", "8", "civics", "Branches of Government", "Legislative (Congress) makes laws. Executive (President) enforces laws. Judicial (Courts) interprets laws."),
            ("social_studies", "9", "world_history", "World War I", "1914-1918. Trench warfare in Europe. Treaty of Versailles ended war, created League of Nations."),
            ("social_studies", "10", "us_history", "Civil Rights Movement", "MLK Jr. led nonviolent protests. Brown v Board (1954) ended school segregation. Civil Rights Act (1964)."),
            ("social_studies", "11", "government", "US Constitution", "Bill of Rights: first 10 amendments protect individual freedoms. 27 amendments total."),
            ("social_studies", "12", "economics", "Supply and Demand", "Price equilibrium where supply meets demand. Elasticity measures responsiveness to price changes."),
        ]
        
        for s, g, d, t, c in samples:
            self.add_doc(s, g, d, t, c)
        
        print(f"✅ {len(samples)} samples created")
        print(f"   Covering K-12 (13 grade levels)")
        print(f"   25+ specific domains")
        
        return samples


def main():
    """Main collection - FULL TEXTBOOK CONTENT"""
    print("="*70)
    print("COMPREHENSIVE K-12 TEXTBOOK COLLECTION")
    print("COLLECTING FULL CHAPTER CONTENT - NO SAMPLING")
    print("="*70)
    
    Path(K12_RAW_DIR).mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Samples
    print("\n" + "="*70)
    print("PHASE 1: K-12 Sample Documents")
    print("="*70)
    sampler = ComprehensiveSampleCollector(K12_RAW_DIR)
    samples = sampler.create_samples()
    
    # Phase 2: Full OpenStax textbooks
    print("\n" + "="*70)
    print("PHASE 2: OpenStax FULL Textbook Collection")
    print("="*70)
    print("⏳ Estimated time: 20-30 minutes")
    print("   Downloading COMPLETE chapters (no limits)...")
    
    collector = FullTextbookCollector(K12_RAW_DIR)
    results, total_chars = collector.collect_all()
    
    # Summary
    print("\n" + "="*70)
    print("✅ COLLECTION COMPLETE")
    print("="*70)
    
    total_chapters = sum(len(b['chapters']) for b in results.values())
    
    print(f"\n📊 Final Statistics:")
    print(f"   OpenStax Books: {len(results)}")
    print(f"   Total Chapters: {total_chapters}")
    print(f"   Total Characters: {total_chars:,}")
    print(f"   Avg per Chapter: {total_chars // total_chapters:,} chars" if total_chapters else "")
    print(f"   Sample Documents: {len(samples)}")
    print(f"   Grade Levels: K-12 (13 levels)")
    print(f"   Domains: 25+ specific topics")
    
    print(f"\n💾 Saved to: {K12_RAW_DIR}")
    print("\n💡 Next steps:")
    print("   python build_k12_graph.py")
    print("   python visualize_k12_graph.py")


if __name__ == "__main__":
    main()
