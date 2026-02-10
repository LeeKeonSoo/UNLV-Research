"""
Khan Academy K-12 Content Collector
Collects concept explanations (not exercises) from Khan Academy
Organized by Grade and Subject
"""

import requests
import json
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


class KhanAcademyCollector:
    """
    Collect K-12 educational content from Khan Academy
    Focus: Concept explanations only (articles, video transcripts)
    """

    def __init__(self, output_dir="khan_k12_raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Khan Academy topic tree API
        self.base_url = "https://www.khanacademy.org/api/v1"
        self.topic_tree_url = f"{self.base_url}/topictree"

        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research)'
        })

    def get_topic_tree(self):
        """Get Khan Academy's complete topic tree"""
        print("\n📥 Fetching Khan Academy topic tree...")

        try:
            response = self.session.get(self.topic_tree_url)
            response.raise_for_status()
            tree = response.json()

            print(f"✅ Topic tree loaded")
            return tree

        except Exception as e:
            print(f"❌ Error fetching topic tree: {e}")
            print("\n⚠️  Khan Academy API might have changed.")
            print("   Falling back to web scraping method...")
            return None

    def extract_k12_subjects(self, tree):
        """Extract K-12 relevant subjects from topic tree"""
        print("\n🔍 Extracting K-12 subjects...")

        k12_keywords = [
            'math', 'science', 'reading', 'grammar', 'history',
            'grade', 'elementary', 'middle-school', 'high-school',
            'algebra', 'geometry', 'biology', 'chemistry', 'physics',
            'english', 'ela', 'social-studies'
        ]

        k12_subjects = []

        def traverse(node, path=[]):
            """Recursively traverse topic tree"""
            if not isinstance(node, dict):
                return

            # Check if this is a K-12 relevant topic
            title = node.get('title', '').lower()
            slug = node.get('slug', '').lower()

            is_k12 = any(kw in title or kw in slug for kw in k12_keywords)

            if is_k12:
                k12_subjects.append({
                    'title': node.get('title'),
                    'slug': node.get('slug'),
                    'path': path + [node.get('title')],
                    'kind': node.get('kind'),
                    'id': node.get('id')
                })

            # Recurse into children
            for child in node.get('children', []):
                traverse(child, path + [node.get('title', '')])

        if tree:
            traverse(tree)

        print(f"   Found {len(k12_subjects)} K-12 topics")
        return k12_subjects

    def collect_web_scraping_method(self):
        """
        Alternative: Web scraping method
        More reliable than API for comprehensive collection
        """
        print("\n🌐 Using web scraping method...")

        # Define K-12 curriculum structure manually
        # Based on Khan Academy's actual structure
        curriculum = {
            'Math': {
                'Early Math (K-2)': [
                    'https://www.khanacademy.org/math/early-math',
                ],
                '3rd Grade': [
                    'https://www.khanacademy.org/math/cc-third-grade-math',
                ],
                '4th Grade': [
                    'https://www.khanacademy.org/math/cc-fourth-grade-math',
                ],
                '5th Grade': [
                    'https://www.khanacademy.org/math/cc-fifth-grade-math',
                ],
                '6th Grade': [
                    'https://www.khanacademy.org/math/cc-sixth-grade-math',
                ],
                '7th Grade': [
                    'https://www.khanacademy.org/math/cc-seventh-grade-math',
                ],
                '8th Grade': [
                    'https://www.khanacademy.org/math/cc-eighth-grade-math',
                ],
                'Algebra 1': [
                    'https://www.khanacademy.org/math/algebra',
                ],
                'Geometry': [
                    'https://www.khanacademy.org/math/geometry',
                ],
                'Algebra 2': [
                    'https://www.khanacademy.org/math/algebra2',
                ],
                'Precalculus': [
                    'https://www.khanacademy.org/math/precalculus',
                ],
                'Calculus': [
                    'https://www.khanacademy.org/math/calculus-1',
                ],
            },
            'Science': {
                'Biology': [
                    'https://www.khanacademy.org/science/biology',
                ],
                'Chemistry': [
                    'https://www.khanacademy.org/science/chemistry',
                ],
                'Physics': [
                    'https://www.khanacademy.org/science/physics',
                ],
                'High School Biology': [
                    'https://www.khanacademy.org/science/high-school-biology',
                ],
            },
            'Reading & Language Arts': {
                'Reading Comprehension': [
                    'https://www.khanacademy.org/ela/cc-2nd-reading-vocab',
                    'https://www.khanacademy.org/ela/cc-3rd-reading-vocab',
                    'https://www.khanacademy.org/ela/cc-4th-reading-vocab',
                    'https://www.khanacademy.org/ela/cc-5th-reading-vocab',
                ],
                'Grammar': [
                    'https://www.khanacademy.org/ela/cc-2nd-writing-language',
                    'https://www.khanacademy.org/ela/cc-3rd-writing-language',
                ],
            },
            'Social Studies': {
                'US History': [
                    'https://www.khanacademy.org/humanities/us-history',
                ],
                'World History': [
                    'https://www.khanacademy.org/humanities/world-history',
                ],
            }
        }

        print(f"\n📚 Curriculum structure:")
        for subject, grades in curriculum.items():
            print(f"   {subject}: {len(grades)} grade levels")

        return curriculum

    def scrape_content_from_url(self, url, subject, grade):
        """
        Scrape content from a Khan Academy URL

        Note: This is a simplified version.
        In production, you'd need:
        1. Selenium/Playwright for dynamic content
        2. Proper parsing of KA's structure
        3. Rate limiting
        """
        print(f"\n   Scraping: {grade} - {subject}")
        print(f"   URL: {url}")

        try:
            # Add delay to be respectful
            time.sleep(1)

            response = self.session.get(url)
            response.raise_for_status()

            # This is a placeholder
            # Real implementation would parse the HTML properly
            content = {
                'url': url,
                'subject': subject,
                'grade': grade,
                'status': 'requires_selenium',
                'note': 'Khan Academy uses React, need Selenium for full scraping'
            }

            return content

        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            return None

    def save_curriculum_structure(self, curriculum):
        """Save curriculum structure for reference"""
        output_file = self.output_dir / "curriculum_structure.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(curriculum, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved curriculum structure: {output_file}")

    def generate_collection_script(self):
        """
        Generate a comprehensive collection script
        that uses proper tools (Selenium/Playwright)
        """
        script_content = '''"""
Khan Academy Complete Scraper with Selenium
Handles dynamic JavaScript content
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

class KhanAcademyScraper:
    def __init__(self):
        # Setup Chrome driver (headless)
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        self.driver = webdriver.Chrome(options=options)

    def scrape_article_content(self, url):
        """Scrape article content from a Khan Academy page"""
        self.driver.get(url)

        # Wait for content to load
        wait = WebDriverWait(self.driver, 10)

        try:
            # Khan Academy articles use specific class names
            article = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "_1vwfr8mh"))
            )

            # Extract text
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Find article content
            content_div = soup.find('div', class_='_1vwfr8mh')

            if content_div:
                return content_div.get_text(strip=True)

        except Exception as e:
            print(f"Error scraping {url}: {e}")

        return None

    def scrape_course(self, course_url):
        """Scrape entire course"""
        self.driver.get(course_url)
        time.sleep(2)

        # Get all article links
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        links = soup.find_all('a', href=True)

        article_links = [
            link['href'] for link in links
            if '/a/' in link['href']  # Articles have /a/ in URL
        ]

        return article_links

    def close(self):
        self.driver.quit()

# Usage:
# scraper = KhanAcademyScraper()
# content = scraper.scrape_article_content("https://...")
# scraper.close()
'''

        script_file = self.output_dir / "khan_scraper_selenium.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)

        print(f"\n📝 Generated advanced scraper: {script_file}")
        print("\n   To use:")
        print("   1. pip install selenium beautifulsoup4")
        print("   2. Install Chrome/ChromeDriver")
        print("   3. Run the generated script")

    def run(self):
        """Main collection process"""
        print("=" * 70)
        print("KHAN ACADEMY K-12 CONTENT COLLECTOR")
        print("=" * 70)

        # Try API first
        tree = self.get_topic_tree()

        if tree:
            k12_subjects = self.extract_k12_subjects(tree)
            # Process subjects...

        # Use web scraping method
        curriculum = self.collect_web_scraping_method()
        self.save_curriculum_structure(curriculum)

        # Generate advanced scraper
        self.generate_collection_script()

        print("\n" + "=" * 70)
        print("COLLECTION SETUP COMPLETE")
        print("=" * 70)

        print("\n⚠️  IMPORTANT:")
        print("   Khan Academy requires Selenium for full content extraction")
        print("   because it's a React-based SPA (Single Page Application)")

        print("\n📋 Next Steps:")
        print("   1. Install: pip install selenium beautifulsoup4 webdriver-manager")
        print("   2. Use the generated khan_scraper_selenium.py")
        print("   3. Or use Khan Academy YouTube channel for video transcripts")

        print("\n💡 Alternative (Easier):")
        print("   Khan Academy videos on YouTube with auto-generated captions")
        print("   - Search: 'Khan Academy [topic] site:youtube.com'")
        print("   - Extract: YouTube transcripts (youtube-transcript-api)")

        return curriculum


def main():
    """Main entry point"""
    collector = KhanAcademyCollector()
    curriculum = collector.run()

    return 0


if __name__ == "__main__":
    exit(main())
