"""
Khan Academy K-12 Article Collector
Extracts ONLY text-based concept explanations (not videos, not exercises)
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from tqdm import tqdm


class KhanArticleCollector:
    """
    Collect text-based concept explanations from Khan Academy
    Focus: Articles only (no videos, no exercises)
    """

    def __init__(self, output_dir="khan_k12_concepts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # No GUI
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')

        print("🌐 Starting Chrome (headless)...")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)

    def get_course_structure(self, course_url):
        """Get all lessons/articles in a course"""
        print(f"\n📖 Analyzing course: {course_url}")

        self.driver.get(course_url)
        time.sleep(5)  # Wait longer for React to render

        # Scroll to load lazy content
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # Find all article links
        # Khan Academy articles have /a/ in their URL
        all_links = soup.find_all('a', href=True)

        article_links = []
        video_links = []

        for link in all_links:
            href = link.get('href', '')

            # Article URLs have this pattern
            if '/a/' in href and href not in article_links:
                full_url = f"https://www.khanacademy.org{href}" if href.startswith('/') else href
                article_links.append(full_url)
            # Also track videos for debugging
            elif '/v/' in href and href not in video_links:
                video_links.append(href)

        print(f"   Found {len(article_links)} articles, {len(video_links)} videos")

        if len(article_links) == 0 and len(video_links) > 0:
            print(f"   ⚠️  No articles, but {len(video_links)} videos exist")
            print(f"   This course might be video-only")

        return article_links

    def extract_article_content(self, article_url):
        """₩
        Extract concept explanation from a Khan Academy article
        """
        try:
            self.driver.get(article_url)
            time.sleep(2)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Extract title
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Untitled"

            # Extract article content
            # Khan Academy uses specific class names for article content
            # This might need adjustment based on their current HTML structure

            # Try multiple selectors
            content_selectors = [
                {'class': '_1vwfr8mh'},  # Common KA article class
                {'class': 'article-content'},
                {'class': 'perseus-article'},
                {'role': 'article'},
            ]

            content_div = None
            for selector in content_selectors:
                content_div = soup.find('div', selector)
                if content_div:
                    break

            # Fallback: get main content
            if not content_div:
                content_div = soup.find('main') or soup.find('article')

            if not content_div:
                print(f"   ⚠️  No content found in: {article_url}")
                return None

            # Extract text
            # Remove script and style elements
            for script in content_div(['script', 'style']):
                script.decompose()

            text = content_div.get_text(separator='\n', strip=True)

            # Clean up
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)

            return {
                'title': title,
                'url': article_url,
                'content': clean_text,
                'word_count': len(clean_text.split()),
                'char_count': len(clean_text)
            }

        except Exception as e:
            print(f"   ❌ Error extracting {article_url}: {e}")
            return None

    def collect_subject(self, subject_name, course_url, grade=None):
        """Collect all articles from a subject"""
        print(f"\n{'='*70}")
        print(f"COLLECTING: {subject_name}")
        if grade:
            print(f"GRADE: {grade}")
        print(f"{'='*70}")

        # Get article list
        article_urls = self.get_course_structure(course_url)

        if not article_urls:
            print("   ⚠️  No articles found")
            return []

        # Collect each article
        articles = []

        for i, url in enumerate(tqdm(article_urls, desc="Extracting articles"), 1):
            article = self.extract_article_content(url)

            if article:
                article['subject'] = subject_name
                article['grade'] = grade
                article['index'] = i
                articles.append(article)

                # Be respectful with rate limiting
                time.sleep(1)

        print(f"\n✅ Collected {len(articles)} articles from {subject_name}")

        # Save subject data
        safe_name = subject_name.replace(' ', '_').replace('/', '-')
        grade_tag = f"_grade_{grade}" if grade else ""
        output_file = self.output_dir / f"{safe_name}{grade_tag}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved: {output_file}")

        return articles

    def collect_all_k12(self):
        """
        Collect all K-12 concept articles
        """
        print("=" * 70)
        print("KHAN ACADEMY K-12 CONCEPT COLLECTION")
        print("=" * 70)

        # Define K-12 curriculum
        # Each entry: (Subject, URL, Grade)
        curriculum = [
            # MATH
            ("Math - Early Math", "https://www.khanacademy.org/math/early-math", "K-2"),
            ("Math - 3rd Grade", "https://www.khanacademy.org/math/cc-third-grade-math", "3"),
            ("Math - 4th Grade", "https://www.khanacademy.org/math/cc-fourth-grade-math", "4"),
            ("Math - 5th Grade", "https://www.khanacademy.org/math/cc-fifth-grade-math", "5"),
            ("Math - 6th Grade", "https://www.khanacademy.org/math/cc-sixth-grade-math", "6"),
            ("Math - 7th Grade", "https://www.khanacademy.org/math/cc-seventh-grade-math", "7"),
            ("Math - 8th Grade", "https://www.khanacademy.org/math/cc-eighth-grade-math", "8"),
            ("Math - Algebra 1", "https://www.khanacademy.org/math/algebra", "9"),
            ("Math - Geometry", "https://www.khanacademy.org/math/geometry", "10"),
            ("Math - Algebra 2", "https://www.khanacademy.org/math/algebra2", "11"),
            ("Math - Precalculus", "https://www.khanacademy.org/math/precalculus", "11-12"),
            ("Math - Calculus", "https://www.khanacademy.org/math/calculus-1", "12"),

            # SCIENCE
            ("Science - Biology", "https://www.khanacademy.org/science/biology", "9-12"),
            ("Science - Chemistry", "https://www.khanacademy.org/science/chemistry", "9-12"),
            ("Science - Physics", "https://www.khanacademy.org/science/physics", "9-12"),
            ("Science - High School Biology", "https://www.khanacademy.org/science/high-school-biology", "9-12"),

            # READING & LANGUAGE ARTS
            ("Reading - 2nd Grade", "https://www.khanacademy.org/ela/cc-2nd-reading-vocab", "2"),
            ("Reading - 3rd Grade", "https://www.khanacademy.org/ela/cc-3rd-reading-vocab", "3"),
            ("Reading - 4th Grade", "https://www.khanacademy.org/ela/cc-4th-reading-vocab", "4"),
            ("Reading - 5th Grade", "https://www.khanacademy.org/ela/cc-5th-reading-vocab", "5"),
            ("Grammar - 2nd Grade", "https://www.khanacademy.org/ela/cc-2nd-writing-language", "2"),
            ("Grammar - 3rd Grade", "https://www.khanacademy.org/ela/cc-3rd-writing-language", "3"),

            # SOCIAL STUDIES
            ("US History", "https://www.khanacademy.org/humanities/us-history", "9-12"),
            ("World History", "https://www.khanacademy.org/humanities/world-history", "9-12"),
        ]

        all_articles = []

        for subject, url, grade in curriculum:
            try:
                articles = self.collect_subject(subject, url, grade)
                all_articles.extend(articles)

                print(f"\n{'='*70}")
                print(f"Progress: {len(all_articles)} total articles collected so far")
                print(f"{'='*70}\n")

            except Exception as e:
                print(f"\n❌ Error collecting {subject}: {e}")
                continue

        # Save combined dataset
        combined_file = self.output_dir / "all_k12_concepts.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("✅ COLLECTION COMPLETE")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Total Articles: {len(all_articles)}")

        # Stats by grade
        by_grade = {}
        for article in all_articles:
            grade = article.get('grade', 'Unknown')
            by_grade[grade] = by_grade.get(grade, 0) + 1

        print(f"\n   By Grade:")
        for grade in sorted(by_grade.keys()):
            print(f"      Grade {grade}: {by_grade[grade]} articles")

        # Stats by subject
        by_subject = {}
        for article in all_articles:
            subject = article.get('subject', 'Unknown').split(' - ')[0]
            by_subject[subject] = by_subject.get(subject, 0) + 1

        print(f"\n   By Subject:")
        for subject, count in sorted(by_subject.items()):
            print(f"      {subject}: {count} articles")

        # Word count stats
        word_counts = [a['word_count'] for a in all_articles]
        print(f"\n   Content Stats:")
        print(f"      Total words: {sum(word_counts):,}")
        print(f"      Avg words/article: {sum(word_counts)/len(word_counts):.0f}")

        print(f"\n💾 Saved combined dataset: {combined_file}")

        return all_articles

    def close(self):
        """Clean up"""
        self.driver.quit()


def main():
    """Main entry point"""
    print("\n⚠️  Requirements:")
    print("   pip install selenium beautifulsoup4 tqdm")
    print("   Chrome/ChromeDriver installed\n")

    try:
        collector = KhanArticleCollector()
        articles = collector.collect_all_k12()
        collector.close()

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install Chrome: https://www.google.com/chrome/")
        print("2. Install ChromeDriver: https://chromedriver.chromium.org/")
        print("3. pip install selenium beautifulsoup4 tqdm")
        return 1


if __name__ == "__main__":
    exit(main())
