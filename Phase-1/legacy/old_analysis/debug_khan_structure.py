"""
Debug Khan Academy structure
Find out what types of content exist and how to identify them
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time


def analyze_page_structure(url):
    """Analyze a Khan Academy page to understand its structure"""

    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')

    print(f"\n🔍 Analyzing: {url}")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        time.sleep(5)  # Wait longer for React

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find ALL links
        all_links = soup.find_all('a', href=True)

        print(f"\n📊 Total links found: {len(all_links)}")

        # Categorize links
        link_types = {
            'article': [],
            'video': [],
            'exercise': [],
            'other': []
        }

        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)

            if not href:
                continue

            # Categorize
            if '/a/' in href:
                link_types['article'].append((href, text))
            elif '/v/' in href:
                link_types['video'].append((href, text))
            elif '/e/' in href:
                link_types['exercise'].append((href, text))
            else:
                if any(keyword in href for keyword in ['/math/', '/science/', '/ela/']):
                    link_types['other'].append((href, text))

        # Print results
        print(f"\n📝 Content types found:")
        print(f"   Articles (/a/): {len(link_types['article'])}")
        print(f"   Videos (/v/): {len(link_types['video'])}")
        print(f"   Exercises (/e/): {len(link_types['exercise'])}")
        print(f"   Other: {len(link_types['other'])}")

        # Show samples
        if link_types['article']:
            print(f"\n✅ Sample articles:")
            for href, text in link_types['article'][:5]:
                print(f"   - {text[:50]}")
                print(f"     {href}")

        if not link_types['article'] and link_types['video']:
            print(f"\n⚠️  No articles found, but {len(link_types['video'])} videos exist")
            print(f"   This course might be video-only")

        if not link_types['article'] and not link_types['video']:
            print(f"\n🔍 Links found (sample):")
            for href, text in list(link_types['other'])[:10]:
                print(f"   - {text[:50]}")
                print(f"     {href}")

        return link_types

    finally:
        driver.quit()


def main():
    """Test multiple Khan Academy pages"""

    test_urls = [
        "https://www.khanacademy.org/math/early-math",
        "https://www.khanacademy.org/math/cc-third-grade-math",
        "https://www.khanacademy.org/math/cc-fourth-grade-math",
        "https://www.khanacademy.org/science/biology",
    ]

    print("=" * 70)
    print("KHAN ACADEMY STRUCTURE ANALYSIS")
    print("=" * 70)

    for url in test_urls:
        link_types = analyze_page_structure(url)
        print("\n" + "=" * 70)
        time.sleep(2)  # Be respectful

    print("\n💡 Recommendations:")
    print("   1. If articles exist: Use article collector")
    print("   2. If only videos: Extract YouTube transcripts")
    print("   3. If mixed: Collect both articles and transcripts")


if __name__ == "__main__":
    main()
