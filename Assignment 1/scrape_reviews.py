import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import urllib3

# Suppress SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.flipkart.com/realme-buds-t200-lite-12-4mm-driver-48hrs-playback-ai-enc-dual-device-pairing-bluetooth/product-reviews/itm1e7ce9f953e30"
PARAMS = {
    "pid": "ACCHAFSFESQZQGBH",
    "lid": "LSTACCHAFSFESQZQGBH8DC7KM",
    "marketplace": "FLIPKART"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.flipkart.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

MAX_PAGES = 15
TARGET_REVIEWS = 120
RETRY_DELAY = 3  # increase delay between requests

all_reviews = []

for page in range(1, MAX_PAGES + 1):
    print(f"Scraping page {page}...")




























































    params = PARAMS.copy()
    params["page"] = page

    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=20, verify=False)

    if response.status_code != 200:
        print(f"Failed on page {page}. Status code: {response.status_code}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    # Try multiple selectors for Flipkart reviews
    review_divs = soup.select("div.ZmyHeo div")
    
    if not review_divs:
        # Fallback selectors
        review_divs = soup.select("div._6HHkWL")
    
    if not review_divs:
        review_divs = soup.select("div.reviewContent")
    
    if not review_divs:
        # Generic selector for any div with review-like text
        review_divs = soup.find_all("div", class_=lambda x: x and "review" in x.lower() if x else False)
    
    print(f"  Found {len(review_divs)} potential review elements")

    page_reviews = []
    for div in review_divs:
        text = div.get_text(" ", strip=True)

        # basic filtering to avoid random short junk
        if text and len(text) > 20:
            page_reviews.append(text)

    # Remove duplicates within the page
    page_reviews = list(dict.fromkeys(page_reviews))

    print(f"Found {len(page_reviews)} candidate reviews on page {page}")

    all_reviews.extend(page_reviews)
    all_reviews = list(dict.fromkeys(all_reviews))

    print(f"Collected {len(all_reviews)} total reviews so far")

    if len(all_reviews) >= TARGET_REVIEWS:
        break

    time.sleep(RETRY_DELAY)

df = pd.DataFrame(all_reviews, columns=["review_text"])
df = df[df["review_text"].str.strip() != ""]
df = df.drop_duplicates().reset_index(drop=True)

# keep only first 100+ if you want a cleaner submission
df = df.head(120)

df.to_csv("flipkart_reviews.csv", index=False, encoding="utf-8-sig")

print(f"Saved {len(df)} reviews to flipkart_reviews.csv")
print(df.head())