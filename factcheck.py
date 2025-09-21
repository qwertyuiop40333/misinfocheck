import os
import requests
from dotenv import load_dotenv
from collections import Counter
from utils import clean_text, is_credible

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def fact_check_with_serp(claim: str, num_results: int = 5):
    """
    Cross-check claim against SERP API (Google News/Wikipedia).
    Returns majority vote + confidence.
    """
    if not SERPAPI_KEY:
        return {"label": "Unverifiable (no API key)", "confidence": 0.0}

    url = "https://serpapi.com/search"
    params = {
        "engine": "google_news",
        "q": claim,
        "api_key": SERPAPI_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return {"label": "Unverifiable", "confidence": 0.0}

        data = resp.json()
        articles = data.get("news_results", [])

        votes = []
        for a in articles[:num_results]:
            link = a.get("link", "")
            snippet = clean_text(a.get("snippet", ""))

            if not snippet:
                continue
            if not is_credible(link):
                continue

            # Simple heuristics: "false/hoax/not true" = vote False
            text_low = snippet.lower()
            if any(x in text_low for x in ["false", "hoax", "not true", "denied"]):
                votes.append(False)
            elif any(x in text_low for x in ["confirmed", "true", "announced", "official"]):
                votes.append(True)
            else:
                # default: weak positive vote
                votes.append(True)

        if not votes:
            return {"label": "Unverifiable", "confidence": 0.0}

        majority = Counter(votes).most_common(1)[0]
        return {
            "label": "Real ✅" if majority[0] else "Fake ❌",
            "confidence": majority[1] / len(votes)
        }

    except Exception as e:
        print("[FactCheck] Error:", e)
        return {"label": "Error", "confidence": 0.0}
