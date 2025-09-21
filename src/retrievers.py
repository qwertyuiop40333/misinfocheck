import os
import requests
from dotenv import load_dotenv
from utils import is_credible, domain_of

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def serpapi_search(query: str, engine="google"):
    """Generic SerpAPI search wrapper."""
    if not SERPAPI_KEY:
        return {}
    try:
        url = "https://serpapi.com/search"
        params = {"engine": engine, "q": query, "api_key": SERPAPI_KEY}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print("[SerpAPI] error:", e)
    return {}

def classify_source_verdict(text: str) -> str:
    """Return True / False / Uncertain based on snippet text."""
    text = text.lower()
    if any(word in text for word in ["fake", "false", "hoax", "myth", "scam", "debunked", "denies"]):
        return "False"
    elif any(word in text for word in ["true", "confirmed", "proven", "verified", "real", "official"]):
        return "True"
    else:
        return "Uncertain"

def vote_on_claim(claim: str):
    votes = {"True": 0, "False": 0, "Uncertain": 0}
    sources_checked = []

    # ---- Google Search ----
    results_web = serpapi_search(claim, engine="google")
    if "organic_results" in results_web:
        for res in results_web["organic_results"][:8]:
            link = res.get("link", "")
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            if not is_credible(link):
                continue
            verdict = classify_source_verdict(title + " " + snippet)
            votes[verdict] += 1
            sources_checked.append(domain_of(link))

    # ---- Google News ----
    results_news = serpapi_search(claim, engine="google_news")
    if "news_results" in results_news:
        for res in results_news["news_results"][:5]:
            link = res.get("link", "")
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            if not is_credible(link):
                continue
            verdict = classify_source_verdict(title + " " + snippet)
            votes[verdict] += 1
            sources_checked.append(domain_of(link))

    # ---- Wikipedia Direct ----
    results_wiki = serpapi_search(claim + " site:wikipedia.org", engine="google")
    if "organic_results" in results_wiki and results_wiki["organic_results"]:
        snippet = results_wiki["organic_results"][0].get("snippet", "")
        verdict = classify_source_verdict(snippet)
        votes[verdict] += 1
        sources_checked.append("wikipedia.org")

    # ---- Final Verdict ----
    if votes["True"] > votes["False"]:
        verdict = "Fact: TRUE ✅"
    elif votes["False"] > votes["True"]:
        verdict = "Fact: FALSE ❌"
    else:
        verdict = "Uncertain ⚠️"

    total_votes = votes["True"] + votes["False"]
    confidence = round((max(votes["True"], votes["False"]) / total_votes) * 100, 2) if total_votes > 0 else 0

    return {
        "label": verdict,
        "confidence": f"{confidence}%",
        "votes": votes,
        "sources": list(set(sources_checked))
    }
