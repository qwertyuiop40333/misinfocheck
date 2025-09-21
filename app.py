import os
import json
import requests
import joblib
import tldextract
from flask import Flask, render_template, request
from dotenv import load_dotenv

# Load API key
load_dotenv()
SERP_API_KEY = os.getenv("SERPAPI_KEY")

app = Flask(__name__)
HISTORY_FILE = "history.json"

# ------------------------------
# Load ML model + vectorizer
# ------------------------------
MODEL_PATH = "models/logreg_model.joblib"
VEC_PATH = "models/tfidf_vectorizer.joblib"
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# ------------------------------
# Helper: load & save history
# ------------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ------------------------------
# Domain Extractor
# ------------------------------
def domain_of(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else url

# ------------------------------
# SerpAPI Helper
# ------------------------------
def serpapi_search(query, engine="google"):
    url = "https://serpapi.com/search.json"
    params = {"engine": engine, "q": query, "api_key": SERP_API_KEY}
    response = requests.get(url, params=params)
    return response.json()

# ------------------------------
# Classifier for snippets
# ------------------------------
def classify_source_verdict(text: str) -> str:
    """Return True / False / Uncertain based on snippet text."""
    text = text.lower()

    # Strong negative cues
    neg_keywords = ["fake", "false", "hoax", "myth", "scam", "debunked", "not true", "incorrect", "wrong", "rumor"]
    if any(kw in text for kw in neg_keywords):
        return "False"

    # Strong positive cues
    pos_keywords = ["confirmed", "proven", "verified", "fact check: true", "is true", "accurate"]
    if any(kw in text for kw in pos_keywords):
        return "True"

    # Otherwise uncertain
    return "Uncertain"

# ------------------------------
# Voting Logic
# ------------------------------
def vote_on_claim(claim):
    votes = {"True": 0, "False": 0, "Uncertain": 0}
    sources_checked = []

    # ---- Google Search ----
    results_web = serpapi_search(claim, engine="google")
    if "organic_results" in results_web:
        for res in results_web["organic_results"][:5]:
            link = res.get("link", "")
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            domain = domain_of(link)

            verdict = classify_source_verdict(title + " " + snippet)
            votes[verdict] += 1
            sources_checked.append(domain)

    # ---- Google News ----
    results_news = serpapi_search(claim, engine="google_news")
    if "news_results" in results_news:
        for res in results_news["news_results"][:5]:
            link = res.get("link", "")
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            domain = domain_of(link)

            verdict = classify_source_verdict(title + " " + snippet)
            votes[verdict] += 1
            sources_checked.append(domain)

    # ---- Wikipedia ----
    results_wiki = serpapi_search(claim + " site:wikipedia.org", engine="google")
    if "organic_results" in results_wiki and results_wiki["organic_results"]:
        snippet = results_wiki["organic_results"][0].get("snippet", "")
        verdict = classify_source_verdict(snippet)
        votes[verdict] += 1
        sources_checked.append("Wikipedia")

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
        "confidence": f"{confidence}%" if confidence > 0 else "0%",
        "votes": votes,
        "sources": list(set(sources_checked))
    }

# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_style = None
    sms_check = None
    history = load_history()

    if request.method == "POST":
        if "claim" in request.form and request.form["claim"].strip():
            claim = request.form["claim"].strip()
            prediction_style = vote_on_claim(claim)

            history.append({
                "type": "News Claim",
                "text": claim,
                "result": prediction_style["label"],
                "confidence": prediction_style["confidence"],
                "sources": prediction_style["sources"]
            })
            save_history(history)

        elif "sms" in request.form and request.form["sms"].strip():
            sms = request.form["sms"].strip().lower()
            if "otp" in sms or "click" in sms or "link" in sms or "account blocked" in sms:
                sms_check = "⚠️ Fraudulent SMS Detected"
            else:
                sms_check = "✅ SMS seems Safe"

            history.append({
                "type": "Bank SMS",
                "text": sms,
                "result": sms_check,
                "confidence": "-"
            })
            save_history(history)

    return render_template("index.html", prediction_style=prediction_style, sms_check=sms_check)

@app.route("/history")
def show_history():
    history = load_history()
    return render_template("history.html", history=history)

if __name__ == "__main__":
    app.run(debug=True)
