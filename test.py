import pandas as pd
import joblib
import re
import os

# ======================
# Load Model + Vectorizer
# ======================
MODEL_PATH = "models/logreg_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully!")
else:
    raise FileNotFoundError("❌ Model or vectorizer not found! Run train.py first.")

# ======================
# Load Test Data (Optional)
# ======================
X_test, y_test = None, None
try:
    X_test = pd.read_csv("data/X_test.csv")['text']
    y_test = pd.read_csv("data/y_test.csv")['label']
    print("✅ Test data loaded successfully!")
except Exception as e:
    print("⚠️ Test data not found, skipping...")

# ======================
# Fake News Detection
# ======================
def detect_fake_news(text: str) -> str:
    """Predict if a given text is fake or real news."""
    if not text.strip():
        return "⚠️ Empty input provided!"
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "📰 FAKE News" if prediction == 1 else "✅ REAL News"

# ======================
# Bank Fraud SMS/Call Detection
# ======================
FRAUD_KEYWORDS = {
    "otp", "kyc", "blocked", "debit", "credit", "urgent",
    "click link", "verify", "update", "account suspended",
    "password", "cvv", "bank call", "loan offer", "refund",
    "free gift", "congratulations", "lottery", "investment scheme"
}

def detect_bank_fraud(message: str) -> str:
    """Detect if an SMS/Call message is likely a bank fraud attempt."""
    if not message.strip():
        return "⚠️ Empty input provided!"
    msg = message.lower()
    for word in FRAUD_KEYWORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", msg):
            return "🚨 Potential Bank Fraud Detected!"
    return "✅ Safe Message"

# ======================
# Demo (Run only when executed directly)
# ======================
if __name__ == "__main__":
    print("\n===== DEMO TESTS =====\n")

    # Fake news example
    news_example = "Breaking: RBI announces new currency ban starting tomorrow!"
    print("News:", news_example)
    print("Prediction:", detect_fake_news(news_example))

    # Real news example
    news_example2 = "The Indian Space Research Organisation launched a new satellite successfully."
    print("\nNews:", news_example2)
    print("Prediction:", detect_fake_news(news_example2))

    # Fraud SMS example
    fraud_msg = "Your bank account is blocked. Update KYC immediately by clicking this link."
    print("\nBank SMS:", fraud_msg)
    print("Prediction:", detect_bank_fraud(fraud_msg))

    # Safe SMS example
    safe_msg = "Dear customer, your monthly bank statement is available on the official app."
    print("\nBank SMS:", safe_msg)
    print("Prediction:", detect_bank_fraud(safe_msg))
