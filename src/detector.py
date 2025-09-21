import os
import joblib

# ======================
# Setup Paths
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root
MODEL_DIR = os.path.join(BASE_DIR, "models")

FAKE_NEWS_MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")   # optional
LOGREG_MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# ======================
# Load Models
# ======================
fake_news_model = None
if os.path.exists(FAKE_NEWS_MODEL_PATH):
    fake_news_model = joblib.load(FAKE_NEWS_MODEL_PATH)

logreg_model = joblib.load(LOGREG_MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

print("✅ Models loaded successfully!")

# ======================
# Fake News Detection
# ======================
def detect_fake_news(text: str) -> str:
    if fake_news_model is None:
        return "⚠️ Fake News model not available."
    
    transformed = tfidf_vectorizer.transform([text])
    prediction = fake_news_model.predict(transformed)[0]
    return "❌ Fake News Detected" if prediction == 1 else "✅ Real News"

# ======================
# Bank/SMS Fraud Detection
# (Rule-based for now, can extend with APIs later)
# ======================
def detect_sms_fraud(sms_text: str) -> str:
    suspicious_keywords = [
        "OTP", "lottery", "prize", "click here",
        "kyc", "account blocked", "verify account", "suspended",
        "password reset", "transaction declined", "unauthorized access"
    ]
    is_fraud = any(word.lower() in sms_text.lower() for word in suspicious_keywords)
    return "⚠️ Fraudulent SMS Detected!" if is_fraud else "✅ Safe SMS"


# ======================
# Quick Test
# ======================
if __name__ == "__main__":
    print(detect_fake_news("Breaking news: Scientists discover water on Mars!"))
    print(detect_sms_fraud("Your account has been suspended, click here to verify"))
