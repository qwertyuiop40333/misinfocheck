import os
from joblib import load

# ======================
# Paths to saved models
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root
MODELS_DIR = os.path.join(BASE_DIR, "models")

VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "logreg_model.joblib")

# ======================
# Load vectorizer + trained logistic regression model
# ======================
try:
    vectorizer = load(VEC_PATH)
    logreg_model = load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"❌ Could not load trained model/vectorizer. Make sure you ran train.py first.\n{e}"
    )

# ======================
# Logistic Wrapper
# ======================
class LogisticWrapper:
    """Wrapper so we can call predict() and predict_proba() easily"""
    def __init__(self, model, vec):   # ✅ fixed here
        self.model = model
        self.vec = vec

    def predict(self, text: str):
        Xv = self.vec.transform([text])
        return self.model.predict(Xv)[0]

    def predict_proba(self, text: str):
        Xv = self.vec.transform([text])
        return self.model.predict_proba(Xv)[0]

# Initialize wrapped model
logreg_wrapper = LogisticWrapper(logreg_model, vectorizer)

# ======================
# Ensemble Prediction
# ======================
def ensemble_predict(claim: str):
    """
    Predict claim using trained logistic regression model.
    Returns:
        (label, confidence)
        label -> 0 (Fake), 1 (Real)
        confidence -> probability score
    """
    pred = logreg_wrapper.predict(claim)
    prob = max(logreg_wrapper.predict_proba(claim))
    return pred, prob

# ======================
# Demo run
# ======================
if __name__ == "__main__":
    test_claims = [
        "NASA confirms water on the moon surface",
        "Aliens landed in Kolkata yesterday night",
    ]
    for claim in test_claims:
        label, confidence = ensemble_predict(claim)
        print(f"📰 Claim: {claim}")
        print(f"   → Prediction: {'Real ✅' if label==1 else 'Fake ❌'} (confidence={confidence:.2f})\n")
