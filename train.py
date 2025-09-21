import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ======================
# Load Dataset
# ======================
DATA_PATH = "data/WELFake_Dataset.csv"
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully!")

# Ensure required columns
if not {"title", "text", "label"}.issubset(df.columns):
    raise ValueError("❌ Dataset must contain 'title', 'text', and 'label' columns!")

# Drop NA values and reset index
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

# Combine title + text for richer input
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

X = df["content"].astype(str)   # input text
y = df["label"].astype(int)     # target labels (0 = Real, 1 = Fake)

# ======================
# Train/Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# Vectorization
# ======================
print("🔄 Vectorizing text... (this may take a while on WELFake)")

vectorizer = TfidfVectorizer(
    max_features=5000,   # cap vocab size → prevents long hangs
    ngram_range=(1, 2),  # unigrams + bigrams
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train.tolist())
X_test_tfidf = vectorizer.transform(X_test.tolist())

print("✅ Vectorization complete!")

# ======================
# Model Training
# ======================
print("🔄 Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=200,
    n_jobs=-1,
    solver="saga"
)
model.fit(X_train_tfidf, y_train)
print("✅ Model trained successfully!")

# ======================
# Save Model + Vectorizer
# ======================
joblib.dump(model, "models/logreg_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("💾 Model + vectorizer saved!")

# ======================
# Evaluate
# ======================
acc = model.score(X_test_tfidf, y_test)
print(f"📊 Test Accuracy: {acc:.4f}")

# Save test split for later use
pd.DataFrame({"text": X_test, "label": y_test}).to_csv("data/X_test.csv", index=False)
pd.DataFrame({"label": y_test}).to_csv("data/y_test.csv", index=False)
print("💾 Test split saved!")
