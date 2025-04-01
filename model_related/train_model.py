import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

# LOAD DATA

COLUMN_NAMES = [
    "id", "label", "statement", "subject", "speaker", "speaker_job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
    "context"
]

train = pd.read_csv("data/train.tsv", sep="\t", names=COLUMN_NAMES)
valid = pd.read_csv("data/valid.tsv", sep="\t", names=COLUMN_NAMES)
test = pd.read_csv("data/test.tsv", sep="\t", names=COLUMN_NAMES)

# Combine for vectorization
combined = pd.concat([train, valid])
texts = combined["statement"].astype(str)
labels = combined["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(texts)
y = labels

# Split back into train/val
X_train = X[:len(train)]
y_train = train["label"]
X_valid = X[len(train):]
y_valid = valid["label"]

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, preds))
print(classification_report(y_valid, preds))

# Save model
joblib.dump(clf, "models/logistic_regression/scam_detector_model.pkl")
joblib.dump(vectorizer, "models/logistic_regression/tfidf_vectorizer.pkl")

# EXPLAINABILITY - SHAP
explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_valid[:100])
feature_names = vectorizer.get_feature_names_out()

# Visualize explanation for first sample
shap.initjs()
print(np.shape(shap_values))  # should show (samples, features, classes)
shap.plots.force(
    explainer.expected_value[0],   # class 0 base value
    shap_values[0][:, 0]           # SHAP values for class 0
)
plt.savefig("shap_explanation.png")
print("Explanation saved as shap_explanation.png")
