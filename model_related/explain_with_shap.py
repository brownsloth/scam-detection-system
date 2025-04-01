# explain_with_shap.py
import shap
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load model and tokenizer
model_path = "distilbert_fakenews_model"  # or wherever you saved it
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Wrapper function to make model SHAP-compatible
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.numpy()

# SHAP explanation function
def explain_with_shap(text, num_features=10):
    explainer = shap.Explainer(predict_proba, tokenizer)
    shap_values = explainer([text])

    tokens = shap_values.data[0]
    scores = shap_values.values[0].mean(axis=0)  # average across classes if multiclass

    # Zip together and sort
    token_scores = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)

    explanation = [{"word": tok, "score": round(score, 4)} for tok, score in token_scores[:num_features]]
    return explanation
