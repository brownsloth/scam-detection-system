import shap
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from shap.maskers import Text

# Load model and tokenizer
model_path = "../model_related/distilbert_fakenews_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Prediction function for SHAP
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        raise ValueError("Input must be string or list of strings.")

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

# SHAP explanation function
def explain_with_shap(text, num_features=10):
    masker = Text(tokenizer)  # SHAP's built-in text masker
    explainer = shap.Explainer(predict_proba, masker)

    # Get SHAP values
    shap_values = explainer([text])

    tokens = shap_values.data[0]  # actual tokens from tokenizer
    scores = shap_values.values[0].mean(axis=0)

    # Combine and sort by importance
    token_scores = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)
    explanation = [{"word": tok, "score": round(score, 4)} for tok, score in token_scores[:num_features]]

    return explanation
