import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load model and tokenizer
model_path = "distilbert_fakenews_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model.eval()

# Label map
id2label = model.config.id2label

# Prediction wrapper for LIME
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.numpy()

# Initialize LIME
explainer = LimeTextExplainer(class_names=[id2label[i] for i in sorted(id2label)], split_expression=r"\W+", mask_string="[MASK]")

# Example input
text = "The senator claimed the earth is flat during the press conference."

# Explain
explanation = explainer.explain_instance(
    text, predict_proba, num_features=10, num_samples=5000
)

# Visualize in terminal
print(f"\n🧠 LIME Explanation for: '{text}'")
for word, weight in explanation.as_list():
    print(f"{word:20} : {weight:.3f}")

# Optionally, show HTML visualization
# explanation.show_in_browser()
