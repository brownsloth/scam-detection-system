# backend/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "../model_related/distilbert_fakenews_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label  # should already be set during training

@app.get("/explain")
def explain(text: str = Query(...)):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        predicted_class = torch.argmax(probs).item()
        label = id2label[predicted_class]

    return {
        "input": text,
        "explanation": [[label, round(probs[predicted_class].item(), 4)]]
    }
