import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None)
    df.columns = [
        "id", "label", "statement", "subject", "speaker", "job", "state", "party",
        "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
        "context"
    ]
    df['text'] = df['statement'].astype(str) + " " + df['context'].astype(str)
    return df[["text", "label"]]

print("📂 Loading datasets...")
train_df = load_data("data/train.tsv")
val_df = load_data("data/valid.tsv")

# Encode labels
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["label"])
val_df["label"] = le.transform(val_df["label"])
label2id = {label: i for i, label in enumerate(le.classes_)}
id2label = {i: label for label, i in label2id.items()}

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)
train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Trainer setup
training_args = TrainingArguments(
    output_dir="./distilbert_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🚀 Training...")
trainer.train()
trainer.save_model("distilbert_fakenews_model")
print("✅ Done training and saved model!")
