import pandas as pd
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
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
test_df = load_data("data/test.tsv")

# Encode the labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])
val_df["label"] = label_encoder.transform(val_df["label"])
test_df["label"] = label_encoder.transform(test_df["label"])
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
id2label = {i: label for label, i in label2id.items()}

# Tokenizer and model
print("🔤 Tokenizing...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load model
print("📦 Loading model...")
model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=len(label2id))
model.config.id2label = id2label
model.config.label2id = label2id

# Training arguments
training_args = TrainingArguments(
    output_dir="./bart_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    max_steps=1000,  # Or any small number to stop early
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Metric function
import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Train
print("🚀 Training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("bart_fakenews_model")
print("✅ Model saved to 'bart_fakenews_model'")
