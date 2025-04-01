import pandas as pd

COLUMN_NAMES = [
    "id", "label", "statement", "subject", "speaker", "speaker_job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
    "context"
]

train_df = pd.read_csv("data/train.tsv", sep="\t", names=COLUMN_NAMES)
valid_df = pd.read_csv("data/valid.tsv", sep="\t", names=COLUMN_NAMES)
test_df = pd.read_csv("data/test.tsv", sep="\t", names=COLUMN_NAMES)

# Confirm
print(train_df.head())
