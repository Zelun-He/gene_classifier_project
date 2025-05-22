import pandas as pd
import os
import random
from kmer_tokenizer import kmer_tokenize

# Input paths
label_file = "data/gene_labels.csv"
sequence_dir = "data/raw_sequences"

# Output paths
os.makedirs("data/processed", exist_ok=True)
train_file = "data/processed/train.tsv"
test_file = "data/processed/test.tsv"

# Load labels
labels = pd.read_csv(label_file).set_index("gene").to_dict()["label"]

data = []

# Read all .txt files
for file in os.listdir(sequence_dir):
    if file.endswith(".txt"):
        gene = file.replace(".txt", "")
        if gene in labels:
            with open(os.path.join(sequence_dir, file), "r") as f:
                raw_seq = f.read()
                kmers = kmer_tokenize(raw_seq, k=6)
                data.append((kmers, labels[gene]))

# Shuffle and split
random.shuffle(data)
split = int(len(data) * 0.8)
train = data[:split]
test = data[split:]

# Save .tsv
with open(train_file, "w") as f:
    for seq, label in train:
        f.write(f"{seq}\t{label}\n")

with open(test_file, "w") as f:
    for seq, label in test:
        f.write(f"{seq}\t{label}\n")

print(f"✅ Wrote {len(train)} train and {len(test)} test samples.")
