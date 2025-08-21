import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset

tokenizer = BertTokenizer(vocab_file="data/vocab/vocab.txt", do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("models/output/checkpoint-916")
model.eval()

sequences = ["ATGCGTAGCTA", "GCTAGCTAGCT"]  # Replace with your sequences
inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

for seq, pred in zip(sequences, predictions):
    label = "TSG" if pred == 0 else "ONC"
    print(f"{seq} → {label}")
