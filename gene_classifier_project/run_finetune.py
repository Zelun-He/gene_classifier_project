import argparse
import logging
import os
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers import set_seed
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)

os.environ["WANDB_API_KEY"] = "c5fe3d558faac39797be8d94b941caa1f8f5b54e"
os.environ["WANDB_PROJECT"] = "gene-sequence-classifier"  # 👈 change to your desired project name
os.environ["WANDB_ENTITY"] = "zelunhe"       # 👈 optional, if you're logging under a team/org
os.environ["WANDB_DISABLED"] = "true"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def load_data(data_dir):
    def read_tsv(filepath):
        sequences, labels = [], []
        with open(filepath, 'r') as f:
            for line in f:
                seq, label = line.strip().split('\t')
                sequences.append(seq)
                labels.append(0 if label == "TSG" else 1)
        return {"seq": sequences, "label": labels}

    train_data = read_tsv(os.path.join(data_dir, "train.tsv"))
    test_data = read_tsv(os.path.join(data_dir, "test.tsv"))
    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data)
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="dnaprom")
    parser.add_argument("--model_type", default="dna")
    parser.add_argument("--tokenizer_name", default="data/vocab")
    parser.add_argument("--model_name_or_path", default="dna6")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    logging.info("Loading tokenizer...")
    tokenizer = BertTokenizer(vocab_file=os.path.join(args.tokenizer_name, "vocab.txt"), do_lower_case=False)

    def tokenize_function(examples):
        return tokenizer(examples["seq"], padding="max_length", truncation=True, max_length=args.max_seq_length)

    logging.info("Loading and tokenizing dataset...")
    datasets = load_data(args.data_dir)
    # Manually tokenize to avoid Windows + multiprocessing hangs
    print("🔄 Tokenizing each sequence manually...")
    train_tok = [tokenize_function({"seq": s}) for s in datasets["train"]["seq"]]
    test_tok = [tokenize_function({"seq": s}) for s in datasets["test"]["seq"]]


    tokenized_datasets = DatasetDict({
        "train": Dataset.from_dict({
            "input_ids": [d["input_ids"] for d in train_tok],
            "attention_mask": [d["attention_mask"] for d in train_tok],
            "label": datasets["train"]["label"],
        }),
        "test": Dataset.from_dict({
            "input_ids": [d["input_ids"] for d in test_tok],
            "attention_mask": [d["attention_mask"] for d in test_tok],
            "label": datasets["test"]["label"],
        })
    })




    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_eval=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics  # This enables accuracy, precision, recall, F1
)


    if args.do_train:
        trainer.train()

    if args.do_eval:
        eval_result = trainer.evaluate()
        print("Evaluation results:", eval_result)

if __name__ == "__main__":
    main()