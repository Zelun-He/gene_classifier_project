# 🧬 Gene Sequence Classification: Oncogenes vs Tumor Suppressors

This project fine-tunes a transformer-based model to classify gene sequences as **oncogenes** or **tumor suppressors** using limited labeled data. Built using Hugging Face Transformers and PyTorch, the model uses a DNABERT-style k-mer tokenizer and supports both custom and pretrained BERT backbones.

---

## 📊 Dataset Format

Each `.tsv` file contains DNA sequences and binary labels:
<sequence> <label>
ATGCATGCATG TSG
CGTAGCTAGCT ONC

- `"TSG"` → Tumor Suppressor (label 0)  
- `"ONC"` → Oncogene (label 1)

---

## ⚙️ Dependencies

Install with:

```bash
pip install transformers datasets scikit-learn
python run_finetune.py \
  --task_name dnaprom \
  --model_type dna \
  --tokenizer_name data/vocab \
  --model_name_or_path bert-base-uncased \
  --do_train --do_eval \
  --data_dir data/processed \
  --output_dir models/output \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 2 \
  --num_train_epochs 4 \
  --learning_rate 2e-5
 Features
✅ Hugging Face Trainer API

✅ Binary classification with custom tokenization

✅ Manual tokenization for Windows compatibility

✅ Evaluation metrics: accuracy, precision, recall, F1
{
  "accuracy": 0.812,
  "precision": 0.800,
  "recall": 0.850,
  "f1": 0.823,
  "eval_loss": 0.423
}
Acknowledgments
Zelun He – Undergraduate Researcher

Dr. Hui Liu – Research Supervisor (Gene Classification)

Model inspiration: DNABERT
Future Work
Fine-tune using real DNABERT weights

Add attention visualization for biological interpretation

Extend to multi-class or multi-label gene function prediction
