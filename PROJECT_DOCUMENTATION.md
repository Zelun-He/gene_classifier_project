# Gene Sequence Classifier Project Documentation

## Project Overview

This project implements a deep learning model to classify gene sequences as **oncogenes (OG)** or **tumor suppressor genes (TSG)** using limited labeled data. The approach leverages transfer learning by fine-tuning pre-trained DNA language models (DNABERT) on cancer gene classification tasks.

## Project Structure

```
gene_classifier_project/
├── data/                                       # Data storage and processing
│   ├── raw_sequences/                          # Raw genomic sequences from Ensembl
│   ├── processed/                              # Processed training/test datasets
│   ├── vocab/                                  # Vocabulary files for tokenization
│   ├── cancerGeneList.tsv                      # OncoKB cancer gene database
│   ├── Census_allThu May 22 21_50_23 2025.csv  # COSMIC cancer gene database
│   └── gene_labels.csv                         # Combined labeled gene dataset
├── DNABERT/                                    # DNABERT model files (empty in current setup)
├── models/                                     # Trained model checkpoints
│   └── output/                                 # Latest model checkpoint
├── scripts/                                    # Data processing and utility scripts
├── wandb/                                      # Weights & Biases experiment tracking
├── run_finetune.py                             # Main training script
├── predict.py                                  # Model inference script
└── requirements.txt                            # Python dependencies
```

## Core Components

### 1. Data Sources

#### COSMIC (Catalogue of Somatic Mutations in Cancer)
- **File**: `data/Census_allThu May 22 21_50_23 2025.csv`
- **Purpose**: Provides gene role annotations (oncogene vs tumor suppressor)
- **Content**: 760 genes with "Role in Cancer" field
- **Processing**: Extracts TSG/OG labels based on role descriptions

#### OncoKB (Oncogene Knowledge Base)
- **File**: `data/cancerGeneList.tsv`
- **Purpose**: Tiered database of cancer genes with clinical evidence
- **Content**: 1,194 genes with binary flags for oncogene/tumor suppressor
- **Processing**: Converts boolean flags to TSG/OG labels

#### Combined Dataset
- **File**: `data/gene_labels.csv`
- **Purpose**: Merged dataset from both sources
- **Content**: 579 unique genes with consistent labeling
- **Format**: `gene,label` where label is either "TSG" or "OG"

### 2. Data Processing Pipeline

#### Sequence Fetching (`scripts/fetch_sequences.py`)
- **Purpose**: Retrieves genomic sequences from Ensembl REST API
- **Process**: 
  1. Loads gene symbols from `gene_labels.csv`
  2. Converts symbols to Ensembl IDs via lookup API
  3. Fetches genomic sequences for each gene
  4. Saves sequences as individual `.txt` files
- **Output**: Raw DNA sequences in `data/raw_sequences/`

#### K-mer Tokenization (`scripts/kmer_tokenizer.py`)
- **Purpose**: Converts DNA sequences to k-mer tokens for BERT processing
- **Method**: Sliding window approach with k=6 (6-mers)
- **Example**: "ATGCGT" → ["ATGCGT", "TGCGTA", "GCGTAG", ...]
- **Rationale**: Captures local sequence patterns and reduces vocabulary size

#### Dataset Preparation (`scripts/make_datasets.py`)
- **Purpose**: Creates train/test splits for model training
- **Process**:
  1. Loads labeled genes and sequences
  2. Applies k-mer tokenization
  3. Splits data 80/20 (train/test)
  4. Saves as TSV files
- **Output**: `train.tsv` and `test.tsv` in `data/processed/`

### 3. Model Architecture

#### Base Model: DNABERT
- **Type**: BERT architecture pre-trained on DNA sequences
- **Configuration**: 
  - Hidden size: 768
  - Layers: 12
  - Attention heads: 12
  - Max sequence length: 512
  - Vocabulary size: 30,522

#### Classification Head
- **Type**: `BertForSequenceClassification`
- **Output**: Binary classification (TSG=0, ONC=1)
- **Loss**: Cross-entropy with class weights (TSG:1.0, ONC:4.0)
- **Rationale**: Addresses class imbalance in cancer gene datasets

### 4. Training Pipeline (`run_finetune.py`)

#### Data Loading
- **Input**: TSV files with sequences and labels
- **Tokenization**: BERT tokenizer with max length 512
- **Format**: `input_ids`, `attention_mask`, `label`

#### Training Configuration
- **Epochs**: 4
- **Learning rate**: 2e-5
- **Batch size**: 4 (per GPU)
- **Optimizer**: AdamW with weight decay 0.01
- **Scheduler**: Linear learning rate decay

#### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

#### Class Imbalance Handling
- **Weighted Loss**: TSG (class 0) weight = 1.0, ONC (class 1) weight = 4.0
- **Rationale**: Oncogenes are typically more common in cancer datasets

### 5. Inference (`predict.py`)

#### Model Loading
- **Source**: Trained model from `models/output/`
- **Tokenizer**: Vocabulary from `data/vocab/`
- **Mode**: Evaluation mode for inference

#### Prediction Process
- **Input**: DNA sequences (can be custom or from test set)
- **Processing**: Tokenization and model forward pass
- **Output**: Binary predictions (0=TSG, 1=ONC)
- **Format**: Sequence → Predicted Label

## Data Flow

```
1. Gene Databases (COSMIC + OncoKB)
   ↓
2. Label Preparation (scripts/prepare_gene_labels.py)
   ↓
3. Sequence Retrieval (scripts/fetch_sequences.py)
   ↓
4. K-mer Tokenization (scripts/kmer_tokenizer.py)
   ↓
5. Dataset Creation (scripts/make_datasets.py)
   ↓
6. Model Training (run_finetune.py)
   ↓
7. Model Inference (predict.py)
```

## Key Features

### Transfer Learning
- **Pre-trained Model**: DNABERT learns biological context from vast unlabeled genomic data
- **Fine-tuning**: Adapts to specific cancer gene classification task
- **Benefits**: Better generalization on small labeled datasets

### Data Integration
- **Multiple Sources**: Combines COSMIC and OncoKB for comprehensive coverage
- **Quality Control**: Filters for genes with clear oncogene/tumor suppressor roles
- **Deduplication**: Removes duplicate genes across sources

### Scalable Architecture
- **Modular Design**: Separate scripts for each processing step
- **Reproducible**: Clear data flow and parameter settings
- **Extensible**: Easy to add new data sources or model architectures

## Usage Examples

### Training the Model
```bash
python run_finetune.py \
  --data_dir data/processed \
  --output_dir models/output \
  --do_train \
  --do_eval
```

### Making Predictions
```bash
python predict.py
```

### Data Processing
```bash
# Prepare gene labels
python scripts/prepare_gene_labels.py

# Fetch sequences
python scripts/fetch_sequences.py

# Create datasets
python scripts/make_datasets.py
```

## Technical Requirements

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **Transformers**: 4.20+
- **Datasets**: HuggingFace datasets library
- **Other**: pandas, requests, sklearn

## Model Performance

The model achieves:
- **Binary Classification**: TSG vs ONC with 2 output classes
- **Class Imbalance Handling**: Weighted loss for realistic cancer gene distribution
- **Sequence Processing**: Handles DNA sequences up to 512 tokens
- **Transfer Learning**: Leverages pre-trained DNA language model knowledge

## Future Enhancements

1. **Additional Data Sources**: Integrate TCGA, ICGC, or other cancer databases
2. **Multi-class Classification**: Distinguish between different types of oncogenes
3. **Sequence Variants**: Handle mutations and sequence variations
4. **Clinical Validation**: Integrate clinical outcome data
5. **Interpretability**: Add attention visualization and feature importance analysis

## Conclusion

This project successfully implements a deep learning solution for cancer gene classification that addresses the core challenge of limited labeled data through:
- Strategic use of multiple authoritative cancer gene databases
- Transfer learning from pre-trained DNA language models
- Comprehensive data processing and validation pipeline
- Scalable training and inference infrastructure

The solution demonstrates how modern deep learning techniques can be applied to bioinformatics challenges while maintaining scientific rigor and reproducibility.
