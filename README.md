# AutoBERT

This repository contains the implementation of our paper **Efficient Cross-Architecture Binary Function Embeddings through Knowledge Distillation**.

## News

- [2025/04/10] We publish the code and models used during our research

## Requirements

- Python 3.11+
- transformers
- sentence_transformers
- CUDA for optimal performance
- Ghidra 11.x

## Models

The pre-trained models (trained on the EKLAVYA and MISA datasets) are available 
on [Google Drive (696 MB)](https://drive.google.com/file/d/1QI7MQnSOvMCOSHrfHUDXN2u8c_7BupEJ/view?usp=sharing).

## Datasets

We used the [EKLAVYA](datasets/EKLAVYA/README.md) and [MISA](datasets/MISA/README.md) 
for our pre-trained models. To train a new model on a custom dataset, the dataset 
should be aligned in the following way:

### Anchor Model
For the pre-training task, use Masked language modeling (MLM) on raw assembly listings.
Prepare a [huggingface dataset](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset)
and start the training at `01_tokenizer.py`.

Start the training at step `02_mlm.py`. Configure your custom paths in the `config`-dictionary.

### Binary Code Similarity Classification (BCSD) Task
For the BCSD downstream task, we used the Triplet Loss for the training process 
(refer to Section 3.5 in the paper). A custom dataset needs to contain the following 
columns:

- sample1
- sample2
- label (e.g. 1.0 for similar samples, 0.0 for dissimilar samples)

Start the training at step `03_sbert.py`. Configure your custom paths in the `config`-dictionary.

### Custom Architecture
To add support for a custom processor architecture, prepare a dataset that aligns
pairs of (anchor, target) assembly code listings (i.e. a [ParallelSentencesDataset](https://www.sbert.net/docs/package_reference/sentence_transformer/datasets.html#parallelsentencesdataset)). If you want to use 
the provided anchor model `bert-x86-eklavya`, the anchor samples should be for the
x86_64 architecture.

Start the training at step `04_mlm.py`. Configure your custom paths in the `config`-dictionary.

## Usage

This guide provides a quick overview of how to use the scripts in the AutoBERT repository.

### 1. Tokenizer Creation (`01_tokenizer.py`)
Build a custom tokenizer from a corpus of assembly code.

```bash
python  -c <path_to_corpus_file(s)> -o <output_path> -e <encoding>
```

- -c: Path(s) to corpus file(s) (multiple allowed).
- -o: Directory to save the tokenizer.
- -e: Encoding of the corpus files (default: utf-8).

### 2. Masked Language Model Training (`02_mlm.py`)
Train a BERT model with masked language modeling on assembly code.

```bash
python 02_mlm.py
```

Configure paths and parameters in the config section of the script:

- tokenizer: Path to the tokenizer created in step 1.
- model_save_path: Directory to save the trained model.

### 3. SBERT Training (03_sbert.py)
Train a Sentence-BERT model for generating embeddings of assembly functions.

```bash
python 03_sbert.py
```

Configure paths and parameters in the config section:

- model_load_path: Path to the pre-trained BERT model.
- model_save_path: Directory to save the trained SBERT model.

### 4. Knowledge Distillation with MSE Loss (04_mse.py)

Distill knowledge from a teacher model to a student model using MSE loss.

```bash
python 04_mse.py
```

Configure paths and parameters in the config section:

- teacher_model: Path to the teacher model.
- student_model: Directory to save the student model.

### 5. Evaluation (05_evaluation.py)

Evaluate models using triplet accuracy and binary classification metrics.

```bash
python 05_evaluation.py
```

Configure paths and parameters in the config section:

- models: List of model paths to evaluate.
- eklavya and misa: Paths to datasets.
