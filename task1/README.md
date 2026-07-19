# Task 1: Mountain Named Entity Recognition (NER)

## Project overview
This project implements a Named Entity Recognition (NER) system for detecting mountain names using a fine-tuned BERT model.

The project includes:

- automatic collection of real mountain names;
- synthetic dataset generation with BIO annotations;
- BERT fine-tuning for NER;
- command-line inference;
- Jupyter notebook demonstrations.


## Features

- Real mountain names collected from the Open Peaks dataset.
- Automatic BIO annotation generation.
- Positive and negative training examples.
- Fine-tuned BERT model for mountain recognition.
- Command-line inference script.
- Interactive Jupyter notebook demo.
- Fully reproducible dataset generation pipeline.


## Project Structure

```bash
task1/
│
├── data/
│   ├── raw/
│   │   └── mountains.csv     # Collected mountain names
│   └── processed/
│       └──ner_dataset.json   # Generated BIO-annotated dataset
│ 
├── models/
│   └── bert_ner/             # Pretrained BERT model (not included)
│
├── notebooks/
│   ├── dataset_creation.ipynb # Dataset creation pipeline
│   └── demo.ipynb             # Pipeline demonstration
│
├── src/
│   ├── config.py             # Project configuration
│   ├── preprocessing.py      # Dataset preprocessing
│   ├── download_mountains.py # Download mountain names
│   ├── dataset_generate.py   # Synthetic dataset generation 
│   └── train.py              # Model training
│
├── inference.py              # Run the complete pipeline
├── requirements.txt          # Project dependencies
└── README.md
```

## Dataset Creation

The dataset is generated automatically in two steps.

### 1. Collect Mountain Names

Download mountain names from the Open Peaks dataset:

```bash 
python src/download_mountains.py
```

This creates:

```bash
data/raw/mountains.csv
```

### 2. Generate the NER Dataset

Generate positive and negative examples with automatic BIO annotations:

```bash
python src/generate_dataset.py
```

This creates:

```bash
data/processed/ner_dataset.json
```

A detailed explanation of the dataset generation process is available in:

```bash
notebooks/dataset_creation.ipynb
```
## Dataset Summary

| Property | Value |
|----------|------:|
| Mountain Names | 2924 |
| Positive Samples | 5000 |
| Negative Samples | 2000 |
| Total Samples | 7000 |
| Annotation Scheme | BIO |

## Model Architecture

The model is based on a pretrained **BERT** encoder fine-tuned for token classification.

Input Text ──> BERT Tokenizer ──> Pretrained BERT Encoder ──> Token Classification Head ───> BIO Labels

The token classification head (dropout and linear classification layer) is provided by the Hugging Face `AutoModelForTokenClassification` implementation.

Unlike text classification, the model predicts a label for every token in the input sequence. 

## Training Strategy

The model was fine-tuned using the Hugging Face Transformers framework.

Training strategy included:

- Fine-tuning a pretrained BERT model.
- Automatic BIO annotation generation.
- Training on a balanced set of positive and negative examples.
- Learning rate scheduling.

## Model Training Configuration

| Parameter  | Value |
|----------- |-------|
| MODEL_NAME | bert-base-uncased |  
| MAX_LENGTH | 128 |
| BATCH_SIZE | 16 |
| TEST_SIZE  | 0.2 |
| LEARNING_RATE | 2e-5 |
| NUM_EPOCHS| 3 |
| WEIGHT_DECAY  | 0.01 |


## Model Training from Scratch

If you prefer to train the model instead of downloading the pretrained model, run:

```bash
python src/train.py
```

## Download Model weights

The fine-tuned BERT model is not included in this repository because of its size.

Download the trained model from Google Drive:

https://drive.google.com/drive/folders/1yoR0vnkr2D3vfnbboaHnhEoxUwxKqh8J?usp=sharing

### After downloading:

**1. Extract the downloaded archive**

**2. Copy model weights in folder models/**

**Final project structure:**

```bash
models/
└── bert_ner/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── training_args.bin
    
```

## Demo Notebook

An interactive notebook demonstration is also available:

```bash
notebooks/demo.ipynb
```

The notebook demonstrates:

- loading the trained model;
- running inference on custom text;
- extracting detected mountain entities.

It can be used to interactively test the pipeline.

# Technologies
- python
- torch
- transformers
- datasets
- evaluate
- seqeval
- scikit-learn
- numpy
- pandas
- matplotlib
- accelerate

# Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Pasha0923/Quantum-test-task.git
cd task1
```
2. **Create virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate
```
3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
4. **Additional setup**

Before running the project, make sure you have:
- downloaded the trained BERT model and placed the `bert_ner/` directory into `models/`.

# Run Inference

Run interactive command-line interface :

```bash
python inference.py
```

Example:

Input:
We climbed Mount Elbrus before visiting Mont Blanc.

Detected mountains:

• **Mount Elbrus**

• **Mont Blanc**
