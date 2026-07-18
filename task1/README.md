# Mountain Named Entity Recognition (NER)

## Project overview
This project implements a Named Entity Recognition (NER) system for detecting mountain names in English text using a fine-tuned BERT model.

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

```
task1/
│
├── data/
│   ├── raw/
│   │   └── mountains.csv
│   │
│   └── processed/
│       └── ner_dataset.json
│
├── models/
│   └── bert_ner/
│
├── notebooks/
│   ├── dataset_creation.ipynb
│   └── demo.ipynb
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── download_mountains.py
│   ├── generate_dataset.py
│   ├── model.py
│   └── train.py
│
├── inference.py
├── requirements.txt
└── README.md