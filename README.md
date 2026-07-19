# Data Science Internship Test Task

This repository contains the solution for the Data Science test assignment. 

The project covers both **Natural Language Processing (NLP)** and **Computer Vision (CV)** tasks and demonstrates the complete machine learning workflow, including data preparation, model development, evaluation, and inference.


```bash
├── docs/
│   └── report.pdf
│
├── task1/
│   ├── notebooks/
│   ├── src/
│   ├── inference.py
│   ├── requirements.txt
│   └── README.md
│
├── task2/
│   ├── assets/
│   ├── notebooks/
│   ├── src/
│   ├── inference.py
│   ├── requirements.txt
│   └── README.md
│
└── .gitignore
```

## Project Overview

### Task 1 – Mountain Named Entity Recognition

Development of a BERT-based Named Entity Recognition model for detecting mountain names 

📄 Documentation: `task1/README.md`

---

### Task 2 – Sentinel Image Matching

Development of a satellite image matching pipeline using Sentinel-2 imagery and a pretrained LoFTR model.

📄 Documentation: `task2/README.md`

---

## Report

The complete project report is available in:
```bash
docs/report.pdf
```

An editable notebook version is also included:
```bash
docs/report.ipynb
```

## Technologies

- Python
- PyTorch
- Transformers
- Kornia (LoFTR)
- Rasterio
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Notes

- Model weights and datasets are excluded from the repository due to their size.
- Each task is self-contained and includes its own `README.md`, `requirements.txt`, and demonstration notebook.