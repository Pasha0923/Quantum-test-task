from pathlib import Path

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MOUNTAINS_PATH = RAW_DATA_DIR / "mountains.csv"
DATASET_PATH = PROCESSED_DATA_DIR / "ner_dataset.json"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "bert_ner"

# ============================================================
# Model
# ============================================================

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

# ============================================================
# Training
# ============================================================

TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01


# ============================================================
# Labels
# ============================================================

LABEL_LIST = [
    "O",
    "B-MOUNTAIN",
    "I-MOUNTAIN",
]

LABEL2ID = {
    label: idx
    for idx, label in enumerate(LABEL_LIST)
}

ID2LABEL = {
    idx: label
    for idx, label in enumerate(LABEL_LIST)
}