import json
from transformers import AutoTokenizer
from datasets import Dataset
from config import (DATASET_PATH,MODEL_NAME,TEST_SIZE,RANDOM_SEED)

def load_dataset() -> Dataset:
    """
    Load NER dataset from JSON file.
    """

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return Dataset.from_list(data)


def split_dataset(dataset: Dataset):
    """
    Split dataset into training and validation sets.
    """

    splits = dataset.train_test_split(
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    train_dataset = splits["train"]
    validation_dataset = splits["test"]

    return train_dataset, validation_dataset


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    """
    Tokenize examples and align BIO labels with BERT tokens.
    """

    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    labels = []

    for i, label in enumerate(examples["ner_tags"]):

        word_ids = tokenized.word_ids(batch_index=i)

        previous_word = None

        label_ids = []

        for word_id in word_ids:

            if word_id is None:

                label_ids.append(-100)

            elif word_id != previous_word:

                label_ids.append(label[word_id])

            else:

                label_ids.append(-100)

            previous_word = word_id

        labels.append(label_ids)

    tokenized["labels"] = labels

    return tokenized


def prepare_datasets():
    """
    Prepare tokenized datasets for Hugging Face Trainer.
    """

    dataset = load_dataset()

    train_dataset, validation_dataset = split_dataset(dataset)

    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )

    return train_dataset, validation_dataset