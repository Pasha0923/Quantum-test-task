import torch

from transformers import (AutoTokenizer,AutoModelForTokenClassification)

from src.config import (MODEL_OUTPUT_DIR,ID2LABEL)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)

model = AutoModelForTokenClassification.from_pretrained(MODEL_OUTPUT_DIR).to(device)

model.eval()


def predict(text: str):
    """
    Predict mountain entities in text.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    labels = [
        ID2LABEL[label.item()]
        for label in predictions[0]
    ]

    return tokens, labels


def print_predictions(text: str):
    """
    Print model predictions.
    """

    tokens, labels = predict(text)

    print(f"\nText: {text}\n")

    for token, label in zip(tokens, labels):

        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        print(f"{token:<20}{label}")


def main():
    while True:

        text = input("\nEnter text (or 'exit' to quit): ")

        if text.lower() == "exit":
            break

        print_predictions(text)

    # examples = [
    #     "I climbed Mount Everest last year.",
    #     "We travelled across the Alps.",
    #     "Snow remained on Hinterer Seelenkogel throughout the year.",
    #     "I like programming in Python.",
    # ]

    # for sentence in examples:
    #     print_predictions(sentence)


if __name__ == "__main__":
    main()