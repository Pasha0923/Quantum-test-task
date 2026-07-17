import torch

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.config import (MODEL_OUTPUT_DIR,ID2LABEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_OUTPUT_DIR).to(device)

model.eval()


def predict(text: str):
    """
    Predict token labels.
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

    predictions = outputs.logits.argmax(dim=-1)[0]

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    labels = [
        ID2LABEL[pred.item()]
        for pred in predictions
    ]
    return tokens, labels


def extract_mountains(tokens, labels):
    """
    Extract mountain names from BIO predictions.
    """

    mountains = []
    current = ""

    for token, label in zip(tokens, labels):

        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        if token.startswith("##"):
            piece = token[2:]
        else:
            piece = token

        if label == "B-MOUNTAIN":

            if current:
                mountains.append(current)

            current = piece

        elif label == "I-MOUNTAIN":

            if not current:
                current = piece

            elif token.startswith("##"):
                current += piece

            else:
                current += " " + piece

        else:

            if current:
                mountains.append(current)
                current = ""

    if current:
        mountains.append(current)

    return mountains


def print_predictions(text: str):
    """
    Print extracted mountain names.
    """

    tokens, labels = predict(text)

    mountains = extract_mountains(tokens, labels)

    print("\n" + "-" * 50)
    print(f"Input text:\n{text}\n")

    if mountains:
        print("Extracted mountain(s):")

        for mountain in mountains:
            print(f"• {mountain.title()}")

    else:
        print("No mountain names found.")

    print("-" * 50)


def predict_mountains(text: str):
    """
    Return extracted mountain names.
    """

    tokens, labels = predict(text)
    return extract_mountains(tokens, labels)

def main():

    while True:

        text = input("\nEnter text ('exit' or 'quit' to finish program): ").strip()

        if text.lower() in ("exit", "quit"):
            break

        if not text:
            continue

        print_predictions(text)


if __name__ == "__main__":
    main()