from pathlib import Path
import json
import random
import re
import pandas as pd
from config import MOUNTAINS_PATH, DATASET_PATH

# Configuration
MAX_MOUNTAINS = 1000
SENTENCES_PER_MOUNTAIN = 5
RANDOM_SEED = 42

LABELS = {
    "O": 0,
    "B-MOUNTAIN": 1,
    "I-MOUNTAIN": 2,
}

# Sentence templates
TEMPLATES = [

    "The expedition reached {mountain} after three days of climbing.",
    "Many hikers dream of visiting {mountain}.",
    "Heavy snowfall covered {mountain} during the night.",
    "The summit of {mountain} attracts experienced climbers.",
    "Tourists enjoyed the breathtaking view from {mountain}.",
    "Scientists conducted research near {mountain}.",
    "The weather around {mountain} changed rapidly.",
    "A rescue team was dispatched to {mountain}.",
    "The documentary featured the history of {mountain}.",
    "Local guides recommended climbing {mountain} in early summer.",
    "Snow remained on {mountain} throughout the year.",
    "Photographers gathered to capture the sunrise over {mountain}.",
    "The trail leading to {mountain} is considered challenging.",
    "Several rare plants grow near {mountain}.",
    "The climbing route to {mountain} requires technical skills.",
    "Our group spent two days exploring {mountain}.",
    "Clouds surrounded {mountain} throughout the afternoon.",
    "Mountaineers celebrated after reaching {mountain}.",
    "The national park includes {mountain} among its main attractions.",
    "Strong winds made climbing {mountain} extremely difficult.",
    "The guide pointed toward {mountain} in the distance.",
    "Wildlife can often be seen near {mountain}.",
    "The highest point of the journey was {mountain}.",
    "Visitors admired the landscape surrounding {mountain}.",
    "The team prepared carefully before attempting {mountain}.",
    "A famous photograph was taken on {mountain}.",
    "The glacier extends from the slopes of {mountain}.",
    "Experienced climbers frequently choose {mountain} for training.",
    "Morning sunlight illuminated {mountain}.",
    "The mountain rescue exercise took place near {mountain}.",
]

NEGATIVE_EXAMPLES = 2000
PEOPLE = [
    "John", "Emma", "Michael", "Sarah",
    "David", "Emily", "Daniel", "Sophia",
]

CITIES = [
    "Paris", "London", "Kyiv", "Berlin",
    "Rome", "Tokyo", "Madrid", "Warsaw",
]

COMPANIES = [
    "OpenAI", "Google", "Microsoft",
    "Apple", "Amazon", "Meta", "NVIDIA",
]

TECHNOLOGIES = [
    "Python",
    "TensorFlow",
    "PyTorch",
    "Docker",
    "Linux",
    "Kubernetes",
]

TOPICS = [
    "artificial intelligence",
    "machine learning",
    "computer vision",
    "natural language processing",
    "data science",
]

ORGANIZATIONS = [
    "NASA",
    "UNESCO",
    "World Health Organization",
    "European Union",
]

NEGATIVE_TEMPLATES = [

    "{person} visited {city} during the summer.",
    "{company} announced a new product yesterday.",
    "{technology} is widely used in modern software development.",
    "{organization} published a new report this week.",
    "Researchers presented a paper about {topic}.",
    "{person} works as a software engineer at {company}.",
    "Students are learning {technology} at the university.",
    "{city} hosted an international technology conference.",
    "Experts discussed {topic} during the workshop.",
    "{company} invested heavily in {topic}.",

]

# Utility functions
def load_mountains(csv_path: Path) -> list[str]:
    """
    Load mountain names from CSV.
    """

    df = pd.read_csv(csv_path)

    mountains = (
        df["mountain"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )
    if not mountains:
        raise ValueError("No mountain names found.")

    return mountains


def sample_mountains(mountains: list[str], max_mountains: int) -> list[str]:
    """
    Randomly select mountain names.
    """

    if len(mountains) <= max_mountains:
        return mountains

    return random.sample(mountains, max_mountains,)

def tokenize(sentence: str) -> list[str]:
    """
    Simple tokenizer.
    Keeps punctuation as separate tokens.
    """

    return re.findall(
        r"\w+(?:[-']\w+)*|[^\w\s]",
        sentence,
    )

def create_bio_labels(tokens: list[str], mountain: str) -> list[int]:
    """
    Create BIO labels.
    """

    mountain_tokens = tokenize(mountain)

    labels = [
        LABELS["O"]
        for _ in tokens
    ]

    length = len(mountain_tokens)

    for i in range(len(tokens) - length + 1):

        if (tokens[i:i + length] == mountain_tokens):

            labels[i] = LABELS["B-MOUNTAIN"]

            for j in range(1,length):

                labels[i + j] = LABELS["I-MOUNTAIN"]

            break


    return labels


def build_dataset(mountains: list[str]) -> list[dict]:
    """
    Generate the complete NER dataset.
    """

    dataset = []

    sample_id = 0

    for mountain in mountains:

        selected_templates = random.sample(TEMPLATES, k=SENTENCES_PER_MOUNTAIN)

        for template in selected_templates:

            sentence = template.format(mountain=mountain)

            tokens = tokenize(sentence)

            labels = create_bio_labels(tokens, mountain)

            dataset.append(
                {
                    "id": sample_id,
                    "sentence": sentence,
                    "tokens": tokens,
                    "ner_tags": labels,
                }
            )

            sample_id += 1

    return dataset


def build_negative_dataset(start_id: int) -> list[dict]:
    """
    Generate negative NER examples without mountain names.
    """

    dataset = []

    sample_id = start_id

    for _ in range(NEGATIVE_EXAMPLES):

        template = random.choice(NEGATIVE_TEMPLATES)

        sentence = template.format(
            person=random.choice(PEOPLE),
            city=random.choice(CITIES),
            company=random.choice(COMPANIES),
            technology=random.choice(TECHNOLOGIES),
            organization=random.choice(ORGANIZATIONS),
            topic=random.choice(TOPICS),
        )

        tokens = tokenize(sentence)

        labels = [LABELS["O"]] * len(tokens)

        dataset.append(
            {
                "id": sample_id,
                "sentence": sentence,
                "tokens": tokens,
                "ner_tags": labels,
            }
        )

        sample_id += 1

    return dataset


def save_dataset(dataset: list[dict], output_path: Path):
    """
    Save dataset as JSON.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def print_statistics(dataset: list[dict],mountains: list[str]):
    """
    Print dataset statistics.
    """

    sentence_lengths = [
        len(sample["tokens"])
        for sample in dataset
    ]

    avg_length = (
        sum(sentence_lengths)
        / len(sentence_lengths)
    )

    print("\nDataset statistics")
    print("-" * 40)

    print(f"Unique mountains: {len(mountains)}")

    print(f"Generated sentences: {len(dataset)}")

    print(f"Average sentence length: {avg_length:.2f} tokens")

    print()

    print("Example sample:\n")
    for sample in dataset[:3]:
        print(
            json.dumps(
                sample,
                indent=4,
                ensure_ascii=False,
        )
    )

def main():
    random.seed(RANDOM_SEED)
    print("Loading mountain names...")

    mountains = load_mountains(MOUNTAINS_PATH)

    print(f"Loaded {len(mountains)} mountains.")

    mountains = sample_mountains(mountains, MAX_MOUNTAINS)

    print(f"Using {len(mountains)} mountains.")

    print()

    print("Generating dataset...")

    positive_dataset = build_dataset(mountains)

    negative_dataset = build_negative_dataset(start_id=len(positive_dataset))

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    for idx, sample in enumerate(dataset):
        sample["id"] = idx

    print()

    print("Saving dataset...")

    save_dataset(dataset,DATASET_PATH)

    print()

    print_statistics(dataset,mountains)

    print()

    print(f"Dataset saved to:\n{DATASET_PATH}")


if __name__ == "__main__":
    main()