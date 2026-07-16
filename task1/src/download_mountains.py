from pathlib import Path

import pandas as pd
import requests


INDEX_URL = (
    "https://raw.githubusercontent.com/open-peaks/data/master/_index.geojson"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mountains.csv"
)


def download_index() -> dict:
    """
    Download the Open Peaks GeoJSON index.

    Returns:
        Parsed GeoJSON dictionary.
    """
    response = requests.get(INDEX_URL, timeout=30)
    response.raise_for_status()

    return response.json()


def extract_mountains(data: dict) -> list[str]:
    """
    Extract unique mountain names from the GeoJSON data.

    Args:
        data: Parsed GeoJSON dictionary.

    Returns:
        Sorted list of unique mountain names.
    """
    mountains = []

    for feature in data.get("features", []):

        name = (
            feature.get("properties", {})
            .get("name", "")
            .strip()
        )

        if name:
            mountains.append(name)

    return sorted(set(mountains))


def save_csv(mountains: list[str], output_path: Path) -> None:
    """
    Save mountain names to CSV.

    Args:
        mountains: List of mountain names.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        {
            "mountain": mountains,
        }
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def main():

    print("Downloading Open Peaks index...")

    data = download_index()

    mountains = extract_mountains(data)

    save_csv(
        mountains,
        OUTPUT_PATH,
    )

    print(f"Successfully saved {len(mountains)} mountain names.")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()