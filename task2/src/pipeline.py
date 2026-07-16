from pathlib import Path
from src.matcher import (evaluate_tiles,load_matcher,select_best_match)
from src.tiling import process_image


def run_pipeline(safe_dir1: Path,safe_dir2: Path):
    """
    Complete matching pipeline.
    """

    print("=" * 60)
    print("Reading Sentinel-2 images...")
    print("=" * 60)

    image1, good_tiles1 = process_image(safe_dir1)
    image2, good_tiles2 = process_image(safe_dir2)

    print()

    print("=" * 60)
    print("Loading LoFTR...")
    print("=" * 60)

    matcher, device = load_matcher()

    print(f"Device: {device}")

    print()
    print("=" * 60)
    print("Matching tiles...")
    print("=" * 60)

    results = evaluate_tiles(good_tiles1,good_tiles2,matcher, device)

    print()
    print("=" * 60)
    print("Selecting best tile...")
    print("=" * 60)

    best = select_best_match(results)

    print()
    print("Top 10 tiles:\n")

    for result in results[:10]:

        print(
            f"{result['coord']}   "
            f"matches={result['matches']}   "
            f"confidence={result['mean_conf']:.3f}"
        )

    print()
    print(f"Best tile: {best['coord']}")
    print(f"Matches: {best['matches']}")
    print(f"Mean confidence: {best['mean_conf']:.3f}")

    return {
        "image1": image1,
        "image2": image2,
        "good_tiles1": good_tiles1,
        "good_tiles2": good_tiles2,
        "tile1": best["tile1"],
        "tile2": best["tile2"],
        "pred": best["pred"],
        "coord": best["coord"],
        "matches": best["matches"],
        "mean_conf": best["mean_conf"],
        "results": results,
    }