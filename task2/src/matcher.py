from pathlib import Path

import torch
import kornia as K
import kornia.feature as KF

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "loftr_outdoor.ckpt"


def load_matcher(weights_path: Path = DEFAULT_WEIGHTS):

    """
    Load pretrained LoFTR model.
    """

    if not weights_path.exists():
        raise FileNotFoundError(
            f"LoFTR weights not found:\n{weights_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    matcher = KF.LoFTR(pretrained=None)

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    matcher.load_state_dict(
        checkpoint["state_dict"]
    )

    matcher = matcher.to(device).eval()

    return matcher, device


def run_loftr(tile1, tile2,matcher,device):
    """
    Run LoFTR on two RGB tiles.
    """

    img1 = torch.from_numpy(tile1)
    img2 = torch.from_numpy(tile2)

    img1 = img1.permute(2, 0, 1).float()[None]
    img2 = img2.permute(2, 0, 1).float()[None]

    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():

        pred = matcher(
            {
                "image0": img1,
                "image1": img2,
            }
        )

    return pred


def evaluate_tiles(good_tiles1: dict,good_tiles2: dict,matcher,device):
    """
    Match all common tiles and rank them.
    """

    common_coords = sorted(
        set(good_tiles1.keys()) &
        set(good_tiles2.keys())
    )

    print(f"Common tiles: {len(common_coords)}")

    results = []

    for coord in common_coords:

        tile1 = good_tiles1[coord]
        tile2 = good_tiles2[coord]

        pred = run_loftr(
            tile1,
            tile2,
            matcher,
            device,
        )

        matches = len(pred["keypoints0"])

        if matches == 0:
            continue

        confidence = (
            pred["confidence"]
            .cpu()
            .numpy()
        )

        mean_conf = float(confidence.mean())

        results.append(
            {
                "coord": coord,
                "tile1": tile1,
                "tile2": tile2,
                "pred": pred,
                "matches": matches,
                "mean_conf": mean_conf,
            }
        )

    print(f"\nProcessed {len(results)} common tiles.")

    results.sort(
        key=lambda x: (
            x["matches"],
            x["mean_conf"],
        ),
        reverse=True,
    )

    return results


def select_best_match(results):
    """
    Return best tile pair.
    """

    if len(results) == 0:
        raise RuntimeError(
            "No matching tiles were found."
        )

    best = results[0]

    print("\nTop 10 tiles:\n")

    for result in results[:10]:

        print(
            f"{result['coord']}   "
            f"matches={result['matches']}   "
            f"confidence={result['mean_conf']:.3f}"
        )

    print(f"Best tile: {best['coord']}")

    print(f"Matches: {best['matches']}")

    print(f"Mean confidence: {best['mean_conf']:.3f}")

    return best