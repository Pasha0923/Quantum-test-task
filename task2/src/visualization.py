import matplotlib.pyplot as plt
import numpy as np


def show_tiles(
    tile1: np.ndarray,
    tile2: np.ndarray,
):
    """
    Display two tiles side by side.
    """

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    ax[0].imshow(tile1)
    ax[0].set_title("Image 1")

    ax[1].imshow(tile2)
    ax[1].set_title("Image 2")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


def draw_matches(
    tile1: np.ndarray,
    tile2: np.ndarray,
    pred,
    top_k: int = 100,
):
    """
    Draw LoFTR matches using Matplotlib.
    """

    mkpts0 = pred["keypoints0"].cpu().numpy()
    mkpts1 = pred["keypoints1"].cpu().numpy()

    confidence = pred["confidence"].cpu().numpy()

    idx = np.argsort(confidence)[::-1][:top_k]

    mkpts0 = mkpts0[idx]
    mkpts1 = mkpts1[idx]

    img1 = np.clip(tile1, 0, 1)
    img2 = np.clip(tile2, 0, 1)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    H = max(h1, h2)
    W = w1 + w2

    canvas = np.ones((H, W, 3), dtype=np.float32)

    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    plt.figure(figsize=(18, 9))

    plt.imshow(canvas)

    plt.scatter(
        mkpts0[:, 0],
        mkpts0[:, 1],
        s=10,
        c="lime",
    )

    plt.scatter(
        mkpts1[:, 0] + w1,
        mkpts1[:, 1],
        s=10,
        c="lime",
    )

    rng = np.random.default_rng(42)

    for p0, p1 in zip(mkpts0, mkpts1):

        color = rng.random(3)

        plt.plot(
            [p0[0], p1[0] + w1],
            [p0[1], p1[1]],
            color=color,
            linewidth=0.8,
            alpha=0.8,
        )

    plt.axis("off")

    plt.tight_layout()

    plt.show()