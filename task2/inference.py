from pathlib import Path

from src.matcher import load_matcher
from src.visualization import (draw_matches,show_tiles)
from src.pipeline import run_pipeline



safe_dir1 = Path("data/S2A_MSIL1C_20160330T082542_N0201_R021_T36UYA_20160330T082810.SAFE")
safe_dir2 = Path("data/S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510.SAFE")

result = run_pipeline(safe_dir1,safe_dir2)

print("\n")
print("=" * 60)
print("Best Match")
print("=" * 60)

print(f"Coordinate      : {result['coord']}")
print(f"Matches         : {result['matches']}")
print(f"Mean confidence : {result['mean_conf']:.3f}")

show_tiles(result["tile1"],result["tile2"],)

draw_matches(
    result["tile1"],
    result["tile2"],
    result["pred"],
    top_k=100,
)


