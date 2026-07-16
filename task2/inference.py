from pathlib import Path

from src.visualization import (draw_matches,show_tiles)
from src.pipeline import run_pipeline

safe_dir1 = Path("data/S2A_MSIL1C_20160621T084012_N0204_R064_T36UYA_20160621T084513.SAFE")
safe_dir2 = Path("data/S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510.SAFE")

result = run_pipeline(safe_dir1,safe_dir2)

show_tiles(result["tile1"],result["tile2"])

draw_matches(result["tile1"],result["tile2"],result["pred"],top_k=100)
