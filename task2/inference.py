from pathlib import Path

from src.tiling import process_image

from src.matcher import load_matcher

safe_dir1 = Path("data/S2A_MSIL1C_20160330T082542_N0201_R021_T36UYA_20160330T082810.SAFE")

safe_dir2 = Path("data/S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510.SAFE")


image1, good_tiles1 = process_image(safe_dir1)
image2, good_tiles2 = process_image(safe_dir2)


# print(type(good_tiles1))
# print(type(good_tiles2))

# print(len(good_tiles1))
# print(len(good_tiles2))



matcher, device = load_matcher()

print(device)
print(type(matcher))