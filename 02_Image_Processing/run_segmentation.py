import cv2
from pathlib import Path

from Segmentation_1 import VinegarEelSegmentation


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\navya\Downloads\CP302work\02_Image_Processing")

IMAGE_FILE = next(BASE_DIR.glob("*.tiff"))


# ============================================================
# READ IMAGE
# ============================================================

print("=" * 60)
print("Running segmentation on:")
print(IMAGE_FILE)
print("=" * 60)

image = cv2.imread(str(IMAGE_FILE), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise RuntimeError(f"Could not read image: {IMAGE_FILE}")

print("Image shape:", image.shape)


# ============================================================
# CREATE SEGMENTATION OBJECT
# ============================================================

segmenter = VinegarEelSegmentation()


# ============================================================
# RUN SEGMENTATION
# ============================================================

print("\nRunning segmentation...")

mask = segmenter.segment(image)

print("Segmentation completed.")


# ============================================================
# SAVE FINAL SEGMENTATION MASK
# ============================================================

output_file = BASE_DIR / "segmentation_mask.png"

cv2.imwrite(str(output_file), mask)

print("Segmentation mask saved:")
print(output_file)


# ============================================================
# SAVE INTERMEDIATE PROCESSING STEPS
# ============================================================

print("\nSaving intermediate processing images...")

segmenter.save_preprocessing_steps(
    image,
    BASE_DIR
)

print("Intermediate processing images saved.")

print("\nFinished!")