import cv2
from pathlib import Path
from tqdm import tqdm

from preprocessing import detect_chamber, preprocess

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(r"C:\Users\navya\Downloads\CP302work\02_Image_Processing")

IMAGE_DIR = BASE_DIR

# =====================================================
# Read images
# =====================================================

files = sorted(IMAGE_DIR.glob("*.tiff"))

print("=" * 50)
print("Image folder :", IMAGE_DIR)
print("Images found :", len(files))
print("=" * 50)

if len(files) == 0:
    raise RuntimeError("No images found!")

# =====================================================
# Read first frame
# =====================================================

first = cv2.imread(str(files[0]))

if first is None:
    raise RuntimeError("Cannot read first image!")

print("Image shape :", first.shape)

# =====================================================
# Detect chamber
# =====================================================

x, y, r = detect_chamber(first)

print(f"Detected ROI : Center=({x},{y}) Radius={r}")

# =====================================================
# Save ROI preview
# =====================================================

preview = first.copy()

cv2.circle(preview, (x, y), r, (0, 255, 0), 3)

cv2.imwrite(
    str(BASE_DIR / "roi_preview.png"),
    preview
)

print("ROI preview saved.")

# =====================================================
# Process images
# =====================================================

print("\nProcessing images...\n")

for file in tqdm(files):

    img = cv2.imread(str(file))

    if img is None:
        print(f"Cannot read {file.name}")
        continue

    try:
        processed = preprocess(img, x, y, r)

    except Exception as e:
        print(f"Skipping {file.name}")
        print(e)
        continue

    outfile = BASE_DIR / f"preprocessed_{file.stem}.png"

    cv2.imwrite(str(outfile), processed)

print("\nFinished!")
print("Preprocessed images saved in:")
print(BASE_DIR)