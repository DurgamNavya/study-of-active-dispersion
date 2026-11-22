"""
===========================================================
SEGMENTATION.PY

Segmentation of Vinegar Eels

Author : OpenAI

Designed for:

1920×1080 images
30 FPS
Bright-field microscopy

Pipeline

Image
 ↓
CLAHE
 ↓
Black-hat enhancement
 ↓
Background removal
 ↓
Threshold
 ↓
Morphology
 ↓
Connected Components
 ↓
Binary Mask

===========================================================
"""

import cv2
import numpy as np

from scipy import ndimage

from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    skeletonize
)

from skimage.measure import (
    label,
    regionprops
)

from pathlib import Path

############################################################
# CLASS
############################################################

class VinegarEelSegmentation:

    def __init__(self):

        ####################################################
        # CLAHE
        ####################################################

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

        ####################################################
        # Blackhat kernel
        ####################################################

        self.blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (21,21)
        )

        ####################################################
        # Morphology kernel
        ####################################################

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3,3)
        )

        ####################################################
        # Minimum object size
        ####################################################

        self.min_area = 80

        ####################################################
        # Maximum object size
        ####################################################

        self.max_area = 80000
    ########################################################

    def normalize(self,image):

        image=image.astype(np.float32)

        image-=image.min()

        image/=image.max()+1e-8

        image*=255

        return image.astype(np.uint8)
        ########################################################

    def enhance(self,image):

        return self.clahe.apply(image)
        ########################################################

    def remove_background(self,image):

        background=cv2.GaussianBlur(
            image,
            (101,101),
            0
        )

        result=cv2.subtract(
            background,
            image
        )

        return self.normalize(result)
        ########################################################

    def blackhat(self,image):

        black=cv2.morphologyEx(

            image,

            cv2.MORPH_BLACKHAT,

            self.blackhat_kernel

        )

        return self.normalize(black)
        ########################################################

    def smooth(self,image):

        return cv2.GaussianBlur(

            image,

            (5,5),

            1.2

        )
        ########################################################

    def threshold(self,image):

        binary=cv2.adaptiveThreshold(

            image,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY,

            51,

            -5

        )

        return binary
        ########################################################
    # MORPHOLOGICAL CLEANING
    ########################################################

    def morphology(self, binary):

        # Remove isolated pixels
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1
        )

        # Connect broken worms
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=2
        )

        return binary


    ########################################################
    # REMOVE SMALL OBJECTS
    ########################################################

    def remove_noise(self, binary):

        binary = binary.astype(bool)

        binary = remove_small_objects(
            binary,
            min_size=self.min_area
        )

        binary = remove_small_holes(
            binary,
            area_threshold=100
        )

        return (binary * 255).astype(np.uint8)


    ########################################################
    # CONNECTED COMPONENT FILTERING
    ########################################################

    def filter_components(self, binary):

        labels = label(binary)

        output = np.zeros(binary.shape, dtype=np.uint8)

        for region in regionprops(labels):

            area = region.area

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            minr, minc, maxr, maxc = region.bbox

            width = maxc - minc
            height = maxr - minr

            if width == 0 or height == 0:
                continue

            aspect = max(width, height) / (min(width, height) + 1)

            # Vinegar eels are elongated
            if aspect < 2:
                continue

            output[labels == region.label] = 255

        return output


    ########################################################
    # DETECT PETRI DISH
    ########################################################

    def detect_dish(self, image):

        blur = cv2.GaussianBlur(image, (9,9), 2)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=1000,
            param1=80,
            param2=35,
            minRadius=350,
            maxRadius=512
        )

        if circles is None:

            h, w = image.shape

            return w//2, h//2, min(h,w)//2-20

        circles = np.uint16(np.around(circles))

        x, y, r = circles[0][0]

        return int(x), int(y), int(r)

      ########################################################
# MASK INSIDE PETRI DISH
########################################################
       ########################################################
    # MASK INSIDE PETRI DISH
    ########################################################

    def apply_dish_mask(self, image, x, y, r):

        # Shrink the detected radius to remove the bright rim
        margin = 80      # Try 50, 70, 80, or 100

        radius = max(r - margin, 0)

        mask = np.zeros(image.shape, dtype=np.uint8)

        cv2.circle(
            mask,
            (x, y),
            radius,
            255,
            -1
        )

        # Keep only pixels inside the circle
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Save for debugging
        cv2.imwrite("dish_mask.png", mask)
        cv2.imwrite("masked_image.png", masked)

        return masked

    ########################################################
    # REMOVE LEFT SCALE BAR
    ########################################################

    def remove_scale(self, binary):

        h, w = binary.shape

        binary[:, :120] = 0

        return binary


    ########################################################
    # REMOVE CENTRAL DEBRIS
    ########################################################

    def remove_center_object(self, binary):

        h, w = binary.shape

        Y, X = np.ogrid[:h, :w]

        cx = w // 2
        cy = h // 2

        radius = 70

        mask = (X-cx)**2 + (Y-cy)**2 <= radius**2

        binary[mask] = 0

        return binary
        ########################################################
    # COMPLETE SEGMENTATION PIPELINE
    ########################################################

    def segment(self, image):

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect Petri dish
        x, y, r = self.detect_dish(gray)

        # Apply dish mask
        gray = self.apply_dish_mask(gray, x, y, r)

        # Enhance contrast
        clahe = self.enhance(gray)

        # Remove uneven illumination
        background = self.remove_background(clahe)

        # Enhance dark worms
        black = self.blackhat(background)

        # Smooth image
        smooth = self.smooth(black)

        # Threshold
        binary = self.threshold(smooth)

        # Morphology
        binary = self.morphology(binary)

        # Remove ruler
        binary = self.remove_scale(binary)

        # Remove central debris
        binary = self.remove_center_object(binary)

        # Remove noise
        binary = self.remove_noise(binary)

        # Keep eel-like components
        binary = self.filter_components(binary)

        return binary
        ########################################################
    # SKELETONIZATION
    ########################################################

    def skeleton(self, binary):

        binary = binary > 0

        skel = skeletonize(binary)

        return (skel.astype(np.uint8) * 255)
        ########################################################
    # OVERLAY MASK
    ########################################################

    def overlay(self, image, binary):

        if len(image.shape) == 2:
            overlay = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )
        else:
            overlay = image.copy()

        overlay[binary > 0] = (0, 255, 0)

        return overlay
        ########################################################
    # DEBUG VISUALIZATION
    ########################################################

    def debug(self, image):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
        else:
            gray = image.copy()

        x, y, r = self.detect_dish(gray)

        masked = self.apply_dish_mask(
            gray,
            x,
            y,
            r
        )

        clahe = self.enhance(masked)

        background = self.remove_background(clahe)

        black = self.blackhat(background)

        smooth = self.smooth(black)

        binary = self.threshold(smooth)

        morph = self.morphology(binary)

        morph = self.remove_scale(morph)

        morph = self.remove_center_object(morph)

        clean = self.remove_noise(morph)

        final = self.filter_components(clean)

        skel = self.skeleton(final)

        return {

            "gray": gray,

            "masked": masked,

            "clahe": clahe,

            "background": background,

            "blackhat": black,

            "smooth": smooth,

            "binary": binary,

            "morphology": morph,

            "clean": clean,

            "final": final,

            "skeleton": skel

        }
        ########################################################
    # SAVE DEBUG IMAGES
    ########################################################

    def save_debug(self, image, out_dir):

        out_dir = Path(out_dir)

        out_dir.mkdir(
            exist_ok=True,
            parents=True
        )

        data = self.debug(image)

        for name, img in data.items():

            cv2.imwrite(
                str(out_dir / f"{name}.png"),
                img
            )
            ########################################################
# PROCESS A FOLDER OF IMAGES
########################################################

    def process_folder(
            self,
            image_folder,
            output_folder):

     image_folder = Path(image_folder)
     output_folder = Path(output_folder)

     mask_folder = output_folder / "mask"
     skeleton_folder = output_folder / "skeleton"
     overlay_folder = output_folder / "overlay"
     debug_folder = output_folder / "debug"

     mask_folder.mkdir(parents=True, exist_ok=True)
     skeleton_folder.mkdir(parents=True, exist_ok=True)
     overlay_folder.mkdir(parents=True, exist_ok=True)
     debug_folder.mkdir(parents=True, exist_ok=True)

    extensions = [
        "*.tif",
        "*.tiff",
        "*.png",
        "*.jpg",
        "*.bmp"
    ]

    files = []

    for ext in extensions:
        files.extend(image_folder.glob(ext))
        files.extend(image_folder.glob(ext.upper()))

    files = sorted(files)

    print("----------------------------------------")
    print("Total Images :", len(files))
    print("----------------------------------------")

    for i, file in enumerate(files):

        print(f"{i+1}/{len(files)} : {file.name}")

        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Cannot read image.")
            continue

        mask = self.segment(image)

        skel = self.skeleton(mask)

        overlay = self.overlay(image, mask)

        cv2.imwrite(
            str(mask_folder / file.name),
            mask
        )

        cv2.imwrite(
            str(skeleton_folder / file.name),
            skel
        )

        cv2.imwrite(
            str(overlay_folder / file.name),
            overlay
        )

    print("----------------------------------------")
    print("Segmentation Completed")
    print("----------------------------------------")
    ########################################################
# PROCESS SINGLE IMAGE
########################################################

    def process_image(
            self,
            image_path,
            output_folder):

     image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise RuntimeError(
            "Cannot read image."
        )

    output_folder = Path(output_folder)

    output_folder.mkdir(
        exist_ok=True,
        parents=True
    )

    mask = self.segment(image)

    skel = self.skeleton(mask)

    overlay = self.overlay(
        image,
        mask
    )

    cv2.imwrite(
        str(output_folder/"mask.png"),
        mask
    )

    cv2.imwrite(
        str(output_folder/"skeleton.png"),
        skel
    )

    cv2.imwrite(
        str(output_folder/"overlay.png"),
        overlay
    )

    self.save_debug(
        image,
        output_folder/"debug"
    )
    ########################################################
# MAIN
########################################################

if __name__ == "__main__":

    IMAGE_FOLDER = r"D:\PhD\Active dispersion\EXP1\PIV\images"
    OUTPUT_FOLDER = r"D:\PhD\Active dispersion\EXP1\PIV\segmentation_results"

    print("========== DEBUG ==========")
    print("Running file:", __file__)
    print("Methods:", [m for m in dir(VinegarEelSegmentation)
                        if not m.startswith("__")])
    print("===========================")

    segmenter = VinegarEelSegmentation()

    segmenter.process_folder(
        IMAGE_FOLDER,
        OUTPUT_FOLDER
    )

