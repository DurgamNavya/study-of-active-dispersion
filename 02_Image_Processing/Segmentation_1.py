###############################################################
# Segmentation.py
#
# Vinegar Eel Segmentation Pipeline
#
# Part 1
# Imports
# Initialization
# Image Reading
# Petri Dish Detection
# Petri Dish Masking
###############################################################

import cv2
import numpy as np

from pathlib import Path

from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    remove_small_holes
)

from scipy import ndimage


###############################################################
# CLASS
###############################################################

class VinegarEelSegmentation:

    ###########################################################
    # INITIALIZATION
    ###########################################################

    def __init__(self):

        # -----------------------------------------------------
        # Petri dish parameters
        # -----------------------------------------------------

        self.mask_margin = 80

        # Set to False if using fixed ROI
        self.auto_detect_dish = True

        # Fixed values (used only if auto_detect_dish=False)
        self.fixed_x = None
        self.fixed_y = None
        self.fixed_radius = None

        # -----------------------------------------------------
        # Morphology parameters
        # -----------------------------------------------------

        self.min_object_size = 80
        self.min_hole_size = 100

        self.blackhat_kernel = (25, 25)

        self.gaussian_sigma = 3

        #######################################################
        # Debug
        #######################################################

        self.save_debug_images = True

    ###########################################################
    # READ IMAGE
    ###########################################################

    def read_image(self, filename):

        image = cv2.imread(
            str(filename),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise RuntimeError(
                f"Cannot read image:\n{filename}"
            )

        return image

    ###########################################################
    # DETECT PETRI DISH
    ###########################################################

    def detect_dish(self, image):

        # -----------------------------------------------------
        # Use fixed ROI if desired
        # -----------------------------------------------------

        if not self.auto_detect_dish:

            return (
                self.fixed_x,
                self.fixed_y,
                self.fixed_radius
            )

        # -----------------------------------------------------
        # Blur image
        # -----------------------------------------------------

        blur = cv2.GaussianBlur(
            image,
            (9, 9),
            2
        )

        circles = cv2.HoughCircles(

            blur,

            cv2.HOUGH_GRADIENT,

            dp=1.2,

            minDist=1500,

            param1=80,

            param2=35,

            minRadius=350,

            maxRadius=600
        )

        if circles is None:

            raise RuntimeError(
                "Petri dish not detected."
            )

        circles = np.uint16(
            np.around(circles)
        )

        x, y, r = circles[0][0]

        return int(x), int(y), int(r)

    ###########################################################
    # PETRI DISH MASK
    ###########################################################

    def apply_dish_mask(
            self,
            image,
            x,
            y,
            r):

        radius = max(
            r - self.mask_margin,
            0
        )

        mask = np.zeros(
            image.shape,
            dtype=np.uint8
        )

        cv2.circle(

            mask,

            (x, y),

            radius,

            255,

            -1
        )

        masked = cv2.bitwise_and(
            image,
            image,
            mask=mask
        )

        return masked

    ###########################################################
    # SAVE PETRI DISH DEBUG
    ###########################################################

    def save_dish_debug(
            self,
            image,
            x,
            y,
            r):

        if not self.save_debug_images:
            return

        debug = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

        cv2.circle(
            debug,
            (x, y),
            r,
            (0, 255, 0),
            2
        )

        cv2.circle(
            debug,
            (x, y),
            r - self.mask_margin,
            (0, 0, 255),
            2
        )

        cv2.imwrite(
            "Detected_Dish.png",
            debug
        )

        mask = np.zeros(
            image.shape,
            dtype=np.uint8
        )

        cv2.circle(
            mask,
            (x, y),
            r - self.mask_margin,
            255,
            -1
        )

        cv2.imwrite(
            "DishMask.png",
            mask
        )
            ###########################################################
    # REMOVE BACKGROUND
    ###########################################################

    def remove_background(self, image):

        # Large Gaussian blur estimates slow-varying background
        background = cv2.GaussianBlur(
            image,
            (101, 101),
            0
        )

        # Keep dark vinegar eels
        result = cv2.subtract(
            background,
            image
        )

        return result

    ###########################################################
    # CONTRAST ENHANCEMENT (CLAHE)
    ###########################################################

    def enhance_contrast(self, image):

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(image)

        return enhanced

    ###########################################################
    # DENOISE
    ###########################################################

    def denoise(self, image):

        image = cv2.GaussianBlur(
            image,
            (5, 5),
            self.gaussian_sigma
        )

        return image

    ###########################################################
    # BLACKHAT ENHANCEMENT
    ###########################################################

    def blackhat(self, image):

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.blackhat_kernel
        )

        result = cv2.morphologyEx(
            image,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        return result

    ###########################################################
    # REMOVE LEFT SCALE BAR
    ###########################################################

    def remove_scale_bar(self, image):

        result = image.copy()

        # Adjust according to your images
        result[:, :120] = 0

        return result

    ###########################################################
    # THRESHOLD
    ###########################################################

    def threshold(self, image):

        _, binary = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    ###########################################################
    # MORPHOLOGICAL CLEANING
    ###########################################################

    def morphology(self, binary):

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        return binary

    ###########################################################
    # REMOVE SMALL OBJECTS
    ###########################################################

    def remove_noise(self, binary):

        binary_bool = binary.astype(bool)

        binary_bool = remove_small_objects(
            binary_bool,
            min_size=self.min_object_size
        )

        binary_bool = remove_small_holes(
            binary_bool,
            area_threshold=self.min_hole_size
        )

        binary = (binary_bool.astype(np.uint8)) * 255

        return binary

    ###########################################################
    # COMPLETE PREPROCESSING PIPELINE
    ###########################################################

    def preprocess(self, image):

        # Detect Petri dish
        x, y, r = self.detect_dish(image)

        # Save debug image
        self.save_dish_debug(image, x, y, r)

        # Mask outside dish
        image = self.apply_dish_mask(
            image,
            x,
            y,
            r
        )

        # Remove scale bar
        image = self.remove_scale_bar(image)

        # Background subtraction
        image = self.remove_background(image)

        # Contrast enhancement
        image = self.enhance_contrast(image)

        # Noise removal
        image = self.denoise(image)

        # Enhance worms
        image = self.blackhat(image)

        return image
        ###########################################################
    # REMOVE OBJECTS TOUCHING IMAGE BORDER
    ###########################################################

    def remove_border_objects(self, binary):

        h, w = binary.shape

        labels, stats = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )[1:3]

        output = np.zeros_like(binary)

        num_labels = stats.shape[0]

        for label in range(1, num_labels):

            component = (labels == label)

            ys, xs = np.where(component)

            if len(xs) == 0:
                continue

            if (
                np.min(xs) == 0 or
                np.max(xs) == w - 1 or
                np.min(ys) == 0 or
                np.max(ys) == h - 1
            ):
                continue

            output[component] = 255

        return output


    ###########################################################
    # FILTER OBJECTS BY AREA
    ###########################################################

    def filter_by_area(
            self,
            binary,
            min_area=30,
            max_area=50000):

        nLabels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )

        output = np.zeros_like(binary)

        for i in range(1, nLabels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            if area > max_area:
                continue

            output[labels == i] = 255

        return output


    ###########################################################
    # FINAL MORPHOLOGY
    ###########################################################

    def final_cleanup(self, binary):

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        return binary


    ###########################################################
    # SEGMENTATION PIPELINE
    ###########################################################

    def segment(self, image):

        #######################################################
        # PREPROCESS
        #######################################################

        pre = self.preprocess(image)

        #######################################################
        # THRESHOLD
        #######################################################

        binary = self.threshold(pre)

        #######################################################
        # MORPHOLOGY
        #######################################################

        binary = self.morphology(binary)

        #######################################################
        # REMOVE SMALL OBJECTS
        #######################################################

        binary = self.remove_noise(binary)

        #######################################################
        # FILTER AREA
        #######################################################

        binary = self.filter_by_area(
            binary,
            min_area=25,
            max_area=40000
        )

        #######################################################
        # REMOVE BORDER OBJECTS
        #######################################################

        binary = self.remove_border_objects(binary)

        #######################################################
        # FINAL CLEANUP
        #######################################################

        binary = self.final_cleanup(binary)

        return binary


    ###########################################################
    # SAVE INTERMEDIATE RESULTS
    ###########################################################

    def save_preprocessing_steps(
            self,
            image,
            folder):

        folder = Path(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        x, y, r = self.detect_dish(image)

        masked = self.apply_dish_mask(
            image,
            x,
            y,
            r
        )

        background = self.remove_background(masked)

        contrast = self.enhance_contrast(background)

        denoise = self.denoise(contrast)

        blackhat = self.blackhat(denoise)

        binary = self.threshold(blackhat)

        morph = self.morphology(binary)

        clean = self.remove_noise(morph)

        cv2.imwrite(
            str(folder / "01_original.png"),
            image
        )

        cv2.imwrite(
            str(folder / "02_masked.png"),
            masked
        )

        cv2.imwrite(
            str(folder / "03_background_removed.png"),
            background
        )

        cv2.imwrite(
            str(folder / "04_contrast.png"),
            contrast
        )

        cv2.imwrite(
            str(folder / "05_blackhat.png"),
            blackhat
        )

        cv2.imwrite(
            str(folder / "06_binary.png"),
            binary
        )

        cv2.imwrite(
            str(folder / "07_morphology.png"),
            morph
        )

        cv2.imwrite(
            str(folder / "08_final_mask.png"),
            clean
        )
        ###############################################################
# SKELETONIZATION
###############################################################

def skeleton(self, binary):

    binary_bool = binary > 0

    skel = skeletonize(binary_bool)

    skel = (skel.astype(np.uint8)) * 255

    return skel


###############################################################
# PCA ORIENTATION
###############################################################

def object_orientation(self, contour):

    pts = contour.reshape(-1,2).astype(np.float32)

    if len(pts) < 5:
        return None

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        pts,
        mean=None
    )

    center = mean[0]

    direction = eigenvectors[0]

    angle = np.arctan2(
        direction[1],
        direction[0]
    )

    angle_deg = np.degrees(angle)

    length = np.sqrt(eigenvalues[0][0])

    return center, direction, length, angle_deg


###############################################################
# FIND ALL ORIENTATIONS
###############################################################

def orientations(self, binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    objects = []

    for c in contours:

        if cv2.contourArea(c) < 25:
            continue

        result = self.object_orientation(c)

        if result is None:
            continue

        objects.append(result)

    return objects


###############################################################
# DRAW ORIENTATIONS
###############################################################

def draw_orientations(
        self,
        image,
        objects):

    if len(image.shape)==2:

        overlay = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        overlay = image.copy()

    for obj in objects:

        center, direction, length, angle = obj

        x = int(center[0])
        y = int(center[1])

        L = int(length * 2)

        x1 = int(
            x - direction[0]*L
        )

        y1 = int(
            y - direction[1]*L
        )

        x2 = int(
            x + direction[0]*L
        )

        y2 = int(
            y + direction[1]*L
        )

        cv2.line(
            overlay,
            (x1,y1),
            (x2,y2),
            (0,0,255),
            2
        )

        cv2.circle(
            overlay,
            (x,y),
            2,
            (0,255,0),
            -1
        )

    return overlay


###############################################################
# GREEN SEGMENTATION OVERLAY
###############################################################

def overlay(self, image, binary):

    if len(image.shape)==2:

        overlay = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        overlay = image.copy()

    overlay[binary>0] = (0,255,0)

    return overlay


###############################################################
# SAVE DEBUG RESULTS
###############################################################

def save_debug(
        self,
        filename,
        image,
        binary,
        skeleton,
        overlay,
        orientation):

    debug = Path("debug")

    debug.mkdir(
        exist_ok=True
    )

    cv2.imwrite(
        str(debug/f"{filename}_mask.png"),
        binary
    )

    cv2.imwrite(
        str(debug/f"{filename}_skeleton.png"),
        skeleton
    )

    cv2.imwrite(
        str(debug/f"{filename}_overlay.png"),
        overlay
    )

    cv2.imwrite(
        str(debug/f"{filename}_orientation.png"),
        orientation
    )
    

        