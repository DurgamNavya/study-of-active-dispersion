import cv2
import numpy as np


def detect_chamber(image):
    """
    Detect circular observation chamber from first frame.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=800,
        param1=100,
        param2=40,
        minRadius=350,
        maxRadius=700,
    )

    if circles is None:
        raise RuntimeError("Could not detect chamber!")

    circles = np.uint16(np.around(circles))
    x, y, r = map(int, circles[0][0])

    return x, y, r


def crop_circle(image, x, y, r):

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.circle(mask, (x, y), r - 10, 255, -1)

    roi = cv2.bitwise_and(image, image, mask=mask)

    x1 = max(x - r, 0)
    x2 = min(x + r, image.shape[1])

    y1 = max(y - r, 0)
    y2 = min(y + r, image.shape[0])

    roi = roi[y1:y2, x1:x2]

    return roi


def preprocess(image, x, y, r):

    roi = crop_circle(image, x, y, r)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Background removal
    background = cv2.GaussianBlur(gray, (51, 51), 0)

    gray = cv2.subtract(gray, background)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 1)

    return gray