import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
from PIL import Image

def extract_features(preprocessed_image_path):
    img = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(img_bin // 255).astype(np.uint8)

    # Minutiae detection
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    convolved = cv2.filter2D(skeleton, -1, kernel)
    minutiae_endings = np.sum(convolved == 11)
    minutiae_bifurcations = np.sum(convolved == 13)

    ridge_density = np.sum(skeleton) / skeleton.size

    # Core/Delta detection
    labels = measure.label(skeleton)
    regions = measure.regionprops(labels)
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        core_y, core_x = largest_region.centroid
        minr, minc, maxr, maxc = largest_region.bbox
        delta_x, delta_y = minc, minr
    else:
        core_x = core_y = delta_x = delta_y = 0

    ridge_count = int(np.sqrt((core_x - delta_x)**2 + (core_y - delta_y)**2) // 2)

    return {
        "minutiae_endings": int(minutiae_endings),
        "minutiae_bifurcations": int(minutiae_bifurcations),
        "ridge_density": float(ridge_density),
        "core_x": float(core_x),
        "core_y": float(core_y),
        "delta_x": float(delta_x),
        "delta_y": float(delta_y),
        "ridge_count_core_delta": ridge_count
    }
