import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
import sys

# ---------- Helper: Orientation Field ----------
def compute_orientation_field(img, block_size=16):
    gy, gx = np.gradient(np.float32(img))
    gxx = cv2.boxFilter(gx*gx, -1, (block_size, block_size))
    gyy = cv2.boxFilter(gy*gy, -1, (block_size, block_size))
    gxy = cv2.boxFilter(gx*gy, -1, (block_size, block_size))
    orientations = 0.5 * np.arctan2(2*gxy, (gxx - gyy + 1e-6))
    return orientations

def detect_core_delta(orient, mag_thresh=0.5):
    # Very basic singularity detector: finds max curvature for core,
    # and highest orientation change for delta
    h, w = orient.shape
    orient_deg = np.degrees(orient)
    # gradient of orientation field
    gy, gx = np.gradient(orient_deg)
    curv = np.abs(gx)+np.abs(gy)
    core = np.unravel_index(np.argmax(curv), curv.shape)
    delta = np.unravel_index(np.argmax(-curv), curv.shape)
    # (y,x) -> (x,y)
    return (core[1], core[0]), (delta[1], delta[0])

# ---------- Helper: Ridge count ----------
def count_ridges_between_core_delta(skel, core, delta):
    mask = np.zeros_like(skel)
    cv2.line(mask, core, delta, 255, 1)
    coords = np.argwhere(mask>0)
    values = [skel[y,x] for (y,x) in coords]
    # count black-white-black transitions
    crossings = 0
    for i in range(1,len(values)):
        if values[i] != values[i-1]:
            crossings += 1
    return crossings // 2

# ---------- Main ----------
def preprocess_and_save(input_path, target_size=(128,128), debug=False):
    # Step 1: Read & grayscale
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image at {input_path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(img_gray)

    # Step 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_norm = clahe.apply(img_gray)

    # Step 3: Denoise
    img_denoised = cv2.medianBlur(img_norm, 3)

    # Step 4: BlackHat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_blackhat = cv2.morphologyEx(img_denoised, cv2.MORPH_BLACKHAT, kernel)
    img_enhanced = cv2.add(img_denoised, img_blackhat)

    # Step 5: Threshold
    if mean_intensity > 180:
        _, img_bin = cv2.threshold(img_enhanced, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        detected_type = "inkpad"
    else:
        img_bin = cv2.adaptiveThreshold(img_enhanced, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 5)
        detected_type = "photo"

    # Step 6: ROI crop
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        roi = img_bin[y:y+h, x:x+w]
    else:
        roi = img_bin

    # Step 7: Normalize + resize
    roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)

    # Step 8: Skeletonization
    if not hasattr(cv2, "ximgproc"):
        raise ImportError("cv2.ximgproc not found. Install opencv-contrib-python.")
    skeleton = cv2.ximgproc.thinning(roi_resized, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Step 9: Orientation field → core/delta
    orientation = compute_orientation_field(roi_resized)
    core, delta = detect_core_delta(orientation)

    # Step 10: Ridge count
    ridge_count = count_ridges_between_core_delta(skeleton, core, delta)

    # Step 11: Minutiae (Crossing Number)
    endings, bifurcations = [], []
    h, w = skeleton.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y,x]==255:
                neighbors = [
                    skeleton[y-1,x-1], skeleton[y-1,x], skeleton[y-1,x+1],
                    skeleton[y,x-1], skeleton[y,x+1],
                    skeleton[y+1,x-1], skeleton[y+1,x], skeleton[y+1,x+1]
                ]
                cn = sum(n==255 for n in neighbors)
                if cn==1 and 10<x<w-10 and 10<y<h-10:
                    endings.append((x,y))
                elif cn>=3 and 10<x<w-10 and 10<y<h-10:
                    bifurcations.append((x,y))

    # Step 12: Cluster (DBSCAN)
    def cluster_points(points, eps=8):
        if not points: return []
        pts = np.array(points)
        labels = DBSCAN(eps=eps, min_samples=1).fit(pts).labels_
        clustered = []
        for lab in set(labels):
            grp = pts[labels==lab]
            clustered.append(tuple(np.mean(grp, axis=0).astype(int)))
        return clustered

    endings = cluster_points(endings)
    bifurcations = cluster_points(bifurcations)

    # Step 13: Overlay
    skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    cv2.circle(skeleton_bgr, core, 3, (255,0,0), -1)    # blue core
    cv2.circle(skeleton_bgr, delta, 3, (0,255,255), -1) # yellow delta
    for (x,y) in endings: cv2.circle(skeleton_bgr, (x,y), 2, (0,0,255), -1)
    for (x,y) in bifurcations: cv2.circle(skeleton_bgr, (x,y), 2, (0,255,0), -1)

    # Step 14: Save PNG to memory
    img_final = Image.fromarray(roi_resized)
    temp_io = BytesIO()
    img_final.save(temp_io, format="PNG")
    temp_io.seek(0)

    # Debug view
    if debug:
        steps = [img_gray, img_norm, img_enhanced, img_bin, roi_resized, skeleton, skeleton_bgr]
        titles = ["Gray","CLAHE","Enhanced","Binarized","ROI","Skeleton","Features"]
        plt.figure(figsize=(20,5))
        for i,(im,t) in enumerate(zip(steps,titles)):
            plt.subplot(1,len(steps),i+1)
            plt.imshow(im if len(im.shape)==2 else cv2.cvtColor(im,cv2.COLOR_BGR2RGB), cmap="gray")
            plt.title(t, fontsize=10)
            plt.axis("off")
        plt.suptitle(
            f"Type={detected_type} | End={len(endings)} Bif={len(bifurcations)} Ridge={ridge_count}",
            fontsize=12)
        plt.tight_layout()
        plt.show()

    return temp_io, detected_type, mean_intensity, endings, bifurcations, core, delta, ridge_count

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    _,dtype,mean,end,bif,core,delta,rc = preprocess_and_save(image_path, debug=True)
    print(f"Mean={mean:.2f}, Type={dtype}, End={len(end)}, Bif={len(bif)}, Core={core}, Delta={delta}, RidgeCount={rc}")
