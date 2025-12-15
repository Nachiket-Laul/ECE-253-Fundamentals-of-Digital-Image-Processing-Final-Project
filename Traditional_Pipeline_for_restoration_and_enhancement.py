import cv2
import numpy as np

# === Step 1: Read Image ===
img = cv2.imread('Input_image.jpg')
if img is None:
    raise ValueError("Image not found!")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Step 2: Mild NLM Denoising ===
denoised = cv2.fastNlMeansDenoisingColored(
    img, None,
    h=6,            
    hColor=6,
    templateWindowSize=7,
    searchWindowSize=21
)

# === Step 3: Scratch Detection  ===
gray_dn = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray_dn, 100, 180)   # increased thresholds = fewer false edges

kernel = np.ones((2, 2), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# === Step 4: Inpainting (Telea recommended for faces) ===
inpainted = cv2.inpaint(
    denoised,
    edges_closed,
    inpaintRadius=2,           # smaller radius = more natural texture
    flags=cv2.INPAINT_TELEA    # Telea works very well on portraits
)

# === Step 5: Gentle CLAHE (LAB) ===
lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=1.2,     # very gentle for portraits
    tileGridSize=(4, 4)
)
cl = clahe.apply(l)

clahe_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# === Step 6: Very Soft Unsharp Mask ===
blur = cv2.GaussianBlur(clahe_img, (3, 3), 1.0)
final = cv2.addWeighted(clahe_img, 1.05, blur, -0.05, 0)

# === Display and Save ===
cv2.imshow("Original", img)
cv2.imshow("Denoised", denoised)
cv2.imshow("Edges", edges_closed)
cv2.imshow("Inpainted", inpainted)
cv2.imshow("Final Restored", final)


cv2.imwrite("restored_old_photo_denoise.jpg", denoised)
cv2.imwrite("restored_old_photo_edge_detection.jpg", edges_closed)
cv2.imwrite("restored_old_photo_inpainted_Telea.jpg", inpainted)
cv2.imwrite("restored_old_photo_clahe.jpg", clahe_img)
cv2.imwrite("restored_photo_final.jpg", final)
cv2.waitKey(0)
cv2.destroyAllWindows()

