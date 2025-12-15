import cv2
import numpy as np
import os

# === CONFIGURATION ===
input_folder = r"Location of the old_photos folder"
output_folder = r"Location of the restored_photos folder"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def add_label(image, text, color=(0, 255, 0)):
    
    h, w = image.shape[:2]
    # Add a small black bar at the top for text clarity
    cv2.rectangle(image, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return image

print(f"Starting comparison processing from: {input_folder}")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        print(f"Processing: {filename}...")

        # Get dimensions for layout construction
        h, w, c = img.shape

        # A. NLM Denoising (The "Standard" for restoration)
        nlm = cv2.fastNlMeansDenoisingColored(img, None, h=6,hColor=6, templateWindowSize=7, searchWindowSize=21)

        # B. Bilateral Filter
        bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        # C. Gaussian Blur
        gaussian = cv2.GaussianBlur(img, (5, 5), 1.2)

        # D. Median Blur
        median = cv2.medianBlur(img, 5)

        
        # GENERATE MASKS (Common for Inpainting)
        
        # We use the NLM result to detect scratches as it's the cleanest
        gray_dn = cv2.cvtColor(nlm, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_dn, 100, 180)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        
        # GENERATE COUNTERPARTS (INPAINTING)
        

        # A. Telea
        telea = cv2.inpaint(nlm, mask, 2, cv2.INPAINT_TELEA)

        # B. Navier-Stokes
        ns_inpaint = cv2.inpaint(nlm, mask, 3, cv2.INPAINT_NS)

       
        # GENERATE COUNTERPARTS (ENHANCEMENT)
        

        # A. Gamma Correction (Applied to Telea result)
        gamma_val = 1.2
        gamma_corrected = np.array(255 * (telea / 255) ** (1 /gamma_val), dtype=np.uint8)

        # B. CLAHE (Applied to Telea result)
        lab = cv2.cvtColor(telea, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
        cl = clahe_obj.apply(l)
        clahe_result = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        
        # BUILD THE LAYOUT (Header + 2x4)
        

        # Labeling images
        img_lbl       = add_label(img.copy(), "ORIGINAL INPUT", (255, 255, 255))

        # Row 1 Labels
        nlm_lbl       = add_label(nlm.copy(), "1. NLM Denoise", (0, 255, 0))
        bilateral_lbl = add_label(bilateral.copy(), "2. Bilateral Filter", (0, 255, 255))
        gaussian_lbl  = add_label(gaussian.copy(), "3. Gaussian Blur",(0, 255, 255))
        median_lbl    = add_label(median.copy(), "4. Median Blur", (0,255, 255))

        # Row 2 Labels
        telea_lbl     = add_label(telea.copy(), "5. Inpaint: Telea",(0, 255, 0))
        ns_lbl        = add_label(ns_inpaint.copy(), "6. Inpaint: NS",(0, 255, 255))
        gamma_lbl     = add_label(gamma_corrected.copy(), "7. Enhance:Gamma", (0, 255, 255))
        clahe_lbl     = add_label(clahe_result.copy(), "8. Enhance:CLAHE", (0, 255, 0))

        # --- Construct 2x4 Grid ---
        row1 = cv2.hconcat([nlm_lbl, bilateral_lbl, gaussian_lbl, median_lbl])
        row2 = cv2.hconcat([telea_lbl, ns_lbl, gamma_lbl, clahe_lbl])
        grid_2x4 = cv2.vconcat([row1, row2])

        # --- Construct Header (Original Image Centered) ---
        total_width = 4 * w  # Width of 4 images side-by-side

        # Create black background for the header
        header_bg = np.zeros((h, total_width, 3), dtype=np.uint8)

        # Calculate centering coordinates
        start_x = (total_width - w) // 2
        end_x = start_x + w

        # Place original image in the center
        header_bg[0:h, start_x:end_x] = img_lbl

        # --- Combine Header and Grid ---
        final_layout = cv2.vconcat([header_bg, grid_2x4])

        # === SAVE ===
        save_name = f"grid_compare_2x4_{filename}"
        save_path = os.path.join(output_folder, save_name)

        cv2.imwrite(save_path, final_layout)
        print(f"Saved Grid: {save_name}")

print("Batch processing complete.")
cv2.destroyAllWindows()