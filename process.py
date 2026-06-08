import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Creating output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# CORE FUNCTIONS (MANUAL IMPLEMENTATIONS)
# ==========================================
def get_histogram(img_array):
    hist = np.zeros(256, dtype=np.int32)
    for pixel in img_array.ravel():
        hist[int(pixel)] += 1
    return hist

def get_he_lut(img_array):
    """Returns the Look-Up Table (Transfer Curve) for HE"""
    hist = get_histogram(img_array)
    cdf = hist.cumsum()
    if (cdf.max() - cdf.min()) == 0:
        return np.arange(256, dtype='uint8')
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return cdf_normalized.astype('uint8')

def power_law_transform(img_array, gamma, c=1.0):
    """s = c * r^gamma"""
    lut = np.array([np.clip(c * ((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)])
    lut = np.round(lut).astype('uint8')
    return lut[img_array], lut

def linear_percentile_stretch(img_array, low_pct=1, high_pct=99):
    """Linear Contrast Stretch remapping 1st-99th percentiles to [0, 255]"""
    flattened = img_array.flatten()
    sorted_pixels = np.sort(flattened)
    
    low_idx = int(len(sorted_pixels) * (low_pct / 100.0))
    high_idx = int(len(sorted_pixels) * (high_pct / 100.0))
    
    p_low = sorted_pixels[low_idx]
    p_high = sorted_pixels[high_idx]
    
    if p_high == p_low:
        lut = np.arange(256, dtype='uint8')
        return img_array.copy(), lut
        
    lut = np.zeros(256, dtype='uint8')
    for i in range(256):
        val = ((i - p_low) * 255.0) / (p_high - p_low)
        lut[i] = np.clip(np.round(val), 0, 255)
        
    return lut[img_array], lut

def histogram_specification(source_img, target_cdf):
    """Performs manual mapping from source CDF directly to a continuous target CDF shape"""
    hist = get_histogram(source_img)
    s_cdf = hist.cumsum() / source_img.size
    
    lut = np.zeros(256, dtype='uint8')
    g_idx = 0
    for r_idx in range(256):
        while g_idx < 255 and target_cdf[g_idx] < s_cdf[r_idx]:
            g_idx += 1
        lut[r_idx] = g_idx
        
    return lut[source_img.astype('uint8')], lut

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
try:
    if not os.path.exists('moon.png'):
        raise FileNotFoundError("Please make sure your moon image is saved as 'moon.png' in this folder!")

    # --- QUESTION 1: HISTOGRAM EQUALIZATION AND CONTRAST STRETCHING ---
    print("Processing Question 1...")
    base = Image.open('moon.png').convert('L')
    img = np.array(base)

    # Generate synthetic degraded profiles matching requirements
    dark = (img * 0.2).astype('uint8')
    bright = (img * 0.5 + 127).astype('uint8')
    low_con = (img * 0.2 + 100).astype('uint8')
    high_con, _ = power_law_transform(img, 2.0)

    images = {'Dark': dark, 'Bright': bright, 'Low_Contrast': low_con, 'High_Contrast': high_con}

    for name, data in images.items():
        he_lut = get_he_lut(data)
        he_res = he_lut[data]
        
        gamma_val = 0.4 if name == 'Dark' else 1.5
        gamma_res, gamma_lut = power_law_transform(data, gamma_val)

        # Plotting satisfying Question 1 matrix requirements (Image, Log Histogram, Active Curves)
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs[0, 0].imshow(data, cmap='gray', vmin=0, vmax=255); axs[0, 0].set_title(f'Original {name}')
        axs[1, 0].hist(data.flatten(), 256, range=[0, 256], color='gray', alpha=0.7)
        axs[1, 0].set_yscale('log')
        
        axs[0, 1].imshow(he_res, cmap='gray', vmin=0, vmax=255); axs[0, 1].set_title('After HE')
        axs[1, 1].hist(he_res.flatten(), 256, range=[0, 256], color='blue', alpha=0.7)
        axs[1, 1].set_yscale('log')
        
        axs[0, 2].plot(he_lut, color='red', lw=2.5, label='HE Curve')
        axs[0, 2].plot(gamma_lut, color='orange', lw=2, linestyle='--', label=f'Gamma {gamma_val}')
        axs[0, 2].set_title('Active Transfer Curves Map')
        axs[0, 2].set_xlim([0, 255]); axs[0, 2].set_ylim([0, 255])
        axs[0, 2].legend(loc='lower right')
        axs[0, 2].grid(True, linestyle='--', alpha=0.5)
        
        axs[1, 2].imshow(gamma_res, cmap='gray', vmin=0, vmax=255); axs[1, 2].set_title(f'Gamma Correction ({gamma_val})')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_analysis.png")
        plt.close()

    # --- QUESTION 2: REAL MOON ENHANCEMENT (MULTIPLE METHODS COMPARED) ---
    print("Processing Question 2...")
    real_moon = np.array(Image.open('moon.png').convert('L'))
    
    # Generate an analytical Gaussian target distribution matching your friend's notebook
    target_pdf = np.zeros(256)
    mean, std = 50, 30
    for i in range(256):
        target_pdf[i] = np.exp(-((i - mean) ** 2) / (2 * (std ** 2)))
    target_pdf /= np.sum(target_pdf)
    target_cdf = np.cumsum(target_pdf)
    
    # Processing pathways collecting both images and lookup-tables
    real_moon_he_lut = get_he_lut(real_moon)
    real_moon_he = real_moon_he_lut[real_moon]
    real_moon_spec, real_moon_spec_lut = histogram_specification(real_moon, target_cdf)
    real_moon_stretch, real_moon_stretch_lut = linear_percentile_stretch(real_moon, 1, 99)
    
    # Corrected Layout Matrix configuration matching array dataset counts explicitly
    fig, axs = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle("Question 2 Master Evaluation Matrix Sheet (Log Counts)", fontsize=14, fontweight='bold')
    
    columns_data = [
        {"img": real_moon, "title": "Original Moon", "color": "gray", "lut": None},
        {"img": real_moon_he, "title": "Global HE", "color": "blue", "lut": real_moon_he_lut},
        {"img": real_moon_spec, "title": "Histogram Specification", "color": "green", "lut": real_moon_spec_lut},
        {"img": real_moon_stretch, "title": "1-99% Contrast Stretch", "color": "orange", "lut": real_moon_stretch_lut}
    ]
    
    for col_idx, col in enumerate(columns_data):
        # Row 1: Visual Outputs
        axs[0, col_idx].imshow(col["img"], cmap='gray', vmin=0, vmax=255)
        axs[0, col_idx].set_title(col["title"], fontsize=12, fontweight='bold')
        axs[0, col_idx].axis('off')
        
        # Row 2: Log-Scaled Histograms
        axs[1, col_idx].hist(col["img"].flatten(), 256, range=[0, 256], color=col["color"], alpha=0.7)
        axs[1, col_idx].set_xlim([0, 255])
        axs[1, col_idx].set_yscale('log')
        axs[1, col_idx].grid(True, linestyle='--', alpha=0.3)
        axs[1, col_idx].set_title(f"{col['title']} Histogram")
        
        # Row 3: Transfer Curve Function Mapping Curves
        if col["lut"] is not None:
            axs[2, col_idx].plot(range(256), col["lut"], color='crimson', lw=2.5)
            axs[2, col_idx].set_title(f"{col['title']} $s = T(r)$ LUT")
        else:
            axs[2, col_idx].plot([0, 255], [0, 255], color='black', linestyle=':')
            axs[2, col_idx].set_title("Identity Line (Unchanged)")
            
        axs[2, col_idx].set_xlim([0, 255])
        axs[2, col_idx].set_ylim([0, 255])
        axs[2, col_idx].grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Question2_Moon_Comparison.png", dpi=150)
    plt.close()

    print("Pipeline Complete! Advanced metric matrices exported into 'output/' folder successfully. :D")

except Exception as e:
    print(f"Pipeline Execution Error: {e}")