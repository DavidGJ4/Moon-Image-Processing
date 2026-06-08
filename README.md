# Moon Image Enhancement — Homework 1

A Python image-processing pipeline that applies histogram equalization, gamma correction, contrast stretching, and histogram specification to a grayscale moon image. All core algorithms are implemented manually using look-up tables (LUTs) rather than high-level enhancement shortcuts.

**Author:** David Guerra  
**Course:** Image Processing — Homework 1

## Overview

The program takes a single input image (`moon.png`) and runs two main pipelines:

1. **Question 1 — Synthetic test cases:** Builds four degraded versions of the moon image (dark, bright, low contrast, high contrast), then applies histogram equalization and gamma correction to each. Produces comparison plots with images, log-scale histograms, and transfer curves.

2. **Question 2 — Real moon enhancement:** Compares four methods on the original moon image:
   - Global histogram equalization
   - Histogram specification (Gaussian target distribution)
   - Linear 1–99% percentile contrast stretch

## Methods Implemented

| Method | Description |
|--------|-------------|
| **Histogram Equalization (HE)** | Maps pixel intensities using the cumulative distribution function (CDF) to spread the histogram across the full 0–255 range. |
| **Power-Law / Gamma Correction** | Non-linear transform \(s = c \cdot r^\gamma\) for shadow lifting or highlight compression. |
| **Linear Percentile Stretch** | Clips the bottom and top percentiles (default 1% and 99%) and linearly rescales the remaining range to [0, 255]. |
| **Histogram Specification** | Matches the source image's CDF to a target distribution (Gaussian, μ=50, σ=30). |

## Requirements

- Python 3.9+
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)

## Setup

```bash
# Clone or download the repository, then:
cd Image-HWK1

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

Place your grayscale moon image in the project root as `moon.png`.

## Usage

```bash
python process.py
```

On success, the script prints progress messages and saves all output plots to the `output/` folder.

## Output Files

| File | Description |
|------|-------------|
| `Dark_analysis.png` | Dark image: HE, gamma (γ=0.4), histograms, and curves |
| `Bright_analysis.png` | Bright image: HE, gamma (γ=1.5), histograms, and curves |
| `Low_Contrast_analysis.png` | Low-contrast image analysis |
| `High_Contrast_analysis.png` | High-contrast image analysis |
| `Question2_Moon_Comparison.png` | Full 3×4 comparison matrix for the real moon image |

## Project Structure

```
Image-HWK1/
├── process.py              # Main pipeline and algorithm implementations
├── moon.png                # Input grayscale moon image (required)
├── moon-documentation.tex  # LaTeX source for the written report
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── output/                 # Generated plots (created on run)
```

## Notes

- The synthetic test images in Question 1 are generated programmatically from `moon.png` using simple intensity scaling and power-law transforms — no separate input files are needed.
- Histogram computation in `get_histogram()` is implemented with an explicit pixel loop, as required by the assignment.
- If `moon.png` is missing, the script exits with a clear error message.
