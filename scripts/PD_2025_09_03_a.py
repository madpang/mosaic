"""
@file: PD_2025_09_03_a.py
@brief: Decrypt an GPG encrypted media file and load it into memory.
@date: [created: 2025-09-03, updated: 2025-09-15]
"""

# %% Environment Setup
# --- Package Imports
import os
import subprocess
import tempfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# --- Function Definitions
def decrypt_and_load_image(file_path):
	"""
	@brief: Decrypt a GPG encrypted image file and load it into memory.
	
	@param[out]:
	- np.ndarray: Loaded image as BGR array (OpenCV format).

	@param[in]:
	- file_path (Path): Path to the encrypted .asc file.

	@note: With this function, the temporary decrypted file will NOT remain on disk and the data is loaded into memory.
	"""
	
	# Create a temporary file for the decrypted image
	with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as temp_file:
		output_path = temp_file.name

		# @note: Since `tempfile` will actually create an empty file, we need to to instruct GPG to overwrite it.
		subprocess.run(["gpg", "--yes", "--decrypt", "--output", output_path, str(file_path)], capture_output=True)

		# Check if the output file has non-zero size
		if os.path.getsize(output_path) == 0:
			raise ValueError(f"Decryption failed: output file '{output_path}' is empty.")

		# Load the decrypted image using OpenCV
		image = cv2.imread(output_path)

		if image is None:
			raise ValueError(f"Failed to load image from {output_path}")

		return image

def ordfilt2(image: np.ndarray, order: int, kernel: np.ndarray, boundary: str = "symmetric") -> np.ndarray:
	"""
	@brief: 2D order-statistic filtering (equivalent to MATLAB's ordfilt2)

	@param[out]:
	- Filtered image (same size as input)
	
	@param[in]:
	- image: Input image (2D numpy array)
	- order: Order statistic (1 = minimum, kernel_size = maximum)
	- kernel: Structuring element (2D numpy array of 0s and 1s)
	- boundary: Boundary condition handling ("symmetric" or "zeros")

	@note: For median filtering, use order = (kernel_size + 1) // 2
	"""
	
	# Get kernel dimensions
	kh, kw = kernel.shape
	ph, pw = kh // 2, kw // 2

	# Pad the image
	if boundary == "symmetric":
		padded = np.pad(image, ((ph, ph), (pw, pw)), mode="symmetric")
	elif boundary == "zeros":
		padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
	else:
		raise ValueError('boundary must be "symmetric" or "zeros"')
	
	# precompute positions where kernel == 1 and the fixed neighborhood size K
	positions = np.argwhere(kernel)                         # list of [ki, kj]
	K = positions.shape[0]
	if K == 0:
		raise ValueError("kernel must contain at least one 1.")
	if not (1 <= order <= K):
		raise ValueError(f"order must be in [1, {K}], got {order}.")
	
	# Initialize output
	H, W = image.shape
	out = np.empty_like(image)

	# fixed-size buffer for neighbors
	neigh = np.empty(K, dtype=image.dtype)
	k0 = order - 1  # zero-based index	
	
	# Apply order filter
	for i in range(H):
		for j in range(W):
			# fill neighborhood values from padded image
			for t, (ki, kj) in enumerate(positions):
				neigh[t] = padded[i + ki, j + kj] # TL of window is (i, j) in padded
			# sort and pick the k0-th element
			out[i, j] = np.sort(neigh, kind="quicksort")[k0]

	return out

# %% Main Execution
# --- Workspace Setup
# @note: Assume the directory structure is `<project_root>/scripts/<this_script>.py`
ws_dir = Path(__file__).parent.parent
# --- Load the encrypted image
img0 = decrypt_and_load_image(ws_dir.joinpath("media").joinpath("beauty-relic.asc"))

# [CHECKPOINT]
# --- Print original image details
print(f"Original Image shape: {img0.shape}, dtype: {img0.dtype}")
# --- Display the original image using Pillow (opens in separate window)
Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)).show(title="Decrypted Original Image")

# %%
# --- Convert image to gray scale using explicit formula
# Note: OpenCV uses BGR format, so we need to extract B, G, R channels
B, G, R = cv2.split(img0)
img1 = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)

# [CHECKPOINT]
# --- Print grayscale image details
print(f"Grayscale Image shape: {img1.shape}, dtype: {img1.dtype}")
# --- Display the grayscale image using Pillow (opens in separate window)
Image.fromarray(img1).show(title="Grayscale Image")

# %%
# --- Remove the watermark (A1 to A2 processing equivalent to MATLAB)
# Extract patch from specific coordinates (MATLAB: A1(2766: 3342, 1614 : 4106))
# @note: MATLAB uses 1-indexed, Python uses 0-indexed, so subtract 1
patch_roi = img1[2765:3342, 1613:4106]  # 577×2493 patch

print(f"Patch shape: {patch_roi.shape}, dtype: {patch_roi.dtype}")
Image.fromarray(patch_roi).show(title="Patch ROI")

# %%
# Apply order-statistic filtering (equivalent to MATLAB's ordfilt2)
# MATLAB: ordfilt2(A1_patch, 1, ones(7, 7), "symmetric")
kernel_7x7 = np.ones((7, 7), dtype=np.uint8)
patch_ordfilt = ordfilt2(patch_roi, 1, kernel_7x7, "symmetric")

print("Applied order-statistic filtering (minimal filtering)")
Image.fromarray(patch_ordfilt).show(title="Patch ROI - After Filtering")

# %%
# Apply 2D median filtering using ordfilt2 (median is the middle order statistic)
# MATLAB: medfilt2(..., [11, 25], "symmetric")
# For median filtering: order = (kernel_size + 1) // 2
kernel_11x25 = np.ones((11, 25), dtype=np.uint8)
median_order = (11 * 25 + 1) // 2  # Middle order for median
patch_filtered = ordfilt2(patch_ordfilt, median_order, kernel_11x25, "symmetric")

print("Applied 2D median filtering")
Image.fromarray(patch_filtered).show(title="Patch ROI - After Filtering")

# %%
# Create A2 by copying A1 and assigning the filtered patch back
img2 = img1.copy()

# Assign filtered patch back with offset and adjustment
# MATLAB: A2(2866: 3332, 1624 : 4066) = A_patch(101 : (size(A_patch, 1) - 10), 11 : (size(A_patch, 2) - 40)) + 4
patch_height, patch_width = patch_filtered.shape
patch_interior = patch_filtered[100:patch_height-10, 10:patch_width-40]  # Extract interior region

# Assign back to img2 with brightness adjustment (+4)
target_h, target_w = patch_interior.shape
img2[2865:2865+target_h, 1623:1623+target_w] = np.clip(patch_interior.astype(np.int16) + 4, 0, 255).astype(np.uint8)

print(f"Watermark removal completed. Processed image shape: {img2.shape}")

# [CHECKPOINT]
# --- Display the processed image using Pillow (opens in separate window)
Image.fromarray(img2).show(title="Processed Image (A2) - Watermark Removed")

# %%
