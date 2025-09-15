"""
@file: PD_2025_09_03_a.py
@brief: Decrypt an GPG encrypted media file and load it into memory.
@date: [created: 2025-09-03, updated: 2025-09-14]
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

# %% Print image details
print(f"Image shape: {img1.shape}, dtype: {img1.dtype}")

# %%
# --- Display the image using matplotlib
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.show()

# %%
