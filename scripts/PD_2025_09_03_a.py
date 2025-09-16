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
	P = padded.shape[1]
	flat = padded.ravel()

	pos = np.argwhere(kernel)
	offsets = pos[:, 0] * P + pos[:, 1]
	K = offsets.size
	if K == 0:
		raise ValueError("kernel must contain at least one 1.")
	if not (1 <= order <= K):
		raise ValueError(f"order must be in [1, {K}], got {order}.")
	k0 = order - 1 # zero-based index

	out = np.empty_like(image)
	neigh = np.empty(K, dtype=image.dtype)  # fixed buffer for neighbors

	# Apply order filter
	for i in range(H):
		base = i * P
		for j in range(W):
			# Gather neighborhood into fixed buffer (no inner Python loop)
			np.take(flat, base + j + offsets, out=neigh)
			# In-place kth selection (no temporary copy)
			neigh.partition(k0)
			out[i, j] = neigh[k0]

	return out

# %% Main Execution
# --- Workspace Setup
# @note: Assume the directory structure is `<project_root>/scripts/<this_script>.py`
ws_dir = Path(__file__).parent.parent

# --- Load the encrypted image
img0 = decrypt_and_load_image(ws_dir.joinpath("media").joinpath("beauty-relic.asc"))

# --- Convert image to gray scale using explicit formula
# @note: OpenCV uses BGR format, so we need to extract B, G, R channels
B, G, R = cv2.split(img0)
img1 = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)

# --- Remove the watermark (A1 to A2 processing equivalent to MATLAB)
# Extract patch from specific coordinates (MATLAB: A1(2766: 3342, 1614 : 4106))
img2 = img1.copy()
# @note: MATLAB uses 1-indexed, Python uses 0-indexed, so subtract 1
patch_roi = img2[2765:3342, 1613:4106]  # 577×2493 patch

img_patch = ordfilt2(
	ordfilt2(patch_roi,
		1,                              # minimal filter
		np.ones((7, 7), dtype=np.uint8),
		"symmetric"),
	(11 * 25 + 1) // 2,					# median filter
	np.ones((11, 25), dtype=np.uint8),  # trial & error parameter
	"symmetric")
	
# Assign filtered patch back with offset and adjustment
p_h, p_w = img_patch.shape
patch_in = img_patch[100:p_h-10, 10:p_w-40]  # Extract interior region

# Assign back to img2 with brightness adjustment (+4)
# @note: Those are trial & error parameters
patch_h, patch_w = patch_in.shape
img2[2865:2865+patch_h, 1623:1623+patch_w] = np.clip(patch_in.astype(np.int16) + 4, 0, 255).astype(np.uint8)

# --- Clip the image
img3 = img2[0:3456, 0:5120]

# [CHECKPOINT]
print(f"img0.shape = {img0.shape}, img1.shape = {img1.shape}, img2.shape = {img2.shape}, img3.shape = {img3.shape}")
# --- Display the processed image using Pillow (opens in separate window)
Image.fromarray(img3).show(title="Processed Image")

# %%
