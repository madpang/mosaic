"""
@file: PD_2025_09_03_a.py
@brief: Create a wallpaper that is suitable for a 3024-by-1964 display from an encrypted media asset
@date: [created: 2025-09-03, updated: 2025-09-17]
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

# --- Clip the image into the core area
img3 = img2[0:3456, 0:5120]

# --- Further clip and resize to fit into the target canvas
img4 = cv2.resize(img3[52:52+2946, 292:292+4536], (3024, 1964), interpolation=cv2.INTER_LINEAR)

# --- Apply manual contrast adjustment curve
# @note: This curve maps 8-bit grayscale values for contrast enhancement
curve = np.array([
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2,
	2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, 8, 9, 9,
	10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19, 20, 21,
	22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37,
	38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56,
	57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 71, 72, 73, 75, 76, 78,
	79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102,
	103, 105, 106, 108, 109, 111, 113, 114, 116, 117, 119, 120, 122, 124, 125, 127,
	128, 130, 131, 133, 135, 136, 138, 139, 141, 142, 144, 146, 147, 149, 150, 152,
	153, 155, 156, 158, 159, 161, 162, 164, 165, 167, 168, 170, 171, 173, 174, 176,
	177, 179, 180, 182, 183, 184, 186, 187, 189, 190, 191, 193, 194, 195, 197, 198,
	199, 201, 202, 203, 204, 206, 207, 208, 209, 210, 212, 213, 214, 215, 216, 217,
	218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 232, 233,
	234, 235, 236, 237, 238, 238, 239, 240, 241, 241, 242, 243, 243, 244, 245, 245,
	246, 246, 247, 247, 248, 249, 249, 249, 250, 250, 251, 251, 252, 252, 252, 253,
	253, 253, 253, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255
], dtype=np.uint8)

# Apply the curve adjustment to img4
img5 = curve[img4]

# [CHECKPOINT]
print(f"img0.shape = {img0.shape}, img1.shape = {img1.shape}, img2.shape = {img2.shape}, img3.shape = {img3.shape}, img4.shape = {img4.shape}")
# --- Display the processed image using Pillow (opens in separate window)
Image.fromarray(img4).show(title="Processed Image")
Image.fromarray(img5).show(title="Processed Image")

# %%
# print length of curve
print(f"Length of curve: {len(curve)}")
# %%
