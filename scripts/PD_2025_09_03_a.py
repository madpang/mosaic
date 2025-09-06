"""
@file: PD_2025_09_03_a.py
@brief: Decrypt an GPG encrypted media file and load it into memory.
@date: [created: 2025-09-03, updated: 2025-09-06]
"""
# %%
import os
import subprocess
import tempfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# === Function Definitions
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

# === Main Execution
# --- Workspace Setup
# @note: Assume the directory structure is `<project_root>/scripts/<this_script>.py`
ws_dir = Path(__file__).parent.parent
# --- Load the encrypted image
img0 = decrypt_and_load_image(ws_dir.joinpath("media").joinpath("beauty-relic.asc"))

# %%
# --- Display the image using matplotlib
plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
