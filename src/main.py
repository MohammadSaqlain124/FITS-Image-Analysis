from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


# Load FITS file
hdul = fits.open('m42_40min_red.fits')

# Show FITS structure
hdul.info()

# Extract image data
image_data = hdul[0].data


# Log Scaling (Enhancement)
image_data_log = np.log(image_data + 1)

vmin = np.percentile(image_data_log, 1)
vmax = np.percentile(image_data_log, 97)


# Brightness Analysis
mean_brightness = np.mean(image_data)
max_brightness = np.max(image_data)
min_brightness = np.min(image_data)

print("\n--- Brightness Analysis ---")
print("Mean Brightness:", mean_brightness)
print("Max Brightness:", max_brightness)
print("Min Brightness:", min_brightness)


# Bright Region Detection
threshold = np.percentile(image_data, 99.5)
bright_mask = image_data > threshold


# COMBINED VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1️⃣ Log Scaled Image
im = axes[0, 0].imshow(image_data_log, cmap='gray', vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Log Scaled FITS Image (M42)")
axes[0, 0].set_xlabel("X Pixels")
axes[0, 0].set_ylabel("Y Pixels")
fig.colorbar(im, ax=axes[0, 0])

# 2️⃣ Histogram
axes[0, 1].hist(image_data.flatten(), bins=100, log=True, color='gray')
axes[0, 1].set_title("Pixel Intensity Distribution")
axes[0, 1].set_xlabel("Intensity")
axes[0, 1].set_ylabel("Frequency (log)")

# 3️⃣ Original (Log Image again for comparison)
axes[1, 0].imshow(image_data_log, cmap='gray', vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Original Image")

# 4️⃣ Bright Regions Highlighted
axes[1, 1].imshow(image_data_log, cmap='gray', vmin=vmin, vmax=vmax)
axes[1, 1].imshow(bright_mask, cmap='jet', alpha=0.5)
axes[1, 1].set_title("Bright Regions Detected")

plt.tight_layout()
plt.show()