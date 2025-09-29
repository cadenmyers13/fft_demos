import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output

# Open default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Downscale for faster computation
frame_width, frame_height = 320, 240

# Set up matplotlib figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
im1 = ax1.imshow(np.zeros((frame_height, frame_width)), cmap='gray', vmin=0, vmax=255)
ax1.set_title("Webcam Feed")
im2 = ax2.imshow(np.zeros((frame_height, frame_width)), cmap='inferno', vmin=0, vmax=1, origin='lower')
# add colorbar to ax2
# plt.colorbar(im2, ax=ax2)
ax2.set_title("Fourier Transform (Magnitude)")
plt.tight_layout()

def update(frame):
    ret, frame = cap.read()
    if not ret:
        return [im1, im2]

    # Resize for speed
    frame_small = cv2.resize(frame, (frame_width, frame_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    im1.set_array(gray)
    
    # Compute FFT and magnitude
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    fshift[gray.shape[0]//2, gray.shape[1]//2] = 0  # remove DC
    magnitude_spectrum = np.log1p(np.abs(fshift))
    magnitude_spectrum /= magnitude_spectrum.max() + 1e-6
    im2.set_array(magnitude_spectrum)

    return [im1, im2]

# Start the animation
ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.show()
