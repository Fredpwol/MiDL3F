

import cv2
import torch
import urllib.request
import numpy as np

import time
from openal import * 
from itertools import cycle
# from PIL import Image
url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

"""Load a model (see [https://github.com/intel-isl/MiDaS/#Accuracy](https://github.com/intel-isl/MiDaS/#Accuracy) for an overview)"""

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

"""Move model to GPU if available"""

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

"""Load transforms to resize and normalize the image for large or small model"""

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

"""Load image and apply transforms"""

# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input_batch = transform(img).to(device)

"""Predict and resize to original resolution"""

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Couldn't open the camera.")
    exit()


def create_segments(matrix, num_splits):
    matrix_rows, matrix_cols = matrix.shape
    split_size = matrix_rows // num_splits
    
    segments = []
    
    for i in range(num_splits):
        for j in range(num_splits):
            segment = matrix[i * split_size: (i + 1) * split_size, j * split_size: (j + 1) * split_size]
            segments.append(segment)
    
    return segments


def get_image_super_pixels(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # Create a SLIC superpixel object
    slic = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=cv2.ximgproc.SLIC, region_size=60, ruler=60.0)

    # Perform superpixel segmentation
    slic.iterate()

    slic.enforceLabelConnectivity()

    # Get the number of superpixels
    num_superpixels = slic.getNumberOfSuperpixels()

    # Get the labels map (superpixel assignments)
    labels = slic.getLabels()

    return labels, num_superpixels


def apply_superpixel_mean(image, labels, num_pixels):
        # Iterate through superpixels
    for superpixel_label in range(num_pixels):
        # Find all pixels with the current superpixel label
        superpixel_mask = (labels == superpixel_label)

        # Extract the pixels within the superpixel
        superpixel_pixels = image[superpixel_mask]

        # Calculate the mean value for the superpixel
        mean_value = np.mean(superpixel_pixels, axis=0)

        # Set all the pixels within the superpixel to the mean value
        image[superpixel_mask] = mean_value

    # Convert the Lab image back to BGR color space
    return image




while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (OpenCV uses BGR)

    labels, num_pix = get_image_super_pixels(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MiDaS
    input_tensor = transform(rgb_frame).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    h, w = output.shape

    n_segments = 3

    # segments = create_segments(output, 4)

    # print(np.array([seg.mean() for seg in segments]).reshape(4, 4))
    # Rescale the depth map for visualization
    depth_colormap = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)

    depth_colormap = (depth_colormap*255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_colormap,  cv2.COLORMAP_MAGMA)

    depth_w_superpixel = apply_superpixel_mean(depth_colormap, labels, num_pix)

    cv2.imshow("Depth Map", depth_w_superpixel)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()



def play_audio(segment_mean):
    x_pos = 5
    sleep_time = 5
    source = oalOpen("tone5.wav")
    source.set_position([0, 0, 0])
    source.set_looping(False)
    source.play()
    listener = Listener()
    listener.set_position([0, 0, 0])

    pos = cycle(np.linspace(0, 360, sleep_time))
    print(np.linspace(0, 360, sleep_time))
    while source.get_state() == AL_PLAYING:
        n = np.radians(next(pos))
        source.set_position([np.cos(n) * 5, np.sin(n) * 2, np.sin(n) * 2])
        print("Playing at: {0}".format(source.position))
        time.sleep(sleep_time)
        x_pos *= -1
    
    oalQuit()
