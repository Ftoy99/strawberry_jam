import argparse
import math
import os.path

from PIL import Image
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import raft_small

import cv2
import numpy as np
import torch

# Device set cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load RAFT model
model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
model.eval()
transform = Raft_Small_Weights.DEFAULT.transforms()


def compute_optical_flow(frame1, frame2, index):
    sigma = 0.15
    # read the images
    frame1 = cv2.imread(frame1)
    frame2 = cv2.imread(frame2)

    # to bw
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Compute optical flow using Farneback method
    # https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 20, 5, 7, 1.2, 0)
    # # Create mask
    # mask = np.zeros((gray1.shape[0], gray1.shape[1], 3), dtype=np.uint8)
    # # Set image saturation to maximum value as we do not need it
    # mask[..., 1] = 255
    #
    # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #
    # # Sets image hue according to the optical flow
    # # direction
    # mask[..., 0] = angle * 180 / np.pi / 2
    #
    # # Sets image value according to the optical flow
    # # magnitude (normalized)
    # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    #
    # # Converts HSV to RGB (BGR) color representation
    # rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    #
    # # Opens a new window and displays the output frame
    # cv2.imshow("dense optical flow", rgb)
    # cv2.waitKey()

    u, v = flow[..., 0], flow[..., 1]

    # Compute motion magnitude and angle
    H, W = gray1.shape

    # opacities
    mag = np.sqrt(u ** 2 + v ** 2)
    m = np.minimum(1, mag / (sigma * np.sqrt(H ** 2 + W ** 2)))

    angle = np.arctan2(v, u)

    # Convert angle to HSV-like color representation
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Map angle to [0, 180]
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = (1 - m * 255).astype(np.uint8)  # Motion magnitude as brightness

    # Convert HSV to BGR
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the optical flow image
    frame_filename = os.path.join("optical_flow_frames", f"of_{index:04d}.png")
    print(f" Saving of_{index:04d}.png")
    cv2.imwrite(frame_filename, flow_bgr)
    return flow, flow_bgr, frame_filename


# Function to preprocess frames
def preprocess_frames(frame1, frame2):
    img1 = Image.fromarray(cv2.imread(frame1))
    img2 = Image.fromarray(cv2.imread(frame2))
    img1, img2 = transform(img1, img2)
    img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
    return img1, img2


def compute_optical_flow_raft(frame1_path, frame2_path, index):
    # Load frames
    frame1, frame2 = preprocess_frames(frame1_path, frame2_path)

    # Compute optical flow
    with torch.no_grad():
        list_of_flows = model(frame1, frame2)

    # Retrieve the final flow map
    flow = list_of_flows[-1].squeeze().cpu().numpy()

    # Save flow visualization
    flow_vis = visualize_optical_flow(flow)
    output_path = os.path.join("optical_flow_frames", f"flow_vis_{index:04d}.png")
    cv2.imwrite(output_path, flow_vis)
    print(f"Saved optical flow visualization: {output_path}")


def visualize_optical_flow(flow):
    sigma = 0.15
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)

    # Optical flow to rgb as described
    H, W = flow.shape[1:]  # dimensions
    u, v = flow[0, ...], flow[1, ...]  # Flows
    m = np.sqrt(u ** 2 + v ** 2) / (sigma * np.sqrt((H ** 2 + W ** 2)))
    print(f"Max magnitude: {np.max(m)}, Min magnitude: {np.min(m)}")

    magnitude = np.clip(m, 0, 1)  # clip instead of min scalar with int
    print(f"Max magnitude after clip: {np.max(magnitude)}")

    angle = np.arctan2(v, u)

    hue = (angle * 180 / np.pi / 2) * 255  # 0 to 180 degreees HSV OPENCV hue
    print(f"Max hue: {np.max(hue)}, Min hue: {np.min(hue)}")

    hsv[..., 0] = hue  # Hue represents flow direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # TODO
    # hsv[..., 2] = magnitude

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print(f"Max BGR value: {np.max(bgr)}, Min BGR value: {np.min(bgr)}")
    return bgr


def split_video_to_frames(video):
    frames = []
    # Create the output directory if it doesn't exist
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Open the video
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image in the 'frames' directory
        frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        frames.append(frame_filename)
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Frames saved to '{frames_dir}'.")
    return frames


if __name__ == '__main__':
    video = "3760967-uhd_3840_2160_25fps.mp4"
    video_path = os.path.join("videos", video)
    frames = split_video_to_frames(video_path)
    for i in range(len(frames)):
        if i != len(frames) - 1:
            compute_optical_flow(frames[i], frames[i + 1], i)
            # compute_optical_flow_raft(frames[i], frames[i + 1], i)
