import cv2
import numpy as np
from scipy.interpolate import Akima1DInterpolator
import time

def frames_to_video(frames, fps, output_path):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def akima_controller(video_path, fps_before, fps_after, output_path):
    start = time.time()

    frames = []
    cap = cv2.VideoCapture(video_path)

    # Read all frames from the video
    print("Reading video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))
    cap.release()

    print(f"Total frames read: {len(frames)}")

    # Perform Akima interpolation for FPS modification
    original_indices = np.arange(len(frames))
    new_frame_count = int(fps_after / fps_before * len(frames))
    new_indices = np.linspace(0, len(frames) - 1, new_frame_count)

    print(f"Interpolating frames using Akima interpolation... Target frame count: {new_frame_count}")

    # Initialize an array for new frames
    height, width, _ = frames[0].shape
    new_frames = []

    # Interpolate each channel separately
    for channel in range(3):  # B, G, R channels
        # Extract channel data for all frames
        channel_data = np.array([frame[:, :, channel] for frame in frames])

        # Perform Akima interpolation
        interpolator = Akima1DInterpolator(original_indices, channel_data)
        interpolated_channel_data = interpolator(new_indices)

        # Store interpolated frames channel-wise
        if channel == 0:
            interpolated_frames = np.zeros((new_frame_count, height, width, 3), dtype=np.float32)
        interpolated_frames[:, :, :, channel] = interpolated_channel_data

    # Clip values and convert to uint8
    new_frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in interpolated_frames]

    # Save the interpolated frames as a new video
    print("Writing interpolated video...")
    frames_to_video(new_frames, fps_after, output_path)

    end = time.time()
    print(f"Successfully interpolated video using Akima interpolation in {end - start:.2f} seconds")