import cv2
import numpy as np
import time

def frames_to_video(frames, fps, output_path):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def nearest_neighbor_controller(video_path, fps_before, fps_after, output_path):
    start = time.time()

    frames = []
    cap = cv2.VideoCapture(video_path)

    # Read all frames from the video
    print("Reading video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Total frames read: {len(frames)}")

    # Perform nearest-neighbor interpolation for FPS modification
    original_indices = np.arange(len(frames))
    new_frame_count = int(fps_after / fps_before * len(frames))
    new_indices = np.linspace(0, len(frames) - 1, new_frame_count)

    print(f"Interpolating frames using nearest neighbor... Target frame count: {new_frame_count}")

    # Find the nearest frame for each target index
    nearest_indices = np.round(new_indices).astype(int)

    # Ensure indices are within bounds
    nearest_indices = np.clip(nearest_indices, 0, len(frames) - 1)

    # Create the new frames by selecting nearest neighbors
    new_frames = [frames[idx] for idx in nearest_indices]

    # Save the interpolated frames as a new video
    print("Writing interpolated video...")
    frames_to_video(new_frames, fps_after, output_path)

    end = time.time()
    print(f"Successfully interpolated video using nearest neighbor in {end - start:.2f} seconds")
