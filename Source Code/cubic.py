import cv2
import numpy as np
import subprocess
import time

def frames_to_video(frames, fps, output_path):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def cubic_controller(video_path, fps_before, fps_after, output_path):
    start = time.time()

    frames = []
    cap = cv2.VideoCapture(video_path)
    # Read all frames from the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Perform cubic interpolation for FPS modification
    new_frames = []
    new_frames_count = int(fps_after / fps_before * len(frames))
    diff = (len(frames) - 1) / (new_frames_count - 1)

    print(len(frames), new_frames_count)

    for i in range(new_frames_count):
        # Calculate the interpolated index
        coord = i * diff
        lower_idx = int(np.floor(coord))
        upper_idx = min(lower_idx + 1, len(frames) - 1)
        next_upper_idx = min(lower_idx + 2, len(frames) - 1)
        prev_lower_idx = max(lower_idx - 1, 0)

        # Calculate interpolation weights
        alpha = coord - lower_idx

        # Fetch frames for cubic interpolation
        p0 = frames[prev_lower_idx].astype(np.float32)
        p1 = frames[lower_idx].astype(np.float32)
        p2 = frames[upper_idx].astype(np.float32)
        p3 = frames[next_upper_idx].astype(np.float32)

        # Perform cubic interpolation for each channel
        def cubic_hermite(t, p0, p1, p2, p3):
            return (
                (-0.5 * t + t ** 2 - 0.5 * t ** 3) * p0 +
                (1 - 2.5 * t ** 2 + 1.5 * t ** 3) * p1 +
                (0.5 * t + 2 * t ** 2 - 1.5 * t ** 3) * p2 +
                (-0.5 * t ** 2 + 0.5 * t ** 3) * p3
            )

        interpolated_frame = cubic_hermite(alpha, p0, p1, p2, p3)
        interpolated_frame = np.clip(interpolated_frame, 0, 255).astype(np.uint8)
        new_frames.append(interpolated_frame)

    # Save the interpolated frames as a new video
    frames_to_video(new_frames, fps_after, output_path)

    end = time.time()

    print(f"Successfully interpolated video using cubic interpolation in {end - start} seconds")