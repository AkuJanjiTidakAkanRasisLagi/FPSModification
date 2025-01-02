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

def linear_controller(video_path, fps_before, fps_after, output_path):
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

    # Perform linear interpolation for FPS modification
    new_frames = []
    new_frames_count = int(fps_after / fps_before * len(frames))
    diff = (len(frames) - 1) / (new_frames_count - 1)
    
    for i in range(new_frames_count):
        # Calculate the interpolated index
        coord = i * diff
        lower_idx = int(np.floor(coord))
        upper_idx = min(lower_idx + 1, len(frames) - 1)

        # Calculate interpolation factor
        alpha = coord - lower_idx

        # Perform blending for interpolation
        lower_frame = frames[lower_idx].astype(np.float32)
        upper_frame = frames[upper_idx].astype(np.float32)
        interpolated_frame = ((1 - alpha) * lower_frame + alpha * upper_frame).astype(np.uint8)

        new_frames.append(interpolated_frame)

    # Save the interpolated frames as a new video
    frames_to_video(new_frames, fps_after, output_path)

    end = time.time()
    print(f"succesfully interpolated video using linear interpolation in {end - start} seconds")
