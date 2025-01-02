import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def pchip_controller(video_path, fps_before, fps_after, output_path, chunk_size=100):
    start = time.time()

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")

    new_frame_count = int(fps_after / fps_before * frame_count)
    print(f"Interpolating to {new_frame_count} frames...")

    # Initialize variables
    chunk_start_idx = 0
    interpolated_frames = []

    while chunk_start_idx < frame_count:
        print(f"Processing chunk starting at frame {chunk_start_idx}...")

        # Read a chunk of frames
        chunk_frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            chunk_frames.append(frame.astype(np.uint8))

        if len(chunk_frames) < 2:
            # Not enough frames for interpolation
            interpolated_frames.extend(chunk_frames)
            break

        # Calculate original indices and new indices for the chunk
        original_indices = np.linspace(chunk_start_idx, chunk_start_idx + len(chunk_frames) - 1, len(chunk_frames))
        chunk_new_indices = np.linspace(chunk_start_idx, chunk_start_idx + len(chunk_frames) - 1,
                                         int(fps_after / fps_before * len(chunk_frames)))

        # Interpolate each channel
        for channel in range(3):  # B, G, R channels
            # Extract channel data for all frames
            channel_data = np.array([frame[:, :, channel] for frame in chunk_frames])

            # Perform PCHIP interpolation
            interpolator = PchipInterpolator(original_indices, channel_data, axis=0)
            interpolated_channel_data = interpolator(chunk_new_indices)

            # Store interpolated frames channel-wise
            if channel == 0:
                new_chunk_frames = np.zeros((len(chunk_new_indices), chunk_frames[0].shape[0],
                                             chunk_frames[0].shape[1], 3), dtype=np.uint8)
            new_chunk_frames[:, :, :, channel] = np.clip(interpolated_channel_data, 0, 255).astype(np.uint8)

        # Append interpolated frames
        interpolated_frames.extend(new_chunk_frames)

        # Update the starting index for the next chunk
        chunk_start_idx += len(chunk_frames)

    cap.release()

    print("Writing all interpolated frames to video...")
    # Write all interpolated frames at once
    height, width, _ = interpolated_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_after, (width, height))

    for frame in interpolated_frames:
        out.write(frame)

    out.release()

    print(f"Successfully interpolated video using PCHIP interpolation in {time.time() - start:.2f} seconds")
