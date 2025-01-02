import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import time

def frames_to_video(frames, fps, output_path):
    """
    Saves a list of frames to a video file.

    Args:
    - frames: List of frames (NumPy arrays).
    - fps: Target frames per second.
    - output_path: Path to save the output video.
    """
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()

def cubic_spline_controller(video_path, fps_before, fps_after, output_path):
    """
    Modifies the FPS of a video using cubic spline interpolation.

    Args:
    - video_path: Path to the input video file.
    - fps_before: Original frames per second of the video.
    - fps_after: Target frames per second of the video.
    - output_path: Path to save the modified video.
    """
    start_time = time.time()

    # Read the video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))
    cap.release()

    print(f"Total frames read: {len(frames)}")

    # Original and target time indices
    original_duration = len(frames) / fps_before  # Duration of the original video
    original_indices = np.linspace(0, original_duration, len(frames))
    new_frame_count = int(original_duration * fps_after)
    new_indices = np.linspace(0, original_duration, new_frame_count)

    print(f"Interpolating {len(frames)} frames to {new_frame_count} frames...")

    # Initialize the new frames list
    height, width, _ = frames[0].shape
    new_frames = []

    # Perform cubic spline interpolation channel-wise for each pixel
    for channel in range(3):  # B, G, R channels
        print(f"Processing channel {channel + 1}/3...")
        channel_data = np.array([frame[:, :, channel] for frame in frames])  # Shape: (frames, height, width)
        interpolated_channel = np.zeros((new_frame_count, height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                print(x, y)
                # Extract pixel values across all frames for (x, y)
                pixel_values = channel_data[:, y, x]

                # Fit a cubic spline to the pixel values
                cs = CubicSpline(original_indices, pixel_values, bc_type='natural')

                # Evaluate the spline at the new time indices
                interpolated_pixel_values = cs(new_indices)
                interpolated_channel[:, y, x] = interpolated_pixel_values

        # Clip the interpolated values and store them in the new frames
        interpolated_channel = np.clip(interpolated_channel, 0, 255).astype(np.uint8)

        # Combine the interpolated channel into frames
        if channel == 0:
            combined_frames = np.zeros((new_frame_count, height, width, 3), dtype=np.uint8)
        combined_frames[:, :, :, channel] = interpolated_channel

    new_frames.extend([frame for frame in combined_frames])

    # Save the modified video
    frames_to_video(new_frames, fps_after, output_path)

    end_time = time.time()
    print(f"Successfully modified FPS using cubic spline interpolation in {end_time - start_time:.2f} seconds.")

# Example Usage
# cubic_spline_fps_modification("input.mp4", 24, 60, "output.mp4")
