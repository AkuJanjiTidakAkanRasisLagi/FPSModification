import cv2
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import time

def frames_to_video(frames, fps, output_path):
    """
    Write a list/array of frames (shape [num_frames, height, width, 3]) to a video file.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()


def polynomial_controller(
    video_path, 
    fps_before, 
    fps_after, 
    output_path, 
    local_window_size=5, 
    max_degree=4,
    row_chunk_size=5
):
    """
    Convert a video from `fps_before` to `fps_after` using *piecewise polynomial interpolation*.
    
    - local_window_size: number of original frames to use around each new frame time (e.g. 5 means 2 on each side).
    - max_degree: max polynomial degree to fit (<= local_window_size - 1).
    - row_chunk_size: how many rows to process at once (for memory control).
    
    Steps:
      1) Read all frames and store as float32.
      2) Compute new frame timestamps.
      3) For each new frame time, pick a small chunk of original frames around that time.
      4) Per-pixel polynomial fit & evaluate, done in row chunks to save memory.
      5) Write the resulting frames to a video with `fps_after`.
    """

    start_time = time.time()

    # --- 1) Read the input video frames ---
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(actual_fps - fps_before) > 0.1:
        print(f"Warning: Video has {actual_fps} FPS, but fps_before is {fps_before}.")

    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame.astype(np.float32))
    cap.release()

    num_frames = len(frames_list)
    if num_frames == 0:
        print("No frames found in the video.")
        return

    print(f"Total original frames read: {num_frames}")

    # --- 2) Setup time indexing ---
    original_duration = num_frames / fps_before  # seconds
    new_frame_count = int(round(original_duration * fps_after))

    original_times = np.linspace(0, original_duration, num_frames)
    new_times = np.linspace(0, original_duration, new_frame_count)

    print(f"Original duration: {original_duration:.2f}s | "
          f"New frame count: {new_frame_count} at {fps_after} FPS "
          f"(=> ~{new_frame_count / fps_after:.2f}s)")

    # Stack frames into a single ndarray: shape (num_frames, height, width, 3)
    video_array = np.stack(frames_list, axis=0)
    height, width, _ = video_array[0].shape

    # Prepare output array
    new_video = np.zeros((new_frame_count, height, width, 3), dtype=np.float32)

    # Precompute half-window
    half_w = local_window_size // 2

    # --- 3) For each new frame, do piecewise polynomial interpolation ---
    print("Interpolating with piecewise polynomial...")

    for i_new, t_new in enumerate(new_times):
        # (A) Identify a local window of frames around t_new
        #     We'll pick frames whose times are nearest to t_new within local_window_size.
        #     A simpler approach: find the closest frame index in original_times, center on that.
        center_idx = np.searchsorted(original_times, t_new)
        # Start index:
        start_idx = max(0, center_idx - half_w)
        end_idx   = min(num_frames, start_idx + local_window_size)
        # Adjust start_idx if end_idx doesn't give us enough frames
        start_idx = max(0, end_idx - local_window_size)

        # Indices of the original frames to use
        idx_range = np.arange(start_idx, end_idx)
        t_range   = original_times[idx_range]  # times for the local window

        # If we have fewer frames than local_window_size (e.g. near edges), 
        # then the actual window is smaller, which is fine.

        # (B) Extract the needed frames from video_array. shape: (win_size, height, width, 3)
        local_frames = video_array[idx_range]

        # --- 4) For memory reasons, we do row-chunking. ---
        for row_start in range(0, height, row_chunk_size):
            row_end = min(row_start + row_chunk_size, height)

            # local_chunk shape: (win_size, row_count, width, 3)
            local_chunk = local_frames[:, row_start:row_end, :, :]

            # We now do polynomial fit per pixel in this chunk.
            # local_chunk[:, :, :, channel] => shape (win_size, row_count, width)
            # We'll fill new_video[i_new, row_start:row_end, :, channel]

            for channel in range(3):
                # shape: (win_size, row_count, width)
                channel_data = local_chunk[:, :, :, channel]

                # Flatten spatial dimensions => shape (win_size, row_count * width)
                flat_data = channel_data.reshape(len(idx_range), -1)

                # For each pixel in that flat layout, fit & evaluate
                for pix_idx in range(flat_data.shape[1]):
                    pix_series = flat_data[:, pix_idx]  # shape (win_size,)

                    # If only 1 frame in the local window, replicate it
                    if len(pix_series) == 1:
                        out_val = pix_series[0]
                    else:
                        # Fit polynomial of degree = min(max_degree, #frames_in_window - 1)
                        deg = min(max_degree, len(pix_series) - 1)
                        poly = Polynomial.fit(t_range, pix_series, deg=deg)
                        out_val = poly(t_new)

                    # Place the result in new_video
                    # We map back pix_idx => row,col within [row_start:row_end, :]
                    row_len = row_end - row_start
                    # row_idx in [0, row_len)
                    # col_idx in [0, width)
                    # row_idx = pix_idx // width
                    # col_idx = pix_idx % width
                    row_idx = pix_idx // width
                    col_idx = pix_idx % width
                    # Write out_val
                    new_video[i_new, row_start + row_idx, col_idx, channel] = out_val

    # --- 5) Convert to uint8 and write the output video ---
    new_video_uint8 = np.clip(new_video, 0, 255).astype(np.uint8)
    print("Writing the output video...")
    frames_to_video(new_video_uint8, fps_after, output_path)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f} seconds.")
    print(f"Output has {new_frame_count} frames at {fps_after} FPS (duration ~ {new_frame_count / fps_after:.2f}s).")
