import cv2
import os
from akima import akima_controller
from cubic_spline import cubic_spline_controller
from cubic import cubic_controller
from nearest_neighbor import nearest_neighbor_controller
from linear import linear_controller
from pchip import pchip_controller
from polynomial import polynomial_controller
# from RBF import RBF_controller

# Path to the video file
video_path = input("Enter the path to the video file: ")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    # Loop through each frame

    os.makedirs(video_path.split(".")[0], exist_ok=True)
  
    fps_before = cap.get(cv2.CAP_PROP_FPS)
    fps_after = int(input(f"Enter the desired FPS (current FPS is {fps_before}): "))

    while (fps_after < 2 or fps_after > 120):
        if fps_after < 2:
            print("Error: FPS must be greater than 1.")
        else:
            print("Error: FPS must be less than 60, laptop gw meledak jir ntar kalo lebih dari 120 FPS.")
        fps_after = int(input("Enter the desired FPS: "))

    cap.release()
    # linear_controller(video_path, fps_before, fps_after, f"{video_path.split('.')[0]}/linear_interpolation_result_2.mp4")
    cubic_controller(video_path, fps_before, fps_after, f"{video_path.split('.')[0]}/cubic_interpolation_result_2.mp4")
    #  cubic_spline_controller(video_path, fps_before, fps_after, f"{video_path.split('.')[0]}/cubic_spline_interpolation_result_2.mp4")
    # nearest_neighbor_controller(video_path, fps_before, fps_after, f"{video_path.split('.')[0]}/nearest_neighbor_interpolation_result_2.mp4")
    # akima_controller(video_path, fps_before, fps_after, f"{video_path.split('.')[0]}/akima_interpolation_result_2.mp4")
    # pchip_controller(video_path, fps_before, fps_after, f"{video_path.split('.')[0]}/pchip_interpolation_result_2.mp4")
    # polynomial_controller(video_path, fps_before, fps_after, f"{video_path.split('.')[0]}/polynomial_interpolation_result_0.mp4")
    # RBF_controller(video_path, fps_before, fps_after)