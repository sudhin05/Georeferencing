import numpy as np
from math import sin, cos, radians
import cv2

"""
    Function to calculate the extrinsic matrix from the camera's perspective to the drone's global frame.

    Args:
    - pitch (float): The pitch angle in degrees.
    - roll (float): The roll angle in degrees.
    - yaw (float): The yaw angle in degrees.

    Set Intrinsic matrix for Raspberry PICAM 
    Set Pixel coordinates manually / from object detection
"""

# Manual inputs for drone data
drone_gps = (latitude, longitude)  # Replace with actual GPS coordinates
altitude = altitude_value  # Replace with actual altitude value
gimbal_angles = [pitch_angle, roll_angle, yaw_angle]  # Replace with actual angles in degrees

camera_matrix = np.array([[382.48419425, 0, 308.82548046],
                          [0, 384.00762005, 215.56477485],
                          [0, 0, 1]], dtype=np.float32)

Z_const = 285

def get_extrinsic_matrix(pitch, roll, yaw):
    yaw, pitch, roll = map(radians, [yaw, pitch, roll])

    rotation = np.array([
        [cos(yaw) * cos(pitch), 
         cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll), 
         cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)],
        
        [sin(yaw) * cos(pitch), 
         sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll), 
         sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)],
        
        [-sin(pitch), 
         cos(pitch) * sin(roll), 
         cos(pitch) * cos(roll)]
    ])

    translation = np.array([0, 0, 0])

    return rotation, translation

# Load image instead of using live feed
image_path = "path_to_image.jpg"  # Replace with actual image path
frame = cv2.imread(image_path)

if frame is None:
    print("Image not found or unable to read")
    exit()

# Display the frame and select the ROI (bounding box)
bbox = cv2.selectROI("Frame", frame, False)
cv2.destroyWindow("Frame")

x, y, w, h = bbox
center_uv_point = np.array([[x + w/2], [y + h/2], [1]], dtype=np.float32) 

rotation_matrix, translation_matrix = get_extrinsic_matrix(gimbal_angles[0], gimbal_angles[1], gimbal_angles[2])

left_side_mat = np.dot(np.linalg.inv(rotation_matrix), np.dot(np.linalg.inv(camera_matrix), center_uv_point))
right_side_mat = np.dot(np.linalg.inv(rotation_matrix), translation_matrix)
s = (Z_const + right_side_mat[2, 0]) / left_side_mat[2, 0]

P = np.dot(rotation_matrix, (s * np.dot(np.linalg.inv(camera_matrix), center_uv_point) - translation_matrix))
print("3D World Coordinates: ", P.ravel())
