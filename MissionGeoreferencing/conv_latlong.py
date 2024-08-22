import numpy as np
from dronekit import connect
from math import sin, cos, radians

"""
    Function to calculate the extrinsic matrix from the camera's perspective to the drone's global frame.

    Args:
    - pitch (float): The pitch angle in degrees.
    - roll (float): The roll angle in degrees.
    - yaw (float): The yaw angle in degrees.

    Set Intrinsic matrix for Raspberry PICAM 
    Set Pixel coordinates manually / from object detection
"""

def get_extrinsic_matrix(pitch, roll, yaw, gps_coords, altitude):
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

    translation = np.array(0,0,0)

    return rotation, translation

vehicle = connect('/dev/ttyAMA0', wait_ready=True, baud=57600) 

drone_gps = (vehicle.location.global_frame.lat, vehicle.location.global_frame.lon)
altitude = vehicle.location.global_relative_frame.alt
gimbal_angles = [vehicle.attitude.pitch, vehicle.attitude.roll, vehicle.attitude.yaw]

camera_matrix = np.array([[382.48419425, 0, 308.82548046],
                          [0, 384.00762005, 215.56477485],
                          [0, 0, 1]], dtype=np.float64)


uv_point = np.array([[363], [222], [1]], dtype=np.float64) 
rotation_matrix, translation_matrix = get_extrinsic_matrix(gimbal_angles[0],gimbal_angles[1],gimbal_angles[2],drone_gps,altitude)

left_side_mat = np.dot(np.linalg.inv(rotation_matrix), np.dot(np.linalg.inv(camera_matrix), uv_point))
right_side_mat = np.dot(np.linalg.inv(rotation_matrix), translation_matrix)

Z_const = 285 #### edit 

s = (Z_const + right_side_mat[2, 0]) / left_side_mat[2, 0]

P = np.dot(rotation_matrix, (s * np.dot(np.linalg.inv(camera_matrix), uv_point) - translation_matrix))
print("3D World Coordinates: ", P.ravel())









