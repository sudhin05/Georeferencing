import cv2
import pandas as pd
import math
import numpy as np
from inference_sdk import InferenceHTTPClient
import sys

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="l8jYuI2kzlX5MAu7uvbK"
)

flight_log_path = 'data/Aug-30th-2022-12-59PM-Flight-Airdata.csv'
flight_log_data = pd.read_csv(flight_log_path)

filtered_log_data = flight_log_data[flight_log_data['isVideo'] == 1].reset_index(drop=True)

vid_path = "data/DJI_0753.MP4"
cap = cv2.VideoCapture(vid_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video_path = 'data/output_DJI_0753.mp4'
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
log_index = 0

def latlong(drone_gps, alti, heading, fov, vw, vh, px, py):
    aspect_ratio = vh / vw
    offset = math.atan2(vh, vw)

    theta = math.radians(fov / 2)
    cdistance = alti * math.tan(theta)

    phi = heading + math.degrees(math.atan2(py - vh / 2, px - vw / 2))
    distance = np.hypot(px - vw / 2, py - vh / 2) * (cdistance / np.hypot(vw / 2, vh / 2))

    delta_lat = distance * math.cos(math.radians(phi))
    delta_lon = distance * math.sin(math.radians(phi))

    new_lat = drone_gps[0] + delta_lat / 111320
    new_lon = drone_gps[1] + delta_lon / (40075000 * math.cos(math.radians(drone_gps[0])) / 360)

    return new_lat, new_lon

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 800, 600)

while cap.isOpened() and log_index < len(filtered_log_data):
    ret, frame = cap.read()
    if not ret:
        break

    result = CLIENT.infer(frame, model_id="aerial-solar-panels/6")

    drone_gps = (filtered_log_data['latitude'].iloc[log_index], filtered_log_data['longitude'].iloc[log_index])
    altitude = filtered_log_data['altitude(feet)'].iloc[log_index]
    bearing = filtered_log_data[' compass_heading(degrees)'].iloc[log_index]
    
    for detection in result['predictions']:
        x, y, width, height, confidence, class_name = detection['x'], detection['y'], detection['width'], detection['height'], detection['confidence'], detection['class']

        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))
        
        gps_coords = latlong(drone_gps, altitude, bearing, 60, width, height, x, y)
        lat, lon = gps_coords

        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        text = f"GPS coordinates({lat:.6f}, {lon:.6f}) with {confidence:.2f}"
        cv2.putText(frame, text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        print(f"Detected solar panel at GPS coordinates: ({lat:.6f}, {lon:.6f}) with confidence {confidence:.2f}")
    
    cv2.imshow("Result", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit early if needed
        break
    
    out_video.write(frame)
    
    frame_count += 1
    log_index += 1

cap.release()
out_video.release()
cv2.destroyAllWindows()
