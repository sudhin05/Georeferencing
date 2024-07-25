import cv2
import torch
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import letterbox
import numpy as np
import pandas as pd
import math

# Load flight log data
flight_log_path = '/mnt/data/Aug-30th-2022-12-59PM-Flight-Airdata.csv'
flight_log_data = pd.read_csv(flight_log_path)

# Select device for YOLO model
device = select_device('')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)  # Load the YOLOv5 model

def latlong(drone_gps, alti, heading, fov, vw, vh, px, py):
    aspect_ratio = vh / vw
    offset = math.atan2(vh, vw)

    theta = math.radians(fov / 2)
    cdistance = alti * math.tan(theta)

    phi = heading + math.degrees(math.atan2(py - vh / 2, px - vw / 2))
    distance = np.hypot(px - vw / 2, py - vh / 2) * (cdistance / np.hypot(vw / 2, vh / 2))

    delta_lat = distance * math.cos(math.radians(phi))
    delta_lon = distance * math.sin(math.radians(phi))

    new_lat = drone_gps[0] + delta_lat / 111320  # Approx meters per latitude degree
    new_lon = drone_gps[1] + delta_lon / (40075000 * math.cos(math.radians(drone_gps[0])) / 360)  # Approx meters per longitude degree

    return new_lat, new_lon

def detect_solar_panels(frame, conf_thres=0.25, iou_thres=0.45):
    img = letterbox(frame, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    return pred, img.shape[2:], frame.shape

# Path to input video
video_path = '/mnt/data/DJI_0753.MP4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Path to output video
out_video_path = '/mnt/data/output_DJI_0753.mp4'
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    pred, img_shape, frame_shape = detect_solar_panels(frame)

    drone_gps = (flight_log_data['latitude'].iloc[frame_count], flight_log_data['longitude'].iloc[frame_count])
    altitude = flight_log_data['altitude(feet)'].iloc[frame_count]
    bearing = flight_log_data['compass_heading(degrees)'].iloc[frame_count]
    
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_shape, det[:, :4], frame_shape).round()
            for *xyxy, conf, cls in det:
                if cls == 1:  # Assuming class 1 is 'solar-panel'
                    x1, y1, x2, y2 = xyxy
                    pixel_x = int((x1 + x2) / 2)
                    pixel_y = int((y1 + y2) / 2)
                    
                    gps_coords = latlong(drone_gps, altitude, bearing, 60, width, height, pixel_x, pixel_y)
                    lat, lon = gps_coords
                    
                    # Log and draw the detection
                    print(f"Detected solar panel at GPS coordinates: ({lat}, {lon}) with confidence {conf:.2f}")
                    cv2.putText(frame, f"Lat: {lat:.6f}, Lon: {lon:.6f}", (pixel_x, pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Write the frame to the output video
    out_video.write(frame)
    
    frame_count += 1

cap.release()
out_video.release()
