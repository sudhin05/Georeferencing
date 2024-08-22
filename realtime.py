from dronekit import connect, VehicleMode, LocationGlobalRelative
import realtime
import numpy as np

K = np.array([[382.48419425, 0, 308.82548046],
              [0, 384.00762005, 215.56477485],
              [0, 0, 1]])

dist_coeffs = np.array([-1.00653100e-03, -2.65340518e+00, 1.41613464e-02, -7.44691182e-03, 7.17526683e+00])


vehicle = connect('/dev/ttyAMA0', wait_ready=True, baud=57600)  # Adjust connection string as needed

drone_gps = (vehicle.location.global_frame.lat, vehicle.location.global_frame.lon)
altitude = vehicle.location.global_relative_frame.alt
gimbal_angles = [vehicle.attitude.yaw, vehicle.attitude.pitch, vehicle.attitude.roll]

