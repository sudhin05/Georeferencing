import math
import numpy as np

def latlong(drone_gps,alti,heading,fov,vw,vh,px,py):
  aspect_ration = vh / vw
  offset = math.atan2(vh,vw)

  theta = math.radians(fov/2)
  cdistance = alti * math.tan(theta)

  phi = heading + math.degrees(math.atan2(py - vh/2,px - vw/2))
  distance = np.hypot(px-vw/2,py-vh/2)*(cdistance/np.hypot(vw/2,vh/2))

  delta_lat = distance * math.cos(math.radians(phi))
  delta_lon = distance * math.sin(math.radians(phi))

  new_lat = drone_gps[0] + delta_lat / 111320  # Approx m per latitude
  new_lon = drone_gps[1] + delta_lon / (40075000 * math.cos(math.radians(drone_gps[0])) / 360)

  return new_lat, new_lon

