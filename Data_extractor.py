import pandas as pd

flight_log_path = "data/Aug-30th-2022-12-59PM-Flight-Airdata.csv"
flight_log = pd.read_csv(flight_log_path)

print(flight_log.info())
