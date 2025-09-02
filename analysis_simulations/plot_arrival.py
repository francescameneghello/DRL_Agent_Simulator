import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

log = pd.read_csv("../output/arrivals/simulated_log_ER_hospital_RANDOM_0.csv")

arrival_times = pd.to_datetime(
    log.loc[log['activity'] == 'start', "start_time"]
)

# Convert to pandas Series
s = pd.Series(arrival_times)

plt.figure(figsize=(10, 4))
plt.scatter(arrival_times, [1]*len(arrival_times), marker="|", s=200)
plt.title("Arrival Times")
plt.xlabel("Time")
plt.yticks([])  # Hide y-axis since it's always 1
plt.grid(True, axis="x")
plt.tight_layout()
plt.show()
