import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# List of CSVs (different runs/dates)
files = [
    "../output/arrivals/simulated_log_ER_hospital_RANDOM_0_1800.csv",
    "../output/arrivals/simulated_log_ER_hospital_RANDOM_0_3600.csv",
    "../output/arrivals/simulated_log_ER_hospital_RANDOM_0_7200.csv"
]

labels = ["Exp1800", "Exp3600", "Exp7200"]

plt.figure(figsize=(12, 5))

for f, label in zip(files, labels):
    log = pd.read_csv(f)

    # Extract arrival times
    arrival_times = pd.to_datetime(
        log.loc[log['activity'] == 'start', "start_time"]
    )

    # Plot each run slightly offset vertically for visibility
    plt.scatter(
        arrival_times,
        [labels.index(label)] * len(arrival_times),  # y offset for each run
        marker="|",
        s=300,
        label=label
    )

plt.title("Arrival Times Across Runs")
plt.xlabel("Time")
plt.yticks(range(len(labels)), labels)  # Show run names on y-axis
plt.grid(True, axis="x")
plt.legend()
plt.tight_layout()
plt.show()
