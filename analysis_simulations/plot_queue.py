import pandas as pd
import matplotlib.pyplot as plt

# List of CSVs (different dates/runs)
files = [
    "../output/queue/queue_progression_ER_hospital_RANDOM_1800.csv",
    "../output/queue/queue_progression_ER_hospital_RANDOM_3600.csv",
    "../output/queue/queue_progression_ER_hospital_RANDOM_7200.csv"
]

files = [
    "../output/output_ER_hospital_CTrue_FIFO_activity/queue_progression_ER_hospital_FIFO_activity.csv",
    "../output/output_ER_hospital_CTrue_RANDOM/queue_progression_ER_hospital_RANDOM.csv",
    "../output/output_ER_hospital_CTrue_FIFO_case/queue_progression_ER_hospital_FIFO_case.csv"
]

labels = ["Exp1800", "Exp3600", "Exp7200"]  # names for legend

labels = ["FIFO_activity", "Random", "FIFO_case"]

# Load all logs
logs = [pd.read_csv(f) for f in files]

# Normalize time
for log in logs:
    print(log)
    log["time"] = log["time"] / 3600

# Get all roles across datasets
all_roles = sorted(set().union(*[log["role"].unique() for log in logs]))

fig, axes = plt.subplots(len(all_roles), 1, figsize=(15, 4 * len(all_roles)), sharex=True, sharey=True)
axes = axes.flatten() if len(all_roles) > 1 else [axes]

# Plot each role
for i, role in enumerate(all_roles):
    ax = axes[i]
    for log, label in zip(logs, labels):
        group = log[log["role"] == role]
        ax.plot(group["time"], group["queue"], marker="o", linestyle="-", label=label)
    ax.set_title(role)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Queue")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

