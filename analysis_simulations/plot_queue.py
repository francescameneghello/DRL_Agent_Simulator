import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("../output/output_ER_hospital_CTrue_T0_RANDOM/queue_progression_ER_hospital_RANDOM.csv")
log["time"] = log["time"] / 3600

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()
roles = log["role"].unique()
# Plot each role in its own subplot
for i, role in enumerate(roles):
    ax = axes[i]
    group = log[log["role"] == role]
    ax.plot(group["time"], group["queue"], marker="o", linestyle="-")
    ax.set_title(role)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Queue")
    ax.grid(True)

# Hide any unused subplots (in case <6 roles in your current data slice)
for j in range(len(roles), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()