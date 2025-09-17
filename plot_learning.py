import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (replace with your filename)
df = pd.read_csv("/home/francesca/Documents/Resource_simulator_distribution/tmp_training/single_reward/defaultMLP_exp1800_ER_hospital_500_CTrue_SUM_WAITING_TIMES_NO_MASKING_ER_hospital/progress.csv")

# Plot mean episode reward vs timesteps
plt.figure(figsize=(10,6))
plt.plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"], label="Mean Episode Reward")

plt.xlabel("Timesteps")
plt.ylabel("Episode Reward (mean)")
plt.title("PPO Learning Progress")
plt.legend()
plt.grid(True)
plt.show()
