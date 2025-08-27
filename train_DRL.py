import os
import numpy as np
from gym_env import gym_env
import datetime
import sys
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from callbacks import SaveOnBestTrainingRewardCallback, EvalPolicyCallback
from callbacks import custom_schedule, linear_schedule
import csv
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, reward_threshold, patience=5, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.best_mean_reward = -float("inf")
        self.counter = 0

    def _on_step(self) -> bool:
        # Only run every 1000 steps or so for efficiency
        if self.n_calls % 1000 == 0:
            # Get training reward history from the logger
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep["r"] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.counter = 0
                else:
                    self.counter += 1

                if self.verbose > 0:
                    print(f"********************************* Step: {self.num_timesteps}, Mean reward: {mean_reward:.2f}, Counter: {self.counter}", "*********************************")

                if self.best_mean_reward >= self.reward_threshold and self.counter >= self.patience:
                    print("Stopping early due to convergence.")
                    return False  # Returning False stops training

        return True


if len(sys.argv) > 1:
    NAME_LOG = sys.argv[1]#'BPI_Challenge_2017_W_Two_TS'
    if not sys.argv[2] == 'from_input_data':    
        N_TRACES = int(sys.argv[2])#2000
    else:
        N_TRACES = sys.argv[2]
    CALENDAR = True if sys.argv[3] == "True" else False
    threshold = int(sys.argv[4])
    postpone = True if sys.argv[5] == "True" else False
    reward_function = sys.argv[6]
else:
    NAME_LOG = 'ER_hospital'
    N_TRACES = 1 #'from_input_data'
    CALENDAR = True
    threshold = 0
    postpone = False

if __name__ == '__main__':
    #if true, load model for a new round of training
    load_model = False
    postpone_penalty = 0
    time_steps = 1000
    #time_steps = 10000
    n_steps = {"BPI_Challenge_2012_W_Two_TS": 1000,
            "confidential_1000": 5120,
            "ConsultaDataMining201618": 5120,
            "PurchasingExample": 5120,
            "BPI_Challenge_2017_W_Two_TS": 48128,
            "Productions": 1280,
            "ER_hospital": 30} ## 5120
    n_steps = n_steps[NAME_LOG] # Number of steps for each network update

    # Create log dir
    now = datetime.datetime.now()
    #log_dir = f"./tmp/{NAME_LOG}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}/"  # Logging training results
    log_dir = f"tmp_training_2/{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{threshold}_P{postpone}_ER_hospital/"
    os.makedirs(log_dir, exist_ok=True)


    ### save n_steps for simulation
    path_step = log_dir + "n_steps.csv"
    with open(path_step, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n_steps"])

    #print(f'Training agent for {config_type} with {time_steps} timesteps in updates of {n_steps} steps.')
    # Create and wrap the environment
    # Reward functions: 'AUC', 'case_task'
    env_simulator = gym_env(NAME_LOG, N_TRACES, CALENDAR, threshold=threshold, postpone=postpone, path_step=path_step)  # Initialize env
    env = Monitor(env_simulator, log_dir)

    # Create the model
    gamma = 0.999
    policy_kwargs = dict(
    net_arch=[dict(pi=[512, 256, 64], vf=[512, 256, 64])] 
    )
    model = MaskablePPO("MlpPolicy", env_simulator, clip_range=0.2, learning_rate=linear_schedule(3e-4), n_steps=int(n_steps), batch_size=256, gamma=gamma, verbose=1)

    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Train the agent
    eval_env = gym_env(NAME_LOG, N_TRACES, CALENDAR, threshold=threshold, postpone=postpone, path_step=path_step)  # Initialize env
    eval_env = Monitor(eval_env, log_dir)
    nr_evaluations = 1 if NAME_LOG != 'BPI_Challenge_2017_W_Two_TS' else 3
    eval_callback = EvalPolicyCallback(check_freq=5*int(n_steps), nr_evaluations=nr_evaluations, log_dir=log_dir, eval_env=eval_env)
    #save_best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    callback = CallbackList([eval_callback])

    model.learn(total_timesteps=int(time_steps), callback=callback)

    model.save(f'{log_dir}/model_final')