import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import csv
from typing import Callable


class EvalPolicyCallback(BaseCallback):

    def __init__(self, check_freq: int, nr_evaluations: int, log_dir: str, eval_env, model_name: str = 'best_model', verbose: int = 1):
        super(EvalPolicyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.nr_evaluations = nr_evaluations
        self.log_dir = log_dir
        self.eval_env = eval_env
        self.save_path = os.path.join(log_dir, model_name)
        self.best_sum_waiting_times = -np.inf
        self.prev_sum_waiting_times = 0
        self.actual_sum_waiting_times = -np.inf
        self.prev_steps = 0
        self.episode_lengths = []
        csv_path = os.path.join(log_dir, "evaluation_monitor.csv")
        self.csv_file = open(csv_path, mode="w", newline="")
        self.csv_monitoring_writer = csv.writer(self.csv_file)
        self.csv_monitoring_writer.writerow(["r_mean"])
        self.csv_file.flush()

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.n_calls - 1) % self.check_freq == 0 and self.n_calls > 1: # After model update
            print('\n-------- Evaluating current policy --------')
            waiting_time_evals = []
            for epoch in range(self.nr_evaluations):
                state, _ = self.eval_env.reset()
                isTerminated = False
                while isTerminated == False:
                    action_masks = self.eval_env.env.action_masks()
                    action, _state = self.model.predict(state, action_masks=action_masks)
                    state, reward, isTerminated, _, info = self.eval_env.step(action)

                waiting_time_evals.append(info['sum_waiting_times'])

            self.actual_sum_waiting_times = np.mean(waiting_time_evals)
            self.csv_monitoring_writer.writerow([self.actual_sum_waiting_times])
            self.csv_file.flush()
            print("Waiting times percentile of current policy --->", self.actual_sum_waiting_times)
            if self.actual_sum_waiting_times > self.best_sum_waiting_times:
                self.best_sum_waiting_times = self.actual_sum_waiting_times
                print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

            print(f"Num timesteps: {self.num_timesteps}")
            print(f"Best percentile waiting times: {self.best_sum_waiting_times:.2f} - Actual percentile waiting_times: {self.actual_sum_waiting_times:.2f} - Last percentile waiting_times: {self.prev_sum_waiting_times:.2f}")
            self.prev_sum_waiting_times = self.actual_sum_waiting_times
            print('---- Finished evaluating current policy ----\n')
        return True

'''
class EvalPolicyCallback(BaseCallback):
    def __init__(self, check_freq: int, nr_evaluations: int, log_dir: str, eval_env, model_name: str = 'best_model', verbose: int = 1):
        super(EvalPolicyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.nr_evaluations = nr_evaluations
        self.log_dir = log_dir
        self.eval_env = eval_env
        self.save_path = os.path.join(log_dir, model_name)
        self.best_sum_waiting_times = -np.inf
        self.prev_sum_waiting_times = 0
        self.actual_sum_waiting_times = -np.inf
        self.prev_steps = 0
        self.episode_lengths = []
        csv_path = os.path.join(log_dir, "evaluation_monitor.csv")
        self.csv_file = open(csv_path, mode="w", newline="")
        self.csv_monitoring_writer = csv.writer(self.csv_file)
        self.csv_monitoring_writer.writerow(["r_mean"])
        self.csv_file.flush()

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.n_calls - 1) % self.check_freq == 0 and self.n_calls > 1: # After model update
            print('\n-------- Evaluating current policy --------')
            waiting_time_evals = []
            for epoch in range(self.nr_evaluations):
                state, _ = self.eval_env.reset()
                isTerminated = False
                while isTerminated == False:
                    action, _state = self.model.predict(state)
                    state, reward, isTerminated, _, info = self.eval_env.step(action)

                waiting_time_evals.append(info['sum_waiting_times'])

            self.actual_sum_waiting_times = np.mean(waiting_time_evals)
            self.csv_monitoring_writer.writerow([self.actual_sum_waiting_times])
            self.csv_file.flush()
            print("Waiting times percentile of current policy --->", self.actual_sum_waiting_times)
            if self.actual_sum_waiting_times > self.best_sum_waiting_times:
                self.best_sum_waiting_times = self.actual_sum_waiting_times
                print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

            print(f"Num timesteps: {self.num_timesteps}")
            print(f"Best percentile waiting times: {self.best_sum_waiting_times:.2f} - Actual percentile waiting_times: {self.actual_sum_waiting_times:.2f} - Last percentile waiting_times: {self.prev_sum_waiting_times:.2f}")
            self.prev_sum_waiting_times = self.actual_sum_waiting_times
            print('---- Finished evaluating current policy ----\n')
        return True
'''

def custom_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > 0.8: #0.95
            return initial_value
        elif progress_remaining > 0.5: #0.9
            return initial_value * 0.5
        else:
            return initial_value * 0.1

    return func

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func