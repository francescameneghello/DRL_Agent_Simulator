import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
from typing import List
import csv
import simpy
import utility
from process import SimulationProcess
#from event_trace import Token
from event_trace_duration import Token
from parameters import Parameters
import sys, getopt
from utility import *
import pm4py
from inter_trigger_timer import InterTriggerTimer
import warnings
import json
import math
import pm4py
from os.path import exists
from datetime import datetime, timedelta
import random
CYCLE_TIME_MAX = 8.64e+6


DEBUG_PRINT = False
if __name__ == "__main__":
    warnings.filterwarnings("ignore")


class gym_env(Env):
    def __init__(self, NAME_LOG, N_TRACES, CALENDAR, POLICY=None, N_SIMULATION=0, threshold=0, postpone=True, path_step=None, features='all') -> None:
        self.name_log = NAME_LOG

        self.CALENDAR = CALENDAR ## "True" If you want to use calendar, "False" otherwise
        self.threshold = threshold
        self.postpone = postpone
        self.features = features
        self.total_reward = 0
        self.reward_count = 0
        self.path_step = path_step

        self.normalization_acc_waiting_times = 5000
        self.policy = POLICY
        self.print = True
        input_file = './example/' + self.name_log + '/input_' + self.name_log + '.json'
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        self.n_simulation = N_SIMULATION

        if N_TRACES == 'from_input_data':
            self.N_TRACES = input_data['traces']
        else:
            self.N_TRACES = N_TRACES

        self.resources = sorted(list(input_data["roles"].keys()))
        self.task_types = sorted(list(input_data["processing_time"].keys())+['PAD'])
        print('ACTIVTIES', self.task_types)
        print("###############################################################")

        # Define the input and output of the agent (neural network)

        # The general inputs are: ### day, hour, roles occupations, ratio_completed_traces,
        #
        # Queue info: role, #tokens_queue, role_occup, role_capacity
        #
        # For each token in the window: activity, len_prefix, acc_cycle_time
        #
        #
        self.WINDOW_SIZE = 40
        self.WINDOW_PREFIX = 0

        ## TIME: 'month', 'day', 'hour', 'estimated_processing_time_', 'remain_cycle_times_',  'queue_waiting_time_', 'acc_waiting_time_'+str(i),
        ## CF: 'actual_activity_'+str(i), 'len_prefix_'+str(i), 'prefix_', 'remain_acts_'
        ## CONGESTION: 'actual_role', roles_occup, 'actual_role', 'role_queue', 'role_occup', 'role_capacity',
            # 'HOL', 'ratio_traces', 'wip_act_queue_',  'queue_waiting_time_', 'acc_waiting_time_'+str(i),
        self.input = ['month', 'day', 'hour']
        self.input += ['actual_role', 'role_queue', 'role_occup', 'role_capacity']

        print('INPUT', self.input, len(self.input))
        print("###############################################################")
        self.output = [i for i in range(0, self.WINDOW_SIZE)]

        self.FEATURE_ROLE = None
        self.PATH_PETRINET = './example/' + self.name_log + '/' + self.name_log + '.pnml'
        PATH_PARAMETERS = input_file

        self.PATH_LOG = './example/' + self.name_log + '/' + self.name_log + '.xes'
        self.params = Parameters(PATH_PARAMETERS, self.N_TRACES, self.name_log, threshold)

        # Observation space
        ### define the minimum and maximum values that each output can have
        lows = np.array([0 for _ in range(len(self.input))])
        highs = np.array([1 for _ in range(len(self.input))])
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.input),),
                                            dtype=np.float64)

        self.action_space = spaces.Discrete(len(self.output))
        print('ACTION SPACE', self.action_space, len(self.output))
        print("###############################################################")

        self.processes = []
        self.last_reward = {}
        self.nr_steps = 0
        self.nr_postpone = 0

        print(f'{self.name_log}, {self.N_TRACES}, calendar={self.CALENDAR}, postpone={self.postpone}', flush=True)
        warnings.filterwarnings("ignore")

    # Reset the environment -> restart simulation

    def reset(self, seed=0, i=None):
        print('-------- Resetting environment --------')
        self.total_reward = 0
        self.reward_count = 0
        self.nr_steps = 0
        self.env = simpy.Environment()
        self.queue_writer = f"output/output_{self.name_log}_C{self.CALENDAR}_{self.policy}/queue_progression_{self.name_log}_{self.policy}.csv"
        self.simulation_process = SimulationProcess(self.env, self.params, self.CALENDAR, self.queue_writer)
        self.completed_traces = []
        self.last_reward = {}
        if self.print:
            calendar = 'CALENDAR' if self.CALENDAR else 'NOT_CALENDAR'
            utility.define_folder_output(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_{self.policy}")            
            f = open(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_{self.policy}/simulated_log_{self.name_log}_{self.policy}_{str(i)}.csv", 'w')
            print(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_{self.policy}/simulated_log_{self.name_log}_{self.policy}_{str(i)}.csv")
            writer = csv.writer(f)
            writer.writerow(Buffer(writer).get_buffer_keys())
        else:
            writer = None
        net, im, fm = pm4py.read_pnml(self.PATH_PETRINET)
        interval = InterTriggerTimer(self.params, self.simulation_process, self.params.START_SIMULATION, self.N_TRACES)
        self.tokens = {}
        prev = 0
        for i in range(0, self.N_TRACES):
            prefix = Prefix()
            itime = interval.get_next_arrival(self.env, i, self.name_log, self.CALENDAR)
            prev = itime
            parallel_object = utility.ParallelObject()
            time_trace = self.env.now
            token = Token(i, net, im, self.params, self.simulation_process, prefix, 'sequential', writer, parallel_object,
                          itime, self.env, self.CALENDAR, None, _print=self.print)
            self.tokens[i] = token
            self.env.process(token.inter_trigger_time(itime))

        ### fix the calendar of roles before start the process
        self.simulation_process.get_state()
        start = self.simulation_process._date_start
        for res in self.simulation_process._resources:
            if res != 'TRIGGER_TIMER':
                role = self.simulation_process._resources[res]
                self.env.process(role.wait_calendar(start))

        self.next_decision_moment(start=True)
        state = self.simulation_process.get_state()
        return self.get_state(), {}

    def action_masks(self): ### missing the definition of Postpone
        mask = [False for _ in range(len(self.output))]
        state = self.simulation_process.get_state()
        for i in range(min(state['role_queue'], self.WINDOW_SIZE)):
            mask[i] = True
        return list(map(bool, mask))

    #!! The algorithm uses this function to interact with the environment
    # Every step is an observation (state, action, reward) which is used for training
    # The agent takes an action based on the current state, and the environment should transition to the next state
    # Take an action at every timestep and evaluate this action in the simulator
    def step(self, action):
        #### action ---> which token can perform the activity (token_id)
        self.nr_steps += 1
        reward = 0
        if action is not None:
            if self.output[action] != 'Postpone':
                res = self.simulation_process.get_state()['role']
                token_id = self.simulation_process.del_token_queue(res, action)
                simulation = self.tokens[token_id].simulation()
                self.env.process(simulation)
                start = self.simulation_process._date_start
                for res in self.simulation_process._resources:
                    if res != 'TRIGGER_TIMER':
                        role = self.simulation_process._get_resource(res)
                        if not role.waiting_for_calendar:
                            self.env.process(role.wait_calendar(start))
                self.next_decision_moment()
            else:
                self.next_decision_moment()

            if token_id in self.last_reward:  ### token finishes process
                reward = self.last_reward[token_id]
            else:
                reward = self.tokens[token_id].last_reward
                #print("######################", reward, self.tokens[token_id]._prefix.get_prefix() ,"######################")

        info = {}
        if len(self.tokens) == 0:
            isTerminated = True
            self.waiting_times = list(self.simulation_process.waiting_times.values())
            info = {'list_wait': self.waiting_times, 'percentile': np.percentile(self.waiting_times, 95)}
            print('Percentile', np.percentile(self.waiting_times, 95), len(self.waiting_times))
            print('Step to finish the simulation', self.nr_steps)
        else:
            isTerminated = False
        return self.get_state(), reward, isTerminated, {}, info

    def step_baseline(self, action):
        ### action is (token_id, priority)
        ### self.tokens ----> { case_id: object_token }
        ### simulation = self.tokens[action[0]].simulation({'priority': priority}) the method has to simulate the execution by requesting the resource with the defined priority
        if action is not None:
            res = self.simulation_process.get_state()['role']
            token_id = self.simulation_process.del_token_queue(res, action)
            simulation = self.tokens[token_id].simulation()
            self.env.process(simulation)
            self.next_decision_moment()
        else:
            ##### se Postpone fai passare un tempo X alla risorsa prima di renderla di nuovo disponibile
            #### simile al calendario
            self.next_decision_moment()

        if token_id in self.last_reward:  ### token finishes process
            reward = self.last_reward[token_id]
        else:
            reward = self.tokens[token_id].last_reward

        if len(self.tokens) == 0:
            isTerminated = True
            self.waiting_times = list(self.simulation_process.waiting_times.values())
            ##### reward finale as percentile
            percentile = np.percentile(self.waiting_times, 95)
            reward = -(1 / (percentile + 1))
            info = {'mean': np.mean(self.waiting_times), 'percentile': np.percentile(self.waiting_times, 95)}
            print('Percentile', np.percentile(self.waiting_times, 95))
            if self.path_step:
                self.save_n_steps_simulation()
        else:
            isTerminated = False
            info = {}
        return self.get_state(), reward, isTerminated, {}, info

    def save_n_steps_simulation(self):
        with open(self.path_step, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.nr_steps]) 

    def delete_tokens_ended(self):
        delete = []
        for i in self.tokens:  ### update tokens ended
            if self.tokens[i].END is True:
                if self.tokens[i]._next_activity is None:
                    self.last_reward[self.tokens[i]._id] = self.tokens[i].last_reward
                    delete.append(i)
        for e in delete:
            del self.tokens[e]

    def check_exception_postpone(self):
        actual_state = self.simulation_process.get_state()
        state_tokens = len(actual_state['traces']['ongoing'])
        ### exception case: all available resources and all tokens already arrived
        if (state_tokens + len(actual_state['traces']['ended'])) == self.params.TRACES and len(actual_state['resource_anvailable']) == 0:
            return True
        else:
            return False

    def handle_postpone_exception(self):
        pre_cycle_time = self.simulation_process.get_state()['traces']['ongoing'].copy()
        next_step = True
        while next_step:
            if self.env.peek() == math.inf:
                next_step = False
            else:
                self.env.step()
            actual_state = self.simulation_process.get_state()['traces']['ongoing']
            if not bool(set(pre_cycle_time) & set(actual_state)):
                next_step = False

    def next_decision_moment(self, start=False):
        #### self.simulation_process.get_state(): resource_available, actual_assignment, traces ongoing and ended with (case_id, accumulate cycle time), time of execution
        next_step = True
        while next_step:
            if not start and self.env.peek() == math.inf: ## no more traces ongoing
                next_step = False
            else:
                previous_state = self.simulation_process.get_state()
                self.env.step()
                start = False
                actual_state = self.simulation_process.get_state()
                state_res = actual_state['resource_available']
                for res in state_res:
                    ### an available resource with at least one token in its queue
                    if state_res[res] > 0 and len(self.simulation_process.role_queues[res]) > 0:
                        next_step = False
                    elif previous_state['resource_available'] != actual_state['resource_available']:
                        resource_release = [True if actual_state['resource_available'][key] > previous_state['resource_available'][key] else False for key in previous_state['resource_available']]
                        time_now = self.simulation_process._date_start
                        if sum(resource_release) > 0:
                            for res in self.simulation_process._resources:
                                if res != 'TRIGGER_TIMER':
                                    role = self.simulation_process._get_resource(res)
                                    if not role.waiting_for_calendar:
                                        self.env.process(role.wait_calendar(time_now))
                    previous_state = actual_state
            self.delete_tokens_ended()

    def get_state(self):
        env_state = self.simulation_process.get_state()
        # Define the input and output of the agent (neural network)

        ### GLOBAL # The general inputs are: month, day, hour, roles occupations, ratio_completed_traces,
        time = [env_state['time'].month/12, env_state['time'].weekday()/6, (env_state['time'].hour*3600 + env_state['time'].minute*60 + env_state['time'].second)/(24*3600)]

        ### ROLE-QUEUE ####  Queue info: role, #tokens_queue, role_occup, role_capacity
        if 'role' in env_state:
            role = (self.resources.index(env_state['role'])/len(self.resources))+1
            queue = 1 / (1 + math.exp(-env_state['role_queue']))
            role_occup = env_state['occupation_roles'][env_state['role']]
            role_capacity = 1 / (1 + math.exp(-env_state['capacity_roles'][env_state['role']]))  ### normalize
        else:
            role = 0
            queue = 1 / (1 + math.exp(1))
            role_occup = 0
            role_capacity = 1 / (1 + math.exp(0))
            tokens_in_queue = []
            hol = 0
            wip_act_queue = []

        array = time
        array += [role, queue, role_occup, role_capacity]
        return np.array(array)

