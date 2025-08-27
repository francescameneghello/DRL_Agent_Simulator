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

        self.normalization_cycle_time = 10000#= self.normalization_cycle_times[self.name_log] if normalization else 0
        self.policy = POLICY
        self.print = True
        if threshold > 0:
            input_file = './example/' + self.name_log + '/input_' + self.name_log + str(threshold) + '.json'
        else:
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
        self.WINDOW_SIZE = 5
        self.WINDOW_PREFIX = 5

        ## TIME: 'month', 'day', 'hour', 'estimated_processing_time_', 'remain_cycle_times_',  'queue_waiting_time_', 'acc_waiting_time_'+str(i),
        ## CF: 'actual_activity_'+str(i), 'len_prefix_'+str(i), 'prefix_', 'remain_acts_'
        ## CONGESTION: 'actual_role', roles_occup, 'actual_role', 'role_queue', 'role_occup', 'role_capacity',
            # 'HOL', 'ratio_traces', 'wip_act_queue_',  'queue_waiting_time_', 'acc_waiting_time_'+str(i),

        if self.features == 'time' or self.features == 'all':
            self.input = ['month', 'day', 'hour']
        if self.features == 'congestion' or self.features == 'all':
            roles_occup = [resource + '_role_occup' for resource in self.resources]
            self.input += ['actual_role', 'role_queue', 'role_occup', 'role_capacity', 'HOL'] + roles_occup + ['ratio_traces'] ## general features
            for i in range(0, len(self.task_types)-1):
                self.input += ['wip_act_queue_'+str(i)]
        for i in range(0, self.WINDOW_SIZE):
            if self.features == 'time':
                self.input += ['acc_waiting_time_'+str(i), 'queue_waiting_time_'+str(i),
                           'remain_acts_'+str(i), 'estimated_processing_time_'+str(i), 'remain_cycle_times_'+str(i)]
            if self.features == 'congestion':
                self.input += ['acc_waiting_time_' + str(i), 'queue_waiting_time_' + str(i)]
            if self.features == 'control_flow':
                self.input += ['actual_activity_' + str(i), 'len_prefix_' + str(i)]
                self.input += ['prefix_' + str(i) for i in range(self.WINDOW_PREFIX)]
            if self.features == 'all':
                self.input += ['actual_activity_'+str(i), 'len_prefix_'+str(i), 'acc_waiting_time_'+str(i), 'queue_waiting_time_'+str(i),
                               'remain_acts_'+str(i), 'estimated_processing_time_'+str(i), 'remain_cycle_times_'+str(i)]  ### info_for_each_token_in_the_window
                self.input += ['prefix_'+str(i) for i in range(self.WINDOW_PREFIX)] ### add also the prefix of the last 5 activities performes

        print('INPUT', self.input)
        print("###############################################################")
        # priority of the token in the queue i.e. a number from 0 to 9
        #self.output = [i for i in range(0, self.WINDOW_SIZE)]
        #if self.postpone: # Add postpone action
        #    self.output += ['Postpone']
        self.output = [i for i in range(0, self.WINDOW_SIZE)]

        path_model = './example/' + self.name_log + '/' + self.name_log
        if exists(path_model + '_diapr_meta.json'):
            self.FEATURE_ROLE = 'all_role'
        elif exists(path_model + '_dispr_meta.json'):
            self.FEATURE_ROLE = 'no_all_role'
        else:
            self.FEATURE_ROLE = None
        self.PATH_PETRINET = './example/' + self.name_log + '/' + self.name_log + '.pnml'
        PATH_PARAMETERS = input_file

        
        self.PATH_LOG = './example/' + self.name_log + '/' + self.name_log + '.xes'
        self.params = Parameters(PATH_PARAMETERS, self.N_TRACES, self.name_log, self.FEATURE_ROLE, threshold)

        print(len(self.input), len(self.output))
        # Observation space
        ### define the minimum and maximum values that each output can have
        lows = np.array([0 for _ in range(len(self.input))])
        highs = np.array([1 for _ in range(len(self.input))])
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.input),),
                                            dtype=np.float64)

        # Action space
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.output))
        print('ACTION SPACE', self.action_space)
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
        self.env = simpy.Environment()
        self.simulation_process = SimulationProcess(self.env, self.params, self.CALENDAR)
        self.completed_traces = []
        #if i != None:

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
        pre_tokens_ended = set(self.last_reward.keys())
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

        info = {}
        if len(self.tokens) == 0:
            isTerminated = True
            self.waiting_times = list(self.simulation_process.waiting_times.values())
            info = {'mean': np.mean(self.waiting_times), 'percentile': np.percentile(self.waiting_times, 95)}
            print('Mean waiting times', np.mean(self.waiting_times), 'Percentile', np.percentile(self.waiting_times, 95))
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
            start = self.simulation_process._date_start
            for res in self.simulation_process._resources:
                if res != 'TRIGGER_TIMER':
                    role = self.simulation_process._get_resource(res)
                    if not role.waiting_for_calendar:
                        self.env.process(role.wait_calendar(start))

            self.next_decision_moment()
        else:
            ##### se Postpone fai passare un tempo X alla risorsa prima di renderla di nuovo disponibile
            #### simile al calendario
            self.next_decision_moment()

        reward = 0
        if len(self.tokens) == 0:
            isTerminated = True
            self.waiting_times = list(self.simulation_process.waiting_times.values())
            ##### reward finale as percentile
            percentile = np.percentile(self.waiting_times, 95)
            reward = -(1 / (percentile + 1))
            info = {'mean': np.mean(self.waiting_times), 'percentile': np.percentile(self.waiting_times, 95)}
            print('Mean waiting times', np.mean(self.waiting_times), 'Percentile',
                  np.percentile(self.waiting_times, 95))
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
                self.env.step()
                start = False
                actual_state = self.simulation_process.get_state()
                state_res = actual_state['resource_available']
                for res in state_res:
                    ### an available resource with at least one token in its queue
                    if state_res[res] > 0 and len(self.simulation_process.role_queues[res]) > 0:
                        next_step = False
            self.delete_tokens_ended()

    def get_state(self):
        env_state = self.simulation_process.get_state()
        # Define the input and output of the agent (neural network)

        ### GLOBAL # The general inputs are: month, day, hour, roles occupations, ratio_completed_traces,
        time = [env_state['time'].month/12, env_state['time'].weekday()/6, (env_state['time'].hour*3600 + env_state['time'].minute*60 + env_state['time'].second)/(24*3600)]
        occupation_roles = [env_state['occupation_roles'][r] for r in env_state['occupation_roles']]
        tokens_running = env_state['ration_completed_trace'] ### already a number from 0 to 1
        
        ### ROLE-QUEUE ####  Queue info: role, #tokens_queue, role_occup, role_capacity
        tokens_features = []
        if 'role' in env_state:
            role = (self.resources.index(env_state['role'])/len(self.resources))+1
            queue = 1 / (1 + math.exp(-env_state['role_queue']))
            role_occup = env_state['occupation_roles'][env_state['role']]
            role_capacity = 1 / (1 + math.exp(-env_state['capacity_roles'][env_state['role']]))  ### normalize
            tokens_in_queue = self.simulation_process.role_queues[env_state['role']]
            hol = (env_state['HOL'] - self.params.WAITING_TIMES_SINGLE['Mean'])/self.params.WAITING_TIMES_SINGLE['Std']
            wip_act_queue = [n/self.WINDOW_SIZE for n in env_state['wip_act_queue']]
        else:
            role = 0
            queue = 1 / (1 + math.exp(1))
            role_occup = 0
            role_capacity = 1 / (1 + math.exp(0))
            tokens_in_queue = []
            hol = 0
            wip_act_queue = []

        # For each token in the window: activity, len_prefix, acc_cycle_time
        # dummy_value == 0
        ## add also the prefix of each token, as the last 5 activities performed
        tokens_in_queue = tokens_in_queue[:self.WINDOW_SIZE] ### keep the first k tokens in the window
        for i in range(len(tokens_in_queue)):
            activity = self.task_types.index(env_state['actual_activity_'+str(i)])
            tokens_features.append(activity/(len(self.task_types)+1)) ## normalize activity
            tokens_features.append(min(1, (env_state['len_prefix_'+str(i)]+1)/self.params.LEN_prefix)) ### len prefix
            normalize_acc = (env_state['acc_waiting_time_'+str(i)] - self.params.WAITING_TIMES_LOG['Mean'])/self.params.WAITING_TIMES_LOG['Std']
            tokens_features.append(normalize_acc) ## acc_waiting time
            normalize_wait_queue = (env_state['queue_waiting_time_' + str(i)] - self.params.WAITING_TIMES_SINGLE['Mean']) / self.params.WAITING_TIMES_SINGLE['Std']
            tokens_features.append(normalize_wait_queue) ## queue waiting time
            tokens_features.append(env_state['remain_acts_'+ str(i)]/max(self.params.remain_activities.values())) ## remain_acts
            normalize_processing = (env_state['estimated_processing_time_' + str(i)] - self.params.cycle_times['Mean']) / \
                            self.params.cycle_times['Std']
            tokens_features.append(normalize_processing) ### estimated processing time
            normalize_cycle = (env_state['remain_cycle_times_' + str(i)] - self.params.cycle_times['Mean']) / \
                            self.params.cycle_times['Std'] ## remain cycle time
            tokens_features.append(normalize_cycle)
            for idx_e in range(self.WINDOW_PREFIX):
                prefix_label = 'prefix_' + str(idx_e)
                if prefix_label in env_state:
                    type = self.task_types.index(env_state[prefix_label])
                else:
                    type = self.task_types.index('PAD')
                tokens_features.append(type/(len(self.task_types)+1))
        size = self.WINDOW_PREFIX + 7
        if len(tokens_features) < self.WINDOW_SIZE*size:
            token_to_add = int((self.WINDOW_SIZE*size - len(tokens_features))/size)
            for i in range(0, token_to_add):
                tokens_features += [0]*size

        array = time + occupation_roles + [tokens_running]
        array += [role, queue, role_occup, role_capacity, hol] + wip_act_queue
        array += tokens_features
        return np.array(array)

