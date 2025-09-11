
from gym_env import gym_env
from sb3_contrib import MaskablePPO
import random
import glob
import pandas as pd
import pm4py
from pm4py.objects.log.util import sorting
import numpy as np
import scipy.stats as st
import json
from collections import defaultdict
import sys, os


def FIFO(simulation_process, state, tokens_pending):
    next_ass = None
    if len(tokens_pending):
        ### first token that arrived in the queue, so position 0 in the queue
        next_ass = 0
    return next_ass


def FIFO_case(simulation_process, state, tokens_pending):
    next_ass = None
    if len(tokens_pending) > 0:
        token_id = min(simulation_process.role_queues[state['role']])
        next_ass = simulation_process.role_queues[state['role']].index(token_id)
    return next_ass


def RANDOM(state, tokens_pending):
    #### action is random position on the queue of the designed role
    next_ass = None
    if len(tokens_pending) > 0:
        next_ass = random.randrange(0, state['role_queue'], 1)
    return next_ass


def SPT(simulation_process, state, tokens_pending, median_processing_time):
    next_ass = None
    if len(tokens_pending) > 0:
        role_queue = simulation_process.role_queues[state['role']]
        estimate_process_time = []
        for token in role_queue:
            act = tokens_pending[token][0]._next_activity
            estimate_process_time.append(median_processing_time[act])
        pos_min = estimate_process_time.index(min(estimate_process_time))
        next_ass = pos_min
    return next_ass


def retrieve_median_processing_time(NAME_LOG):
    path = './example/' + NAME_LOG + '/' + NAME_LOG + '.xes'
    log = pm4py.read_xes(path)
    activities = pm4py.get_event_attribute_values(log, "concept:name")
    median_processing_time = {a: [] for a in activities}
    for index, row in log.iterrows():
        activity = row['concept:name']
        time = (row['time:timestamp'] - row['start:timestamp']).total_seconds()
        median_processing_time[activity].append(time)
    for a in median_processing_time:
        median_processing_time[a] = np.mean(median_processing_time[a])
    return median_processing_time


def confidence_interval(data):
    n = len(data)
    C = 0.95
    alpha = 1 - C
    tails = 2
    q = 1 - (alpha / tails)
    dof = n - 1
    t_star = st.t.ppf(q, dof)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    ci_upper = x_bar + t_star * s / np.sqrt(n)
    ci_lower = x_bar - t_star * s / np.sqrt(n)
    print(x_bar, ci_lower, ci_upper)
    return ci_lower, ci_upper


def evaluate(NAME, POLICY, data, calendar, threshold):
    #"output/output_{}_{}_{}/simulated_log_{}_{}_{}".format(self.name_log, calendar, self.threshold, self.name_log, self.policy, str(i)) + ".csv", 'w'
    path = f'output/test/simulated_log_{NAME}_{POLICY}'# + '*.csv'
    #path = 'output/output_' + NAME + '/ENTIRE_LOG_CALENDAR/'+POLICY+'/simulated_log_' + NAME + '_' + POLICY + '*.csv'
    #all_file = glob.glob(path)
    all_file = [path + '_' + str(i) + '.csv' for i in range(0, 1)]
    cycle_time = []
    for file in all_file:
        simulated_log = pd.read_csv(file)
        simulated_log['start_time'] = pd.to_datetime(simulated_log['start_time'], format="%Y-%m-%d %H:%M:%S")
        simulated_log['end_time'] = pd.to_datetime(simulated_log['start_time'], format="%Y-%m-%d %H:%M:%S")
        simulated_log = pm4py.format_dataframe(simulated_log, case_id='id_case', activity_key='activity',
                                               timestamp_key='end_time', start_timestamp_key='start_time')
        simulated_log = pm4py.convert_to_event_log(simulated_log)
        simulated_log = sorting.sort_timestamp_log(simulated_log)
        for trace in simulated_log:
            cycle_time.append((trace[-1]['end_time'] - trace[0]['start_time']).total_seconds())
    #print('evaluate',cycle_time)
    ci_lower, ci_upper = confidence_interval(cycle_time)
    data[POLICY] = {'values': cycle_time, 'mean': [np.mean(cycle_time), ci_lower, ci_upper]}


def run_simulation(NAME_LOG, POLICY, N_SIMULATION, threshold=0, postpone=True, reward_function='inverse_CT', model=None, median_processing_time=None):
    output_QUEUE_progression = ''
    env_simulator = gym_env(NAME_LOG, N_TRACES, CALENDAR, POLICY=POLICY, threshold=threshold, postpone=postpone)
    if POLICY in ['FIFO_activity', 'FIFO_case', 'SPT', 'RANDOM']:
        file = f'./output/result_{NAME_LOG}_C{CALENDAR}_T{THRESHOLD}_{POLICY}.json'
    else:
        file = f'./output/result_{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{THRESHOLD}_P{postpone}_{reward_function}_{POLICY}.json'
    cycle_times = {}
    percentile_wait_simulations = []
    mean = []
    for i in range(0, N_SIMULATION):
        obs = env_simulator.reset(i=i)
        isTerminated = False
        ##### simulation #####
        while not isTerminated:
            state = env_simulator.get_state()
            if POLICY == 'FIFO_activity':
                action = FIFO(env_simulator.simulation_process, env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            elif POLICY == 'FIFO_case':
                action = FIFO_case(env_simulator.simulation_process, env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            elif POLICY == 'SPT':
                action = SPT(env_simulator.simulation_process, env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending, median_processing_time)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            elif POLICY == 'RANDOM':
                action = RANDOM(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            else:
                mask = env_simulator.action_masks()
                action, _states = model.predict(state, action_masks=mask)
                state, reward, isTerminated, dones, info = env_simulator.step(action)
        cycle_times[f'simulation_{i}'] = {'percentile': info['percentile'], 'mean': info['mean']}
        percentile_wait_simulations.append(info['percentile'])
        mean.append(info['mean'])
    with open(file, 'w') as f:
        cycle_times['percentile_wait_simulations'] = percentile_wait_simulations
        json.dump(cycle_times, f, indent=2)

    print('Mean percentile simulations', np.mean(percentile_wait_simulations))
    print('END simulation')


### Path DRL model
traces = {"BPI_Challenge_2012_W_Two_TS":1200,
        "confidential_1000":159,
        "ConsultaDataMining201618":181,
        "PurchasingExample":101,
        "BPI_Challenge_2017_W_Two_TS":5789,
        "Productions":45,
        "ER_hospital": 1000}

if len(sys.argv) > 1:
    NAME_LOG = sys.argv[1]#'BPI_Challenge_2017_W_Two_TS'
    if not sys.argv[2] == 'from_input_data':    
        N_TRACES = int(sys.argv[2])#2000
    else:
        N_TRACES = sys.argv[2]
    CALENDAR = True if sys.argv[3] == "True" else False
    THRESHOLD = int(sys.argv[4])
    postpone = True if sys.argv[5] == "True" else False
    reward_function = sys.argv[6]
    POLICY = sys.argv[7]

    if POLICY == 'None':
        model = MaskablePPO.load(os.getcwd() + "/tmp/" + f"{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{THRESHOLD}_P{postpone}_{reward_function}" + "/best_model.zip")
    N_SIMULATION = 100
else:
    model = None
    NAME_LOG = "ER_hospital" #'PurchasingExample'#'BPI_Challenge_2017_W_Two_TS', 'confidential_1000', 'ConsultaDataMining201618','PurchasingExample'
    N_TRACES = 1
    CALENDAR = True
    THRESHOLD = 'Exp7200'
    POLICY = 'FIFO_activity'
    N_SIMULATION = 1

if POLICY == 'SPT':
    median_processing_time = retrieve_median_processing_time(NAME_LOG)
    print(median_processing_time)
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, THRESHOLD, median_processing_time=median_processing_time)
elif POLICY == 'FIFO_activity' or POLICY == 'FIFO_case' or POLICY == 'RANDOM':
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, THRESHOLD)
else:
    model = MaskablePPO.load('/home/francesca/Documents/Resource_simulator_DRL/tmp_training_2/BPI_Challenge_2012_W_Two_TS_from_input_data_CTrue_T20_PFalse_bpi12_all_2/best_model.zip')
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, threshold=THRESHOLD, postpone=False, reward_function='inverse_CT', model=model)
