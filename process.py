'''
Class to manage the resources shared by all the traces in the process.

<img src="../docs/images/process_class.png" alt="Alt Text" width="780">
'''
import simpy
from role_simulator import RoleSimulator
import math
from parameters import Parameters
from datetime import datetime, timedelta
import random


class SimulationProcess(object):

    def __init__(self, env: simpy.Environment, params: Parameters, calendar):
        self._env = env
        self._params = params
        self._date_start = params.START_SIMULATION
        self._resources = self.define_single_role()
        self._resource_events = self._define_resource_events(env)
        self._resource_trace = simpy.Resource(env, math.inf)
        self._am_parallel = []
        self._actual_assignment = []
        self.traces = {"ongoing": [], "ended": []}
        self.resources_available = True ### check if there are resource to assign one event
        self.tokens_pending = {} ### dictionary keyv = id_case, element = tokenOBJ
        self.next_assign = None
        self.calendar = calendar
        #self.predictor = Predictor((self._params.MODEL_PATH_PROCESSING, self._params.MODEL_PATH_WAITING), self._params)
        #self.predictor.predict()
        self.waiting_times = {}
        self.role_queues = {res: [] for res in self._resources}
        self.WINDOW_PREFIX = 5 ### da sistemare
        del self.role_queues['TRIGGER_TIMER']
        self.len_queues = []

    def add_token_queue(self, resources, token_id):
        res = self._get_resource(resources)
        self.role_queues[res._name].append(token_id)

    def del_token_queue(self, res, pos):
        token_id = self.role_queues[res][pos]
        del self.role_queues[res][pos]
        return token_id

    def update_tokens_pending(self, token):
        self.tokens_pending[token._id] = [token, token._time_last_activity]

    def del_tokens_pending(self, id):
        del self.tokens_pending[id]

    def define_single_role(self):
        """
        Definition of a *RoleSimulator* object for each role in the process.
        """
        set_resource = list(self._params.ROLE_CAPACITY.keys())
        dict_role = dict()
        for res in set_resource:
            #if res in self._params.RESOURCE_TO_ROLE_LSTM.keys():
            res_simpy = RoleSimulator(self._env, res, self._params.ROLE_CAPACITY[res][0],
                                      self._params.ROLE_CAPACITY[res][1])
            dict_role[res] = res_simpy
        return dict_role

    def get_occupations_single_role(self, resource):
        """
        Method to retrieve the specified role occupancy in percentage, as an intercase feature:
        $\\frac{resources \: occupated \: in \:role}{total\:resources\:in\:role}$.
        """
        occup = self._resources[resource]._get_resource().count / self._resources[resource]._capacity
        return round(occup, 2)

    def get_occupations_all_role(self, role):
        """
        Method to retrieve the occupancy in percentage of all roles, as an intercase feature.
        """
        occup = []
        if self._params.FEATURE_ROLE == 'all_role':
            for key in self._params.ROLE_CAPACITY_LSTM:
                if key != 'SYSTEM' and key != 'TRIGGER_TIMER':
                    occup.append(self.get_occupations_single_role_LSTM(key))
        else:
            occup.append(self.get_occupations_single_role_LSTM(role))
        return occup

    def get_occupations_single_role_LSTM(self, role):
        """
        Method to retrieve the specified role occupancy in percentage, as an intercase feature:
        $\\frac{resources \: occupated \: in \:role}{total\:resources\:in\:role}$.
        """
        occup = 0
        for res in self._resources:
            #if res != 'TRIGGER_TIMER' and res in self._params.RESOURCE_TO_ROLE_LSTM and self._params.RESOURCE_TO_ROLE_LSTM[res] == role:
            if res != 'TRIGGER_TIMER':
                occup += self._resources[res]._get_resource().count
        #occup = occup / self._params.ROLE_CAPACITY_LSTM[role][0]
        occup = occup / len(self._params.ROLE_CAPACITY[role])
        return round(occup, 2)


    def get_state(self):
        state = {'resource_available': {}, 'resource_unavailable': {}}
        for res in self._resources:
            if res != 'TRIGGER_TIMER':
                role = self._resources[res]
                if role.waiting_for_calendar:
                    state['resource_unavailable'][res] = role._get_resource().capacity
                    state['resource_available'][res] = 0
                else:
                    state['resource_unavailable'][res] = role._get_resource().count
                    state['resource_available'][res] = role._get_resource().capacity - role._get_resource().count

        state['actual_assignment'] = self._actual_assignment
        state['traces'] = self.traces ### {'ongoing: [(caseid, cycle_time)], 'ended': []}
        state['time'] = self._date_start + timedelta(seconds=self._env.now)


        ### state: GLOBAL (time, resource_occupation_per_role, wip, completed_tokens/tokens_to_execute),
        #state['wip'] = self._resource_trace.count
        state['occupation_roles'] = {res: 0 if self._resources[res]._get_resource().count == 0 else self._resources[res]._get_resource().count/self._resources[res]._get_resource().capacity for res in self._resources}
        del state['occupation_roles']['TRIGGER_TIMER']
        state['ration_completed_trace'] = 0 if len(self.traces['ended']) == 0 else len(self.traces['ended'])/self._params.TRACES

        state['capacity_roles'] = { res: self._resources[res]._get_resource().capacity for res in self._resources}
        del state['capacity_roles']['TRIGGER_TIMER']

        #### find the Role free with a not empty queue
        role_designed = None
        for res in self.role_queues:
            if state['resource_available'][res] > 0 and len(self.role_queues[res]):
                role_designed = res

        if role_designed:
            state['role_queue'] = len(self.role_queues[role_designed])
            self.len_queues.append(state['role_queue'])
            state['role_occup'] = state['occupation_roles'][role_designed]
            state['role'] = role_designed
            state['wip_act_queue'] = {res: 0 for res in self._resource_events}

            #### state: window of tokens: activity, len_prefix, acc_cycle_time
            hol = []
            tokens_in_queue = self.role_queues[role_designed]
            for idx, token_id in enumerate(tokens_in_queue):
                prefix = self.tokens_pending[token_id][0]._prefix.get_prefix()
                activity = self.tokens_pending[token_id][0]._trans.label
                state['actual_activity_'+str(idx)] = activity
                state['len_prefix_' + str(idx)] = len(prefix)
                ### acc_waiting_time in the queue and in total
                state['acc_waiting_time_'+str(idx)] = self.tokens_pending[token_id][0].acc_waiting_times
                state['queue_waiting_time_' + str(idx)] = self._env.now - self.tokens_pending[token_id][0].time_entered_in_queue
                ### TO ADD: remaining_cycle_time, fix remain_acts
                remain_acts = 2 #self._params.remain_activities[self.tokens_pending[token_id][0]._trans.name]
                state['remain_acts_' + str(idx)] = remain_acts
                state['estimated_processing_time_' + str(idx)] = self._params.median_processing_time[self.tokens_pending[token_id][0]._trans.label]
                state['wip_act_queue'][activity] += 1
                hol.append(self.tokens_pending[token_id][0].time_entered_in_queue)
                ### remain_cycle_times
                max_processing = max(self._params.median_processing_time.values())
                state['remain_cycle_times_' + str(idx)] = remain_acts * max_processing
                for idx, act in enumerate(prefix[-self.WINDOW_PREFIX:]):
                    state['prefix_' + str(idx)] = act

            state['HOL'] = self._env.now - min(hol)
            state['wip_act_queue'] = list(state['wip_act_queue'].values())
        return state

    def _get_resource(self, resources):
        ### da controllare se libera quella giusta!
        if isinstance(resources, list):
            now = self._date_start + timedelta(seconds=self._env.now)
            if now.weekday() > 4:
                res = [item for item in resources if "weekend" in item][0]
            elif now.hour > 13:
                res = [item for item in resources if "night" in item][0]
            else:
                res = [item for item in resources if "day" in item][0]
        else:
            res = resources
        return self._resources[res]
    def set_actual_assignment(self, id, activity, res):
        self._actual_assignment.append((id, activity, res))

    def _get_resource_event(self, task):
        return self._resource_events[task]

    def _get_resource_trace(self):
        return self._resource_trace

    def _update_state_traces(self, id, env):
        self.traces["ongoing"].append((id, env.now))

    def _release_resource_trace(self, id, time, request_resource, wait):
        tupla = ()
        for i in self.traces["ongoing"]:
            if i[0] == id:
                tupla = i
        #previous_time = tupla[1]
        self.traces["ongoing"].remove(tupla)
        self.traces["ended"].append((id, time))
        self.waiting_times[id] = wait
        self._resource_trace.release(request_resource)

    def update_kpi_trace(self, id, time):
        tupla = ()
        for i in self.traces["ongoing"]:
            if i[0] == id:
                tupla = i
        #previous_time = tupla[1]
        self.traces["ongoing"].remove(tupla)
        self.traces["ongoing"].append((id, time))

    def _define_resource_events(self, env):
        resources = dict()
        for key in self._params.PROCESSING_TIME.keys():
            resources[key] = simpy.Resource(env, math.inf)
        return resources

    def _set_single_resource(self, resource_task):
        return self._resources[resource_task]._get_resources_name()

    def _release_single_resource(self, id, res, activity):
       if res != 'TRIGGER_TIMER':
            self._actual_assignment.remove((id, activity, res))

    def get_predict_processing(self, cid, pr_wip, transition, ac_wip, rp_oc, time):
        return self.predictor.processing_time(cid, pr_wip, transition, ac_wip, rp_oc, time)