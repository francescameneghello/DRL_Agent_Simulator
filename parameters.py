"""
    Class for reading simulation parameters
"""
import json
import math
import os
from datetime import datetime


class Parameters(object):

    def __init__(self, path_parameters: str, traces: int, name_log: str, feature_role: str, threshold=0):
        self.TRACES = traces
        """TRACES: number of traces to generate"""
        self.PATH_PARAMETERS = path_parameters
        """PATH_PARAMETERS: path of json file for others parameters. """
        self.FEATURE_ROLE = feature_role
        if self.FEATURE_ROLE == 'all_role':
            self.prefix = ('_diapr', '_dpiapr', '_dwiapr')
        else:
            self.prefix = ('_dispr', '_dpispr', '_dwispr')
        self.threshold = threshold
        self.NAME_EXP = name_log
        self.read_metadata_file()

    def read_metadata_file(self):
        '''
        Method to read parameters from json file, see *main page* to get the whole list of simulation parameters.
        '''
        if os.path.exists(self.PATH_PARAMETERS):
            with open(self.PATH_PARAMETERS) as file:
                data = json.load(file)
                roles_table = data['resource_table']
                self.START_SIMULATION = self._check_default_parameters(data, 'start_timestamp')
                self.SIM_TIME = self._check_default_parameters(data, 'duration_simulation')
                self.PROBABILITY = data['probability'] if 'probability' in data.keys() else []
                self.PROCESSING_TIME = data['processing_time']
                self.WAITING_TIME = data['waiting_time'] if 'waiting_time' in data.keys() else []
                self.INTER_TRIGGER = data["interTriggerTimer"]
                self.ROLE_CALENDAR = data["role_calendars"]
                self.ROLE_ACTIVITY = dict()
                for elem in roles_table:
                    self.ROLE_ACTIVITY[elem] = roles_table[elem]

                if 'calendar' in data['interTriggerTimer'] and data['interTriggerTimer']['calendar']:
                    self.ROLE_CAPACITY = {'TRIGGER_TIMER': [math.inf, {'days': data['interTriggerTimer']['calendar']['days'], 'hour_min': data['interTriggerTimer']['calendar']['hour_min'], 'hour_max': data['interTriggerTimer']['calendar']['hour_max']}]}
                else:
                    self.ROLE_CAPACITY = {'TRIGGER_TIMER': [math.inf, []]}
                self._define_roles_resources(data['roles'])
                self.LEN_prefix = data['LEN_prefix']
                self.WAITING_TIMES_LOG = data["WAITING_TIMES_LOG"]
                self.WAITING_TIMES_SINGLE = data["SINGLE_WAITING_TIMES_LOG"]
                self.remain_activities = data["Remain_activities"]
                self.median_processing_time = data["Median_processing_time"]
                self.cycle_times = data["CYCLE_TIMES"]
        else:
            raise ValueError('Parameter file does not exist')

    def _define_roles_resources(self, roles):
        for idx, key in enumerate(roles):
            #self.ROLE_CAPACITY[key] = [roles[key]['resources'], {'days': roles[key]['calendar']['days'],
            #                                                          'hour_min': roles[key]['calendar']['hour_min'],
            #                                                          'hour_max': roles[key]['calendar']['hour_max']}]
            self.ROLE_CAPACITY[key] = [roles[key], self.ROLE_CALENDAR[key]]

    def _check_default_parameters(self, data, type):
        if type == 'start_timestamp':
            value = datetime.strptime(data['start_timestamp'], '%Y-%m-%d %H:%M:%S') if type in data else datetime.now()
        elif type == 'duration_simulation':
            value = data['duration_simulation'] if type in data else 31536000
        return value
