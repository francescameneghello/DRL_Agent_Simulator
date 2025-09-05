"""
Class to manage the arrivals times of tokes in the process.
"""

import numpy as np
from datetime import datetime, timedelta
from parameters import Parameters
from process import SimulationProcess
import custom_function as custom
import pandas as pd
from prophet.serialize import model_to_json, model_from_json
from numpy.random import triangular as triang
import json
from scipy.stats import truncnorm
from scipy.stats import expon

class InterTriggerTimer(object):

    def __init__(self, params: Parameters, process: SimulationProcess, start: datetime, n_traces: int):
        self._process = process
        self._start_time = start
        self._type = params.INTER_TRIGGER['type']
        self._previous_date = start
        self._previous = 0
        if self._type == 'distribution':
            """Define the distribution of token arrivals from specified in the file json"""
            self.name_distribution = params.INTER_TRIGGER['name']
            self.params = params.INTER_TRIGGER['parameters']
        else:
            self.params = params
        self._interval = 0
        #self.generate_arrivals(n_traces, start)

    def get_next_arrival(self, env, case, name_log, calendar):
        """Generate a new arrival from the distribution and check if the new token arrival is inside calendar,
        otherwise wait for a suitable time."""
        if self._type == 'distribution':
            if self.name_distribution == "truncated_normal":
                lower = self.params["lower"]
                upper = self.params["upper"]
                mu = self.params["loc"]
                sigma = self.params["scale"]
                a, b = (lower - mu) / sigma, (upper - mu) / sigma
                arrival = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1)[0]
            elif self.name_distribution == "exponential":
                scale = self.params["scale"]
                min_val = self.params["lower"]
                max_val = self.params["upper"]
                def truncated_exponential_inverse(scale, min_val, max_val, size=1000):
                    cdf_min = expon.cdf(min_val, scale=scale)
                    cdf_max = expon.cdf(max_val, scale=scale)
                    u = np.random.uniform(cdf_min, cdf_max, size=size)
                    return expon.ppf(u, scale=scale)
                arrival = truncated_exponential_inverse(scale, min_val, max_val, size=1)[0]
            else:
                arrival = getattr(np.random, self.name_distribution)(**self.params, size=1)[0]
            resource = self._process._get_resource('TRIGGER_TIMER')
            if calendar and resource._get_calendar():
                stop = resource.to_time_schedule(self._start_time + timedelta(seconds=env.now + arrival))
                self._interval = stop + arrival
            else:
                self._interval = arrival
        elif self._type == 'custom':
            arrival = self.custom_arrival(self._start_time + timedelta(seconds=env.now))
        else:
            raise ValueError('ERROR: Invalid arrival times generator')
        self._previous += arrival
        #self._previous_date = self._start_time + timedelta(seconds=self._interval)
        return self._previous

    def custom_arrival(self, time):
        """
        Call to the custom functions in the file custom_function.py.
        """
        return custom.custom_arrivals_time(time)

