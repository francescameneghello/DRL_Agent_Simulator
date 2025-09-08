'''
Class to define the role object within the simulation.
The role is defined by its name (*name*), the amount of resources available (*capacity*),
and the schedule assigned to it (*calendar*).

Each role needs to be defined within the json file in the following format.
The example describes *Role 2*, Ellen and Sue, who work Monday
through Saturday, from 8 a.m. to 7 p.m.
```shell
    "resource": {
        ........
        "name_of_role": {
            "resources": #List of resources,
            "calendar": {
                "days": #List of working days of the role,
                        #given as integer 0 means Sunday and 6 means Saturday.
                "hour_min": #the hour of the start of the working day,
                "hour_max": #the hour of the end of the working day
            }
        }
        ........
    }
```

```json
    "Role 2": {
        "resources": ["Ellen", "Sue"],
        "calendar": {
            "days": [1, 2, 3, 4, 5, 6],
            "hour_min": 8,
            "hour_max": 19
        }
    }
```
Finally, in the simulation parameters file, we have to indicate the role assigned to the execution of each activity.
(Here to view complete examples)

```json
    "resource_table": [
        {
            "role": "Role 1",
            "task": "A_SUBMITTED"
        },
        {
            "role": "Role 2",
            "task": "A_PARTLYSUBMITTED"
        }
    ]
```

'''
from datetime import timedelta
import simpy
import random


class RoleSimulator(object):

    def __init__(self, env: simpy.Environment, name: str, capacity, calendar: dict):
        self._env = env
        self._name = name
        self._resources_name = capacity
        self._defined_resource = capacity if type(capacity) == float else set(capacity)
        self._capacity = capacity if type(capacity) == float else len(capacity) #1
        self._calendar = calendar
        self._resource_simpy = self._resource_simpy = simpy.Resource(env, self._capacity)
        self._queue = []
        self.waiting_for_calendar = True

    def wait_calendar(self, start_process, time=1):
        stop = self.to_time_schedule(start_process + timedelta(seconds=self._env.now))
        self.waiting_for_calendar = True if stop > 0 else False
        yield self._env.timeout(stop)
        self.waiting_for_calendar = False

    def _get_name(self):
        return self._name

    def get_name_single_resource(self):
        single_resource = self._defined_resource.pop()
        return single_resource

    def _set_name_single_resource(self, single_resource):
        self._defined_resource.add(single_resource)

    def _get_capacity(self):
        return self._capacity

    def _get_resource(self):
        return self._resource_simpy

    def _get_calendar(self):
        return self._calendar

    def release(self, request, single_resource):
        """
        Method to release the role resource that was used to perform the activity.
        """
        self._resource_simpy.release(request)
        self._set_name_single_resource(single_resource)

    def request(self):
        """
        Method to require a resource of the role needed to perform the activity.
        """
        self._queue.append(self._resource_simpy.queue)
        return self._resource_simpy.request()

    def _check_day_work(self, timestamp):
        return True if (timestamp.weekday() in self._calendar['days']) else False

    def _check_hour_work(self, timestamp):
        hour_min = self._calendar['hour_min_weekend'] if timestamp.weekday() > 4 else self._calendar['hour_min_week']
        hour_max = self._calendar['hour_max_weekend'] if timestamp.weekday() > 4 else self._calendar['hour_max_week']
        return True if (hour_min <= timestamp.hour < hour_max) else False

    def _define_stop_weekend(self, timestamp):
        monday = 7 - timestamp.weekday()
        new_start = timestamp.replace(hour=self._calendar['hour_min_week'], minute=0, second=0) + timedelta(days=monday)
        return (new_start-timestamp).total_seconds()

    def _define_stop_between_days(self, timestamp):
        hour_min = self._calendar['hour_min_weekend'] if timestamp.weekday() > 4 else self._calendar['hour_min_week']
        if timestamp.hour < hour_min:
            stop = (timestamp.replace(hour=hour_min, minute=0, second=0) - timestamp).total_seconds()
        else:
            next_hour_min = self._calendar['hour_min_week'] if (timestamp.weekday() + 1) > 6 else self._calendar['hour_min_weekend']
            new_day = timestamp.replace(hour=next_hour_min, minute=0, second=0) + timedelta(days=1)
            stop = (new_day - timestamp).total_seconds()
        return stop

    ### previous version with jus tone interval for calendar
    def _define_stop_week(self, timestamp):
        if timestamp.hour < self._calendar['hour_min']:
            stop = (timestamp.replace(hour=self._calendar['hour_min'], minute=0, second=0) - timestamp).total_seconds()
        else:
            new_day = timestamp.replace(hour=self._calendar['hour_min'], minute=0, second=0) + timedelta(days=1)
            stop = (new_day - timestamp).total_seconds()
            if new_day.weekday() not in self._calendar['days']:
                stop = stop + self._define_stop_weekend(new_day)
        return stop

    def to_time_schedule(self, timestamp):
        """
            Method to check the schedule of the requested resource and
            eventually it returns the time to wait before executing the activity.
        """
        if not self._check_day_work(timestamp):
            stop = self._define_stop_weekend(timestamp)
        elif not self._check_hour_work(timestamp):
            stop = self._define_stop_between_days(timestamp) #self._define_stop_week(timestamp)
        else:
            stop = 0
        return stop


    #def _get_resources_name(self):
    #    choiced = self._resources_name[0]
    #    self._resources_name.remove(choiced)
    #    return choiced

    def _release_resource_name(self, resource):
        self._resources_name.append(resource)