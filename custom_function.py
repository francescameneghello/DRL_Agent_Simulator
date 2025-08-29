'''
This file contains all the customizable functions, which the user can define,
and which are called by the simulator in the specific steps.

The following table describes the case and inter-case features that can be
used as input from a predictive model.

| Feature      | Description  |
|:------------:|:-------------------------- |
| id_case | Case id of the trace to which the event belongs. |
| activity | Represents the next activity to be performedg. |
| enabled_time | Timestamp of when the activity requests the role to be executed. |
| start_time |  Timestamp of when the activity starts to run.   |
| end_time |   Timestamp of when the activity ends to run.    |
| role |   Designated role to perform the next activity.   |
| resource |  Role resource available to perform the activity.   |
| wip_wait | Represents the number of traces running in the simulation before the waiting time.  |
| wip_start |  Represents the number of traces running in the simulation once an available resource is obtained to run the activity. |
| wip_end | Represents the number of traces running in the simulation at the end of the activity execution. |
| wip_activity |  Represents the number of events running in the simulation that perform the same activity, once an available resource is obtained.   |
| ro_total |    The percentage of occupancy in terms of resources in use for the roles defined in the simulation.   |
| ro_single |   The percentage of occupancy in terms of resources in use for the role defined for the next activity.     |
| queue |  Represents the length of the queue for the required resource.  |
| prefix |  List of activities already performed.  |
| attribute_case |  Attributes defined for the trace.      |
| attribute_event |  Attributes defined for the next event to be executed.   |

'''

from statsmodels.tsa.ar_model import AutoRegResults
from utility import Buffer
import random
import pickle
from datetime import datetime
import os
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import expon


def case_function_attribute(case: int, time: datetime):
    """
        Function to add one or more attributes to each trace.
        Input parameters are case id number and trace start timestamp and return a dictionary.
        For example, we generate a trace attribute, the requested loan amount, to simulate the process from the BPIChallenge2012A.xes log.
    """
    '''Parameters:
        - age(int or float): Age of the patient
        - expert(int: 1 = expert, 0 = non - expert) ### fix after during the simulation
        - citizen(int: 1 = citizen, 0 = not citizen)
        - insurance(int: 1 = has private insurance, 0 = no private insurance)'''
    return {"age": random.randint(18, 99), "expert": 0, "citizen": random.choices([0, 1], [0.10, 0.90])[0], "insurance": random.choices([0, 1], [0.80, 0.20])[0]}


def event_function_attribute(case: int, time: datetime):
    """
        Function to add one or more attributes to each event.
        Input parameters are case id number and track start timestamp and return a dictionary.
        In the following example, we assume that there are multiple bank branches where activities
        are executed by day of the week. From Monday to Wednesday, activities are executed in the
        Eindhoven branch otherwise in the Utrecht one.
    """
    #bank = "Utrecht" if time.weekday() > 3 else "Eindhoven"
    return {}


def custom_arrivals_time(case, previous, name_log):
    """
    Function to define a new arrival of a trace. The input parameters are the case id number and the start timestamp of the previous trace.
    For example, we used an AutoRegression model for the *arrivals example*.
    """
    path = './example/' + name_log + '/inter_arrival_rate.csv'
    arrival = pd.read_csv(path)
    next = datetime.strptime(arrival.loc[case].at["timestamp"], '%Y-%m-%d %H:%M:%S')
    interval = (next - previous).total_seconds()
    return interval

def custom_resource(state, tokens_pending, time):
    '''
    INPUT
        state --->
            STATE {'resource_available': ['Sue'], 'resource_anvailable': ['Sara', 'Mike', 'Pete'],
            'actual_assignment': [(7, 'A_REGISTERED', 'Pete'), (8, 'A_DECLINED', 'Mike'), (7, 'A_APPROVED', 'Sara')],
             'traces': {'ongoing': [(0, 32960.446648234916), (2, 32541.33625239031), (1, 36255.4695204107), (8, 39628.923973052784),
            (9, 51419.70748698439), (7, 52302.5958864416), (6, 52497.56907047473)], 'ended': [(5, 30940.946641046834), (4, 31159.896296280713), (3, 38733.15122274552)]}}

        token_pending ===> list of waiting tokens. For each token in the list you can get: "token._buffer" (information about token explained on https://francescameneghello.github.io/RIMS_tool/custom_function.html),
        "token._next_activity" the next activity to perform, "token._id" id of token

    OUPUT
        (token._id, token._next_activity, random.choice(state['resource_available'])) ====> tupla(id_case, next_activity,resource)
    '''
    pass


def custom_processing_time(buffer: Buffer):
    """
    Define the processing time of the activity (return the duration in seconds).
    Example of features that can be used to predict:

    ```json
            {
                "id_case": 23,
                "activity": "A_ACCEPTED",
                "enabled_time": "2023-08-23 11:14:13",
                "start_time": "2023-08-23 11:14:13",
                "end_time": "2023-08-23 11:20:13",
                "role": "Role 2",
                "resource": "Sue",
                "wip_wait": 3,
                "wip_start": 3,
                "wip_end": 3,
                "wip_activity": 1,
                "ro_total": [0.5, 1],
                "ro_single": 1,
                "queue": 0,
                "prefix": ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED"],
                "attribute_case": {"AMOUNT": 59024},
                "attribute_event": {"bank_branch": "Eindhoven"}
            }
    ```
    """
    parameters = buffer.get_feature("activity")
    lower = 0
    upper = 10800
    scale = 7200
    def truncated_exponential_inverse(scale, min_val, max_val, size=1000):
        cdf_min = expon.cdf(min_val, scale=scale)
        cdf_max = expon.cdf(max_val, scale=scale)
        u = np.random.uniform(cdf_min, cdf_max, size=size)
        return expon.ppf(u, scale=scale)
    duration = truncated_exponential_inverse(scale, lower, upper, size=1)[0]
    return duration


def custom_waiting_time(buffer: Buffer):
    """ Define the waiting time of the activity (return the duration in seconds).
    Example of features that can be used to predict:
    ```json
    {
        "id_case": 15,
        "activity": "A_PARTLYSUBMITTED",
        "enabled_time": "None",
        "start_time": "None",
        "end_time": "None",
        "role": "Role 2",
        "resource": "None",
        "wip_wait": 21,
        "wip_start": -1,
        "wip_end": -1,
        "wip_activity": 1,
        "ro_total": [0.5, 1],
        "ro_single": 1,
        "queue": 13,
        "prefix": ["A_SUBMITTED"],
        "attribute_case": {"AMOUNT": 18207},
        "attribute_event": {"bank_branch": "Eindhoven"}

    }
    ```
    """
    input_feature = list()
    buffer.print_values()
    input_feature.append(buffer.get_feature("wip_wait"))
    input_feature.append(buffer.get_feature("wip_activity"))
    input_feature.append(buffer.get_feature("enabled_time").weekday())
    input_feature.append(buffer.get_feature("enabled_time").hour)
    input_feature.append(buffer.get_feature("ro_single"))
    input_feature.append(buffer.get_feature("queue"))
    loaded_model = pickle.load(
        open(os.getcwd() + '/example/example_process_times/waiting_time_random_forest.pkl', 'rb'))
    y_pred_f = loaded_model.predict([input_feature])
    return int(y_pred_f[0])


def custom_decision_mining(buffer: Buffer):
    """
    Function to define the next activity from a decision point in the Petri net model.
    For example, we used a Random Forest model for the *decision mining* example.

    Example of features that can be used to predict:
    ```json
        {
            "id_case": 43,
            "activity": "A_FINALIZED",
            "enabled_time": "2023-08-24 16:24:29",
            "start_time": "2023-08-24 19:37:31",
            "end_time": "2023-08-24 20:03:34",
            "role": "Role 2",
            "resource": "Sue",
            "wip_wait": 16,
            "wip_start": 4,
            "wip_end": 4,
            "wip_activity": 2,
            "ro_total": [0.0, 1],
            "ro_single": 1,
            "queue": 15,
            "prefix": ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED", "A_ACCEPTED"],
            "attribute_case": {"AMOUNT": 86061},
            "attribute_event": {"bank_branch": "Eindhoven"}

        }
    ```
    """
    input_feature = list()
    prefix = buffer.get_feature("prefix")
    if "Register" in prefix[-1]:
        ##### gateway to decide Examination or Expert Examination
        actual_time = buffer.get_feature("end_time")
        if actual_time.weekday() > 4 or actual_time.hour > 14:
            return 0 #Examination
        else:
            #['Activity_0j3kufo', 'Activity_0va6x20'] ---> ["Examination", "Expert Examination"]
            prob = [0.70, 0.30]
            value = [*range(0, len(prob), 1)]
            return int(random.choices(value, prob)[0])
    else:
        ### decision point for Treatment unsuccessful or Treatment successful
        beta_0 = -1.0  # intercept
        beta_age = -0.05  # older age reduces success probability slightly
        beta_expert = 1.5  # expert doctor increases probability
        beta_citizen = 0.2  # german citizen has slight increase
        beta_insurance = 0.7  # insurance increases probability
        linear_score = (beta_0
                        + beta_age * buffer.get_feature("attribute_case")["age"]
                        + beta_expert * buffer.get_feature("attribute_case")["expert"]
                        + beta_citizen * buffer.get_feature("attribute_case")["citizen"]
                        + beta_insurance * buffer.get_feature("attribute_case")["insurance"])

        # Logistic function
        #probability = 1 / (1 + np.exp(-linear_score))
        probability = 0.80
        # ["Activity_1368wm0", "Activity_1swf9sg"] ----> ["Treatment successful", "Treatment unsuccessful"]
        prob = [probability, 1-probability]
        value = [*range(0, len(prob), 1)]
        return int(random.choices(value, prob)[0])