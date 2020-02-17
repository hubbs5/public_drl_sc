

import numpy as np
import calendar
import string
import datetime
from .demand_utils import softmax
from argparse import ArgumentTypeError


def get_default_test_demand_settings(settings):
    '''
    Inputs
    =========================================================================
    settings: dictionary of experimental settings

    Outputs
    =========================================================================
    settings: dictionary of experimental settings which includes demand model
    '''
    defaults = {
    'ORDER_SIZE': 1,
    'LOAD_LEVEL': 1,
    'MEAN_A': 75,
    'MEAN_B': 25
    }

    for key in defaults.keys():
        if key not in settings.keys():
            settings[key] = defaults[key]
        elif defaults[key] is not None:
            settings[key] = type(defaults[key])(settings[key])
        else:
            pass

    return settings

def generate_test_orders(env):
    mean_a_base = env.settings['MEAN_A']
    n_orders = int(env.n_days * 1.5)
    doc_nums = np.arange(n_orders)
    c = 0
    while c <= 0:
        mean_a = mean_a_base + np.random.normal(loc=0, scale=2)
        mean_b = 100 - mean_a
        if mean_b < 0:
            continue
        c += 1
        prob_array = np.array([mean_a, mean_b])
        probs = prob_array / prob_array.sum()
        gmids = np.random.choice([1, 2], p=probs, size=n_orders)
        doc_create_time = np.random.choice(np.arange(env.n_days), size=n_orders)
        planned_gi_time = np.random.choice(np.arange(1, 6), size=n_orders) + doc_create_time
        marg = np.array([int(env.product_data[env.gmid_index_map[i], 
                          env.prod_data_indices['variable_margin']] + np.random.normal(loc=2)) 
                for i in gmids])
        ob = np.vstack([doc_nums, doc_create_time, planned_gi_time, np.ones(n_orders), gmids,
                        np.zeros(n_orders), np.ones(n_orders), marg, np.zeros(n_orders),
                        np.zeros(n_orders), np.zeros(n_orders), np.zeros(n_orders), 
                        np.ones(n_orders)]).T
    return ob