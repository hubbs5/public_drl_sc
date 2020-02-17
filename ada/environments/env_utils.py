# utils.py
# Christian Hubbs
# christiandhubbs@gmail.com
# 13.03.2018

# This file contains numerous utilities for implementing the PPM simulation.

import numpy as np
import pandas as pd
import sys
import os
import matplotlib as mpl
if sys.platform == 'linux':
    if os.environ.get('DISPLAY', '') == '':
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import string
from copy import copy
from datetime import datetime, date, timedelta
from dateutil import parser
import calendar
import time
import pickle
import warnings
from str2bool import str2bool
import openpyxl as pyxl
import re
from simpledbf import Dbf5
from .demand_models.generate_orders import check_demand_settings
pd.options.mode.chained_assignment = None

from ada.Loggermixin import *

# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger=Loggermixin.get_default_logger()

# Check environment settings to ensure everything is enumerated and revert
# to defaults where values are missing.
def check_env_settings(settings=None, *args, **kwargs):
    '''
    Input
    settings: dict of values required to parameterize the simulation 
        environment. Missing values or None is permissible as these will
        be populated with defaults.

    Output
    settings: dict of values required to completely specifiy the simulation
        environment.
    '''
    defaults = {
         'SYS_START_TIME': datetime.strftime(
            datetime.fromtimestamp(time.time()), "%Y-%m-%d"),
         'RANDOM_SEED': int(time.time()),
         'N_PRODUCTS': 6,
         'TRANSITION_MATRIX_SETTING': 'RANDOM',
         'BASE_TIME_INTERVAL': 1,
         'BASE_TIME_UNIT': 'DAY',
         'FIXED_PLANNING_HORIZON': 7,
         'LOOKAHEAD_PLANNING_HORIZON': 14,
         'MAINTENANCE_MODEL': 'UNIFORM_STOP',
         'DEMAND_MODEL': 'SEASONAL_DEMAND',
         'PRODUCT_DATA_PATH': None, # Enter file path to import data if available
         'SHUTDOWN_PROB': 0.0,
         'START_TIME': '2018-01-01',
         'END_TIME': '2018-04-01',
         'WEEKEND_SHIPMENTS': True,
         'REWARD_FUNCTION': 'OTD1',
         'STATE_SETTING': 'INV_BALANCE_PRODUCTION',
         'ORDER_BOOK': None,
         'WORKING_CAPITAL_PERCENTAGE': 0.1/365,
         'LATE_PENALTY': 25,
         'TRAIN': True,
         'IMPOSE_MIN_CAMPAIGN': True}
    if settings is None:
        settings = defaults
    else:
        for key in defaults.keys():
            if key not in settings.keys():
                settings[key] = defaults[key]
            elif key == 'START_TIME' or key == 'END_TIME':
                # Ensure standard formatting of YYYY-MM-DD
                settings[key] = str(parser.parse(settings[key]).date())
            elif defaults[key] is not None:
                if type(defaults[key]) == bool:
                    settings[key] = str2bool(str(settings[key]))
                else:
                    settings[key] = type(defaults[key])(settings[key])
            elif 'PATH' in key and settings[key] is not None:
                settings[key] = str(settings[key])

    if settings['ENVIRONMENT'] == 'TARTAN':
        assert settings['BASE_TIME_UNIT'] == 'DAY', 'BASE_TIME_UNIT = {}. \
        Tartan model only supports Days as the base time interval'.format(
            settings['BASE_TIME_UNIT'])
    elif settings['ENVIRONMENT'] == 'GOPHER':
        assert settings['BASE_TIME_UNIT'] == 'HOUR', 'BASE_TIME_UNIT = {}. \
        Gopher model only supports hours as the base time interval'.format(
            settings['BASE_TIME_UNIT'])

    settings = check_demand_settings(settings)

    return settings

def load_scenario_data(path, env=None):
    print(path)
    supported_extensions = ['.pkl', '.xlsx', '.yaml']
    
    # Check for both relative and absolute paths
    if os.path.exists(path):
        pass
    elif os.path.exists(os.path.join(os.getcwd(), path)):
        path = os.path.join(os.getcwd(), path)
    else:
        raise FileNotFoundError('File not found: {}'.format(
            path))
    filename, file_ext = os.path.splitext(path)
    if file_ext.lower() in supported_extensions:
        zfin_to_gmid = None
        zfin_data = None
        # Pickle Files
        if file_ext.lower() == '.pkl':
            print(path)
            prod_data, trans_mat = pickle.load(open(path, 'rb'))
            zfin = np.arange(prod_data.shape[0])
            zfin_to_gmid = {i: i for i in zfin}
            zfin_data = {i for i in zfin}   
        # Excel Files
        elif file_ext.lower() == '.xlsx':
            prod_data, trans_mat, zfin, \
                zfin_to_gmid, zfin_data = load_product_data_from_excel(path)
        # Yaml Files
        elif file_ext.lower() == '.yaml':
            prod_data, trans_mat = load_yaml_data(env)
            zfin = np.arange(prod_data.shape[0])
            zfin_to_gmid = {i: i for i in zfin}
            zfin_to_data = {i for i in zfin}
    else:
        raise ValueError('File extension {} not currently supported'.format(
            file_ext))
    return prod_data, trans_mat, zfin, zfin_to_gmid, zfin_data

def aggregate_orders(env):
    # TODO: There may be situations where low demand is encountered
    # and there are no orders. This will raise an error down the line
    pred_orders = env.order_book[np.where(
        (env.order_book[:, 
            env.ob_indices['doc_create_time']]<=env.sim_time) & 
        (env.order_book[:,
            env.ob_indices['shipped']]==0) &
        (env.order_book[:,
            env.ob_indices['doc_num']]>0))].astype(float)

    unique_order, unique_order_id = np.unique(pred_orders[:, 
        env.ob_indices['gmid']], return_inverse=True)
    order_pred_qty = np.bincount(unique_order_id, 
        pred_orders[:, env.ob_indices['order_qty']])

    return order_pred_qty, unique_order

def get_net_forecast(env):
    # Aggregate orders by month and gmid
    current_month = env.sim_day_to_date[env.sim_time][0]
    net_forecast = np.zeros((12, env.n_products))
    for m, month in enumerate(range(1, 13)):
        if month < current_month:
            continue
        for j, g in enumerate(env.gmids):
            pred_orders = env.order_book[np.where(
                (env.order_book[:, 
                    env.ob_indices['doc_create_time']]<=env.sim_time) & 
                (env.order_book[:,
                    env.ob_indices['shipped']]==0) &
                (env.order_book[:,
                    env.ob_indices['doc_num']]>0) &
                (env.order_book[:, 
                    env.ob_indices['planned_gi_month']]==month) &
                (env.order_book[:, 
                    env.ob_indices['gmid']]==g))]

            unique_order, unique_order_id = np.unique(pred_orders[:, 
                env.ob_indices['gmid']], return_inverse=True)
            order_pred_qty = np.bincount(unique_order_id, 
                pred_orders[:, env.ob_indices['order_qty']])
            try:
                net_forecast[m, j] = max(env.monthly_forecast[m, j] - order_pred_qty.sum(), 0)
            except IndexError:
                # Indicates no forecast for the month, leave at 0
                continue
    return net_forecast

# TODO: evaluate breaking into individual functions
def get_current_state(env, schedule=None, day=None):
    # Get copy of the inventory
    inv = env.inventory.copy()

    if day is None:
        day = env.sim_time

    # If schedule is passed, get the expected inventory total and
    # add it to the current inventory
    if schedule is not None:
        if type(schedule) == tuple:
            h_logger.debug("Schedule is tuple:", day)
        # Get expected, unbooked production from the schedule
        pred_production = schedule[np.where(
            (schedule[:,
                env.sched_indices['cure_end_time']]<=day) &
            (schedule[:,
                env.sched_indices['booked_inventory']]==0))]
        # Sum scheduled, unbooked production
        un_prod, un_prod_id = np.unique(pred_production[:,
            env.sched_indices['gmid']], return_inverse=True)
        pred_prod_qty = np.bincount(un_prod_id, 
            pred_production[:, env.sched_indices['prod_qty']])
        # Add pred_production to inventory
        if len(pred_prod_qty) < 1:
            # No predicted production to add
            pass
        else:
            prod_idx = np.array([env.gmid_index_map[p] 
                for p in un_prod.astype(int)])
            inv[prod_idx] += pred_prod_qty

        # Get current product
        current_prod = schedule[
            schedule[:,
                env.sched_indices['prod_start_time']]==day,
                env.sched_indices['gmid']].astype(int)
        # Check to ensure that a value exists for the scheduled slot, 
        # if not, this indicates shutdown
        if current_prod.size == 0:
            current_prod = 0
    else:
        current_prod = 0

    # Ensure current product is an integer value and not array
    if type(current_prod) is np.ndarray:
        current_prod = current_prod.take(0)

    if day == env.sim_time:
        env.current_prod = copy(current_prod)

    # The state is defined only by the inventory available
    if env.state_setting == 'INVENTORY':
        state = inv
    
    # The current state is defined as the ratio of inventory to open
    # orders.
    elif env.state_setting == 'IO_RATIO':
        # Aggregate open orders on the books
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        # Calculate state prediction
        # Note: this formulation ignores off-grade as the state
        inv_ratios = np.array([inv[env.gmid_index_map[i]] / order_pred_qty[k] for 
                                     k, i in enumerate(gmids_to_update)])
        state_inv = np.zeros(env.n_products)
        indices_to_update = np.array([env.gmid_index_map[int(i)] 
            for i in gmids_to_update])
        state_inv[indices_to_update] += inv_ratios

        state = state_inv
  
    # inv_balance_production setting defines the state as the inventory
    # balance concatenated with the current production state.
    elif env.state_setting == 'INV_BALANCE_PRODUCTION':
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        state_inv = inv.copy()
        indices_to_update = np.array([env.gmid_index_map[int(i)] 
            for i in gmids_to_update])
        # Ensure array is not empty
        if gmids_to_update.size > 0:
            state_inv[indices_to_update] -= order_pred_qty

        one_hot = np.zeros(env.n_products)

        # 0 vector if plant is shut down
        if current_prod != 0:
            one_hot[env.gmid_index_map[current_prod]] = 1
        
        state = np.hstack([one_hot, state_inv])

    # The current state is defined as the ratio of inventory
    # to open orders available for that day and the current
    # production
    elif env.state_setting == 'IO_PRODUCT':            
        # Aggregate open orders on the books
        # Filtering by sim_time ensures all orders are already entered
        # Filtering shipped ensures that only open orders
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        
        # Calculate state prediction
        # Note: this formulation ignores off-grade as the state
        prod_index = np.array([env.gmid_index_map[i] 
            for i in gmids_to_update])
        inv_ratios = np.array([inv[i] / order_pred_qty[k] for 
            k, i in enumerate(prod_index)])
        state_inv = np.zeros(env.n_products)
        # Subtract 1 from the index to ignore off-grade levels
         
        state_inv[prod_index] += inv_ratios

        # Include product to be produced in state prediction as one-hot 
        # vector
        # 0 vector if plant is shut down
        one_hot = np.zeros(env.n_products)     
        if current_prod != 0:
            one_hot[env.gmid_index_map[current_prod]] = 1
        
        state = np.hstack([one_hot, state_inv])

    # Add net forecast to state
    elif env.state_setting == 'CONCAT_FORECAST':
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        state_inv = inv
        indices_to_update = np.array([env.gmid_index_map[int(i)] 
            for i in gmids_to_update])
        # Ensure array is not empty
        if gmids_to_update.size > 0:
            state_inv[indices_to_update] -= order_pred_qty
        
        one_hot = np.zeros(env.n_products)
        if current_prod != 0:
            one_hot[env.gmid_index_map[current_prod]] = 1
        net_forecast = get_net_forecast(env).flatten()
        state = np.hstack([one_hot, state_inv, net_forecast])
    
    return state

# Get state definition
# TODO: Possibly easier to simply get the state and return the dimensions
def observation_space(env):
    observation_space = []
    # Return inventory only for state definition
    if env.state_setting == 'INVENTORY':
        [observation_space.append(int(x)) for x in 
            env.product_data[:,env.prod_data_indices['gmid']]]

    # Get ratio of inventory/open orders
    elif env.state_setting == 'IO_RATIO' or env.state_setting == 'INV_BALANCE':
        [observation_space.append(int(x)) for x in 
            env.product_data[:,env.prod_data_indices['gmid']]]

    # Get inventory/order ratio and current product as one-hot vector
    elif env.state_setting == 'IO_PRODUCT' or env.state_setting == 'INV_BALANCE_PRODUCTION':
        [observation_space.append(int(x)) for x in 
            env.product_data[:,env.prod_data_indices['gmid']]]
        observation_space = 2 * observation_space

    elif env.state_setting == 'CONCAT_FORECAST':
        [observation_space.append(int(x)) for x in 
            env.product_data[:,env.prod_data_indices['gmid']]]
        observation_space = 2 * observation_space
        observation_space = np.hstack([np.array(observation_space),
            np.zeros(12*env.n_products)])

    return np.array(observation_space)

# Calculate the customer service level
def get_cs_level(env):
    # Get orders that are due
    orders_due = env.order_book[np.where(
            env.order_book[:,env.ob_indices['planned_gi_time']]<=env.sim_time)]
    # Calculate number of orders that are on-time, late, or haven't shipped
    on_time = orders_due[np.where((orders_due[:,env.ob_indices['shipped']]==1) &
                                  (orders_due[:,env.ob_indices['on_time']]==1))].shape[0]
    late = orders_due[np.where((orders_due[:,env.ob_indices['shipped']]==1) &
                                  (orders_due[:,env.ob_indices['on_time']]==-1))].shape[0]
    not_shipped = orders_due[np.where((orders_due[:,env.ob_indices['shipped']]==0))].shape[0]
    
    if orders_due.shape[0] == 0:
        cs_level = np.array([0, 0, 0])
    else:
        cs_level = np.array([on_time, late, not_shipped]) / orders_due.shape[0]

    return cs_level

# Calculate cost of holding inventory
def calculate_inventory_cost(env):
    env.order_book = env.order_book.astype(float)
    # Aggregate orders based on material code
    # unique_gmid is an array of the unique materials
    # gmid_locs is an array giving the locations of the unique identifiers
    unique_gmid, gmid_locs, gmid_counts = np.unique(
        env.order_book[:, env.ob_indices['gmid']],
        return_inverse=True,
        return_counts=True)
    beta_i = np.bincount(gmid_locs, env.order_book[:,env.ob_indices['var_std_margin']]) / gmid_counts

    # Add 0 as placeholder for beta_i if lengths are unequal
    if len(beta_i) < len(env.gmid_index_map):
        for i in env.gmids:
            if i not in unique_gmid:
                beta_i = np.insert(beta_i, env.gmid_index_map[i], 0)
    # Check gmid maps to determine if OG is to be included in the calculation.
    # Where len(gmid_map) < len(inventory): OG is to be excluded, if equal: include
    _og_flag = len(env.inventory) - len(env.gmid_index_map)
    assert _og_flag >= 0, "Discrepancy between GMID's and inventory mapping: {}".format(_og_flag)
    assert _og_flag <= 1, "Discrepancy between GMID's and inventory mapping: {}".format(_og_flag)
    return sum([env.inventory[env.gmid_index_map[i] + _og_flag] * beta_i[env.gmid_index_map[i]] 
                for i in unique_gmid]) * env.working_capital_per * -1

def plot_gantt(env, save_location=None):
    # Get available products from the environment
    labels = env.product_data[:,
        env.prod_data_indices['product_name']]
    unique_products = env.product_data[:,
        env.prod_data_indices['gmid']].astype(int)

    # Find products that have not been scheduled, if any to ensure proper
    # labeling.
    unscheduled_products = [p for p in unique_products if 
        p not in env.containers.actions]

    # Combine actual schedule with unscheduled products 
    extended_schedule = np.hstack([env.containers.actions, 
        unscheduled_products])

    # Organize products in a matrix where the row indexes the product and
    # the columns index the day
    gantt_matrix = np.zeros((env.n_products, extended_schedule.shape[0]))

    # Populate matrix with values
    for i, j in enumerate(extended_schedule):
        for k in range(env.n_products):
            if j == k + 1:
                gantt_matrix[k, i] = j

    # Set color scheme
    cmap = mpl.cm.get_cmap('Paired')
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    # Cycle through products and plot accordingly
    for i in range(gantt_matrix.shape[0]):
        # Cycle through time slots for each product
        for j, k in enumerate(gantt_matrix[i]):
            if k != 0:
                start = j
                # Later update to match on GMID rather than simply the index
                # because the GMID is unlikely to have a one-to-one
                # correspondence to the location.
                prod_duration = env.product_data[int(k - 1), 
                    env.prod_data_indices['min_run_time']].astype(int)
                ax.barh(i, prod_duration, left=start, color=c[i])

    # Format plot
    ax.invert_yaxis()
    ax.grid(color='k', linestyle=':')
    pos = np.arange(gantt_matrix.shape[0]) + 0.5
    locsy, labelsy = plt.yticks(pos, unique_products)
    plt.title('Gantt Chart')
    plt.xlabel('Day')
    plt.xlim([0, len(env.containers.actions)])
    plt.ylabel('Product')
    if save_location is not None:
        plt.savefig(save_location)
    plt.show()

def get_state_labels(env, predicted=False):
    # TODO: Update function to make selections based on state setting not length of obs_space
    state_labels = []
    obs_space = env.observation_space.shape[0]
    prod_names = env.product_data[:, env.product_data_cols.index('product_name')].astype(str)
    state_labels = ['state_' + i.lower() for i in prod_names]
    if obs_space == env.n_products:
        pass
    elif obs_space == env.n_products + 1:
        state_labels.insert(0, 'state_og')

    elif obs_space == 2 * env.n_products + 1:
        prod_state_labels = ['state_production_' + i.lower() for i in prod_names]
        prod_state_labels.insert(0, 'state_production_shut_down')
        state_labels = prod_state_labels + state_labels

    elif env.state_setting == 'IO_PRODUCT' or env.state_setting == 'INV_BALANCE_PRODUCTION':
        prod_state_labels = ['state_production_' + i.lower() for i in prod_names]
        state_labels = prod_state_labels + state_labels

    elif env.state_setting == 'CONCAT_FORECAST':
        prod_state_labels = ['state_production_' + i.lower() for i in prod_names]
        forecast_labels = ['net_forecast_' + str(j) + '_' + i 
            for i in calendar.month_abbr[1:] for j in env.gmids]
        state_labels = prod_state_labels + state_labels + forecast_labels
    else:
        raise ValueError("No labeling rule for {} state data of length {} and {} products.".format(
            env.settings['STATE_SETTING'], obs_space, env.n_products))
    # Label predictions to differentiate from actual values
    if predicted:
        state_labels = ['predicted_' + i for i in state_labels]
    return state_labels

# Get planning_data headers
def get_planning_data_headers(env):
    names = env.containers.get_names()
    col_names = ['planning_day', 'heuristic_flag']
    # Append action probabilities and predicted state labels
    for a in env.action_list:
        col_names.append('action_prob_' + str(a))

    for i in names:
        if np.array(getattr(env.containers, i)).size > 0:
            x = np.vstack(getattr(env.containers, i))
            # Generate column names
            dims = x.shape[1]
            if i == 'state':
                state_labels = get_state_labels(env, predicted=False)
                col_names = col_names + state_labels
            elif i == 'predicted_state':
                state_labels = get_state_labels(env, predicted=True)
                col_names = col_names + state_labels
            elif dims > 1 and dims <= env.n_products:
                for j in range(dims):
                    col_names.append(str(i) + '_' + string.ascii_lowercase[j])
            elif dims > env.n_products:
            # Off-grade needs to be added to the first value
            # Add flag to differentiate between state settings
                for j in range(dims):
                    if j == 0:
                        col_names.append(str(i) + '_og')
                    else:
                        col_names.append(str(i) + '_' + string.ascii_lowercase[j - 1])
            else:
                col_names.append(str(i))

    # Also return a dict of names and indices for easy reference
    planning_data_indices = {k: i for i, k in enumerate(col_names)}

    return col_names, planning_data_indices

def get_mpc_data_headers(env):
    names = env.containers.get_names()
    col_names = []
    for i in names:
        if len(getattr(env.containers, i)) > 0:
            x = np.vstack(getattr(env.containers, i))
            dims = x.shape[1]
            if dims > 1 and dims <= env.n_products:
                for j in range(dims):
                    col_names.append(str(i) + '_' + string.ascii_lowercase[j])
            elif dims > env.n_products:
                for j in range(dims):
                    if j == 0:
                        col_names.append(str(i) + '_og')
                    else:
                        col_names.append(str(i) + '_' + string.ascii_lowercase[j - 1])
            else:
                col_names.append(str(i))
    
    planning_data_indices = {k: i for i, k in enumerate(col_names)}
    
    return col_names, planning_data_indices

def load_current_state_data(settings, path=None):
    if path is None:
        path = 'production_models/bahia_blanca/'
    files = os.listdir(path)
    inv_path, prod_path, f_path, order_path = None, None, None, None
    for f in files:
        if 'inventory' in f:
            inv_path = os.path.join(path, f)
        elif 'product' in f:
            prod_path = os.path.join(path, f)
        elif 'forecast' in f:
            f_path = os.path.join(path, f)
        elif 'orders' in f:
            order_path = os.path.join(path, f)
    # Both pd and tm should be the same as what the agent was trained on
    # production_data, transition_matrix = load_current_production_data(
    #     settings, prod_path) 
    order_book = load_current_order_data(order_path)
    forecast = load_current_forecast_data(f_path)
    inventory = load_current_inventory_data(inv_path)
    return inventory, order_book, forecast

def load_current_production_data(settings, path):
    # Ensure testing and training values are identical
    try:
        train_prods, train_tm = load_scenario_data(settings['PRODUCT_DATA_PATH'])
        training_loaded = True
    except KeyError:
        warnings.warn('No training environment found, cannot guarantee environments match.')
        answer = input('Continue? y/n')
        if answer == False:
            sys.exit('Program exited.')
    test_prods, test_tm = load_scenario_data(path)
    if training_loaded:
        assert np.array_equal(train_prods, test_prods), 'Product data for test and train environments do not match.'
        assert np.array_equal(train_tm, test_tm), 'Transition matrices for test and train environments do not match.'

    return test_prods, test_tm

def load_current_schedule(env):
    sched_dbf_path = os.path.split(env.settings['PRODUCT_DATA_PATH'])[0]
    sched_dbf_path = os.path.join(sched_dbf_path, "EXPORT.DBF")
    print("Loading Current Schedule from {0}".format(sched_dbf_path))
    dbf = Dbf5(sched_dbf_path)
    df = dbf.to_dataframe()
    # Build Schedule
    sched = []
    b_id = 0
    booked_inv = 0.0
    off_grade = 0.0
    actual_prod = 0.0
    sched_start_row = df.iloc[0,:]
    start_split = sched_start_row["START_DATE"].split("-")
    if len(sched_start_row["START_TIME"]) > 3:
        start_hour = int(sched_start_row["START_TIME"][:2])
    else:
        start_hour = int(sched_start_row["START_TIME"][0])
    start_min  = int(sched_start_row["START_TIME"][-2:])
    sched_start = datetime(int(start_split[0]),int(start_split[1]),int(start_split[2]),start_hour, start_min)    
    sched_end_dt = sched_start
    idx = 0 
    ## Cut current schedule to only include fixed planning horizon elements
    while sched_end_dt < sched_start + timedelta(hours = 24.0*env.fixed_planning_horizon):
        row = df.iloc[idx,:]
        gmid = int(row["GMID"])
        prod_rate = row["PROD_RATE"]
        prod_qty  = row["QUANTITY"] 
        prod_time = prod_qty / prod_rate
        start_split = row["START_DATE"].split("-")
        if len(row["START_TIME"]) > 3:
            start_hour = int(row["START_TIME"][:2])
        else:
            start_hour = int(row["START_TIME"][0])
        start_min  = int(row["START_TIME"][-2:])
        datetime_start = datetime(int(start_split[0]),int(start_split[1]),int(start_split[2]),start_hour, start_min)
        prod_start = datetime_start - sched_start
        prod_start = prod_start.total_seconds() / (60*60)
        prod_end = int(prod_start + prod_time)
        cure_time = 24
        cure_end = prod_end + cure_time
        inv_index = env.gmids.index(gmid) + 1
        sched_row = [b_id,gmid,prod_rate,prod_qty, prod_time, prod_start, prod_end, 
                    cure_time, cure_end, booked_inv, inv_index, off_grade, actual_prod]
        b_id += 1
        sched.append(sched_row)
        idx += 1
        sched_end_dt = datetime_start 
    schedule = np.stack(sched)
    return schedule

def load_current_order_data(path=None):
    try:
        orders = _load_current_order_data()
    except NotImplementedError:
        # warnings.warn('Current order data connections currently not' + \
        #  'implemented. Loading most recent manual order file.')
        orders = _load_state_data_by_file(path, 'Orders', True)

    return orders

def load_current_forecast_data(path=None):
    try:
        forecast = _load_current_forecast_data()
    except NotImplementedError:
        # warnings.warn('Current order data connections currently not' + \
        #  'implemented. Loading most recent manual order file.')
        forecast = _load_state_data_by_file(path, 'Forecast', True)

    return forecast

def load_current_inventory_data(path=None):
    try:
        # Get data from system
        inventory = _load_current_inventory_data()
    except NotImplementedError:
        # warnings.warn('Current inventory data connections currently not implemented.' + \
        #     ' Loading most recent manual inventory file.')
        inventory = _load_state_data_by_file(path, 'Inventory', True)

    return inventory

# TODO: Complete the following function
def _load_current_inventory_data():
    # Loads current inventory from SAP HANA or relevant internal system
    raise NotImplementedError('Inventory data system not defined.')

# TODO: Complete the following function
def _load_current_forecast_data():
    #Load current inventory from SAP HANA or relevant internal system
    raise NotImplementedError('Inventory data system not defined.')
    # extension = os.path.basename(path).split('.')[-1].lower()
    # if extension == 'xlsx':
    #     pass
    # return forecast

# TODO: Complete the following function
def _load_current_order_data():
    # Loads current inventory from SAP HANA or relevant internal system
    raise NotImplementedError('Order data system not defined.')

def _load_state_data_by_file(path, dtype='', pandas=False):
    # Check to see when file was last modified. Prompt the user to continue
    # if file is old.
    today = datetime.now().date()
    file_last_modified = datetime.utcfromtimestamp(os.path.getmtime(path)).date()
    if today > file_last_modified:
        user_input = input('{} file was last modified on {}'.format(
            dtype, file_last_modified) + ' Do you want to continue working with' + \
            ' this data? (y/n)\n>>>>>>\t')
        if str2bool(user_input) == False:
            sys.exit('Program exited.')
    supported_extensions = ['csv', 'xlsx', 'pkl']
    # Load relevant xlsx workbook or csv file with inventory levels.    
    extension = os.path.basename(path).split('.')[-1]
    if extension.lower() == 'csv':
        data = pd.read_csv(path)
        if data.shape[1] <= 1:
            # Try another separator
            data = pd.read_csv(path, sep=';')
    elif extension.lower() == 'xlsx':
        data = pd.read_excel(path, dtype=str)
    elif extension.lower() == 'pkl':
        data = pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Extension {} not supported.'.format(extension) + \
            ' Ensure file is in one of the following formats: {}'.format(
                supported_extensions))
    if type(data) == pd.core.frame.DataFrame:
        try:
            data = data.drop('Unnamed: 0', axis=1)
        except KeyError:
            pass
        if pandas == False:
            data = data.values
    return data

def XLTableExpandToDataFrame(location, limit=1000, index=1):
    '''
    Inputs
    =========================================================================
    location: openpyxl.cell.cell.cell to give location of the top left
        corner of the relevant table
    limit: int that limits the table size
    index: 0 or 1 where 0 indicates a numeric index the size of the frame
        and 1 indicates the first column of the table is used as the index
    '''
    assert index==0 or index==1, 'Index value must be either 0 or 1.'
    frame = []
    frame_cell = location
    cols_count = 0
    rows_count = 0
    frame_cols =  frame_cell
    frame_rows =  frame_cols
    while not frame_rows.value is None and rows_count < limit:
        train_frame_row = []
        while not frame_cols.value is None and cols_count < limit:
            train_frame_row.append(frame_cols.value)
            cols_count += 1
            frame_cols =  frame_cell.offset(rows_count,cols_count)
        frame.append(train_frame_row)
        cols_count = 0
        rows_count += 1
        frame_rows = frame_cell.offset(rows_count,cols_count)
        frame_cols = frame_rows
    
    frame = np.vstack(frame)
    if index==1:
        frame = pd.DataFrame(data=frame[1:,1:], columns=frame[0,1:], 
                             index=frame[1:,0])
    else:
        frame = pd.DataFrame(data=frame[1:,:], columns=frame[0,:], 
                         index=np.arange(frame.shape[0]-1))
    
    frame = frame.apply(pd.to_numeric, downcast="float", errors="ignore")
    return frame  

def load_product_data_from_excel(product_data_path):
    wb = pyxl.load_workbook(product_data_path, data_only=True)
    # Load train data
    trains_loc = wb['Overview'][
        wb.defined_names['Trains'].value.split("!")[1]].offset(1,0) 
    trains_df = XLTableExpandToDataFrame(trains_loc)
    
    # Load production data
    prod_loc = wb['Overview'][
        wb.defined_names['Products'].value.split('!')[1]].offset(1,0)
    prod_df = XLTableExpandToDataFrame(prod_loc, index=0)
    prod_df.insert(0, 'train', trains_df['train_number'].values[0].astype(int))
    
    # Load transition data
    trans_loc = wb['Overview'][wb.defined_names[
        'ProductsTransition'].value.split('!')[1]].offset(1,0)
    trans_df = XLTableExpandToDataFrame(trans_loc, index=1)
    
    # Transform transition data
    max_losses = prod_df['batch_size'].max().astype(str)
    transition_matrix = replace_chars_vec(max_losses, 
        trans_df.values).astype(float)
    transition_matrix = np.hstack([prod_df['startup'].values.reshape(-1, 1), 
                                   transition_matrix])
    transition_matrix = np.vstack([np.hstack([0, prod_df['startup']]),
                                   transition_matrix])

    # Get final products
    zfin_loc = wb['Overview'][wb.defined_names[
        'ProductsFinished'].value.split('!')[1]].offset(1,0)
    zfin_list = XLTableExpandToDataFrame(zfin_loc)['gmid'].astype(int).values

    # Get ZFIN-ZEMI/GMID mappings
    zfin_loc = wb['Overview'][wb.defined_names['ProductsFinished'].value.split('!')[1]].offset(1,0)
    zfin_df = XLTableExpandToDataFrame(zfin_loc, index=0)
    zemi = prod_df['product_name'].map(lambda x: ' '.join(x.split(' ')[:-2]))
    zfin = zfin_df['product_name'].map(lambda x: ' '.join(x.split(' ')[:-2]) 
                            if x.split(' ')[-1] == 'KG' else
                           ' '.join(x.split(' ')[:-1]))
    prod_df2 = prod_df.copy()
    prod_df2['zemi'] = zemi
    zfin_df['zemi'] = zfin
    # Merge frames
    merged = zfin_df.merge(prod_df2, on='zemi', how='left')
    merged['packaging'] = merged['product_name_x'].map(
        lambda x: parse_packaging(x))
    merged['inventory_index'] = np.arange(len(merged))
    zfin_to_gmid = {i[0]: i[1] 
        for i in merged[['gmid_x', 'gmid_y']].values.astype(int)}
    zfin_data = {int(i[0]): [int(i[1]), i[2], i[3], i[4], i[5], i[6]]
        for i in merged[['gmid_x', 'gmid_y', 
            'product_name_x', 'product_name_y', 
            'packaging', 'batch_size_x',
            'inventory_index']].values}

    return prod_df.values, transition_matrix, zfin_list, zfin_to_gmid, zfin_data
    
def replace_chars(replacement_value, val):
    if type(val) != str:
        val = str(val)
    return re.sub("[a-zA-Z]+", replacement_value, val)

# Vectorize replace_chars function
replace_chars_vec = np.vectorize(replace_chars)

def parse_packaging(desc):
    if 'BG6025' in desc:
        return 'bag'
    elif 'BB1200' in desc:
        return 'ss'
    elif 'BLK' in desc:
        return 'bulk'
    else:
        return ''
    
def process_forecast_data(forecast_data, env):
    # Check if forecast has already been processed 
    if forecast_data.shape[1] == len(env.gmids):
        if type(forecast_data) == pd.core.frame.DataFrame:
            forecast_data = forecast_data.values
    else:
        df = forecast_data.loc[forecast_data['Field-03'].isin(env.zfin.astype(str))]
        assert len(df) > 0, "No matching ZFIN GMID's found in forecast."
        # Melt frames separately and recombine on the country
        melt_cats = ['ACTD', 'RSLF', 'HFSF', 'UAH7']
        id_vars = ['Field-03'] #, 'Field-04']
        join_cols = ['Field-03', 'Month', 'Year']
        df_reshape = None
        for cat in melt_cats:
            melt_cols = [col for col in df.columns if cat in col]
            [melt_cols.append(i) for i in id_vars]
            
            _df_sub = df.loc[:, melt_cols]
            df_sub = pd.DataFrame()
            # Ensure numerical columns are formatted as such
            for col in _df_sub.columns:
                if col in id_vars:
                    df_sub = pd.concat([df_sub, _df_sub[col]], axis=1)
                else:
                    df_sub = pd.concat([df_sub, _df_sub[col].astype(float)], axis=1)
            df_agg = df_sub.groupby(id_vars).sum()
            df_agg.reset_index(inplace=True)
            df_melt = pd.melt(df_agg, id_vars=id_vars)
            df_melt['Month'] = df_melt['variable'].map(
                lambda x: x.split(' ')[-1])
            df_melt['Year'] = df_melt['Month'].map(
                lambda x: x.split('/')[-1])
            df_melt['Month'] = df_melt['Month'].map(
                lambda x: x.split('/')[0])
            df_melt.drop('variable', axis=1, inplace=True)
            col_list = df_melt.columns.tolist()
            col_list[col_list.index('value')] = cat
            df_melt.columns = col_list
            if df_reshape is None:
                df_reshape = df_melt.copy()
            else:
                df_reshape = df_reshape.merge(df_melt, on=join_cols, 
                    how='outer')
                
        df_reshape.fillna(0, inplace=True)
        col_list = df_reshape.columns.tolist()
        col_list[col_list.index('Field-03')] = 'ZFIN'
        # col_list[col_list.index('Field-04')] = 'ZFIN Name'
        df_reshape.columns = col_list
        new_order = ['ZFIN', 'Year', 'Month', 'ACTD',
            'RSLF', 'HFSF', 'UAH7']
        df_reshape = df_reshape.loc[:, new_order].copy()
        # Aggregate values by ZFIN and date
        agg = df_reshape.groupby(['ZFIN', 'Year', 'Month'])[
            ['ACTD', 'RSLF', 'HFSF', 'UAH7']].sum()
        agg = agg.loc[agg.sum(axis=1).values!=0].reset_index()
        agg['GMID'] = agg['ZFIN'].map(lambda x: env.zfin_to_gmid_map[int(x)])

        fcast_agg = agg.groupby(['GMID', 'Year', 'Month'])[
            'RSLF'].sum().reset_index()
        fcast_agg['year_mon'] = fcast_agg.apply(
                     lambda x: datetime.strptime(
                         str(x.Year) + '-' + str(x.Month),'%y-%m'), axis=1)

        # Get first day of current month
        # Convert to pd.Timestamp to avoid error
        now = pd.Timestamp(date.today().replace(day=1))
        next_year = pd.Timestamp(now.replace(year=now.year+1, 
            month=now.month - 1))
        fcast = fcast_agg.loc[(fcast_agg['year_mon']>=now) &
                      (fcast_agg['year_mon']<=next_year)]
        # Convert data types
        fcast['GMID'] = fcast['GMID'].astype(str)
        forecast_data = np.zeros((12, env.n_products))
        for g in env.gmid_index_map.keys():
            for i, m in enumerate(range(1, 13)):
                forecast_data[i, env.gmid_index_map[g]] = fcast.loc[
                    (fcast['Month']==str(m)) &
                    (fcast['GMID']==str(int(g)))]['RSLF']

    return forecast_data

def keep_base_name(s):
    split = s.split(' ')[:4]
    if split[-1] == 'HF':
        return ' '.join(split)
    else:
        return ' '.join(split[:3])

def process_order_data(order_data, env):
    # Check to see if order_book is already in the proper format
    if order_data.shape[1] == len(env.order_cols):
        return order_data
    if type(order_data) != pd.core.frame.DataFrame:
        raise ValueError("order_data loaded as {}; type not supported".format(
            type(order_data)))
    # Rename columns
    order_data.columns = [j if 'Unnamed:' not in j 
            else order_data.columns[i-1] + ' Desc' 
            for i, j in enumerate(order_data.columns)]
    
    # Filter orders by ZFIN
    orders_sub = order_data.loc[order_data['Material'].isin(env.zfin.astype(str))]
    # Filter orders by doc type
    doc_types = ['ZOR', 'ZSO', 'ZBRJ', 'ZFD', 'ZRI', 'ZBRI', 'ZVER', 'ZLOR']
    orders_sub = orders_sub.loc[orders_sub['Sales Doc. Type'].isin(doc_types)]
    # Convert volumes from KG to MT
    orders_sub['order_qty'] = np.round(
        orders_sub['Weight - Net (w/o UoM)'].astype(float) / 1000, 3)
    
    # Key corresponds to data column, value corresponds to new order book column
    time_cols = {'Dt - (OI) Customer Requested Del (Confirmed)': 
                     'cust_req_date', # See note above 
                 'Dt - (DH) Goods Issue Actual': 'actl_gi_time',
                 'Dt - (DH) Goods Issue Plan': 'planned_gi_time',
                 'Dt - (OH) Created On': 'doc_create_time'}
    # Convert time strings to datetime object
    # Note # is used for missing values, we set to some large, future value for now
    for key in time_cols.keys():
        orders_sub[time_cols[key]] = orders_sub[key].map(
            lambda x: datetime.strptime(str(x), '%m/%d/%Y') if x != '#' 
                else datetime.strptime('01/01/2100', '%m/%d/%Y'))
        if key == 'Dt - (DH) Goods Issue Plan':
            orders_sub['planned_gi_month'] = orders_sub[key].map(
                lambda x: datetime.strptime(str(x), '%m/%d/%Y').month)
        times = (orders_sub[time_cols[key]] - env.start_time).map(
            lambda x: x.days)
        if env.settings['BASE_TIME_UNIT'] == 'HOUR':
            times *= 24
        orders_sub[time_cols[key]] = times
        
    col_name_map = {'Sales Document': 'doc_num', 
                'Material': 'gmid'}

    for k in col_name_map.keys():
        orders_sub[col_name_map[k]] = orders_sub[k].copy()
        
    orders_sub['shipped'] = 1
    orders_sub.loc[orders_sub['actl_gi_time']>365*2]['shipped'] = 0
    
    orders_sub['on_time'] = 0
    orders_sub['late_time'] = 0
    orders_sub['cust_segment'] = 1
    orders_sub['var_std_margin'] = 0
    orders_sub['doc_num'] = orders_sub['doc_num'].astype(int)
    orders_sub['late_time'] = orders_sub['actl_gi_time'] - orders_sub['cust_req_date']
    
    # Add ZEMI and ZFIN
    orders_sub['gmid'] = orders_sub['gmid'].map(lambda x: 
        env.zfin_to_gmid_map[int(x)])
    orders_sub['zfin'] = orders_sub['Material'].copy()

    orders_sub['shipped'] = 1
    orders = orders_sub[env.order_cols].values
    # Set shipped for future orders to 0
    orders[np.where(orders[:, env.ob_indices['actl_gi_time']]>=365*2)[0],
           env.ob_indices['shipped']] = 0
    orders[np.where(orders[:, env.ob_indices['shipped']]==0)[0],
          env.ob_indices['late_time']] = 0
    
    return orders

def determine_date_format(date_series):
    labels = ['%m', '%d', '%Y'] # strptime labels
    dates = np.vstack(date_series.map(lambda x: re.split('\W+', x))).astype(int)
    d = {j: labels[i] for i, j in enumerate(np.argsort(np.max(dates, axis=0)))}
    date_format_string = '-'.join([d[k] for k in range(3)])
    return date_format_string

def convert_date_series(date, date_format_string):
    date_re = re.sub('\W+', '-', date)
    return datetime.strptime(date_re, date_format_string)

def process_inventory_data(inventory_data, env):
    if inventory_data.shape[0] == 1:
        # Already in format, simply return inventory data
        return inventory_data
    else:
        date_format_string = determine_date_format(inventory_data['Calendar Day'])
        inventory_data['datetime'] = inventory_data['Calendar Day'].map(
            lambda x: convert_date_series(x, date_format_string))
        data_recent = inventory_data.loc[
            inventory_data['datetime']==inventory_data['datetime'].max()]
        data_recent = data_recent.loc[data_recent['Material'].isin(env.zfin)]
        remove = string.punctuation
        remove = remove.replace("-", "") # don't remove hyphens
        remove = remove.replace(".", "") # don't remove periods
        pattern = r"[{}]".format(remove) # create the regex pattern
        data_recent['quantity'] = data_recent['Inventory Balance (KG)'].map(
            lambda x: float(re.sub(pattern, "", x)))
        data_recent['gmid'] = data_recent['Material'].map(
            lambda x: env.zfin_to_gmid_map[int(x)])
        inventory = data_recent.groupby(['gmid'])['quantity'].sum() / 1000
        inventory_sorted = np.zeros(len(inventory))
        for i in env.gmid_index_map.keys():
            inventory_sorted[env.gmid_index_map[i]] += inventory[i]
            
        return inventory_sorted

def load_yaml_data(env):
    if float(env.settings["OFFGRADE_GMID"]) == type(1.0):
        env.OffGradeGMID = int(env.settings["OFFGRADE_GMID"])
    else:
        env.OffGradeGMID = 9999

    # Parameter Yaml File   
    file_name = "{0}.yaml".format(env.settings["PARAMETER_FILE"])
    par_file  = os.path.join(os.path.dirname(os.path.abspath(__file__)),"parameter_files",file_name)
    stream    = open(par_file, 'r')
    env.product_data_yaml = yaml.load(stream,Loader=yaml.FullLoader)
    stream.close()       
    env.products = env.product_data_yaml['products']
    # env.products.insert(0, "offgrade")
    env.n_products = len(env.products)

    StandardBatchTimes = dict()
    StandardBatchSizes = dict()
    StandardCureTimes  = dict()

    StandardBatchSizes[env.OffGradeGMID] = 0 ## ADD GMID FOR OFFGRADE
    StandardCureTimes[env.OffGradeGMID]  = 0 ## ADD GMID FOR OFFGRADE        
    StandardBatchTimes[env.OffGradeGMID] = 1 ## 1 Hours of OffGrade to Schedule Delay ADD GMID FOR OFFGRADE      
    env.gmids = {}
    env.gmidProdIndexDict = dict()
    env.gmids["offgrade"] = env.OffGradeGMID

    env.gmidProdIndexDict[env.OffGradeGMID ] = 0 ## ADD GMID FOR OFFGRADE
    p_ind = 1
    for p in env.products:
        env.gmids[p] = int(env.product_data_yaml[str(p) + "_GMID" ])
        env.gmidProdIndexDict[int(env.product_data_yaml[str(p) + "_GMID" ])] = p_ind
        StandardBatchSizes[int(env.product_data_yaml[str(p) + "_GMID" ])] = env.product_data_yaml[str(p) + "_Batch Size"]
        StandardCureTimes[int(env.product_data_yaml[str(p) + "_GMID" ])]  = env.product_data_yaml[str(p) + "_CureTime"]            
        StandardBatchTimes[int(env.product_data_yaml[str(p) + "_GMID" ])]  = env.product_data_yaml[str(p) + "_Batch Size"] / env.product_data_yaml[str(p) + "_Rate"] 
        p_ind += 1
    # Standard Batch Time // Standard Cure Times --> Arrays 
    startupMat = np.zeros([len(env.products)])
    ind_p = 0 
    for p in env.products:
        val = env.product_data_yaml[str(p) + "_Startup" ]
        startupMat[ind_p] = val
        ind_p += 1

    env.transition_matrix = np.empty([len(env.products) + 1,len(env.products) + 1])
    env.transition_matrix[0,0] = 0
    env.transition_matrix[1:,0] =  startupMat
    env.transition_matrix[0,1:] =  startupMat
    ind_p = 1 
    for p in env.products:
        ind_pp = 1
        for pp in env.products:
            val = env.product_data_yaml[str(p) + "_" + str(pp) ]
            if not val == 'x' and not val == 'X':
                env.transition_matrix[ind_p, ind_pp] = val
            else:
                env.transition_matrix[ind_p, ind_pp] = 100000#np.inf
            ind_pp += 1
        ind_p += 1

    RatesMat = {} #np.empty([len(env.products)])
    RatesMat[env.OffGradeGMID] = 1 # No Rate for off grade product
    for p in env.products:
        val = env.product_data_yaml[str(p) + "_Rate" ]
        g = env.gmids[p]
        if not val == None:
            RatesMat[g] = val
        else:
            RatesMat[g] = np.inf

    data = [[asset_number, "{0}".format(p).lower(), int(env.gmids[p]),
            StandardBatchTimes[int(env.gmids[p])], prod_time_uom, RatesMat[int(env.gmids[p])], run_rate_uom,
            StandardBatchSizes[int(env.gmids[p])], size_uom] 
            for p in env.products]

    return np.array(data), env.transition_matrix.copy()