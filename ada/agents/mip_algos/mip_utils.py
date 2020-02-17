# opt_utils: Contains various helper functions to extract data from 
# optimized Pyomo models.
# Author: Christian Hubbs
# Contact: christiandhubbs@gmail.com
# Date: 02.08.2018

import numpy as np 
import sys
import os
import matplotlib as mpl
if sys.platform == 'linux':
    if os.environ.get('DISPLAY', '') == '':
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle
from os.path import exists
from pathlib import Path
from datetime import datetime
from ada.Loggermixin import *

# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger=Loggermixin.get_default_logger()

# Discount orders based on their lateness
def discount_order_dict(model):
    horizon_limit = 1 + model.K + model.sim_time
    order_dict_disc = {}
    for n in model.order_dict.keys():
        planned_gi_time = model.order_dict[n][
            model.order_cols.index('planned_gi_time')]
        vsm = model.order_dict[n][
            model.order_cols.index('var_std_margin')]

        order_dict_disc[n] = {t: 
            [model.order_dict[n][model.order_cols.index(idx)]
            if idx != 'var_std_margin'
            else vsm * (1 - model.alpha/100*(t-planned_gi_time))
                if t >= planned_gi_time else 0
            for idx in model.order_cols]
            for t in range(model.sim_time, horizon_limit)}
            
    return order_dict_disc


# Convert schedule to dict of binary values
def schedule_to_binary(env, model, schedule=None):
    if schedule is not None:
        current_schedule_vector = [int(i)
            for i in schedule[:, env.sched_indices['gmid']]]
        current_schedule_dict = {(i, t): 1 if i==current_schedule_vector[t]
                                 else 0
                                 for i in model.i
                                 for t in range(len(current_schedule_vector))}
    else:
        current_schedule_dict = None
    return current_schedule_dict

# Assign binaries to production slots according to schedule
def convert_schedule_to_vars(env, model, schedule=None):
    current_schedule = schedule_to_binary(env, model, schedule=schedule)
    # Fix future y[0, t>env.sim_time] == 0 to prevent the model from choosing
    # to shutdown
    for t in model.t:
        if t >= env.sim_time:
            model.y[0, t].fix(0)
    # Scheduling constraint forces the schedule to be maintained for a 
    # given slot
    if schedule is not None:
        for t in model.t:
            for i in model.i:
                if t < 0:
                    # Fix values before simulation to 0
                    model.y[i, t].fix(0)
                else:
                    try:
                        model.y[i, t].fix(current_schedule[i, t])
                    except KeyError:
                        pass
    # If schedule is None, first two time intervals of t == 0
    else:
        current_schedule = {}
        for t in model.t:
            for i in model.i:
                if t < 0:
                    model.y[i, t].fix(0)
                    current_schedule[i, t] = 0

    model.current_schedule = current_schedule
                    
    return model

# Assign binaries to production slots for each scenario in the stoch prog
def convert_schedule_to_stochastic_vars(env, model, schedule=None):
    current_schedule = schedule_to_binary(env, model, schedule=schedule)
    # Fix future y[0, t>env.sim_time] == 0 to prevent the model from choosing
    # to shutdown
    for s in model.s:
        for t in model.t:
            if t >= env.sim_time:
                model.y[0, t, s].fix(0)
    # Scheduling constraint forces the schedule to be maintained for a 
    # given slot
    if schedule is not None:
        for s in model.s:
            for t in model.t:
                for i in model.i:
                    if t < 0:
                        # Fix values before simulation to 0
                        model.y[i, t, s].fix(0)
                    else:
                        try:
                            model.y[i, t, s].fix(current_schedule[i, t])
                        except KeyError:
                            pass
    # If schedule is None, first two time intervals of t == 0
    else:
        current_schedule = {}
        for s in model.s:
            for t in model.t:
                for i in model.i:
                    if t < 0:
                        model.y[i, t, s].fix(0)
                        current_schedule[i, t] = 0              

    model.current_schedule = current_schedule

    return model

# Builds dictionary of orders
def build_order_dict(env, order_book):
    order_dict = {}
    order_cols = ['gmid', 'order_qty', 'var_std_margin', 
                'planned_gi_time', 'shipped']
    for col in order_cols:
        col_idx = env.ob_indices[col]
        for n in order_book[:, env.ob_indices['doc_num']]:
            order_index = np.where(order_book[:, env.ob_indices['doc_num']]==n)[0]
            if n not in order_dict:
                order_dict[n] = [order_book[order_index, col_idx][0]]
            else:
                order_dict[n].append(order_book[order_index, col_idx][0])
    # Append empty column to track lateness value
    for n in order_book[:, env.ob_indices['doc_num']]:
        order_dict[n].append(0)
    order_cols.append('late')
        
    return order_dict, order_cols

def build_stochastic_order_dict(env, n_scenarios):
    order_dict = {}
    order_cols = ['gmid', 'order_qty', 'var_std_margin', 
        'planned_gi_time', 'shipped']
    for s in range(n_scenarios):
        env.get_forecast()
        order_book = subset_orderbook(env).copy()
        for col in order_cols:
            col_idx = env.ob_indices[col]
            for n in order_book[:, env.ob_indices['doc_num']]:
                order_index = np.where(order_book[:, env.ob_indices['doc_num']]==n)[0]
                if (s, n) not in order_dict.keys():
                    order_dict[(s, n)] = [order_book[order_index, col_idx][0]]
                else:
                    order_dict[(s, n)].append(order_book[order_index, col_idx][0])
        # Append empty column to track lateness value
        for n in order_book[:, env.ob_indices['doc_num']]:
            order_dict[(s, n)].append(0)
    order_cols.append('late')
        
    return order_dict, order_cols
    
# Reduces orderbook to orders that are due within the planning horizon
def subset_orderbook(env, K):
    extended_horizon = max(K, 7) + env.sim_time
    # extended_horizon = max(K, 7) + env.sim_time
    order_book = env.order_book[np.where(
        (env.order_book[:, env.ob_indices['doc_create_time']]<=env.sim_time) &
        (env.order_book[:, env.ob_indices['shipped']]==0) &
        (env.order_book[:, env.ob_indices['planned_gi_time']]<=extended_horizon))[0]]
    
    return order_book

# Converts inventory to dictionary for MILP
def get_initial_inventory_dict(env):
    inventory_init = env.inventory.copy()
    inv_init_dict = {}
    for idx, gmid in enumerate(
        env.product_data[:, env.prod_data_indices['gmid']]):
        inv_init_dict[int(gmid)] = inventory_init[idx]
        
    return inv_init_dict

# Gets dict of run rates for MILP model
def get_run_rate_dict(env):
    run_rate_dict = {}
    for idx, gmid in enumerate(
        env.product_data[:, env.prod_data_indices['gmid']]):
        run_rate_dict[int(gmid)] = float(env.product_data[idx, 
            env.prod_data_indices['run_rate']]) 
    
    return run_rate_dict

# Plot Gantt Chart from Pyomo model
def mip_gantt_plot(env, model, time_step=None, 
    color_scheme=None, save=False, path=None):
    time_stamp = env.sim_time if time_step is None else time_step
    gantt_matrix = np.zeros((max(model.j), model.K + env.sim_time + 1))

    product_names = env.product_data[:, env.prod_data_indices['product_name']]

    production_object = getattr(model, 'y')
    for idx in production_object:
        row = np.where(model.gmids==int(idx[0]))[0]
        gantt_matrix[row, idx[1]] = production_object[idx].value

    if color_scheme is None:
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        cmap = mpl.cm.get_cmap(color_scheme)
        c = [cmap(i) for i in range(len(product_names))]

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    # Cycle through products (i)
    for i in range(gantt_matrix.shape[0]):
        # Plot production times (k)
        for j, k in enumerate(gantt_matrix[i]):
            start = (j + 1) * k 
            if start != 0:
                ax.barh(i, 1, left=start - 1, color=c[i])

    ax.invert_yaxis()
    ax.grid(color='k', linestyle=':')
    pos = np.arange(gantt_matrix.shape[0])
    locsy, product_namesy = plt.yticks(pos, product_names)
    plt.setp(product_namesy, fontsize=14)
    plt.title('Production Gantt Chart Day {:d}'.format(time_stamp))
    plt.xlabel('Day')
    plt.ylabel('Product')
    plt.axvline(x=max(model.h), c='k')
    plt.annotate('Planning Horizon',
        xy=(max(model.h) - 0.5, gantt_matrix.shape[0] / 2 - 1),
        rotation=90, 
        size=20, color='k')
    if save:
        fig.savefig(path + '/MIP_gantt_day_' + str(time_stamp) + '.png', 
            bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def mip_inventory_plot(env, model, time_step=None,
    color_scheme=None, save=False, path=None):
    time_stamp = env.sim_time if time_step is None else time_step
    inventory_matrix = np.zeros((max(model.j), model.K + 1))

    product_names = env.product_data[:, env.prod_data_indices['product_name']]

    inventory_object = getattr(model, 'inventory')
    for idx in inventory_object:
        row = np.where(model.gmids==int(idx[0]))[0]
        inventory_matrix[row, idx[1]] = inventory_object[idx].value

    if color_scheme is None:
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        cmap = mpl.cm.get_cmap(color_scheme)
        c = [cmap(i) for i in range(len(product_names))]

    # Plot inventory
    fig = plt.figure(figsize=(12,8))
    x_axis = np.arange(min(model.k), model.K + 1)
    for p in range(inventory_matrix.shape[0]):
        plt.plot(x_axis, inventory_matrix[p, min(model.k):], label=product_names[p])

    plt.legend(loc='best')
    plt.title('Inventory from Day {:d}'.format(time_stamp))
    plt.xlabel('Time Interval (days)')
    plt.ylabel('Inventory Quantity (MT)')
    plt.axvline(x=max(model.h), color='k')
    plt.annotate('Planning Horizon',
        xy=(max(model.h) - 0.5, inventory_matrix.max() / 2 + 0.15 * \
            inventory_matrix.max()),
        rotation=90, 
        size=20,
        color='k')
    if save:
        fig.savefig(path + '/MIP_inventory_day_' + str(time_stamp) + '.png', 
            bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def mip_sales_plot(env, model, time_step=None,
    color_scheme=None, save=False, path=None):
    time_stamp = env.sim_time if time_step is None else time_step
    sales_matrix = np.zeros((max(model.j), model.K + 1))

    product_names = env.product_data[:, env.prod_data_indices['product_name']]

    sales_object = getattr(model, 'sales')
    
    for idx in sales_object:
        row = np.where(model.gmids==int(idx[0]))[0]
        sales_matrix[row, idx[1]] += sales_object[idx].value

    if color_scheme is None:
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        cmap = mpl.cm.get_cmap(color_scheme)
        c = [cmap(i) for i in range(len(product_names))]

    fig, ax = plt.subplots(env.n_products, sharex=True, figsize=(12,8))

    for s in range(env.n_products):
        ax[s].plot(sales_matrix[s], label=product_names[s],
            color=c[s])
        ax[s].legend(loc='best')
        if s == 0:
            ax[s].set_title('Sales from Day {:d}'.format(time_stamp))
        if s == env.n_products / 2:
            ax[s].set_ylabel('Sales Quantity (MT)')

    plt.xlabel('Time Interval (days)')
    if save:
        fig.savefig(path + '/MIP_sales_day_' + str(time_stamp) + '.png', 
            bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def mip_shipment_totals_plot(env, model, time_step=None,
    color_scheme=None, save=False, path=None):
    time_stamp = env.sim_time if time_step is None else time_step
    # Sum shipments to see what is late vs what was shipped on time
    opt_shipment_matrix = np.zeros((max(model.j), model.K + 1))
    planned_shipment_matrix = opt_shipment_matrix.copy()

    product_names = env.product_data[:, env.prod_data_indices['product_name']]

    shipment_object = getattr(model, 'x')
    for idx in shipment_object:
        row = np.where(model.gmids==int(idx[0]))[0]
        opt_shipment_matrix[row, idx[1]] += shipment_object[idx].value if \
            shipment_object[idx].value is not None else 0

    if color_scheme is None:
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        cmap = mpl.cm.get_cmap(color_scheme)
        c = [cmap(i) for i in range(len(product_names))]

    for n in model.n:
        day = model.order_dict[n][model.order_cols.index('planned_gi_time')]
        product = model.order_dict[n][model.order_cols.index('gmid')]
        product_index = np.where(model.gmids==product)[0][0]
        planned_shipment_matrix[product_index, day-1] += 1
        
    # Shipment bar plots
    # Show actual goods issue date vs. requested gi dates by totaling shipments
    # and products for those days
    c_r = [i for i in reversed(c)]
    product_names_p = ['Planned ' + i for i in product_names]
    product_names_a = ['Actual ' + i for i in product_names] 
    fig, ax = plt.subplots(env.n_products, sharex=True, figsize=(12,8))

    # Cycle through products
    for i in range(planned_shipment_matrix.shape[0]):
        # Get planned shipment totals
        planned_day = np.arange(time_stamp, 
            time_stamp + planned_shipment_matrix.shape[1]) + 0.75
        planned_num = planned_shipment_matrix[i]

        # Get actual shipment totals
        actual_day = np.arange(time_stamp, 
            time_stamp + planned_shipment_matrix.shape[1]) + 0.25
        actual_num = opt_shipment_matrix[i]

        # Plot planned shipments
        ax[i].bar(planned_day, planned_num, label=product_names_p[i], color=c_r[i], width=0.5)

        # Plot actual shipments
        ax[i].bar(actual_day, actual_num, label=product_names_a[i], color=c[i], width=0.5)

        ax[i].legend(loc='best')
        if i == 0:
            ax[i].set_title('Planned GI Dates and Actual GI Dates from Day {:d}'.format(
                time_stamp))
        if i == round(env.n_products / 2):
            ax[i].set_ylabel('Total Product Shipments')

    if save:
        fig.savefig(path + '/MIP_shipment_day_' + str(time_stamp) + '.png', 
            bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def save_data(model_list, settings):
    assert type(model_list) is list
    attrs = getattr(model_list[0], '_decl')
    names = [i for i in attrs.keys()]
    for name in names:
        data = None
        data_dims = [len(model_list)]
        # These values grow in two dimensions as orders are added in time
        if name == 'x' or name == 'sales':
            max_orders = max(model_list[-1].n.value) + 1
            
        for i, m in enumerate(model_list):
            # Save inventory results
            pyomo_object = getattr(m, name)
            if pyomo_object.type() is pyomo.core.base.objective.Objective:
                if data is None:
                    data = [pyomo_object.expr()]
                else: 
                    data.append(pyomo_object.expr())
                continue
                
            if pyomo_object.type() is not pyomo.core.base.var.Var:
                break
            
            try:
                dims = [max(pyomo_object)[j] + 1 for j in range(pyomo_object.dim())]
            except TypeError:
                # Catch error associated with 1-D variables, e.g. off-grade
                dims = [max(pyomo_object) + 1]
                
            if name == 'x' or name == 'sales':
                dims[-1] = max_orders
                
            _data_matrix = np.zeros(tuple(dims))
            if data is None:
                # Set up matrix to hold data
                [data_dims.append(j) for j in dims]
                data_dims = tuple(data_dims)
                data = np.zeros(data_dims)
            
            try:
                for idx in pyomo_object:
                    _data_matrix[idx] = pyomo_object[idx].value
                if name == 'z':
                    data_matrix = _data_matrix[:,:, -(m.K + 1):]
                else:
                    data_matrix = _data_matrix[:,-(m.K + 1):]
                
                data[i] = data_matrix.copy()
            except IndexError:
                h_logger.debug(name)
            
        # Save results
        if not exists(settings['DATA_PATH']):
            data_path = Path(settings['DATA_PATH'])
            data_path.mkdir(parents=True, exist_ok=True)
        
        if data is not None:
            file = open(
                os.path.join(settings['DATA_PATH'] + '/' + name + '.pkl'), 'wb')
            pickle.dump(data, file)

def check_settings(settings=None):
    '''
    Input
    settings: dict of values required to parameterize the mip. 
        Missing values or None is permissible as these will
        be populated with defaults.

    Output
    settings: dict of values required to completely specifiy the mip.
    '''
    defaults = {
         'TIME_LIMIT': None,
         'GAP': None,
         'SOLVER': 'GLPK',
         'MIP_ALGO': 'MPC',
         'REWARD_FUNCTION': 'FULL'}
    if settings is None:
        settings = defaults
    else:
        for key in defaults.keys():
            if key not in settings.keys():
                settings[key] = defaults[key]
            # None is a valid argument for GAP
            elif key == 'GAP' and (settings[key] =='None' or settings[key] == 'NONE'):
                    settings[key] = None
            elif defaults[key] is not None:
                settings[key] = type(defaults[key])(settings[key])
    # Set forecast to 0 for GOD MIP
    if 'GOD' in settings['MIP_ALGO']:
        settings['FORECAST'] = False
                
    if 'DATA_PATH' not in settings.keys():
        default_data_path = os.getcwd() + "/RESULTS/" + settings['MIP_ALGO'].upper() + '/'\
            + datetime.now().strftime("%Y_%m_%d_%H_%M")
        settings['DATA_PATH'] = default_data_path

    os.makedirs(settings['DATA_PATH'], exist_ok=True)

    return settings

