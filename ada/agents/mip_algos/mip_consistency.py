#!/usr/bin/env python

# This file contains functions to ensure the proper behavior of the MIP models.
# Most functions compare binar

import numpy as np
from warnings import warn
from ada.Loggermixin import *

# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger=Loggermixin.get_default_logger()

# Run all consistency tests
def check_mip_consistency(mpc):
    check_og_production(mpc)
    check_mass_balance(mpc)
    check_production_planning(mpc)
    check_early_shipments(mpc)
    check_transition_count(mpc)

# Get time index from an array
def get_t_index(x, t):
    return np.where(x==t)[0][0]

#############################################################################
# Single period tests
#############################################################################
def sp_check_early_shipments(model):
    shipment_count = 0
    d = {}
    for n in model.n:
        due_date = model.order_dict[n][model.order_cols.index('planned_gi_time')]
        model_ship_time = [t for i in model.i 
            for t in model.t if model.sales[i, t, n].value >= 1]
        if len(model_ship_time) == 1:
            if due_date > model_ship_time[0]:
                print('Order {} shipped by model {} day(s) early.'.format(
                    n, model_ship_time[0]))
                shipment_count += 1
        elif len(model_ship_time) > 1:
            print('Multiple shipments for order {}'.format(n))
            shipment_count += len(model_ship_time)
    d[(model.sim_time, 'Early Shipment Count')] = shipment_count
    return d

# Ensure that binary variables align with sales variables
def sp_check_sales_binary_agreement(model):
    # Update to handle a series of models
    sales, x = [], []
    idx_list = []
    for idx in model.x:
        idx_list.append(idx)
        sales.append(1 if model.sales[idx].value > 1e-6 else 0)
        # x should only be none for values where t < env.sim_time
        x.append(model.x[idx].value if model.x[idx].value is not None else 0)
    passed = np.allclose(sales, x)
    d = {'Sales-Binary Agreement': [passed, sum(sales) - sum(x)]}
    return d
    # if passed:
    #     # print('Sales variables agree with binary variables')
    #     # print('Test passed.')
    #     return passed, d
    # else:
    #     sales_mask = np.where(np.array(sales) > 0, 1, 0)
    #     failed = np.where(sales_mask != x)[0]
    #     f_index = [idx_list[i] for i in failed]
    #     _ = [print(f, model.sales[f].value, model.x[f].value)
    #          for f in f_index]
    #     print('Discrepancies between sales and binary variables')
    #     print('Test failed.')
    #     return passed

# Get OG values from z variable
def sp_check_z_og(model):
    z_og_list = []
    z_og_t = []
    z_transitions = []
    for idx in model.z:
        if model.z[idx].value == 1:
            og = model.transition_losses[idx[0], idx[1]]
            z_og_list.append(og)
            z_transitions.append([idx, og])
        else:
            continue
        z_og_t.append(idx[2])
    
    # Return ordered OG vector and ordered time period vector
    return np.array(z_og_list)[np.argsort(z_og_t)], np.array(z_og_t)[np.argsort(z_og_t)]

# Get OG values from y variable
def sp_check_y_og(model):
    _y_list = []
    y_t = []
    y_transitions = []
    for idx in model.y:
        if model.y[idx].value == 1 and idx[-1] >= model.sim_time - model.delay:
            _y_list.append(idx)
        else:
            continue
        y_t.append(idx[-1])
    y_list = np.array(_y_list)[np.argsort(y_t)]
    
    y_og = []
    y_og_t = []
    for i, t in enumerate(y_list[:,1]):
        if i > 0 and t >= model.sim_time - model.delay:
            y_og.append(model.transition_losses[y_list[i-1,0], y_list[i,0]])
            y_og_t.append(t)
    return np.array(y_og), np.array(y_og_t)

# Check OG production directly
def sp_check_og_(model):
    og_array = np.zeros(max(model.t)+1)
    og_min = min(model.t)
    og_max = max(model.t)
    og_array = np.zeros(og_max+1-min(og_min, 0))
    for idx in model.og:
        og_array[idx[-1]] += model.og[idx].value
        
    return og_array[max(og_min,0):og_max+1-min(og_min,0)], np.arange(og_min, og_max+1)

# Single period OG check returns dict of results
def sp_check_og(model):
    og, og_t = sp_check_og_(model)
    y_og, y_og_t = sp_check_y_og(model)
    z_og, z_og_t = sp_check_z_og(model)
    # Slice arrays over same time horizon
    t_i = max(og_t.min(), y_og_t.min(), z_og_t.min())
    t_f = min(og_t.max(), y_og_t.max(), z_og_t.max())
    _og = np.round(og[get_t_index(og_t, t_i):get_t_index(og_t, t_f)], 2)
    _y_og = np.round(y_og[get_t_index(y_og_t, t_i):get_t_index(y_og_t, t_f)], 2)
    _z_og = np.round(z_og[get_t_index(z_og_t, t_i):get_t_index(z_og_t, t_f)], 2)
    # All three off-grade measurements should match to pass the test
    d = {}
    d['OG-Y Match'] = [np.allclose(_og, _y_og), (_og - _y_og).sum()]
    d['OG-Z Match'] = [np.allclose(_og, _z_og), (_og - _z_og).sum()]
    d['Y-Z Match'] = [np.allclose(_y_og, _z_og), (_y_og - _z_og).sum()]
    return d

def get_schedule_from_mip(model):
    sched_object = getattr(model, 'y')
    sched_array = np.zeros((model.I, max(model.t) + 1))
    for idx in sched_object:
        row = np.where(model.gmids==int(idx[0]))[0]
        sched_array[row, idx[1]] = sched_object[idx].value
        
    return sched_array

#############################################################################
# Multi-period tests compare model results with simulation data
#############################################################################

# Ensure shipment constraints hold
def check_early_shipments(mpc):
    results = {}
    for m in mpc.solved_models:
        results.update(sp_check_early_shipments(m))

    failures = np.array([(key, results[key]) 
        for key in results.keys() if results[key]>0])
    if len(failures) > 0:
        print("Early Shipment Constraint Violation Encountered for days:")
        [print("Solution Day: {} \tNumber of early shipments: {}".format(
            f[0][0], f[1])) for f in failures]
    else:
        print("No Shipment Constraint Violations Encountered")

# Ensure that the sales variables and the binary variables x match
def check_sales_binary_agreement(mpc):
    results = {}
    for m in mpc.solved_models:
        results_day = sp_check_sales_binary_agreement(m)
        results.update(results_day)

    failures = np.array([(results[key], key[0]) 
        for key in results.keys() if results[key][0]==False])
    if len(failures) > 0:
        print("Sales-Binary Agreement Violation Encountered for days:")
        [print("Solution Day: {} Number of discrepancies: {}".format(
            f[1], f[0][1])) for f in failures]
    else:
        print("No Sales-Binary Violations Encountered")

def check_og_production(mpc):
    results = {}
    for m in mpc.solved_models:
        results_day = sp_check_og(m)
        results.update(results_day)
    failures = np.array([(results[key], key[0], key[1]) 
        for key in results.keys() if results[key][0]==False])
    if len(failures) > 0:
        if max(failures[:,1].astype(float)) > m.delay:
            print('Off-grade test violations found for the following tests:')
            [print('Solution Day: {} Test {} OG Difference: {} MT'.format(
                f[1], f[2], f[0][1]))
             for f in failures]
        else:
            warn('Off-grade test violations found only for initial periods: {}'.format(
                np.unique(failures[:,1])))
#             [print('Solution Day: {} Test {} OG Difference: {} MT'.format(
#                 f[1], f[2], f[0][1]))
#              for f in failures]
    else:
        print('No Off-Grade Violations Encountered.')

def check_mass_balance(mpc):
    max_production = max(mpc.env.product_data[:, 
                    mpc.env.product_data_cols.index('run_rate')]).astype(int)
    violations_prod_qty = 0
    violation_qty = 0

    for iteration, model in enumerate(mpc.solved_models):
        # Check planned production volume vs actual
        prod_object = getattr(model, 'p')
        og_object = getattr(model, 'og')
        prod_qty = np.zeros((model.I, max(model.t) + 1))
        og_qty = []

        for idx in prod_object:
            row = np.where(model.gmids==int(idx[0]))[0]
            prod_qty[row, idx[1]] = prod_object[idx].value

        for idx in og_object:
            og_qty.append(og_object[idx].value)
            
        #print("Total Production Value")
        daily_prod = prod_qty.sum(axis=0)
        x = np.where(daily_prod > max_production)[0]
        if x.size > 0:
            violations_prod_qty += x.size
            violation_qty += daily_prod[x] - max_production
            print("Solution Day {:d} Production Limit Exceeded".format(iteration))
            print("Violated Days: {}".format(x))
            
    if violations_prod_qty > 0:
        print("{:d} Production Quantity Violations Encountered".format(violations_prod_qty))
        print("{:d} MT Total Production Volume Difference".format(int(violation_qty)))
    else:
        print("No Production Violations Encountered")

def check_production_planning(mpc):
    violations_planning = 0

    for iteration, model in enumerate(mpc.solved_models):
        # Check planned production volume vs actual
        sequence_object = getattr(model, 'y')
        sequence = np.zeros((model.I, max(model.t) + 1))
        for idx in sequence_object:
            row = np.where(model.gmids==int(idx[0]))[0]
            sequence[row, idx[1]] = sequence_object[idx].value
            
        num_planned_products = sequence.sum(axis=0)
        x = np.where(num_planned_products > 1)[0]
        if x.size > 0:
            violations_planning += x.size
            print("Solution Day {:d} Limit on Number of Planned Products Exceeded".format(iteration))
            print("Violated Days {}".format(x))
    if violations_planning > 1:
        print("{:d} Product Planning Violation Encountered".format(violations_planning))
    else:
        print("No Production Planning Violations Encountered")

# def check_og_production(mpc):
#     violations_og = 0
#     violation_qty = 0

#     for iteration, model in enumerate(mpc.solved_models):
#         violations_og, violation_qty, og_sim, og_mip = sp_check_og_production(
#             model, violations_og, violation_qty, iteration)
#     if violations_og > 0:
#         print("{:d} Off-Grade Errors Encountered".format(violations_og))
#         if violation_qty > 0:
#             print("MIP overestimates OG vs sim by {:.1f} MT".format(violation_qty))
#         else:
#             print("MIP underestimates OG vs sim by {:.1f} MT".format(-violation_qty))
#     else:
#         print("No Off-Grade Errors Encountered")

def check_transition_count(mpc):
    violations_transitions = 0

    for iteration, model in enumerate(mpc.solved_models):
        trans_obj = getattr(model, 'z')
        if min(model.t) < 0:
            trans_array = np.zeros(max(model.t) - min(model.t) + 1)
        else:
            trans_array = np.zeros(max(model.t) + 1)
        for idx in trans_obj:
            if trans_obj[idx].value is not None:
                if trans_obj[idx].value > 0:
                    trans_array[idx[2]] += 1
        
        x = np.where(trans_array > 1)[0]
        if x.size > 0:
            violations_transitions += x.size
            print("Solution Day {:d} Limit on Number of Product Transitions Exceeded".format(iteration))
            print("Violated Days {}".format(x))

    if violations_transitions > 1:
        print("{:d} Product Transition Violations Encountered".format(violations_transitions))
    else:
        print("No Product Transition Violations Encountered")