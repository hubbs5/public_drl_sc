#!/usr/bin/env python3

# This file contains utility functions designed to assist with deploying the 
# RL and MILP models in a production environment.

import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta

# Converts dates and times from integer values to real-world values
def convert_schedule_times(agent):
	start_time = datetime.strptime(agent.env.settings['START_TIME'], '%Y-%m-%d').date()
	base_time_unit = agent.env.settings['BASE_TIME_UNIT']
	schedule = agent.env.get_schedule(agent.schedule)
	if base_time_unit == 'DAY':
		schedule['prod_start_time'] = schedule['prod_start_time'].apply(
			lambda x: start_time + timedelta(days=x))
		schedule['prod_end_time'] = schedule['prod_end_time'].apply(
			lambda x: start_time + timedelta(days=x))
	elif base_time_unit == 'HOUR':
		schedule['prod_start_time'] = schedule['prod_start_time'].apply(
			lambda x: start_time + timedelta(hours=x))
		schedule['prod_end_time'] = schedule['prod_end_time'].apply(
			lambda x: start_time + timedelta(hours=x))

	return schedule

def order_schedule_cols(schedule):
	df = pd.concat([schedule['prod_start_time'], 
		schedule['prod_end_time'],
		schedule['gmid'],
		schedule['prod_qty']], axis=1)
	df['unit'] = 'MT'
	return df

def determine_packaging(env, schedule):
	# TODO: Replace with current ZFIN Inventory
	zfin_inventory = np.zeros(len(env.zfin))
	zfin_inventory_temp = zfin_inventory.copy()
	zfin_data = env.zfin_data
	demand = env.get_orders()
	demand['package'] = demand['zfin'].map(lambda x: env.zfin_data[x][3])
	for n in demand['doc_num']:
	    order_data = demand.loc[demand['doc_num']==n]
	    lot_size = zfin_data[order_data['zfin'].values[0]][4]
	    pack_type = order_data['package'].values[0]
	    lot_type = pack_type + '_qty_remaining'
	    qty = order_data['order_qty'].values[0]
	    # Fill value from inventory first
	    if zfin_inventory_temp[
	        zfin_data[order_data['zfin'].values[0]][-1]] >= order_data['order_qty'].values[0]:
	        zfin_inventory_temp[zfin_data[order_data['zfin']][-1]] -= order_data['order_qty']
	    # Assign capacity
	    else:
	        # Get matching indices for product availability
	        idxs = schedule.loc[
	            (schedule['qty_remaining']>=qty) &
	            (schedule['gmid']==order_data['gmid'].values[0])].index
	        # Check lot sizes first for bags and ss
	        for idx in idxs:
	            if pack_type != 'bulk':
	                current_lot = schedule.loc[idx, lot_type]
	                if current_lot < qty:
	                    if schedule.loc[idx, 'qty_remaining'] < lot_size:
	                        # No material to assign, move to next entry in schedule
	                        continue
	                    elif schedule.loc[idx, 'qty_remaining'] >= lot_size:
	                        # Assign a lot to the packaging type
	                        schedule.loc[idx, lot_type] += lot_size - qty
	                        schedule.loc[idx, pack_type] += lot_size
	                        schedule.loc[idx, 'qty_remaining'] -= lot_size
	                        break
	                elif current_lot >= qty:
	                    schedule.loc[idx, lot_type] -= qty
	                    break

	# Assign unallocated quantities to bulk
	schedule['bulk'] = schedule['bulk'] + schedule['qty_remaining']
	schedule.drop('qty_remaining', inplace=True, axis=1)
	schedule.drop('bag_qty_remaining', inplace=True, axis=1)
	schedule.drop('ss_qty_remaining', inplace=True, axis=1)
	schedule.bulk = schedule.bulk.map(lambda x: np.round(x, 1))
	schedule.bag = schedule.bag.map(lambda x: np.round(x, 1))
	schedule.ss = schedule.ss.map(lambda x: np.round(x, 1))
	return schedule

# def determine_packaging(schedule):
# 	# Warning: Currently random function. Replace by competent algorithm
#     vals = pd.DataFrame(np.vstack(
#     	schedule['prod_qty'].map(
#         lambda x: np.round(x*f())).values), 
#         columns=['Bulk', 'Bag', 'SS'])
#     return pd.concat([schedule, vals], axis=1)


def f():
    f0 = np.random.rand()
    f1 = np.random.rand() 
    if f1 + f0 >= 1:
        f2 = 0
        f1 = 1 - f0
    else:
        f2 = 1 - f1 - f0

    t = np.array([f0, f1, f2])
    return np.random.choice(t, len(t), replace=False)
