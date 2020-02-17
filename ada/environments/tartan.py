# Tartan V0.4: Production Planning Model
# Author: Christian Hubbs
# Email: christiandhubbs@gmail.com
# Date: 15.05.2018

# V04 Updates: 
#   Time basis of model is set by production run rates. 
#   Refactor breaks out maintenance model for more modularity.
# V03 Updates: 
#   Adjust cost function to account for sales of late products and
#   to maximize profit as (sales - cost). 
#   Added flag to disable off-grade and another to remove restrictions
#   on weekend shipments.
#   Added containers to track late penalties, state, actions, etc.
#   This version introduced the Null action of 0 into the action 
#   space.
#   Adjusted return from the step function to only return the
#   schedule because rewards are logged in containers.
#   Updated the order generation function to properly account for the
#   shipment calendar and moved the function to a sub-module. 

#############################################################################
# This model is designed to simulate the basic dynamics and processes of an
# LDPE production system. There have been a number of simplifying assumptions
# to keep it tractable for current research goals. 
# The system runs with 24 hour time intervals.
# All batches take 24 hours to produce and 24 hours for curing time.
# The number of products is adjustable from 2 to n products.
# A scaling factor is used to determine the approximate volume of demand the
# system will experience in a simulated year with 1 indicating that supply is
# equal to demand. Values greater than 1 indicate more demand than supply and
# values less than one indicate excess capacity. This is a parameter to be 
# adjusted when generating orders.

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from numpy import inf
from copy import copy
import string
import datetime
from .env_utils import *
from .calendars.calendar_functions import create_shipping_calendar
# from .demand_models.generate_orders import generate_orders
from .maintenance_models import *
from .ppm import core

class productionFacility(core):
    
    def __init__(self, settings=None):
        if settings is None:
            print("No settings provided. Loading defaults.")
            settings = {'ENVIRONMENT': 'TARTAN'}
        try:
            if settings['ENVIRONMENT'] != 'TARTAN':
                raise ValueError('Environment settings and facility mismatch. {} defined and Tartan called.'.format(
                    settings['ENVIRONMENT']))
        except KeyError:
            settings['ENVIRONMENT'] = 'TARTAN'

        settings = check_env_settings(settings)
        core.__init__(self, settings)

        # Generalize to n_steps for either daily or hourly model
        self.n_steps = self.n_days
        # _, self.cs_labels = self.get_cs_level()

    # Get calendar day at anytime during the simulation
    def get_current_date(self):
        date_string = str(self.year) + "+" + str(self.day) + "+" \
         + str(self.hour)
        d = datetime.strptime(date_string, "%Y+%j+%H")
        return pd.to_datetime(d)

    # Run the simulation
    # Each time step recieves the schedule from the agent in
    # order to produce the 
    def step(self, schedule):
        # Add to action times container
        self.containers.action_times.append(self.sim_time)
        # Get copy of inventory levels
        prev_inv = self.inventory.copy()

        # Get current state and log it relevant container
        state = get_current_state(self, schedule)
        self.containers.state.append(state)
        
        # Update inventory based on yesterday's production
        # Calculate inventory cost and return reward
        schedule = self.update_inventory(schedule)
        if len(self.containers.inventory) < self.n_days:
            self.containers.inventory.append(self.inventory.copy())

        # TODO: If shut down occurs on 1st day of the simulation, the 
        # schedule becomes an empty array. Current product ought to equal 0, 
        # catch error here and set self.current_prod = 0
        # Log scheduled product
        try:
            self.containers.actual_action.append(int(self.current_prod))
        except TypeError:
            self.current_prod = 0
            self.containers.actual_action.append(int(self.current_prod))

        # Ship orders each day
        if self.sim_time in self.shipping_calendar:
            # Ship orders and update order book
            # return reward
            self.ship_orders()
        else:
            self.containers.shipment_rewards.append(0)
            self.containers.late_penalties.append(0)
            try:
                # Late orders shouldn't change due to shipping holiday,
                # take value from previous day
                prev_late_order_count = self.containers.late_orders_on_books[-1]
            except IndexError:
                prev_late_order_count = 0
            self.containers.late_orders_on_books.append(prev_late_order_count)
        
        reduced_schedule = schedule[np.where(schedule[:,self.sched_indices["prod_start_time"]] >= self.sim_time)]
        reduced_schedule = reduced_schedule[0,:]

        self.containers.inventory_cost.append(self.get_inventory_cost())
        
        self.containers.total_reward.append(self.get_rewards())
        self.day += 1
        self.sim_time += 1
        
        # self.sim_time = reduced_schedule[self.sched_indices["prod_end_time"]].astype(int)
        
        # Update forecast
        if self.settings['FORECAST']:
            self.get_forecast()
            
        return schedule
    