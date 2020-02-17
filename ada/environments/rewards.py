# rewards
# Christian Hubbs
# christiandhubbs@gmail.com
# 13.03.2018

# This file implements containers to track the rewards for the Production
# Planning Model (PPM). 
# Rewards to track:
# Late penalties
# Inventory costs
# On-time shipment rewards

import numpy as np
import pandas as pd
from ada.Loggermixin import *

# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger=Loggermixin.get_default_logger()

class containers(object):

  def __init__(self):
    self.late_penalties = []
    self.late_orders_on_books = []
    self.inventory_cost = []
    self.inventory = []
    self.shipment_rewards = []
    self.total_reward = []
    self.actions = []
    self.state = []
    self.planned_production = []
    self.actual_action = []
    self.actual_production = []
    self.predicted_state = []
    self.action_times = []
    self.planning_day = []

  def reset(self):
    self.late_penalties = []
    self.late_orders_on_books = []
    self.inventory_cost = []
    self.inventory = []
    self.shipment_rewards = []
    self.total_reward = []
    self.actions = []
    self.state = []
    self.planned_production = []
    self.actual_action = []
    self.actual_production = []
    self.predicted_state = []
    self.action_times = []
    self.planning_day = []

  def get_names(self):
    # Returns container names
    names = [i for i in self.__dict__.keys()]
    return names

  def stack_values(self, limit=None):
    # Returns stacked numpy array of all container values
    data = None
    names = self.get_names()
    for i in names:
      if len(getattr(self, i)) == 0:
        continue
      _data = np.vstack(getattr(self, i))
      if type(limit) is int:
        _data = _data[:limit]
      
      if data is None:
        data = _data.copy()
      else:
        try:
          data = np.hstack([data, _data])
        except ValueError:
          h_logger.debug('from stack_values')
          h_logger.debug(i, data.shape, _data.shape)
    return data

  # Reward function
  def reward_function(self):
    if 'OTD' in self.reward_function:
      return get_OTD_reward(self)
    elif 'VALUE' in self.reward_function:
      return get_fin_reward(self)
    else:
      # Add other reward functions and their classifications...
      pass

def get_fin_reward(env):
  # Returns summation of product
  if env.reward_function == 'VALUE_ADD':
    late_penalties = np.sum(env.containers.late_penalties[-1])
    inventory_cost = np.sum(env.containers.inventory_cost[-1])
    shipment_rewards = np.sum(env.containers.shipment_rewards[-1])
    reward = late_penalties + inventory_cost + shipment_rewards
  else:
    raise NotImplementedError('{} is not implemented'.format(env.reward_function))

  return reward

def get_OTD_reward(env):
    # Define time range for OTD level
    orders_due = env.order_book[np.where(
            env.order_book[:,
            env.ob_indices['planned_gi_time']]==env.sim_time)]
    # Calculate number of orders that are on-time, late, or haven't shipped
    on_time = orders_due[np.where((orders_due[:,env.ob_indices['shipped']]==1) &
                                  (orders_due[:,env.ob_indices['on_time']]==1)
                                  )].shape[0]
    # OTD1 returns daily OTD
    if env.reward_function == 'OTD1':
      try:
        return on_time / orders_due.shape[0]
      except ZeroDivisionError:
        return 0
    # OTD2 returns OTD at the end of the episode
    elif env.reward_function == 'OTD2':
        if env.sim_time == env.n_days - 1:
            orders_due = env.order_book[np.where(
                    env.order_book[:,
                    env.ob_indices['planned_gi_time']]<=env.sim_time)]
        else:
            return 0
    # OTD3 returns value of 1 if the OTD score is greater than a given threshold
    elif env.reward_function == 'OTD3':
      orders_due = env.order_book[np.where(
          env.order_book[:,
          env.ob_indices['planned_gi_time']]==env.sim_time)]
      if on_time >= 0.85:
        reward = 1
      else:
        reward = -1
    # OTD4 returns OTD for every month
    elif env.reward_function == 'OTD4':
      raise NotImplementedError('{} is not implemented'.format(env.reward_function))
    else:
      raise NotImplementedError('{} is not implemented'.format(env.reward_function))
    
    return reward