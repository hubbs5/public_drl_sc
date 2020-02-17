# PPM V0.1
# Author: Christian Hubbs
# Email: christiandhubbs@gmail.com
# Date: 02.02.2019

'''
PPM contains the core class to be inherited by different production
environments. This is designed to contain a base of methods that are
consistent across environments to allow development of flexible models
to represent our various production constraints and requirements.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import inf
from copy import copy
import string
import datetime
import math
from .env_utils import *
from .calendars.calendar_functions import *
from .demand_models.generate_orders import generate_orders
from .maintenance_models import *
from .rewards import *
from .demand_models.forecast import *

class core():
    
  def __init__(self, settings):
      
    self.settings = settings
    if self.settings['N_PRODUCTS'] < 2:
        raise ValueError("Enter more than one product.")
    else:
        self.n_products = self.settings['N_PRODUCTS']

    self.product_data_cols = ['train',
      'product_name', 
      'gmid',
      'run_rate', 
      'run_rate_unit',
      'min_run_time',  
      'startup',
      'cure_time', 
      'variable_margin', 
      'batch_size', 
      'margin_velocity', 
      'min_campaign_length', 
      'min_campaign_length_unit']

    # Get column indices for product data
    cols = self.product_data_cols
    # Store column indices in ob_indices dict for easy access
    self.prod_data_indices = {}
    for col in cols:
        self.prod_data_indices[col] = cols.index(col)
    
    self.get_product_data()
        
    self.fixed_planning_horizon = self.settings['FIXED_PLANNING_HORIZON']
    self.lookahead_planning_horizon = self.settings['LOOKAHEAD_PLANNING_HORIZON']
    self.shutdown_prob = settings['SHUTDOWN_PROB']
    self.state_settings = ['INVENTORY', # State is just the inventory level
        'IO_RATIO', # Ratio of inventory to orders
        'IO_PRODUCT', # Ratio of inventory to orders + current product
        'INV_BALANCE_PRODUCTION', # Inventory - open orders and current product
        'CONCAT_FORECAST' # INV_BALANCE_PRODUCTION and flattened vector of monthly net forecast
        ] 
    self.state_setting = self.settings['STATE_SETTING']
    self.order_size = self.settings['ORDER_SIZE']
    self.forecast_model = self.settings['FORECAST']
    self.forecast_accuracy = self.settings['FORECAST_ACCURACY']

    # List column names for the production schedule
    self.sched_cols = ['batch_num', 'gmid', 'production_rate',
                      'prod_qty', 'prod_time', 'prod_start_time',
                      'prod_end_time', 'cure_time', 'cure_end_time',
                      'booked_inventory', 'action', #'inventory_index',
                      'off_grade_production', 'actual_production']
    
    # List column names for the order book
    self.order_cols = ['doc_num', 'doc_create_time', 'planned_gi_time',
      'planned_gi_month', 'gmid', 'zfin', 'order_qty', 'var_std_margin',
      'actl_gi_time', 'shipped', 'on_time', 'late_time', 'cust_segment']
    self.cs_labels = ['on_time', 'late', 'not_shipped']

    # Initialize calendar and time settings
    self.start_time = datetime.strptime(
        self.settings['START_TIME'],'%Y-%m-%d')
    self.end_time = datetime.strptime(
      self.settings['END_TIME'], '%Y-%m-%d')
    self.n_days = int((self.end_time - self.start_time).days)
    self.n_hours = int((self.end_time - self.start_time).total_seconds()/3600)
    self.stop_hours = 0 # Length of a production outtage
    self.stop_count = 0 # Counts the number of production outtages
    self.total_hours_down = 0 # Total hours of downtime
    self.sim_hour = 0
    self.sim_time = 0
    self.call_scheduler = True # Flag to call network scheduler for a given time step

    # reward_function changes the reward function that is used
    # in the get_rewards() function.
    # otd1 -> reward is on-time delivery percentage and returned
    # every shipping day.
    # otd2 -> reward is on-time delivery percentage and returned
    # at the end of every episode.
    self.reward_function = self.settings['REWARD_FUNCTION']
    self.maintenance_model = self.settings['MAINTENANCE_MODEL']
    self.working_capital_per = self.settings['WORKING_CAPITAL_PERCENTAGE']

    # Set up time
    self.year = self.start_time.year
    self.shipping_calendar, self.shipping_dict, self.sim_day_to_date = \
      create_shipping_calendar(self.start_time, self.n_days, 
        settings['WEEKEND_SHIPMENTS'])
    self.day = min(self.shipping_calendar)
    self.date_to_day = {tuple(i[1]): i[0] for i in self.sim_day_to_date.items()}
    self.month_to_sim_day = get_month_to_sim_day_dict(self.sim_day_to_date)
    self.n_months = max(self.shipping_dict.keys())

    # Set order_book to none
    self.order_book = None
    self.ob_indices = None
    self.schedule = None
    self.sched_indices = None
    # Set custom order generation stats to replace this
    self.order_statistics = None

    # Initialize without a product
    self.current_prod = 0
    self.batch_num = 0
    self.curing_time = 1 # Day
    self.containers = containers()

    self.avg_og_val = 500
    self.inv_unit_cost = 10

    # Get action information
    # TODO: Update to generalize to multiple trains
    self.action_dict_cols = ["min_run_time", 'run_rate', "batch_size", "inv_index","prod_name","gmid"]
    try:
      self.action_dict = self.get_action_info()['1']
    except KeyError:
     self.action_dict = self.get_action_info()[1]

    # action_list contains the actions that are available to the agent.
    # Action 0 is reserved for outtages and cannot be chosen in this model
    # This list is only useful for a single train model
    self.action_list = []
    [self.action_list.append(act) if act != 0 else None 
        for act in self.action_dict]
    self.gmid_index_map = self.get_gmid_index_map()  
    self.gmids = [int(k) for k in self.gmid_index_map.keys() if k != 0]
    self.gmid_action_map = {self.action_dict[i][self.action_dict_cols.index("gmid")]: i 
      for i in self.action_dict.keys() 
      if self.action_dict[i][self.action_dict_cols.index("gmid")] in self.gmids}
    self.index_gmid_map = {self.gmid_index_map[k]: k 
    	for k in self.gmid_index_map.keys()}
    self.action_gmid_map = {i:j for i, j in zip(self.action_list, self.gmids)}
    self.inventory = np.zeros(self.n_products)
    self.off_grade = 0
    # Add additional transition costs when relevant
    self.transition_costs = np.zeros(self.transition_matrix.shape)
    
    # Get column indices for order book
    if self.ob_indices == None:
        cols = self.order_cols
        # Store column indices in ob_indices dict for easy access
        self.ob_indices = {}
        for col in cols:
            self.ob_indices[col] = cols.index(col)
            
    # Get column indices for schedule
    if self.sched_indices == None:
        cols = self.sched_cols
        # Store column indices in ob_indices dict for easy access
        self.sched_indices = {}
        for col in cols:
            self.sched_indices[col] = cols.index(col)

    # Get observation_space size
    self.observation_space = observation_space(self)
    
    # Impose Min Campaign Length 
    self.impose_min_campaign_len = self.settings["IMPOSE_MIN_CAMPAIGN"]
    self.min_campaign_len = [1] ## Action 0 campaign length
    for i in range(self.n_products):
        min_camp = float(self.product_data[i,
          self.prod_data_indices["min_campaign_length"]]) / \
          float(self.product_data[i,
            self.prod_data_indices["batch_size"]])
        self.min_campaign_len.append(int(math.ceil(min_camp)))      
    self.min_campaign_len = np.array(self.min_campaign_len)
    self.reset()

  def generate_transition_matrix(self):
    # TODO: Preserve Nonetype for transition_matrix_setting
    # Choices provide the off-grade ranges for the model in MT
    # with forbidden transitions creating a full silo of off-grade
    if self.settings['TRANSITION_MATRIX_SETTING'] == 'RANDOM':
        choices = [0, 5, 10, 15, 20, self.product_data[
            :,self.product_data_cols.index(
                'min_campaign_length')].astype(float).max().round(0)]
        # Create transition matrix for all product changes including 
        # from start-up (i.e. product = 0)
        transition_matrix = np.random.choice(choices, 
            (self.n_products + 1) ** 2).reshape(
            self.n_products + 1, self.n_products + 1)
        # Fill diaganal with 0's so there is no cost for remaining 
        # with the same product
        np.fill_diagonal(transition_matrix, 0)
    elif self.settings['TRANSITION_MATRIX_SETTING'] == "PARAMETER_FILE":
      transition_matrix = np.zeros((self.n_products + 1, 
            self.n_products + 1))     
    # Set transition costs to 0
    elif self.settings['TRANSITION_MATRIX_SETTING'] is None or \
      self.settings['TRANSITION_MATRIX_SETTING'] == 'None' or \
      self.settings['TRANSITION_MATRIX_SETTING'] == 'NONE':
        transition_matrix = np.zeros((self.n_products + 1, 
            self.n_products + 1))
    
    return transition_matrix
  
  def generate_product_data(self):

    if self.settings['BASE_TIME_UNIT'] == 'HOUR':
      prod_time_uom = 'Hr'
      run_rate = 10
      run_rate_uom = 'MT/Hr'
      size_uom = 'MT'
      batch_size = 240
      cure_time = 24
    
    elif self.settings['BASE_TIME_UNIT'] == 'DAY':
      prod_time_uom = 'Days'
      run_rate = 240
      run_rate_uom = 'MT/Day'
      size_uom = 'MT'
      cure_time = 1

    train = 1
    batch_size = 240
    startup = 2
    # TODO: synthesize VSM here vs in order book
    variable_standard_margin = 1000 # ($/MT)
    min_campaign_length = 240
    min_campaign_length_uom = 'MT'
    # Programmatically generate production data
    
    data = [[train, # Train number
      list(string.ascii_uppercase)[n], # Product Name
      n + 1, # GMID
      run_rate, # Run rate
      run_rate_uom, # MT/Time
      np.round(batch_size / run_rate, 2), # Min Run Time
      # prod_time_uom, # 
      startup, # Losses due to startup and shutdown (MT)
      cure_time,
      variable_standard_margin,
      batch_size, 
      np.round(variable_standard_margin / run_rate, 2), # Margin velocity
      min_campaign_length,
      min_campaign_length_uom] 
        for n in range(self.n_products)]

    return np.array(data)

  def get_product_data(self):
    try:
      if self.settings['PRODUCT_DATA_PATH'] is not None:
        self.product_data, self.transition_matrix, self.zfin, \
          self.zfin_to_gmid_map, \
          self.zfin_data = load_scenario_data(
            self.settings['PRODUCT_DATA_PATH'], self)
        self.n_products = len(self.product_data)
        self.settings['N_PRODUCTS'] = self.n_products
      else:
        self.product_data = self.generate_product_data()
        self.transition_matrix = self.generate_transition_matrix()
        self.zfin = self.product_data[:, self.prod_data_indices['gmid']]
    except KeyError:
      self.product_data = self.generate_product_data()
      self.transition_matrix = self.generate_transition_matrix()
      self.zfin = self.product_data[:, self.prod_data_indices['gmid']]

  # The action_dict is a dictionary object that contains the available
  # actions for each train which are accessed by entering the key value
  # associated with the name of the train. For example, Train 1 is data
  # can be accessed by entering `action_dict['1']` which will return
  # another dictionary object with the available actions. action=0
  # returns None for all trains and corresponds to turnarounds or shutdowns.
  # Each action contains a list of numbers which correspond to:
  # [min_run_time, run_rate, inventory_index, product_name].
  def get_action_info(self, train=None):
      product_data = self.product_data
      product_data_cols = self.product_data_cols
      if train is not None:
          product_data = product_data[np.where(
                  product_data[:,0]==str(train))[0]]
          prod_list = list(np.unique(product_data[:,1]))
      else:
          # If multiple trains are used, there is the potential
          # for product overlap, so the inventory index may not
          # match the action value.
          prod_list = list(np.unique(product_data[:,
            self.prod_data_indices['product_name']]))
          # Additional inventory slot to account for off-grade
          inv_index_ = np.zeros(product_data.shape[0], dtype='int')
          for i in range(product_data.shape[0]):
              inv_index_[i] = prod_list.index(product_data[i,1])
          inv_count = 0

      action_dict = {}
      min_run_time_index = product_data_cols.index('min_run_time')
      run_rate_index     = product_data_cols.index('run_rate')
      batch_size_index   = product_data_cols.index('batch_size')
      data_col = [min_run_time_index, run_rate_index, batch_size_index]

      # Loop through product data by train
      for i in np.unique(product_data[:,0]):
          # Index products for each train
          train_data_index = np.where(product_data[:,
            self.prod_data_indices['train']]==i)[0]
          train_prod = product_data[train_data_index,
            self.prod_data_indices['product_name']]
          train_gmid = product_data[train_data_index,
            self.prod_data_indices['gmid']]

          train_dict = {}
          # Action = 0 -> None, reserved for outtages and turnarounds
          train_dict[0] = [None for _ in self.action_dict_cols]
          for j, k in enumerate(train_data_index):
              train_data = np.round(
                  product_data[k,data_col].astype('float'), 1)
              train_data = list(train_data)
              # Append inventory index
              train_data.append(j)
              if train is None:
                inv_count += 1

              train_data.append(train_prod[j]) # Append product name
              train_data.append(int(train_gmid[j])) # Append GMID
              train_dict[j + 1] = train_data
              action_dict[i] = train_dict

      if train is not None:
          action_dict = action_dict[train]
      return action_dict
  
  # Aggregate the inventory to be released in case multiple silos are released during
  # the same time step such as following a production stop and restart
  def aggregate_inventory(self, schedule, to_inv_index):
      unique_products, unique_prod_id = np.unique(
          schedule[to_inv_index, self.sched_indices['gmid']], 
          return_inverse=True)
      
      # Sum actual production
      act_prod = np.zeros(self.n_products)
      planned_prod = np.zeros(self.n_products)
      production_sum = np.bincount(
          unique_prod_id, schedule[to_inv_index, 
          self.sched_indices["actual_production"]])
      planned_prod_sum = np.bincount(
          unique_prod_id, schedule[to_inv_index, 
          self.sched_indices["prod_qty"]])
      off_grade_production = np.bincount(
          unique_prod_id, schedule[to_inv_index,
           self.sched_indices["off_grade_production"]])
      # act_prod[0] += off_grade_production.sum()
      for i in self.gmids:
        if i in unique_products:
            self.inventory[self.gmid_index_map[i]] += production_sum[
                np.where(unique_products==i)[0]]
            # Store actual production value
            act_prod[self.gmid_index_map[i]] += production_sum[
                np.where(unique_products==i)[0]]
            planned_prod[self.gmid_index_map[i]] += planned_prod_sum[
                np.where(unique_products==i)[0]]
      self.containers.actual_production.append(act_prod)
      self.containers.planned_production.append(planned_prod)
      # self.containers.actual_og_production.append(off_grade_production.sum())

  def update_inventory(self, schedule):
      # Get indices for cured products that are ready to move to inventory
      to_inv_index = np.where(
        (schedule[:,self.sched_indices["cure_end_time"]]<=self.sim_time) & 
        (schedule[:,self.sched_indices["booked_inventory"]]==0))[0]
      # Update inventory values if there are finished products
      if len(to_inv_index) > 0:
        # Add to inventory
        self.aggregate_inventory(schedule, to_inv_index)
          
        # Change inventory release flag
        schedule[to_inv_index, self.sched_indices["booked_inventory"]] = 1
        # Update current batch number
        self.batch_num = schedule[to_inv_index[-1],
          self.sched_indices["batch_num"]]
        # Book off-grade production
        self.off_grade += np.sum(schedule[to_inv_index,
         self.sched_indices["off_grade_production"]])
      else:
        # Log zeros if array is empty
        self.containers.actual_production.append(
          np.zeros(self.n_products))
        self.containers.planned_production.append(
          np.zeros(self.n_products))

      return schedule

  def ship_orders(self):
    if self.order_book is None:
      raise ValueError("No orders in system. Call generate_orders method.")

    # Count the number of late orders currently on the books and the penalties/rewards
    num_of_late_orders = 0
    ship_reward = 0
    late_penalty = 0
    # Function called at end of day to ship open orders
    # Check order book for open orders that are due to ship
    # ots = orders-to-ship
    self.order_book = self.drop_dummy_orders()
    ots = self.order_book[np.where(
      (self.order_book[:,self.ob_indices["shipped"]]==0) & # Select orders that haven't been shipped/GI
      (self.order_book[:,self.ob_indices["planned_gi_time"]]<=self.sim_time))]

    # Order by due date
    ots = ots[ots[:,self.ob_indices["planned_gi_time"]].argsort()]
    
    # Update each order in turn
    for i, gmid in enumerate(ots[:,self.ob_indices["gmid"]]):
      ord_num = ots[i,self.ob_indices["doc_num"]]
      # Calculate number of days late
      #hours_late = self.hour - ots[i,self.ob_indices["planned_gi_time"]]
      days_late = self.sim_time - ots[i, self.ob_indices["planned_gi_time"]]
      ots[i,self.ob_indices["late_time"]] = days_late
      ots[i,self.ob_indices["actl_gi_time"]] = self.sim_time
      prod = self.gmid_index_map[gmid]
      # Ensure enough inventory to fulfill order
      if self.inventory[int(prod)] >= ots[i,self.ob_indices["order_qty"]]:
          # Update inventory
          self.inventory[int(prod)] -= ots[i,self.ob_indices["order_qty"]]
          # Update order info
          ots[i,self.ob_indices["shipped"]] = 1

          # OTD reward
          # On time deliveries are rewarded each day based on the 
          # product of their value and customer segment
          ship_reward += (ots[i,self.ob_indices["var_std_margin"]] * 
            ots[i,self.ob_indices["order_qty"]] *
            ots[i,self.ob_indices["cust_segment"]])
          # Mark order as late or on-time
          if days_late <= 0:
              ots[i,self.ob_indices["on_time"]] = 1

      # Deal with orders that cannot be shipped
      else:
          # Impose costs on late shipments
          # Costs are imposed on late shipments each day they are late
          # resulting in a penalty equal to 10% of the total value
          # times the number of days late
          late_penalty -= (ots[i,self.ob_indices["var_std_margin"]] *
            ots[i,self.ob_indices["order_qty"]] * 
            ots[i,self.ob_indices["cust_segment"]] * self.settings['LATE_PENALTY']/100)
          num_of_late_orders += 1

          # Mark late orders
          if ots[i,self.ob_indices["on_time"]] == 0:
              ots[i,self.ob_indices["on_time"]] = -1

      # Update order book with new results
      self.order_book[np.where(self.order_book[:,
        self.ob_indices["doc_num"]]==ord_num)] = ots[i]

    self.containers.shipment_rewards.append(ship_reward)
    self.containers.late_penalties.append(late_penalty)
    self.containers.late_orders_on_books.append(num_of_late_orders)

  def get_gmid_index_map(self):
    idx = 0
    gmid_index_map = {}
    for k in self.action_dict.keys():
      # if self.action_dict[k][2] is None:
      #   gmid_index_map[k] = 0
      if self.action_dict[k][self.action_dict_cols.index("gmid")] is not None:
        gmid_index_map[self.action_dict[k][self.action_dict_cols.index("gmid")]] = idx
        idx += 1
    return gmid_index_map

  # Calculate on-time delivery
  def otd_performance(self):
      orders = pd.DataFrame(self.order_book, columns=self.order_cols)
      # On time percentage
      counts = pd.DataFrame(orders['on_time'].value_counts().rename("number"))
      status = pd.DataFrame({"status" : pd.Series(["on_time", "late", 
          "not_delivered"], index = [1, -1, 0])})
      status = status.merge(counts, left_index=True, right_index=True)
      status['order_percentage'] = round(status['number'] / len(orders) * 100, 2)
      return status
  
  # Resets the model while maintaining consistent transition matrix, order probabilities,
  # production data, number of products, etc. so that the simulation can be run repeatedly 
  # for training purposes.
  def reset(self):
      # Reset counters
      self.sim_time = 0
      self.day = 0
      
      if self.settings['TRAIN']:
        # Get generated orders
        self.order_book = generate_orders(self)
        self.monthly_forecast = build_monthly_forecast(self)
        self.get_forecast()
        self.inventory = np.zeros(self.n_products)
      elif self.settings['TRAIN'] == False:
        # TODO: Move data set up to ppm
        # self.monthly_forecast = load_forecast(self.settings)
        pass
      self.stop_count = 0
      self.stop_time = 0
      # Reset containers
      self.containers.reset()

  def check_min_campaign(self, schedule, action):
      if schedule is None: ## else if schedule is none then repeat action min campaign length
          return self.min_campaign_len[action]
      else:   
          cur_prod = schedule[-1,self.sched_indices["gmid"]].astype(int)  
          cur_prod = int(self.gmid_action_map[cur_prod])        
          if action == cur_prod:        
              return 1
          else:
              return self.min_campaign_len[action]

  # The append schedule function will take the selected next action
  # from the algorithm and fill this data into the schedule. The off_grade
  # value is used to determine whether or not an action is allowed.
  # To save repetition, the action is vetted before going to the schedule
  # and thus the off-grade amount is passed as an argument.
  def append_schedule(self, schedule, action):
    # Look up off-grade production value
    off_grade = self.transition_matrix[0, int(action)]

    ## Check number of lots neede to impose min campaign length and scale batch size and batch time to those values
    if self.impose_min_campaign_len:
        num_lots = self.check_min_campaign(schedule, action)
    else:
        num_lots = 1
    ### Calculate batch size and batch time for num_lots required -- use these values to build schedule
    batch_size = num_lots * (self.action_dict[action][
      self.action_dict_cols.index("batch_size")])
    rate = float(
      self.action_dict[action][self.action_dict_cols.index("run_rate")])
    batch_time = int(math.ceil(batch_size / rate))
    if schedule is None:
      # Initialize schedule
      next_sched_entry = np.array([
        self.sched_indices["batch_num"], # Batch number
        self.action_dict[action][self.action_dict_cols.index("gmid")], # GMID
        self.action_dict[action][self.action_dict_cols.index("run_rate")], # Production rate
        batch_size, # Production quantity
        batch_time, # Production time
        self.sim_time, # Start day
        self.stop_hours + batch_time, # End time
        self.curing_time, # Curing time 
        0, # Curing completion time placeholder
        0, # Curing Completion flag
        # self.action_dict[action][3], # Inventory index
        action, # Action
        off_grade, # Off-grade production
        max((batch_size - off_grade), 0) # Actual production quantity
      ])
      # Calculate curing completion time
      next_sched_entry[self.sched_indices["cure_end_time"]] = (
        next_sched_entry[self.sched_indices["prod_end_time"]] + 
        next_sched_entry[self.sched_indices["cure_time"]])
      schedule = next_sched_entry.reshape(1,-1)

    else:
      # Look up off-grade production value
      off_grade = self.transition_matrix[
      	int(schedule[-1, self.sched_indices['action']]), 
        int(action)]
      
      next_sched_entry = np.array([
        schedule[-1,self.sched_indices["batch_num"]] + 1, # Batch Number
        self.action_dict[action][self.action_dict_cols.index("gmid")], # GMID
        self.action_dict[action][self.action_dict_cols.index("run_rate")], # Production rate
        batch_size, # Production quantity
        batch_time, # Production time
        self.stop_hours + schedule[-1, self.sched_indices["prod_end_time"]], # Start time
        0,  # End time placeholder
        self.curing_time, # Curing time
        0, # Curing completion time placeholder
        0, # Curing Completion flag
        # self.action_dict[action][3], # Inventory index
        action, # Action
        off_grade, # Off-grade production
        max((batch_size - off_grade), 0) # Actual production quantity
      ])
      # Calculate silo end time
      next_sched_entry[self.sched_indices["prod_end_time"]] = (
        next_sched_entry[self.sched_indices["prod_start_time"]] + 
        next_sched_entry[self.sched_indices["prod_time"]])
      # Calculate curing completion time
      next_sched_entry[self.sched_indices["cure_end_time"]] = (
        next_sched_entry[self.sched_indices["prod_end_time"]] + 
        next_sched_entry[self.sched_indices["cure_time"]])
      schedule = np.vstack([schedule, next_sched_entry.reshape(1,-1)])
        
      # Remove stop hours from calculation
      self.stop_hours = 0
        
    return schedule

  def drop_dummy_orders(self):
    # Exclude dummy orders
    return self.order_book[np.where(
        self.order_book[:,self.ob_indices['doc_num']]>0)[0]]

  def aggregate_monthly_visible_demand(self):
    self.order_book = self.drop_dummy_orders()
    # Subset to get real orders currently visible in the system
    orders_in_view = self.order_book[np.where(
        self.order_book[:,self.ob_indices['doc_create_time']]<=self.sim_time)[0]].copy()
    
    # Aggregate monthly totals
    current_month = self.sim_day_to_date[self.sim_time][0]
    months = list(self.shipping_dict.keys())
    agg_qty= np.zeros((len(months), self.n_products))
    for i, month in enumerate(months):
        # Aggregate monthly demand by gmid
        for g in self.gmids:
            agg_qty[i, self.gmid_index_map[int(g)]] = orders_in_view[
                np.where((orders_in_view[:,self.ob_indices['gmid']]==int(g)) &
                    (orders_in_view[:,self.ob_indices['planned_gi_month']]==month))[0],
                self.ob_indices['order_qty']].sum()
    agg_monthly_demand = int(agg_qty.sum())
    agg_ob_demand = int(orders_in_view[np.where(
        orders_in_view[:,self.ob_indices['planned_gi_month']]<=max(months))[0],
        self.ob_indices['order_qty']].sum())
    # Ensure aggregate values match
    assert agg_monthly_demand == agg_ob_demand, \
        "Quantities do not match.\nDay: {}\nOrder Book: {}\nAgg Monthly Demand: {}".format(
            self.sim_time, agg_ob_demand, agg_monthly_demand)
    
    return agg_qty.astype(int), orders_in_view
  
  # Call reward function
  def get_rewards(self):
    return containers.reward_function(self)

  def get_inventory_cost(self):
    return calculate_inventory_cost(self)

  def get_orders(self):
    return pd.DataFrame(self.order_book, columns=self.order_cols)

  def get_schedule(self, schedule):
    return pd.DataFrame(schedule, columns=self.sched_cols)

  def plot_gannt(self):
    plot_gannt(self)

  def production_stop(self, schedule):
    return production_stop(self, schedule)

  def get_cs_level(self):
    return get_cs_level(self)

  def get_forecast(self):
    select_forecast(self)