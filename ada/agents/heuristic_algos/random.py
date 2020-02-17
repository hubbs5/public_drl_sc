# Random Agent
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 20.02.2019

import numpy as np 
from ada.environments.demand_models.demand_utils import *

class random_agent():

	def __init__(self, env):

		self.action_space = env.action_list
		self.env = env

	def get_action(self):
		return np.random.choice(self.action_space)

	def train(self):
		schedule = None
		for step in range(self.env.n_steps):
			# Get planning horizon limit and current time
			planning_limit = self.env.sim_time + self.env.fixed_planning_horizon
			planning_time = np.max(schedule[:, 
				self.env.sched_indices['prod_start_time']]) if schedule is not None else self.env.sim_time
			while planning_time < planning_limit:
				action = self.get_action()
				schedule = self.env.append_schedule(schedule, action)
				planning_time = np.max(schedule[:, 
					self.env.sched_indices['prod_start_time']])
			schedule = self.env.step(schedule)
			check_forecast_consistency(self.env)

