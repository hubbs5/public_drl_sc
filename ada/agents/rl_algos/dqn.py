# dq: Deep-Q Learning
# Christian Hubbs
# 11.07.2018

import torch
from torch import nn
import numpy as np 
import copy
import time
import os

from ada.scheduler.rl_scheduler.rl_utils import *
from ..network_scheduler import *
from ..ppm.utils import get_cs_level, get_planning_data_headers

class DQLearning():

	def __init__(self, env, agent):

		self.env = env
		self.agent = agent
		self.target_net = copy.deepcopy(agent)
		self.action_space = np.arange(env.action_space.n)

		self.loss, self.grads = [], []

	def log_planning_data(self, data, algo='dqn'):
		_log_planning_data(self, data, algo)

	def train(self, gamma=0.99, sync_freq=1000, buffer_size=10000,
		batch_size=32, num_episodes=None, level=None, epsilon=0.5, 
		epsilon_final=0.01, epsilon_decay_rate=1E-5):

		self.epsilon = epsilon
		self.epsilon_start = epsilon
		self.buffer = ExperienceBuffer(buffer_size)
		self.level = level
		self.num_episodes = num_episodes
		self.sync_freq = sync_freq

		if self.num_episodes is None and self.level is None:
			self.num_episodes = 2000

		self.ep_rewards = []
		self.ep_reward = 0
		self.ep_loss = []
		self._states = []
		self._qvals = []
		self.ep_counter = 0
		self.step_counter = 0
		self.schedule = None

		training = True
		while training:
			
			self.epsilon = max(self.epsilon_start * (1 - self.step_counter \
				* epsilon_decay_rate), epsilon_final)

			for day in range(self.env.n_days):

				try:
				self.schedule, planning_data = q_scheduler(self.env,
					self.agent, self.schedule, self.epsilon)
				except ValueError:
					print('nan found in output')
					training = False
					break

				if planning_data is not None:
					_planning_data.append(planning_data)
					# Extract values for buffer
					states = planning_data[-env.]

				self.schedule = self.env.step(self.scheduler)

				if self.buffer.get_length() < buffer_size:
					continue

				# Sync networks
				if self.step_counter % self.sync_freq == 0:
					self.target_net.load_state_dict(
						self.agent.state_dict())

				batch = self.buffer.sample(batch_size)
				loss = self.agent.update(batch, self.target_net, gamma)
				self.ep_loss.append(loss)

			training = self.check_for_completion()
			self.log_planning_data(_planning_data)