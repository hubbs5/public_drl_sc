# RL Agent: Builds RL agent according to specified algorithms
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 04.02.2019

from copy import deepcopy

from .rl_algos import rl_utils
from .rl_algos import a2c

def create_agent(env):
	# Check for default RL values
	env.settings = rl_utils.check_settings(env.settings)

	# Get algorithm specific settings and hyperparameters
	if env.settings['RL_ALGO'] == 'A2C':
		agent = a2c.a2c(env)
		# settings = a2c.check_settings(env, settings)
	elif env.settings['RL_ALGO'] == 'DQN':
		raise ValueError('RL_ALGO {} not yet implemented'.format(
			env.settings['RL_ALGO']))
	elif env.settings['RL_ALGO'] == 'PPO':
		raise ValueError('RL_ALGO {} not yet implemented'.format(
			env.settings['RL_ALGO']))
	elif env.settings['RL_ALGO'] == 'TRPO':
		raise ValueError('RL_ALGO {} not yet implemented'.format(
			env.settings['RL_ALGO']))
	else:
		raise ValueError('RL_ALGO {} not recognized'.format(
			env.settings['RL_ALGO']))

	return agent