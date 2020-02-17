# Heuristic Agent: Builds agents according to heuristic algorithms
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 21.02.2019

from .heuristic_algos import heuristic_utils
from .heuristic_algos import random

def create_agent(env): 
	# Check for default RL values
	#settings = heuristic_utils.check_settings(settings)

	# Get algorithm specific settings and hyperparameters
	if env.settings['HEURISTIC_ALGO'] == 'RANDOM':
		agent = random.random_agent(env)
	else:
		raise ValueError('HEURISTIC_ALGO {} not recognized'.format(
			env.settings['HEURISTIC_ALGO']))

	return agent