# Opt Agent: Builds optimization agent according to specified algorithms
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 04.02.2019

from .mip_algos import mip_utils
from ..scheduler.mip_scheduler import *

def create_agent(env, schedule=None):
	# Check for default MIP values
	env.settings = mip_utils.check_settings(env.settings)
	# Get algorithm specific settings and hyperparameters
	if env.settings['MIP_ALGO'] == 'MPC':
		from .mip_algos.deterministic_mip import buildDeterministicMIP
		agent = schedulingMPC(env, buildDeterministicMIP, schedule=schedule)
	elif 'GOD' in env.settings['MIP_ALGO']:
		from .mip_algos.god_mip import buildGodMIP
		# TODO: Fix planning horizon properly
		env.fixed_planning_horizon = env.n_days
		agent = schedulingMPC(env, buildGodMIP, schedule=schedule)
	elif env.settings['MIP_ALGO'] == 'SMPC': # Import stochastic MPC
		from .mip_algos.stochastic_mip import buildStochasticMIP
		agent = schedulingMPC(env, buildStochasticMIP, schedule=schedule)
	else:
		raise ValueError('MIP_ALGO {} not recognized'.format(
			env.settings['MIP_ALGO']))

	return agent