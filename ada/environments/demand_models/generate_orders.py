# This file contains two functions to tie together all of the relevant 
# demand models. Update the import statement and the if/else when new
# models are introduced.

import numpy as np 

from .seasonal_demand import *
from .historic_demand import *
from .forecast import *
from .test_demand import *

def generate_orders(env):
	env.settings = check_demand_settings(env.settings)
	demand_model = env.settings['DEMAND_MODEL']
	if demand_model == 'SEASONAL_DEMAND':
		return generate_seasonal_orders(env)
	elif demand_model == 'EXCEL_DEMAND':
		return load_excel_demand_file(env)		
	elif demand_model == 'HISTORY_MODEL':
		return generate_history_model_orders(env)	
	elif demand_model == 'TEST':
		return generate_test_orders(env)
	else:
		raise ValueError('Unrecognized demand model requested')

# def generate_forecast(env):
# 	try:
# 		# Default forecast value
# 		if env.settings['FORECAST'] == 'UNIFORM' or env.settings['FORECAST'] == True:
# 			build_uniform_forecast(env)
# 		elif env.settings['FORECAST'] == 'UNIFORM_HORIZON':
# 			build_uniform_horizon_forecast(env)
# 		elif env.settings['FORECAST'] == 'AGGREGATE_HORIZON':
# 			build_aggregated_horizon_forecast(env)
# 		elif env.settings['FORECAST'] == 'STOCHASTIC_UNIFORM':
# 			build_stochastic_uniform_forecast(env)
# 		elif env.settings['FORECAST'] == False:
# 			pass
# 		elif env.settings['FORECAST'] == 'DETERMINISTIC_AVERAGE_DEMAND':
# 		else:
# 			raise NotImplementedError('Forecast model: {} not yet implemented.'.format(
# 				env.settings['FORECAST']))
# 	except KeyError:
# 		pass

def check_demand_settings(settings):
	demand_model = settings['DEMAND_MODEL']
	if demand_model == 'SEASONAL_DEMAND':
		settings = get_default_seasonal_demand_settings(settings)
	elif demand_model == 'UNIFORM_DEMAND':
		settings = get_default_uniform_demand_settings(settings)
	elif demand_model == 'EXCEL_DEMAND':
		if settings['EXCEL_DEMAND_FILE'] is None:
			raise ValueError('No demand file provided in settings when DEMAND_MODEL is set to EXCEL_DEMAND')	
	# elif demand_model == 'HISTORY_MODEL':
		# pass
		# settings = get_default_history_model_demand_settings(settings)
	elif demand_model == 'TEST':
		settings = get_default_test_demand_settings(settings)
	else:
		raise ValueError('Unrecognized demand model requested')
	return settings