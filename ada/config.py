#!/usr/bin/env python3

import sys 
import os
import string
import re
import numpy as np
import torch
import pandas as pd
import warnings
from copy import copy
from argparse import ArgumentParser
from datetime import datetime, timedelta

from ada.environments import env_utils
from ada.agents.rl_algos import rl_utils
from ada.agents.mip_algos import mip_utils
from ada.agents.heuristic_algos import heuristic_utils
from ada.Loggermixin import *

# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger=Loggermixin.get_default_logger()

# Define configuration here
config = {
    'AGENT_CLASS': 'RL',
    'ENVIRONMENT': 'TARTAN',
    'N_PRODUCTS': 6,
    'START_TIME': '2018-01-01',
    'END_TIME': '2018-12-31',
    'REWARD_FUNCTION': 'OTD1'
    # And so on...
}

def parse_cl_args(argv):
    config_args = ['default', '<path>']
    config_help = 'Define the configuration file. Acceptable arguments are:'
    # Parse command line arguments
    parser = ArgumentParser(description="Import configuration file or define" + 
        " the relevant parameters.")
    parser.add_argument('--config', metavar='CONFIGPATH', type=str, default='default',
        help='Define the configuration file. Acceptable arguments are: ' + 
        ', '.join(config_args))

    return parser.parse_args()

def parse_config_file(config_data):
    # Function to ensure config files contain all relevant information
    # Raise warning if the following configuration requirements are not met
    # then populate with defaults.
    # Simulation environment
    # Algorithm
    config_data = capitalize_config_values(config_data)
    #[h_logger.debug(key, ': ', config_data[key]) for key in config_data.keys()]
    if 'AGENT_CLASS' not in config_data.keys():
        warn_string = '\nNo agent_class defined in configuration.' + \
            ' Loading A2C by default.'
        warnings.warn(warn_string)
        config_data['AGENT_CLASS'] = 'RL'
    if 'ENVIRONMENT' not in config_data.keys():
        warn_string = '\nNo environment defined in configuration.' + \
            ' Loading TARTAN by default.'
        warnings.warn(warn_string)
        config_data['ENVIRONMENT'] = 'TARTAN'
    
    # Check remaining configuration settings
    config_data = env_utils.check_env_settings(config_data)
    if config_data['AGENT_CLASS'] == 'RL':
        config_data = rl_utils.check_settings(config_data)
    elif config_data['AGENT_CLASS'] == 'MIP':
        config_data = mip_utils.check_settings(config_data)
    elif config_data['AGENT_CLASS'] == 'HEURISTIC':
        config_data = heuristic_utils.check_settings(config_data)
    else:
        raise ValueError('AGENT_CLASS {} not recognized.'.format(
            config_data['AGENT_CLASS']))

    return config_data

def save_config_file(config_data):
    filepath = config_data['DATA_PATH'] # Data Path set by agent settings
    file = open(os.path.join(filepath, 'config_settings.txt'), 'w')
    file.writelines('parameter,value')
    # Save and print primary configuration settings first
    primary_settings = ['AGENT_CLASS', 'MIP_ALGO', 'RL_ALGO', 'ENVIRONMENT',
        'SYS_START_TIME', 'DATA_PATH']

    # Save config data
    [file.writelines("\n{},{}".format(key, config_data[key])) 
        for key in primary_settings if key in config_data.keys()]
    [file.writelines("\n{},{}".format(key, config_data[key])) 
        for key in config_data.keys() if key not in primary_settings]
    file.close()
    [h_logger.debug("{}: {}".format(key, config_data[key]))
        for key in primary_settings if key in config_data.keys()]
    [h_logger.debug("{}: {}".format(key, config_data[key]))
        for key in config_data.keys() if key not in primary_settings]

def capitalize_config_values(config_data):
    assert type(config_data) == dict, "Configuration data not in dictionary format."
    for key in config_data.keys():
        # Skip paths as they may be case sensitive
        if 'PATH' in key:
            continue
        if type(config_data[key]) == str:
            config_data[key] = config_data[key].upper()
    return config_data

def load_config_file(filepath):
    config_data = None
    supported_extensions = ['.csv', '.txt', '.xls', '.xlsx']
    # Check config file extension
    user_input = filepath
    filename, file_ext = os.path.splitext(user_input)
    h_logger.debug(user_input)
    count = 5
    while config_data is None:
        try:
            # Load config file
            if file_ext == '.csv' or file_ext == '.txt':
                data = pd.read_csv(user_input, header=0)
                data = data.set_index(data[data.columns[0]])
                config_data = data.xs(data.columns[1], 
                    axis=1, drop_level=True).to_dict()
            elif file_ext == '.xlsx' or file_ext == '.xls':
                data = pd.read_excel(user_input, header=0)
                data = data.set_index(data[data.columns[0]])
                config_data = data.xs(data.columns[1], 
                    axis=1, drop_level=True).to_dict()
        except FileNotFoundError:
            h_logger.debug("No valid configuration file found.")
            user_input = input("Enter path to file, or 1 to load default." +
                " 2 to quit. Be sure to escape any backslashes " + 
                "on Windows.\n>>>> ")
            filename, file_ext = os.path.splitext(user_input)
            if user_input == str(1):
                config_data = config
                break
            if user_input == str(2):
                sys.exit('No valid file entered. Exiting.')
            
        if file_ext not in supported_extensions:
            h_logger.debug("Invalid file format from {}. Valid extensions are:".format(
                user_input))
            [h_logger.debug(ext) for ext in supported_extensions]
            count += 1
            if count >= 5:
                sys.exit('No valid file entered. Exiting.')

    if type(config_data) == pd.core.frame.DataFrame:
        return config_data.to_dict() 
    elif type(config_data) == dict:
        return config_data
    
# Useful for deleting unnecessary results folders during testing
def get_results_path(path, target):
    path_name = None
    while path_name != target:
        path_split = os.path.split(path)
        path_name = path_split[1]
        path = path_split[0]

    return os.path.join(path, path_name)

def set_up_sim(args, default_path=None, config_dict=None):
    # Use explicitly supplied configuration if provided
    if config_dict is not None:
        if type(config_dict) == dict:
            config_data = config_dict
        else:
            raise ValueError("config_dict not dict type. {} passed.".format(type(config_dict)))
    # process the args, if supplied
    elif args is not None and args.config is not None:
        if args.config.lower() == 'default':
            if default_path is not None:
                config_data = load_config_file(default_path)
            else:
                raise ValueError("--config=default but no default path supplied")
        else:
            config_data = load_config_file(args.config)
    # otherwise fall back to the default path
    elif default_path is not None:
        config_data = load_config_file(default_path)
    else: # otherwise throw an error
        raise ValueError("No config supplied")

    config_data = parse_config_file(config_data)
    
    np.random.seed(config_data['RANDOM_SEED'])
    torch.manual_seed(config_data['RANDOM_SEED'])
    try:
        if config_data['DEVICE'] == 'GPU':
            torch.cuda.manual_seed.all(config_data['RANDOM_SEED'])
    except KeyError:
        config_data['DEVICE'] = 'CPU'
    # Initialize environment
    if config_data['ENVIRONMENT'] == 'TARTAN':
        from ada.environments.tartan import productionFacility
        env = productionFacility(config_data)
    elif config_data['ENVIRONMENT'] == 'GOPHER':
        from ada.environments.gopher import productionFacility
        env = productionFacility(config_data)
    else:
        raise ValueError('Environment name {} not recognized.'.format(
            config_data['ENVIRONMENT']))

    # Initialize agent
    if config_data['AGENT_CLASS'] == 'RL':
        from ada.agents.rl_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'MIP':
        from ada.agents.opt_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'HEURISTIC':
        from ada.agents.heuristic_agent import create_agent
        agent = create_agent(env)
    else:
        raise ValueError('AGENT_CLASS {} not recognized.'.format(
            config_data['AGENT_CLASS']))

    # Save config data for reference
    save_config_file(config_data)

    return agent

    # if args.config.lower() != 'default':
    #     config_data = load_config_file(args.config)
    # elif default_path is not None:
    #     config_data = load_config_file(default_path)
    # # elif default_path is not None:
    # #     config_data = load_config_file(default_path)
    # elif config_dict is not None:
    #     assert type(config_dict) == dict, "config_dict not dict type. {} passed.".format(
    #         type(config_dict))
    #     config_data = config_dict
    # else:
    #     config_data = config

    # config_data = parse_config_file(config_data)
    
    # np.random.seed(config_data['RANDOM_SEED'])
    
    # # Initialize environment
    # if config_data['ENVIRONMENT'] == 'TARTAN':
    #     from ada.environments.tartan import productionFacility
    #     env = productionFacility(config_data)
    # elif config_data['ENVIRONMENT'] == 'GOPHER':
    #     raise ValueError('Environment not yet implemented.')
    # else:
    #     raise ValueError('Environment name {} not recognized.'.format(
    #         config_data['ENVIRONMENT']))
    # try:
    #     if config_data['ORDER_BOOK_PATH'] is not None:
    #         env.order_book = env_utils.load_scenario_data(
    #             config_data['ORDER_BOOK_PATH'])
    # except KeyError:
    #     pass
    # try:
    #     if config_data['PRODUCT_DATA_PATH'] is not None:
    #         env.product_data, env.transition_matrix, env.zfin = env_utils.load_scenario_data(
    #             config_data['PRODUCT_DATA_PATH'])
    # except KeyError:
    #     pass

    # # Initialize agent
    # if config_data['AGENT_CLASS'] == 'RL':
    #     from ada.agents.rl_agent import create_agent
    #     agent = create_agent(env)
    #     torch.manual_seed(config_data['RANDOM_SEED'])
    #     if config_data['DEVICE'] == 'GPU' or config_data['DEVICE'] == 'CUDA':
    #         torch.cuda.manual_seed.all(config_data['RANDOM_SEED'])
    # elif config_data['AGENT_CLASS'] == 'MIP':
    #     from ada.agents.opt_agent import create_agent
    #     agent = create_agent(env)
    # elif config_data['AGENT_CLASS'] == 'HEURISTIC':
    #     from ada.agents.heuristic_agent import create_agent
    #     agent = create_agent(env)
    # else:
    #     raise ValueError('AGENT_CLASS {} not recognized.'.format(
    #         config_data['AGENT_CLASS']))

    # # Save config data for reference
    # save_config_file(config_data)

    # return agent

# TODO: This has grown to be an ad hoc mess and ought to be moved
# to ppm.py where these helper functions and methods can be called
# when ORDER_BOOK_PATH, PRODUCT_DATA_PATH, or other values are populated.
def set_up_production_environment(args, default=None):
    if args.config.lower() == 'default' and default is not None:
        config_data = load_config_file(default)
    elif args.config.lower() != 'default':
        config_data = load_config_file(args.config)
    else:
        raise ValueError('No configuration file provided.')

    config_data = parse_config_file(config_data)
    today = datetime.now()
    config_data['START_TIME'] = str(today.date())
    config_data['END_TIME'] = str(today.date() + 
        timedelta(days=config_data['LOOKAHEAD_PLANNING_HORIZON']))
    
    np.random.seed(config_data['RANDOM_SEED'])
    torch.manual_seed(config_data['RANDOM_SEED'])
    if config_data['DEVICE'] == 'GPU':
        torch.cuda.manual_seed.all(config_data['RANDOM_SEED'])
    # Initialize environment
    if config_data['ENVIRONMENT'] == 'TARTAN':
        from ada.environments.tartan import productionFacility
        env = productionFacility(config_data)
    elif config_data['ENVIRONMENT'] == 'GOPHER':
        from ada.environments.gopher import productionFacility
        env = productionFacility(config_data)
    else:
        raise ValueError('Environment name {} not recognized.'.format(
            config_data['ENVIRONMENT']))

    inventory, order_book, forecast = \
        env_utils.load_current_state_data(config_data)
    env.order_book = env_utils.process_order_data(order_book, env)
    inventory = env_utils.process_inventory_data(inventory, env)
    env.inventory = copy(inventory.flatten().astype(float))

    # Forecast requires separate preprocessing
    env.monthly_forecast = env_utils.process_forecast_data(forecast, env)

    # Get Current Schedule
    env.schedule = env_utils.load_current_schedule(env)
    
    # Initialize agent
    if config_data['AGENT_CLASS'] == 'RL':
        from ada.agents.rl_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'MIP':
        from ada.agents.opt_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'HEURISTIC':
        from ada.agents.heuristic_agent import create_agent
        agent = create_agent(env)
    else:
        raise ValueError('AGENT_CLASS {} not recognized.'.format(
            config_data['AGENT_CLASS']))


    # Save config data for reference
    # save_config_file(config_data)

    return agent