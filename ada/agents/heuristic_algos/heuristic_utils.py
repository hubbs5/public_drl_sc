# heuristic_utils: Contains various helper functions to extract data from 
# heuristic functions and set default configurations.
# Author: Christian Hubbs
# Contact: christiandhubbs@gmail.com
# Date: 20.02.2019

import os 
from datetime import datetime

def check_settings(settings=None):
    '''
    Input
    settings: dict of values required to parameterize the mip. 
        Missing values or None is permissible as these will
        be populated with defaults.

    Output
    settings: dict of values required to completely specifiy the mip.
    '''
    defaults = {
    	'HEURISTIC_ALGO': 'RANDOM'
    	}
    if settings is None:
        settings = defaults
    else:
        for key in defaults.keys():
            if key not in settings.keys():
                settings[key] = defaults[key]
            elif defaults[key] is not None:
                settings[key] = type(defaults[key])(settings[key])
                
    if 'DATA_PATH' not in settings.keys():
        default_data_path = os.getcwd() + "/RESULTS/" + settings['HEURISTIC_ALGO'].upper() + '/'\
            + datetime.now().strftime("%Y_%m_%d_%H_%M")
        settings['DATA_PATH'] = default_data_path

    os.makedirs(settings['DATA_PATH'], exist_ok=True)

    return settings
