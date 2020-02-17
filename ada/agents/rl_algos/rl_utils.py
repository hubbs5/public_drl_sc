# rl_utils: Reinforcement Learning Utilities
# Christian Hubbs
# 10.03.2018

# Update 27.03.2018: added check_settings to manage RL settings dictionaries

# This file contains a collection of utilities to assist with developing RL
# agents.

# discountReturns: takes a vector of rewards and a discount factor to return
# the discounted returns

# generate_search_dict: develops a dictionary to run a grid search of
# network hyperparameters.

# log_data:

# check_settings:

import numpy as np
import os
from os.path import exists
from datetime import datetime
import time
from pathlib import Path
import warnings
import collections
import pickle
from ...environments.env_utils import get_planning_data_headers

# discountReturns
def discount_returns(returns, gamma=0.99):
    discounted_returns = np.zeros_like(returns)
    cumulative_returns = 0
    for i in reversed(range(0, len(returns))):
        cumulative_returns = cumulative_returns * gamma + returns[i]
        discounted_returns[i] = cumulative_returns
    return discounted_returns

# Get search parameters for the ANN
def generate_search_dict(layer_min, node_min, layer_max=None, node_max=None,
                         learning_rate=0.001, value_estimator=True,
                         algo='a2c', num_episodes=2000, gamma=0.99, 
                         convergence_tol=0.02, save_path='default'):
    t = time.localtime()
    run_date = ''.join([str(t.tm_year), str(t.tm_mon), str(t.tm_mday)])
    search_dict = {}
    if layer_max is None:
        layer_max = layer_min + 1
    else: layer_max += 1
    if node_max is None:
        node_max = node_min + 1
    else: node_max += 1
    layer_range = [layer_min, layer_max]
    node_range = [node_min, node_max]
    cwd = os.getcwd()
    
    k = 0
    for layer in range(min(layer_range), max(layer_range)):
        for node in range(min(node_range), max(node_range)):
            if save_path == 'default':
                data_path = str(cwd + "/" + run_date + "/" + algo + 
                                "_" + str(layer) + "_" + str(node))
                #ckpt_path = str(cwd + "/" + run_date + "/" + algo + 
                #                "_" + str(layer) + "_" + str(node))
                if value_estimator:
                    data_path = data_path + '_baseline'
                    
            else:
                _save_path = save_path
            search_dict[k] = {'n_hidden_layers': layer,
                              'n_hidden_nodes': node,
                              'learning_rate': learning_rate,
                              'value_estimator': value_estimator,
                              'num_episodes': num_episodes,
                              'gamma': gamma,
                              'convergence_tol': convergence_tol,
                              'checkpoint_path': data_path,
                              'DATA_PATH': data_path}
            k += 1
    
    return search_dict

def log_data(data, settings, env, val_names=None):
    model_name = (settings['ENVIRONMENT'] + '_' + str(settings['N_PRODUCTS']) \
        + '_' + settings['RL_ALGO'])
    # Maybe come up with a way to deal with lists as well
    if type(data) == list:
        data = np.array(data)
    file_name = settings['DATA_PATH'] + '/' + model_name + '.txt'
    if not exists(file_name):
        path = Path(settings['DATA_PATH'])
        path.mkdir(parents=True, exist_ok=True)
        with open(file_name, 'w') as file:
            #file.write(env.spec.id + "\n")
            file.write("Training began at: {:s}\n".format(settings['START_TIME']))
            # Insert data here
            file.write("Training completed at: \n")
            file.write(settings['RL_ALGO'] + " algorithm\n")
            file.write("Number of Products {}\n".format(env.n_products))
            file.write("State Setting: {}\n".format(env.state_setting))
            file.write("Reward function: {}\n".format(env.reward_function))
            file.write("Planning Time Horizon = {}\n".format(env.fixed_planning_horizon))
            # if env.transition_matrix_setting == 'random':
            #     file.write("Transition Matrix randomly generated.")
            # elif env.transition_matrix_setting is None:
            #     file.write("No transition losses.")
            file.write("Random seed: \n")
            file.write("="*78 + "\n")
            file.write("Episode Results: \n")
            # Write data headers
            if val_names is not None:
                [file.write("{:s}\t".format(name)) for name in val_names]
                file.write("\n")
            # Append data
            if data.ndim < 2:
                [file.write("{:f}\r".format(entry)) for entry in data]
            if data.ndim == 2:
                for row in data:
                    [file.write("{:f}\t".format(col)) for col in row]
                    file.write("\r")
            if data.ndim > 2:
                raise ValueError('More than 2 dimensions in data to be written.')
        file.close()
    else:
        with open(file_name, 'a') as file:
            # Append data
            if data.ndim < 2:
                [file.write("{:f}\r".format(entry)) for entry in data]
            if data.ndim == 2:
                for row in data:
                    [file.write("{:f}\t".format(col)) for col in row]
                    file.write("\r")
            if data.ndim > 2:
                raise ValueError('More than 2 dimensions in data to be written.')
        file.close()

    # Ensure existence of checkpoint path
    ckpt_path = settings['DATA_PATH']
    if not exists(ckpt_path):
        ckpt_path = Path(ckpt_path)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    order_stats_file = settings['DATA_PATH'] + '/order_statistics.pkl'
    # Save order statistics
    if not exists(order_stats_file):
        order_stats_file = open(order_stats_file, 'wb')
        pickle.dump(env.order_statistics, order_stats_file)

def check_for_convergence(data, episode, settings):
    # The model is said to have converged when the average of the last X% of
    # of episodes is within the tolerance of the preceeding X% of episodes.
    # For example, say that we have 100 episodes and set the percentage_check
    # value to 10% with a tolerance of 1% and the model declares convergence
    # after 50 episodes. That would mean that the average results of episodes
    # 40 to 50 within the range +/- (1 + epsilon) * average(ep_30 to ep_40).
    percentage_check = 0.1
    if percentage_check > 0.5:
        return False
    if 'convergence_tol' not in settings.keys():
        return False
    # Ensure at least 2 * percentage_check of episodes are complete
    elif episode >= settings['num_episodes'] * (2 * percentage_check):
        last_10_per = len(data) - int(settings['num_episodes'] * percentage_check)
        trailing_10_per = len(data) - int(settings['num_episodes'] * 2 * percentage_check)
        mean_last_10 = np.mean(data[-int(settings['num_episodes'] * percentage_check):])
        mean_trailing_10 = np.mean(data[-int(settings['num_episodes'] * 2 * percentage_check):
            -int(settings['num_episodes'] * percentage_check)])
        if mean_last_10 <= (1 + settings['convergence_tol']) * mean_trailing_10 and (
            1 - settings['convergence_tol']) * mean_trailing_10:
            print("Policy converged after: {:d}".format(episode))
            print("Mean last 10%: {:.5f}".format(mean_last_10))
            print("Mean trailing 10%: {:.5f}".format(mean_trailing_10))
            return True
    else:
        return False


# Check RL settings to ensure all values are properly entered and available
def check_settings(settings=None):
    '''
    Input
    settings: dict of values required to parameterize the simulation 
        environment. Missing values or None is permissible as these will
        be populated with defaults.

    Output
    settings: dict of values required to completely specifiy the simulation
        environment.
    '''
    defaults = {
        'RL_ALGO': 'A2C',
        'N_EPISODES': 1000,
        'BATCH_SIZE': 10,
        'GAMMA': 0.99,
        'N_HIDDEN_NODES': 32,
        'N_HIDDEN_LAYERS': 8,
        'ACTIVATION_FUNCTION': 'ELU',
        'LEARNING_RATE': 1E-2,
        'ACTOR_LR': 0.0,
        'CRITIC_LR': 0.0,
        'BIAS': True,
        'BETA': 1E-3,
        'GRADIENT_CLIPPING': False,
        'DEVICE': 'CPU',
        'FRAMEWORK': 'PYTORCH',
        'PERIODIC_TESTING': True # Enables testing to occur at 20% intervals
    }
    if settings is None:
        settings = defaults
    else:
        for key in defaults.keys():
            if key not in settings.keys():
                settings[key] = defaults[key]
            elif defaults[key] is not None:
                settings[key] = type(defaults[key])(settings[key])
                
    # TODO: Evaluate whether or not this is the best level to locate the data_path generation
    if 'DATA_PATH' not in settings.keys():
        default_data_path = os.getcwd() + "/RESULTS/" + settings['RL_ALGO'].upper() + '/'\
            + datetime.now().strftime("%Y_%m_%d_%H_%M")
        settings['DATA_PATH'] = default_data_path

    os.makedirs(settings['DATA_PATH'], exist_ok=True)

    return settings

def z_norm(x, axis=1):
    if np.ndim(x) <= 1 and axis == 1:
        x = x.reshape(-1, 1)
    try:
        norms = (x - x.mean(axis=axis)) / x.std(axis=axis)
    except Warning:
        norms = (x - x.mean(axis=axis)) / (
            x.std(axis=axis) + 1E-6)
    return norms

class ExperienceBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.experience = collections.namedtuple('Experience',
            field_names=['state', 'action', 'reward', 
                'step_number', 'next_state'])

    def get_length(self):
        return (len(self.buffer))

    def append(self, state, action, reward, step_number, next_state):
        exp = self.experience(state, action, reward, step_number, next_state)
        self.buffer.append(exp)

    def sample(self, batch_size):
        indices = np.random.choice(self.get_length(), 
            batch_size, replace=False)
        states, actions, rewards, step_numbers, next_states = zip(*
            (self.buffer[i] for i in indices))
        states = np.array([np.array(states[i]) for i in range(batch_size)])
        next_states = np.array([np.array(next_states[i]) 
            for i in range(batch_size)])
        actions = np.array(actions, dtype=np.int)
        rewards = np.array(rewards, dtype=np.float)
        steps = np.array(step_numbers)

        return states, actions, rewards, steps, next_states

def _log_planning_data(model, data, algo):
    data = np.vstack(data)[:model.env.n_days]
    if algo == 'dqn':
        data = np.hstack([data,
            model.env.containers.stack_values()[:model.env.n_days],
            model._qvals
            ])
        if model.planning_data_headers is None:
            headers, _ = get_planning_data_headers(model.env)
            for a in model.env.action_list:
                headers.append('qval_' + str(a))

            planning_data_indices = {k: i for i, k in enumerate(headers)}
            model.planning_data_headers = headers
            model.planning_data_indices = planning_data_indices

    if model.planning_data is None:
        model.planning_data = np.dstack([data])
    else:
        model.planning_data = np.dstack([model.planning_data,
            data])

def torchToNumpy(tensor, device='cpu'):
    if device=='cuda':
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

def check_device_settings(settings):
    if settings['DEVICE'].upper() == 'CPU':
        return 'cpu'
    elif settings['DEVICE'].upper() == 'CUDA' or settings['DEVICE'].upper() == 'GPU':
        return 'cuda'
    else:
        raise ValueError('Device {} not recognized. Define either CPU or GPU.'.format(settings['DEVICE']))


