# a2c: runs actor critic algorithm
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 06.02.2019

import numpy as np 
import warnings
from .rl_utils import *
import time
import os
import pandas as pd
from datetime import datetime
import torch
from ..networks.networks import policyEstimator, valueEstimator
from ...scheduler.network_scheduler import network_scheduler, estimate_schedule_value
from ...environments.env_utils import get_planning_data_headers

# Define a2c class
class a2c():

    def __init__(self, env):

        self.env = env
        self.settings = self.check_a2c_settings(self.env.settings)
        self.policy_est = policyEstimator(self.env, self.settings)
        self.value_est = valueEstimator(self.env, self.settings)
        self.planning_data = None
        self.planning_data_headers = None
        # Set up containers for policy, value, entropy, and total loss
        # TODO: see if namedtuple or other modules in containers yield better
        # performance for data logging.
        self.loss, self.policy_loss, self.policy_grads = [], [], []
        self.entropy_loss, self.value_loss, self.value_grads = [], [], []
        self.kl_div = []

    def check_a2c_settings(self, settings):
        # Add a2c specific settings here
        return settings

    def test(self, checkpoint, n_tests=10):
        path = Path('scenarios')
        if os.path.exists(path) == False:
            path.mkdir(parents=True, exist_ok=True)

        # Get scenarios from order_book data
        order_books = [s for s in os.listdir(path) if 'order' in s]

        # TODO: Build a scenario generating function in case none are found
        test_data = pd.DataFrame()
        count = 0
        for s in order_books:
            ob_path = os.path.join('scenarios', s)
            self.env.reset()
            self.env.order_book = pickle.load(open(ob_path, 'rb'))

            # Run single-episode experiment
            schedule = None
            test_planning_data = []
            for day in range(self.env.n_days):
                schedule, _planning_data = network_scheduler(self.env, 
                    self.policy_est, schedule, test=True)
                schedule = self.env.step(schedule)
                test_planning_data.append(_planning_data)
            cs_level = self.env.get_cs_level()

            inv_cost = np.round(sum(self.env.containers.inventory_cost), 0)
            late_penalties = np.round(sum(self.env.containers.late_penalties), 0)
            shipment_rewards = np.round(sum(self.env.containers.shipment_rewards), 0)
            total_rewards = inv_cost + late_penalties + shipment_rewards

            test_data_dict = {'scenario': ob_path.split('_')[-1].split('.')[0],
                         'algo': self.env.settings['RL_ALGO'],
                         'product_availability': np.round(cs_level[0], 3),
                         'delayed_order': np.round(cs_level[1], 3),
                         'not_shipped': np.round(cs_level[2], 3),
                         'total_rewards': total_rewards,
                         'inv_cost': inv_cost,
                         'late_penalties': late_penalties,
                         'shipment_rewards': shipment_rewards}
            test_data = pd.concat([test_data, pd.DataFrame(test_data_dict, index=[count])])
            count += 1
        # Save test data
        test_data.to_csv(self.settings['DATA_PATH'] + '/checkpoint_test_' \
            + str(checkpoint) + '.csv')
        if checkpoint != 100:
            torch.save(self.value_est.state_dict(), 
                self.settings['DATA_PATH'] + '/critic_' + str(checkpoint) + '.pt')
            torch.save(self.policy_est.state_dict(), 
                self.settings['DATA_PATH'] + '/actor_' + str(checkpoint) + '.pt')

    def train(self):
        # Set up data storage
        data_log_header = ['total_reward']
        [data_log_header.append(x) for x in self.env.cs_labels]

        # Set up data lists
        self.training_rewards = []
        self.training_smoothed_rewards = []
        self.training_cs_level = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_value_log = []
        
        action_space = self.env.action_list

        self.GAMMA = self.env.settings['GAMMA']
        N_EPISODES = self.env.settings['N_EPISODES']
        for ep in range(N_EPISODES):
            self.env.reset()
            self.value_log = [] 

            self.schedule = None
            _planning_data = []

            self.step = 0 ## init step counter
            while True and self.step < 1e6:

                if self.env.call_scheduler == False:
                    continue
                    
                self.schedule, planning_data = network_scheduler(self.env,
                    self.policy_est, self.schedule)
   
                if planning_data is not None:
                    _planning_data.append(planning_data)

                if self.value_est is not None:
                    self.value_log.append(estimate_schedule_value(self.env, 
                        self.value_est, self.schedule))

                self.schedule = self.env.step(self.schedule)
                self.step += 1
                if self.env.sim_time >= self.env.n_steps:
                    break

            self.log_episode_data(_planning_data, ep)

            # Kill if stuck producing one product
            if len(np.unique(self.env.containers.actions)) == 1:
                break

            # UPDATE NETWORKS
            # =============================================================== 
            # Update gradients based on batch_size
            if ((ep + 1) % self.env.settings['BATCH_SIZE'] == 0 and ep > 0) or (
                ep == N_EPISODES - 1 and ep > 0):
                self.update_networks()

                # Save data since last batch
                data_since_last_batch = np.array(
                    self.training_rewards[-self.env.settings['BATCH_SIZE']:]).reshape(-1,1)
                cs_level_batch = np.vstack(self.training_cs_level)[-self.env.settings['BATCH_SIZE']:]
                data_since_last_batch = np.hstack((data_since_last_batch, 
                    cs_level_batch))

                log_data(data_since_last_batch, self.env.settings, self.env, data_log_header)
                self.log_policy_data()

                # Pickle planning data
                planning_data_file = self.env.settings['DATA_PATH'] + '/planning_data.pkl'
                data = open(planning_data_file, 'wb')
                pickle.dump(self.planning_data, data)
                data.close()

            # Test policy
            if ep % (N_EPISODES/5) == 0 or ep == N_EPISODES - 1:
                chkpt = 100 if ep == N_EPISODES - 1 else int(ep/N_EPISODES*100)
                self.test(chkpt)

            converged = check_for_convergence(self.training_rewards, ep, self.env.settings)

            # Print episode progress
            max_percentage = 100
            training_completion_percentage = int((ep + 1) / 
                self.env.settings['N_EPISODES'] * 100)

            if converged:
                print("Policy Converged")
                print("Episodes {:2d}\nMean Reward (last 100 episodes): {:2f}".format(
                    ep +1, self.training_smoothed_rewards[ep]))
                break

        # Save network parameters
        path = os.path.join(self.env.settings['DATA_PATH'])
        self.policy_est.saveWeights(path)
        if self.value_est is not None:
            path = os.path.join(self.env.settings['DATA_PATH'])
            self.value_est.saveWeights(path)

        return print("Network trained")

    def predict(self):
        print("Building schedule until {}".format(self.settings['END_TIME']))
        # Set up data storage
        data_log_header = ['total_reward']
        [data_log_header.append(x) for x in self.env.cs_labels]

        # Set up data lists
        self.training_rewards = []
        self.training_smoothed_rewards = []
        self.training_cs_level = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_value_log = []
        
        action_space = self.env.action_list

        self.value_log = []       
        
        # Loaded current schedule already
        self.schedule = self.env.schedule
        _planning_data = []

        self.step = 0 ## init step counter
        while True and self.step < 1e6:

            if self.env.call_scheduler == False:
                continue
                
            self.schedule, planning_data = network_scheduler(self.env,
                self.policy_est, self.schedule)
            
            if planning_data is not None:
                _planning_data.append(planning_data)

            if self.value_est is not None:
                self.value_log.append(estimate_schedule_value(self.env, 
                    self.value_est, self.schedule))

            self.schedule = self.env.step(self.schedule)
            self.step += 1
            if self.env.sim_time >= self.env.n_steps:
                break 

    def log_episode_data(self, data, episode):
        self.value_log = np.array(self.value_log)
        # Log batch and episode data at end of episode
        discounted_rewards = discount_returns(
            np.array(self.env.containers.total_reward),
            gamma=self.GAMMA)
        self.batch_rewards.extend(discounted_rewards)
        self.batch_actions.extend(self.env.containers.actions)
        self.batch_states.append(self.env.containers.predicted_state)
        self.batch_value_log.extend(self.value_log)            
        # Ensure planning data has correct dimensions 
        data = np.vstack(data)[:self.step]

        data = np.hstack([data, 
            self.env.containers.stack_values()[:self.step],
            self.value_log.reshape(-1, 1)[:self.step]][:self.step])
        
        if self.planning_data is None:
            self.planning_data = {}
            self.planning_data[episode] = np.dstack([data])
        else:
            self.planning_data[episode] = np.dstack([data])
        # Log planning headers if haven't already
        self.planning_data_headers, self.planning_data_indices = get_planning_data_headers(self.env)
        if self.value_est:
            self.planning_data_headers.append('value_log')
            self.planning_data_indices['value_log'] = max(self.planning_data_indices.values()) + 1

        
        cs_level_ep = self.env.get_cs_level()
        self.training_cs_level.append(cs_level_ep)
        if self.env.reward_function == 'OTD1':
            # Get the average daily otd score
            self.training_rewards.append(
                sum(self.env.containers.total_reward) / self.env.n_days)
        else:
            self.training_rewards.append(sum(self.env.containers.total_reward))
        self.training_smoothed_rewards.append(np.mean(
            self.training_rewards[-self.env.settings['BATCH_SIZE']:]))

    def update_networks(self):
        self.batch_states = np.vstack(self.batch_states)
        self.batch_rewards = np.vstack(self.batch_rewards).ravel()
        self.batch_actions = np.vstack(self.batch_actions).ravel()
        self.batch_value_log = np.vstack(self.batch_value_log).ravel()
        # Normalize rewards
        if 'VALUE' in self.env.settings['REWARD_FUNCTION']:
            try:
                self.batch_rewards = z_norm(self.batch_rewards, axis=0)
            except IndexError:
                print(self.batch_rewards.shape, type(self.batch_rewards))
                raise ValueError
        
        if self.value_est is not None:
            value_loss, value_grads = self.value_est.update(
                states=self.batch_states,
                returns=self.batch_rewards)

            self.batch_rewards = self.batch_rewards - self.batch_value_log

        current_probs = self.policy_est.predict(self.batch_states)
        
        loss, policy_loss, entropy_loss, policy_grads = self.policy_est.update(
                states=self.batch_states,
                actions=self.batch_actions,
                returns=self.batch_rewards)

        # Calculate KL Divergence
        new_probs = self.policy_est.predict(self.batch_states)
        new_probs = torchToNumpy(new_probs, device=self.policy_est.device)
        current_probs = torchToNumpy(current_probs, device=self.policy_est.device)
        kl = -np.sum(current_probs * np.log(new_probs / (current_probs + 1e-5)))
        if np.isnan(kl):
            kl = 0

        # Append loss values
        self.loss.append(np.mean(loss))
        self.policy_loss.append(np.mean(policy_loss))
        self.entropy_loss.append(np.mean(entropy_loss))
        self.value_loss.append(np.mean(value_loss))
        self.policy_grads.append(policy_grads)
        self.value_grads.append(value_grads)
        self.kl_div.append(kl)

        # Clear batch holders
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_value_log = []

    def log_policy_data(self):
        batch_size = self.settings['BATCH_SIZE']
        model_name = (self.settings['ENVIRONMENT'] + '_' + \
            self.settings['RL_ALGO'] + '_' + 'LOSS_DATA')
        file_name = self.settings['DATA_PATH'] + '/' + model_name + '.txt'
        val_names = ['total_loss', 'policy_loss', 'value_loss', 
            'entropy_loss', 'kl_divergence']
        policy_data = np.array(
            [self.loss[-batch_size:],
            self.policy_loss[-batch_size:],
            self.value_loss[-batch_size:],
            self.entropy_loss[-batch_size:],
            self.kl_div[-batch_size:]]).reshape(-1, len(val_names))

        if not exists(file_name):
            path = Path(self.settings['DATA_PATH'])
            path.mkdir(parents=True, exist_ok=True)
            with open(file_name, 'w') as file:
                # Write data headers
                if val_names is not None:
                    [file.write("{:s}\t".format(name)) for name in val_names]
                    file.write("\n")
                # Append data
                if policy_data.ndim < 2:
                    [file.write("{:f}\r".format(entry)) for entry in policy_data]
                if policy_data.ndim == 2:
                    for row in policy_data:
                        [file.write("{:f}\t".format(col)) for col in row]
                        file.write("\r")
                if policy_data.ndim > 2:
                    raise ValueError('More than 2 dimensions in data to be written.')
            file.close()
        else:
            with open(file_name, 'a') as file:
                # Append data
                if policy_data.ndim < 2:
                    [file.write("{:f}\r".format(entry)) for entry in policy_data]
                if policy_data.ndim == 2:
                    for row in policy_data:
                        [file.write("{:f}\t".format(col)) for col in row]
                        file.write("\r")
                if policy_data.ndim > 2:
                    raise ValueError('More than 2 dimensions in data to be written.')
            file.close()