# pt_networks.py : PyTorch implementations of netowrks to be run with the ada
# simulation library.
# Christian Hubbs
# christiandhubbs@gmail.com
# 11.07.2018

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
import os
from warnings import warn
from ..rl_algos.rl_utils import check_device_settings, torchToNumpy

class policyEstimator(nn.Module):
    def __init__(self, env, settings):
        super(policyEstimator, self).__init__()

        self.device = check_device_settings(settings)
        self.n_inputs = env.observation_space.shape[0]
        try:
            self.n_outputs = env.action_space.n 
        except AttributeError:
            self.n_outputs = len(env.action_list)
        self.n_hidden_nodes = settings['N_HIDDEN_NODES']
        if settings['ACTOR_LR'] <= 0:
            self.learning_rate = settings['LEARNING_RATE']
            warn("LEARNING_RATE will be disabled in a future version." + \
                 " Please use ACTOR_LR to avoid warning in future.",
                 DeprecationWarning)
        else:
            self.learning_rate = settings['ACTOR_LR']
        self.grad_clip = settings['GRADIENT_CLIPPING']
        self.clip = 0.1
        self.bias = settings['BIAS']
        self.beta = settings['BETA']
        self.action_space = np.arange(self.n_outputs)

        # Double layers to account for activation functions
        self.n_hidden_layers = 2 * settings['N_HIDDEN_LAYERS']
        self.layers = OrderedDict()
        for i in range(self.n_hidden_layers + 1):
            # Define linear network with no hidden layers
            if self.n_hidden_layers == 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs, 
                    self.n_outputs,
                    bias=self.bias)
            # Define input layer for multi-layer network
            elif i % 2 == 0 and i == 0 and self.n_hidden_layers != 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs, 
                    self.n_hidden_nodes,
                    bias=self.bias)
            # Define intermediate layers
            elif i % 2 == 0 and i < self.n_hidden_layers:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_hidden_nodes,
                    bias=self.bias)
            # Define ouput linear layer
            elif i % 2 == 0 and i == self.n_hidden_layers:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_outputs,
                    bias=self.bias)
            # Odd layers are activation functions
            else:
                self.layers[str(i)] = nn.ReLU()

            self.net = nn.Sequential(self.layers)
            if self.device == 'cuda':
                self.net.cuda()
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                lr=self.learning_rate)

    def get_logits(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        return self.net(state_t)

    def predict(self, state):
        logits = self.get_logits(state)
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        action_probs = self.predict(state).detach().numpy()
        return np.random.choice(self.action_space, p=action_probs)

    def calc_loss(self, states, actions, returns):
        returns_t = torch.FloatTensor(returns).to(self.device)
        actions_t = torch.LongTensor(np.array(actions) - 1).to(self.device).reshape(-1, 1)
        log_probs = F.log_softmax(self.get_logits(states), dim=-1)
        log_prob_actions = returns_t * torch.gather(log_probs, 1, actions_t).squeeze()
        p_loss = -log_prob_actions.mean()
        action_probs = self.predict(states)
        e_loss = -self.beta * (action_probs * log_probs).sum(dim=1).mean()
        return p_loss, e_loss

    def update(self, states, actions, returns):
        self.optimizer.zero_grad()
        p_loss, e_loss = self.calc_loss(states, actions, returns)
        loss = p_loss - e_loss
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.optimizer.step()
        grads = [g.grad.data for g in self.net.parameters()]
        total_loss = torchToNumpy(loss, device=self.device)
        p_loss = torchToNumpy(p_loss, device=self.device)
        e_loss = torchToNumpy(e_loss, device=self.device)
        return total_loss, p_loss, e_loss, grads

    def saveWeights(self, path):
        os.makedirs(path, exist_ok=True)
        # self.logger.debug(path)
        torch.save(self.net.state_dict(), path + '/actor.pt')

class valueEstimator(nn.Module):
    def __init__(self, env, settings):
        super(valueEstimator, self).__init__()

        self.device = check_device_settings(settings)
        self.n_inputs = env.observation_space.shape[0]
        
        self.n_outputs = 1
        self.n_hidden_nodes = settings['N_HIDDEN_NODES']
        if settings['CRITIC_LR'] <= 0:
            self.learning_rate = settings['LEARNING_RATE']
            warn("LEARNING_RATE will be disabled in a future version." + \
                 " Please use CRITIC_LR to avoid warning in future.",
                 DeprecationWarning)
        else:
            self.learning_rate = settings['CRITIC_LR']
        self.grad_clip = settings['GRADIENT_CLIPPING']
        self.clip = 0.1
        self.bias = settings['BIAS']
        self.beta = settings['BETA']
        self.action_space = np.arange(self.n_outputs)

        # Double layers to account for activation functions
        self.n_hidden_layers = 2 * settings['N_HIDDEN_LAYERS']
        self.layers = OrderedDict()
        for i in range(self.n_hidden_layers + 1):
            # Define linear network with no hidden layers
            if self.n_hidden_layers == 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs, 
                    self.n_outputs,
                    bias=self.bias)
            # Define input layer for multi-layer network
            elif i % 2 == 0 and i == 0 and self.n_hidden_layers != 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs, 
                    self.n_hidden_nodes,
                    bias=self.bias)
            # Define intermediate layers
            elif i % 2 == 0 and i < self.n_hidden_layers:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_hidden_nodes,
                    bias=self.bias)
            # Define ouput linear layer
            elif i % 2 == 0 and i == self.n_hidden_layers:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_outputs,
                    bias=self.bias)
            # Odd layers are activation functions
            else:
                self.layers[str(i)] = nn.ReLU()

            self.net = nn.Sequential(self.layers)
            if self.device == 'cuda':
                self.net.cuda()
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                lr=self.learning_rate)

    def predict(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        return self.net(state_t)

    def calc_loss(self, states, returns):
        returns = torch.FloatTensor(returns).to(self.device)
        value = self.predict(states).view(-1)
        return nn.MSELoss()(value, returns)

    def update(self, states, returns):
        self.optimizer.zero_grad()
        loss = self.calc_loss(states, returns)
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.optimizer.step()
        grads = [g.grad.data for g in self.net.parameters()]
        loss = torchToNumpy(loss, device=self.device)
        return loss, grads

    def saveWeights(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), path + '/critic.pt')

class deepQNet(nn.Module):
    def __init__(self, env, n_hidden_layers=0, n_hidden_nodes=4,
    			 learning_rate=0.01, grad_clip=False, bias=False,
                 device='cpu'):
        super(deepQNet, self).__init__()
        
        self.device = check_device_settings(settings)
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.n_hidden_nodes = n_hidden_nodes
        # Add extra layer to account for activation function
        self.n_hidden_layers = 2 * n_hidden_layers
        self.layers = OrderedDict()
        for i in range(self.n_hidden_layers + 1):
            # Define input layer
            if self.n_hidden_layers == 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs,
                    self.n_outputs,
                    bias=self.bias)
            elif i % 2 == 0 and i == 0 and self.n_hidden_layers != 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs,
                    self.n_hidden_nodes,
                    bias=self.bias)
            elif i % 2 == 0 and i < self.n_hidden_layers:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_hidden_nodes,
                    bias=self.bias)
            elif i % 2 == 0 and i == self.n_hidden_layers:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_outputs,
                    bias=self.bias)
            else:
                self.layers[str(i)] = nn.ReLU()
                
            self.net = nn.Sequential(self.layers)
            self.target_net = copy.deepcopy(self.net)

            self.optimizer = torch.optim.Adam(self.net.parameters(),
            	lr=self.learning_rate)
            
    def predict(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        return self.net(state_t)

    def calc_loss(self, batch, gamma):
    	states, actions, rewards, dones, next_states = batch
    	states_t = torch.FloatTensor(states).to(self.device)
    	next_states_t = torch.FloatTensor(next_states).to(self.device)
    	actions_t = torch.LongTensor(actions).to(self.device)
    	rewards_t = torch.FloatTensor(rewards).to(self.device)
    	done_mask = torch.ByteTensor(dones).to(self.device)

    	state_action_vals = self.net.predict(states_t).gather(
    		1, actions_t.unsqueeze(-1)).squeeze(-1)
    	next_state_vals = self.target_net.predict(next_states_t).max(1)[0]

    	next_state_vals[done_mask] = 0
    	next_state_vals = next_state_vals.detach()
    	expected_values = next_state_vals * gamma + rewards_t
    	loss = nn.MSELoss()(state_action_vals, expected_values)
    	return loss

    def update(self, batch, gamma):
    	self.optimizer.zero_grad()
    	loss = self.calc_loss(batch, gamma)
    	loss.backward()
    	self.optimzer.step()
    	return loss
