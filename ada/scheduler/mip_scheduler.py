# milp_scheduler: a set of functions that allows a schedule to be generated
# generated from a Pyomo output
# Christian Hubbs
# 30.07.2018

import numpy as np 
from pyomo.opt import SolverFactory
import imageio
import pickle
import cloudpickle # Works for Pyomo Objects
import os

from .schedule_utils import build_schedule
from ..agents.mip_algos.mip_utils import *

class schedulingMPC():
    
    def __init__(self, env, model_function, schedule=None, *args, **kwargs):
        self.env = env
        # TODO: Eliminate passing settings multiple times through model
        self.settings = env.settings
        self.build_model = model_function
        # Placeholder for solved model
        self.m_solved = None
        self.solved_models = []
        self.model_results = []
        self.episodes = self.env.n_steps
        self.schedule = schedule
        
    def train(self):
        
        # Set up data lists
        self.time_stamps = []

        if 'GOD' in self.env.settings['MIP_ALGO']:
            m = self.build_model(self.env, schedule=self.schedule)
            self.schedule, self.m_solved, self.results = build_milp_schedule(self.env, m,
                schedule=self.schedule, settings=self.settings)
            print("\nSolution for simulation horizon found.")
            for t in range(self.env.n_steps):
                self.time_stamps.append(self.env.sim_time)
                self.schedule = self.env.step(self.schedule)
                self.solved_models.append(self.m_solved)
                self.model_results.append(self.results)        
        else:
            for t in range(self.env.n_steps):
                m = self.build_model(self.env, schedule=self.schedule)
                if self.env.settings['MIP_ALGO'] == 'SMPC':
                    self.schedule, self.m_solved, self.results = build_smpc_schedule(
                        self.env, m, schedule=self.schedule, settings=self.settings)
                else:
                    self.schedule, self.m_solved, self.results = build_milp_schedule(
                        self.env, m, schedule=self.schedule, settings=self.settings)
                print("\nSolution from Day: {:d}".format(self.env.sim_time))
                self.time_stamps.append(self.env.sim_time)
                self.schedule = self.env.step(self.schedule)
                # if 'SMPC' not in self.env.settings['MIP_ALGO']:
                    # self.solved_models.append(self.m_solved)
                self.model_results.append(self.results)

        # Save environment data
        # TODO: Change saving functions to an option
        self.planning_data = self.env.containers.stack_values(limit=self.env.n_days)
        model_name = str(self.env.settings['MIP_ALGO'] + ' with ' + str(self.env.n_products) + \
            'prods')
        # planning_data_file = self.env.settings['DATA_PATH'] + '/planning_data_' + \
        #     model_name.replace(' ', '_') + '.pkl'
        # pickle.dump(self.planning_data, open(planning_data_file, 'wb'))
 
        # # Pickle planning environment
        # planning_env_file = self.env.settings['DATA_PATH'] + '/env_' + \
        #     model_name.replace(' ', '_') + '.pkl'
        # pickle.dump(self.env, open(planning_env_file, 'wb'))

        # Pickle MPC solutions
        agent_file = self.env.settings['DATA_PATH'] + '/agent_' + \
            model_name.replace(' ', '_') + '.pkl'
        if os.path.exists(os.path.basename(agent_file)) == False:
            os.makedirs(os.path.basename(agent_file), exist_ok=True)
        if self.env.settings['MIP_ALGO'] != 'SMPC':
            cloudpickle.dump(self, open(agent_file, 'wb'))

    def generate_gifs(self):
        plots = ['gantt', 'inventory', 'sales', 'shipment']
        for model, time in zip(self.solved_models, self.time_stamps):
            self.plot_gantt(model, time, save=True)
            self.plot_inventory(model, time, save=True)
            self.plot_sales(model, time, save=True)
            self.plot_shipments(model, time, save=True)

            # Build GIF at last time step
            if time == self.env.n_steps - 1:
                for plot_type in plots:
                    files = [settings['DATA_PATH'] + 
                        '/MIP_{:s}_{:d}.png'.format(plot_type, x) 
                        for x in range(1, self.episodes + 1)]
                    images = []
                    for file in files:
                        images.append(imageio.imread(file))
                    imageio.mimsave(self.settings['DATA_PATH'] + \
                        '/{:s}_animation.gif'.format(plot_type),
                        images, duration=0.7)
            
    def plot_gantt(self, model, time_step=None, color_scheme=None, save=False):
        if self.m_solved is None:
            raise ValueError('Solve model by running .train() before viewing the plot.')
        mip_gantt_plot(self.env, model, time_step, color_scheme=color_scheme, 
            save=save, path=self.settings['DATA_PATH'])
        
    def plot_inventory(self, model, time_step=None, color_scheme=None, save=False):
        if self.m_solved is None:
            raise ValueError('Solve model by running .train() before viewing the plot.')
        mip_inventory_plot(self.env, model, time_step,
            color_scheme=color_scheme, save=save, path=self.settings['DATA_PATH'])
        
    def plot_sales(self, model, time_step=None, color_scheme=None, save=False):
        if self.m_solved is None:
            raise ValueError('Solve model by running .train() before viewing the plot.')
        mip_sales_plot(self.env, model, time_step, 
            color_scheme=color_scheme, save=save, path=self.settings['DATA_PATH'])
        
    def plot_shipments(self, model, time_step=None, color_scheme=None, save=False):
        if self.m_solved is None:
            raise ValueError('Solve model by running .train() before viewing the plot.')
        mip_shipment_totals_plot(self.env, model, time_step,
            color_scheme=color_scheme, save=save, path=self.settings['DATA_PATH'])

def build_milp_schedule(env, model, schedule=None, settings=None, *args, **kwargs):
    model, results = solve_milp_scheduler(env, model, schedule=schedule)
    schedule = build_schedule(env, get_action_from_milp_scheduler, 
        schedule, model=model)
    return schedule, model, results

def build_smpc_schedule(env, model, schedule=None, settings=None, *args, **kwargs):
    model, results = solve_milp_scheduler(env, model, schedule=schedule)
    schedule = build_schedule(env, get_action_from_smip_scheduler, 
        schedule, model=model)
    return schedule, model, results

# def get_action_from_milp_scheduler(env, model, schedule=None, 
#         planning_time=None, *args, **kwargs):
#     #model = solve_milp_scheduler(env, model, schedule)
#     production = getattr(model, 'y')
#     if planning_time is None:
#         planning_time = env.sim_time
#     prod_plan = np.zeros((env.sim_time + model.K + 1))
#     # Cycle through Pyomo production results
#     # Not that it is indexed by product i and time interval, t
#     for idx in production:
#         if production[idx].value == 1:
#             # TODO: add action_dict_indices to env to reference that
#             # value here instead of the raw index
#             prod_plan[idx[1]] = env.action_dict[idx[0]][2]
#     action = prod_plan[int(planning_time)]
#     return action

def get_action_from_milp_scheduler(env, model, planning_time=None,
    *args, **kwargs):
    production = getattr(model, 'y')
    for idx in production:
        if planning_time == idx[1] and production[idx].value == 1:
            action = env.gmid_action_map[idx[0]]
            return action

# Define function to extract scenario closest to expected value
def get_smip_scenario(model):
    scen_rew = getattr(model, 'scenario_reward')
    rewards = np.zeros(model.n_scenarios)
    for i, idx in enumerate(scen_rew):
        rewards[i] = scen_rew[idx].value

    # Get value closest to objective
    scenario = np.argmin(np.abs(rewards - model.reward.expr()))
    return scenario

def get_action_from_smip_scheduler(env, model, planning_time=None, 
    *args, **kwargs):
    scenario = get_smip_scenario(model)
    production = getattr(model, 'y')
    for idx in production:
        if idx[-1] == scenario:
            if planning_time == idx[1] and production[idx].value == 1:
                action = env.gmid_action_map[idx[0]]
                return action

def solve_milp_scheduler(env, model, schedule=None, 
    print_output=False, *args, **kwargs):
    settings = env.settings
    if settings is None or settings['SOLVER'] is None:
        print('Defaulting to GLPK solver.')
        solver = 'glpk'
    else:
        solver = settings['SOLVER'].lower()

    model_solver = SolverFactory(solver)

    if solver == 'glpk' or solver is None:
        if settings['GAP'] is not None:
            model_solver.options['mipgap'] = settings['GAP']
        if settings['TIME_LIMIT'] is not None:
            model_solver.options['tmlim'] = settings['TIME_LIMIT']
    elif solver == 'cplex':
        if settings['GAP'] is not None:
            model_solver.options['absmipgap'] = settings['GAP']
    elif solver == 'gurobi':
        if settings['GAP'] is not None:
            model_solver.options['MIPGap'] = settings['GAP']

        model_solver.options['logfile'] = 'gurobi_log.txt'
    
    if solver == 'gams':
        results = model_solver.solve(model, tee=print_output, solver='cplex')
    else:
        results = model_solver.solve(model, tee=False)
        
    return model, results
