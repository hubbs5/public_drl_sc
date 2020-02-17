# stochastic_mip: contains Mixed-Integer Programming models used to 
# solve the tartan environment. Solves over multiple scenarios to
# determine the best solution among the possible scenarios and implements
# said solution.
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 01.08.2018

# Update: 14.02.2019: Introduce new constraints to tighten model and linearize
#   lateness factor.
# Update: 03.09.2018: Properly take care of offgrade by indexing parameters
#   by GMID value.
from pyomo.environ import *
import numpy as np 
import copy
from .mip_utils import *

def buildStochasticMIP(env, schedule=None, *args, **kwargs):
    # Build model
    m = ConcreteModel()

    # Import settings values
    obj_funcs = ['prod_availability', 'revenue']
    if env.settings['REWARD_FUNCTION'] not in obj_funcs:
        reward = 'FULL'
    else:
        reward = env.settings['REWARD_FUNCTION']
    m.alpha = env.settings['LATE_PENALTY']
    m.n_scenarios = 20

    # Parameters
    # Fixed planning horizon
    m.H = env.fixed_planning_horizon 
    m.delay = 2 # Time to produce product + curing time
    # Extended horizon to avoid poor end states. Minimum of 1 week.
    m.K = max(env.sim_time + 2 * m.H, 
        env.sim_time + 7)
    m.I = env.n_products
    m.gmids = env.product_data[:, env.prod_data_indices['gmid']].astype('int')
    m.production_states = np.insert(m.gmids, 0, 0) # Make state = 0 idle
    m.fixed_horizon = env.sim_time + m.H
    m.working_capital_per = env.working_capital_per
    m.sim_time = copy.copy(env.sim_time)

    # Create order dictionary for easy indexing with Pyomo
    m.order_book = subset_orderbook(env).copy()
    unique_gmid, gmid_locs, gmid_counts = np.unique(
        env.order_book[:, env.ob_indices['gmid']],
        return_inverse=True,
        return_counts=True)
    mean_values = np.bincount(gmid_locs,
        env.order_book[:,env.ob_indices['var_std_margin']]) / gmid_counts
    m.beta = {gmid: mean_values[idx] for idx, gmid in enumerate(m.gmids)}
    m.order_dict, m.order_cols = build_stochastic_order_dict(env, 
        n_scenarios=m.n_scenarios)
    m.order_dict_disc = discount_order_dict(m)
    # Get list of keys to avoid key errors
    m.order_keys = list(m.order_dict_disc.keys())

    ################### SETS #############################################
    # Adding -1 to the time index because we need to know the transition 
    # delay + 1 periods ago to get the off-grade from that transition state.
    # Means that first OG entry is invalid. May be other way to fix this, but
    # it seems this is reasonably simple.
    m.t = RangeSet(env.sim_time - m.delay, m.K) # Schedule time intervals t
    m.h = RangeSet(env.sim_time, env.sim_time + m.H) # Fixed time intervals h
    m.i = Set(initialize=m.gmids)
    m.j = Set(initialize=m.production_states)
    m.k = Set(initialize=m.production_states)
    m.n = Set(initialize=np.unique(np.vstack(m.order_keys)[:,1]))
    m.s = RangeSet(0, m.n_scenarios-1)

    m.gmid_to_index = {i: idx for idx, i in enumerate(m.i)}
    m.index_to_gmid = {idx: i for idx, i in enumerate(m.i)}
    
    m.inventory_init = Param(m.gmids, 
        initialize=get_initial_inventory_dict(env))
    m.run_rates = get_run_rate_dict(env)

    m.transition_losses = {(j, i): env.transition_matrix[idx, jdx] 
        for idx, i in enumerate(m.production_states)
        for jdx, j in enumerate(m.production_states)}
    
    ################### VARIABLES #########################################
    # Binary Variables
    m.x = Var(m.i, m.t, m.n, m.s, within=Binary)
    m.y = Var(m.j, m.t, m.s, within=Binary)
    m.z = Var(m.k, m.j, m.t, m.s, within=Binary)

    # Continuous Variables
    m.p = Var(m.j, m.t, m.s, within=NonNegativeReals) # Production QTY
    m.og = Var(m.j, m.t, m.s, within=NonNegativeReals) # OG Production
    m.inventory = Var(m.i, m.t, m.s, within=NonNegativeReals) # Inventory
    m.sales = Var(m.i, m.t, m.n, m.s,  within=NonNegativeReals) # Sales
    m.inventory_cost = Var(m.t, m.s, within=NonNegativeReals)
    m.scenario_reward = Var(m.s, within=Reals) # Placeholder for rewards
    # Fix binary variables if schedule is passed    
    m = convert_schedule_to_stochastic_vars(env, m, schedule=schedule)
    
    ################## CONSTRAINTS #########################################
    @m.Constraint(m.i, m.t, m.s)
    def inventory_constraint(m, i, t, s):
        if t == min(m.t):
            return m.inventory[i, t, s] - m.inventory_init[i] \
                + sum(m.sales[i, t, n, s] for n in m.n) == 0
        elif t > min(m.t) and t < min(m.t) + m.delay:
            return m.inventory[i, t, s] - m.inventory[i, t-1, s] \
                + sum(m.sales[i, t, n, s] for n in m.n) == 0
        elif t >= min(m.t) + m.delay:
            return m.inventory[i, t, s] - m.inventory[i, t-1, s] \
                - m.p[i, t-m.delay, s] + sum(m.sales[i, t, n, s]
                    for n in m.n) == 0

    @m.Constraint(m.t, m.s)
    def production_assignment_constraint(m, t, s):
        return sum(m.y[j, t, s] for j in m.j) == 1

    @m.Constraint(m.j, m.t, m.s)
    def production_qty_constraint(m, j, t, s):
        if j != 0:
            return m.p[j, t, s] - m.run_rates[j] * m.y[j, t, s] + m.og[j, t, s] == 0
        else:
            return m.p[j, t, s] - 0 * m.y[j, t, s] == 0

    @m.Constraint(m.i, m.t, m.n, m.s)
    def sales_qty_constraint(m, i, t, n, s):
        # Not all orders exist across all scenarios because of stoch nature 
        # of forecast. Check order_keys first
        if t >= m.sim_time:
            if (s, n) in m.order_keys:
                planned_gi_time = m.order_dict_disc[(s,n)][t][
                    m.order_cols.index('planned_gi_time')]
                order_qty = m.order_dict_disc[(s,n)][t][
                    m.order_cols.index('order_qty')]
                gmid = m.order_dict_disc[(s,n)][t][
                    m.order_cols.index('gmid')]
                if i == gmid and t >= planned_gi_time:
                    return m.sales[i, t, n, s] - order_qty * m.x[i, t, n, s] == 0
        return m.sales[i, t, n, s] == 0

    @m.Constraint(m.n, m.s)
    def shipment_constraint(m, n, s):
        return sum(m.x[i, t, n, s] for i in m.i for t in m.t if t >= env.sim_time) <= 1

    # Transition constraints
    # Note we transition from j -> k
    @m.Constraint(m.j, m.t, m.s)
    def transition_constraint1(m, j, t, s):
        return sum(m.z[k, j, t, s] for k in m.k) - m.y[j, t, s] == 0

    @m.Constraint(m.k, m.t, m.s)
    def transition_constraint2(m, k, t, s):
        if t > min(m.t):
            return sum(m.z[k, j, t, s] for j in m.j) - m.y[k, t-1, s] == 0
        else:
            return Constraint.NoConstraint

    @m.Constraint(m.t, m.s)
    def transition_assignment_constraint(m, t, s):
        return sum(m.z[k, j, t, s] for j in m.j for k in m.k) == 1

    # Note we transition from j -> k
    @m.Constraint(m.k, m.t, m.s)
    def transition_loss_constraint(m, k, t, s):
        return sum(m.transition_losses[k, j] * m.z[j, k, t, s]
            for j in m.j) - m.og[k, t, s] == 0

    @m.Constraint(m.t, m.s)
    def inv_cost_calculation(m, t, s):
        return sum(m.beta[i]*m.inventory[i, t, s]
            for i in m.i)*m.working_capital_per - m.inventory_cost[t, s] == 0

    #add NACs
    # if schedule is not None:
    @m.Constraint(m.j, m.t, m.s)
    def nac(m, j, t, s):
        if t <= env.sim_time + m.H and s < m.n_scenarios - 1 :
            return m.y[j, t, s] == m.y[j, t, s+1]
        else:
            return Constraint.NoConstraint
 
    # Break rewards down by scenario
    @m.Constraint(m.s)
    def scenario_reward_constraint(m, s):
        return m.scenario_reward[s] - sum(
            m.sales[i, t, n, s] *
            m.order_dict_disc[(s, n)][t][m.order_cols.index('var_std_margin')]
                for i in m.i
                for t in m.t if t >= env.sim_time
                for n in m.n if (s, n) in m.order_keys) + \
            m.working_capital_per * sum(
                m.beta[i] * m.inventory[i, t, s]
                for i in m.i
                for t in m.t if t >= env.sim_time) == 0

    if reward == 'FULL':
        m.reward = Objective(expr=
            1 / m.n_scenarios * sum(m.scenario_reward[s] for s in m.s),
            sense=maximize)
        # m.reward = Objective(expr=
        #     (1 / m.n_scenarios * \
        #     sum(
        #         sum(
        #         m.sales[i, t, n, s] * m.order_dict_disc[(s, n)][t][
        #         m.order_cols.index('var_std_margin')]
        #         for i in m.i 
        #         for t in m.t if t >= env.sim_time
        #         for n in m.n if (s, n) in m.order_keys) - \
        #     m.working_capital_per * sum(
        #         m.beta[i] * m.inventory[i, t, s] 
        #         for i in m.i
        #         for t in m.t if t >= env.sim_time) for s in m.s)),
        #     sense=maximize)
    else:
        raise ValueError('Reward function setting {} not defined'.format(reward))
   
    return m