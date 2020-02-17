# mip: contains Mixed-Integer Programming models used to solve the tartan
# scheduling environment on a rolling horizon.
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

def buildDeterministicMIPRH(env, schedule=None, *args, **kwargs):
    # Build model
    m = ConcreteModel()

    # Import settings values
    obj_funcs = ['prod_availability', 'revenue']
    if env.settings['REWARD_FUNCTION'] not in obj_funcs:
        reward = 'FULL'
    else:
        reward = env.settings['REWARD_FUNCTION']
    m.alpha = env.settings['LATE_PENALTY']

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
    m.order_book = subset_orderbook(env)
    unique_gmid, gmid_locs, gmid_counts = np.unique(
        env.order_book[:, env.ob_indices['gmid']],
        return_inverse=True,
        return_counts=True)
    mean_values = np.bincount(gmid_locs,
        env.order_book[:,env.ob_indices['var_std_margin']]) / gmid_counts
    m.beta = {gmid: mean_values[idx] for idx, gmid in enumerate(m.gmids)}
    m.order_dict, m.order_cols = build_order_dict(env, m.order_book)
    m.order_dict_disc = discount_order_dict(m)

    ################### SETS #############################################
    # Adding -1 to the time index because we need to know the transition 
    # delay + 1 periods ago to get the off-grade from that transition state.
    # Means that first OG entry is invalid. May be other way to fix this, but
    # it seems this is reasonably simple.
    # m.t = RangeSet(env.sim_time - m.delay - 1, m.K) # Schedule time intervals t
    m.t = RangeSet(env.sim_time - m.delay, m.K) # Schedule time intervals t
    m.h = RangeSet(env.sim_time, env.sim_time + m.H) # Fixed time intervals h
    m.i = Set(initialize=m.gmids)
    m.j = Set(initialize=m.production_states)
    m.k = Set(initialize=m.production_states)
    order_numbers = list(set(m.order_dict.keys()))
    m.n = Set(initialize=order_numbers)

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
    m.x = Var(m.i, m.t, m.n, within=Binary)
    m.y = Var(m.j, m.t, within=Binary)
    m.z = Var(m.k, m.j, m.t, within=Binary)

    # Continuous Variables
    m.p = Var(m.j, m.t, within=NonNegativeReals) # Production QTY
    m.og = Var(m.j, m.t, within=NonNegativeReals) # OG Production
    m.inventory = Var(m.i, m.t, within=NonNegativeReals) # Inventory
    m.sales = Var(m.i, m.t, m.n, within=NonNegativeReals) # Sales
    m.inventory_cost = Var(m.t, within=NonNegativeReals)
    # Fix binary variables if schedule is passed    
    m = convert_schedule_to_vars(env, m, schedule=schedule)
    
    ################## CONSTRAINTS #########################################
    @m.Constraint(m.i, m.t)
    def inventory_constraint(m, i, t):
        if t == min(m.t):
            return m.inventory[i, t] - m.inventory_init[i] \
                + sum(m.sales[i, t, n] for n in m.n) == 0
        elif t > min(m.t) and t < min(m.t) + m.delay:
            return m.inventory[i, t] - m.inventory[i, t-1] \
                + sum(m.sales[i, t, n] for n in m.n) == 0
        elif t >= min(m.t) + m.delay:
            return m.inventory[i, t] - m.inventory[i, t-1] \
                - m.p[i, t-m.delay] + sum(m.sales[i, t, n]
                    for n in m.n) == 0

    @m.Constraint(m.t)
    def production_assignment_constraint(m, t):
        return sum(m.y[j, t] for j in m.j) == 1

    @m.Constraint(m.j, m.t)
    def production_qty_constraint(m, j, t):
        if j != 0:
            return m.p[j, t] - m.run_rates[j] * m.y[j, t] + m.og[j, t] == 0
        else:
            return m.p[j, t] - 0 * m.y[j, t] == 0

    @m.Constraint(m.i, m.t, m.n)
    def sales_qty_constraint(m, i, t, n):
        if t >= m.sim_time:
            planned_gi_time = m.order_dict_disc[n][t][
                m.order_cols.index('planned_gi_time')]
            order_qty = m.order_dict_disc[n][t][
                m.order_cols.index('order_qty')]
            gmid = m.order_dict_disc[n][t][
                m.order_cols.index('gmid')]
            if i == gmid and t >= planned_gi_time:
                return m.sales[i, t, n] - order_qty * m.x[i, t, n] == 0
        return m.sales[i, t, n] == 0

    @m.Constraint(m.n)
    def shipment_constraint(m, n):
        return sum(m.x[i, t, n] for i in m.i for t in m.t if t >= env.sim_time) <= 1

    # Transition constraints
    # Note we transition from j -> k
    @m.Constraint(m.j, m.t)
    def transition_constraint1(m, j, t):
        return sum(m.z[k, j, t] for k in m.k) - m.y[j, t] == 0

    @m.Constraint(m.k, m.t)
    def transition_constraint2(m, k, t):
        if t > min(m.t):
            return sum(m.z[k, j, t] for j in m.j) - m.y[k, t-1] == 0
        else:
            return Constraint.NoConstraint

    @m.Constraint(m.t)
    def transition_assignment_constraint(m, t):
        return sum(m.z[k, j, t] for j in m.j for k in m.k) == 1

    # Note we transition from j -> k
    @m.Constraint(m.k, m.t)
    def transition_loss_constraint(m, k, t):
        return sum(m.transition_losses[k, j] * m.z[j, k, t]
            for j in m.j) - m.og[k, t] == 0

    @m.Constraint(m.t)
    def inv_cost_calculation(m, t):
        return sum(m.beta[i]*m.inventory[i, t]
            for i in m.i)*m.working_capital_per - m.inventory_cost[t] == 0

    if reward == 'FULL':
        m.reward = Objective(expr=
            sum(
                m.sales[i, t, n] * m.order_dict_disc[n][t][
                m.order_cols.index('var_std_margin')]
                for i in m.i 
                for t in m.t if t >= env.sim_time
                for n in m.n) - \
            m.working_capital_per * sum(
                m.beta[i] * m.inventory[i, t] 
                for i in m.i
                for t in m.t if t >= env.sim_time),
            sense=maximize)
    else:
        raise ValueError(
            'Reward function setting {} not defined'.format(reward))
   
    return m