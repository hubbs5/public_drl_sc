# Christian Hubbs
# 29.01.2018

# Update 16.05.2018
# Adjust the inventory portion of the state representation to be the inventory
# divided by the order required within the planning horizon.

# This function takes a network as an argument to generate schedules.

import numpy as np
from copy import copy
from ada.agents.rl_algos.rl_utils import torchToNumpy
from ada.environments.env_utils import get_current_state

def build_network_schedule(env, network, schedule=None):
    try:
        schedule = network_scheduler(env, network, schedule)
    except ValueError:
        schedule = network_scheduler(env, network, None)

    return schedule

# State prediction to determine what the state will look like at a future date
# based solely on information available in the order book and the current 
# schedule. This could be improved in the future to include ML/statistical
# predictions rather than simply a summation of orders in the books.
def predict_state(env, schedule=None):
    # Get copy of state information
    inv_prediction = env.inventory.copy()
    # Get the last scheduled day
    last_scheduled_hour = env.sim_time
    if schedule is not None:
        last_scheduled_hour = schedule[-1, 
            env.sched_indices['prod_start_time']]
        # Extract unbooked production up to the schedule limit
        pred_production = schedule[np.where(
                (schedule[:, 
                    env.sched_indices['cure_end_time']]<=last_scheduled_hour) & 
                (schedule[:, 
                    env.sched_indices['booked_inventory']]==0))]
        # Sum scheduled, unbooked production
        un_prod, un_prod_id = np.unique(pred_production[:, 
            env.sched_indices['gmid']], return_inverse=True)
        prod_pred_qty = np.bincount(un_prod_id, 
                                    pred_production[:, 
                                    env.sched_indices['prod_qty']])
        # Sum off-grade production
        pred_og_prod = pred_production[:, 
        env.sched_indices['off_grade_production']].sum()

        # Add scheduled production
        inv_prediction[un_prod.astype('int')] += prod_pred_qty
        # Add off-grade
        inv_prediction[0] += pred_og_prod
    
    # Aggregate orders on the books
    # Filtering by sim_time ensures all orders are already entered
    # Filtering by last_scheduled_hour gives an inventory prediction
    # for that specific day.
    pred_orders = env.order_book[np.where(
            (env.order_book[:, 
                env.ob_indices['doc_create_date']]<=env.sim_time) & 
            (env.order_book[:, 
                env.ob_indices['planned_gi_date']]<=last_scheduled_hour) &
            (env.order_book[:,
                env.ob_indices['shipped']]==0))]
   
    un_order, un_order_id = np.unique(pred_orders[:, 
        env.ob_indices['material_code']], return_inverse=True)
    order_pred_qty = np.bincount(un_order_id, 
        pred_orders[:, env.ob_indices['order_qty']])
    
    # Calculate state prediction
    # Note: this formulation ignores off-grade as the state

    state_prediction = np.array([inv_prediction[i] / order_pred_qty[k] for 
                                 k, i in enumerate(un_order)])
    state_pred = np.zeros(env.n_products)
    # Subtract 1 from the index to ignore off-grade levels
    state_pred[un_order-1] += state_prediction

    # Include product to be produced in state prediction as one-hot vector
    one_hot = np.zeros(env.n_products + 1)
    if schedule is None:
        current_prod = 0
    # NOTE: Should current_prod be set on prod_end_time or prod_start_time?
    else:
        current_prod = schedule[
            schedule[:, 
            env.sched_indices['prod_end_time']]==last_scheduled_hour,
            env.sched_indices['gmid']].astype('int')
        # Check to see if there is nothing scheduled i.e. in case of 
        # shut-down or start-up
        if current_prod.size == 0:
            current_prod = 0
    one_hot[current_prod] = 1
    
    state = np.hstack([one_hot, state_pred])
    
    return state

# Generate schedule from policy network
def network_scheduler(env, network, schedule, confidence_level=None, test=False):
    '''
    Inputs
    =========================================================================
    env: productionFacility object
    network: policy network object
    schedule: numpy array containing schedule
    confidence_level: float or None. If the max probability that comes 
        from the policy network is below the confidence_level, then the 
        schedule defaults to the heuristic.
    '''
    a_probs = []
    heuristic_selection = []
    predicted_state = []

    # TODO: TEMPORARY BUG FIX!
    # Sometimes schedule is an empty array and not None.
    # Check here to see if it is an empty array, if so, set schedule = None
    # In the future, see what is causing this problem upstream
    # Note that this fails with shut_downs. Gives rise to dimensional
    # mis-match during update (i.e. one less action than reward)
    try:
        if schedule.shape[0] < 1:
            schedule = None
    except AttributeError:
        pass
    
    # Get last scheduled day
    if schedule is None:
        planning_day = env.sim_time
    else:
        planning_day = np.max(schedule[:, 
            env.sched_indices['prod_end_time']])

    planning_limit = env.sim_time + env.fixed_planning_horizon
    
    while planning_day < planning_limit:
        state = get_current_state(env, schedule=schedule, 
            day=planning_day)
        action_probs = torchToNumpy(network.predict(state))

        if action_probs.ndim > 1:
            action_probs = np.squeeze(action_probs)

        # nan should not appear in action probabilities
        if any(np.isnan(action_probs)):
            raise ValueError("nan found in action probability output. {}".format(
                action_probs))

        # Run heuristic
        if confidence_level is not None and action_probs.max() < confidence_level:
           action = heuristic_scheduler(env)
           heuristic_selection.append([planning_day, 1])
        elif test == True:
            action = env.action_list[np.argmax(action_probs)]
            heuristic_selection.append([planning_day, 0])
        else:
           action = np.random.choice(env.action_list, p=action_probs)
           heuristic_selection.append([planning_day, 0])
        a_probs.append(action_probs)
        predicted_state.append(state)

        schedule = env.append_schedule(schedule, action)

        # TODO: this loop leaves actions and predicted_state with one less
        # entry than it ought to in the presence of a production outtage over
        # the course of an episode. This causes problems with data logging.

        if planning_day < env.n_steps: # Log actions inside simulation horizon
            if not planning_day in env.containers.planning_day:
                env.containers.planning_day.append(planning_day)
                env.containers.actions.append(int(action))
                env.containers.predicted_state.append(state)
            else:
                idc = env.containers.planning_day.index(planning_day)   
                env.containers.actions[idc] = int(action)
                env.containers.predicted_state[idc] = state
        
        sched_end = np.max(schedule[:,env.sched_indices['prod_end_time']])
        planning_day = sched_end 

    # Reshape probs and heuristics
    if len(heuristic_selection) == 0 or len(a_probs) == 0:
        # Occurs if schedule is unchanged
        planning_data = None
    else:
        heuristic_selection = np.vstack(heuristic_selection)
        a_probs = np.vstack(a_probs)
        planning_data = np.hstack([heuristic_selection, a_probs])

    return schedule, planning_data

# Get schedule value estimation
def estimate_schedule_value(env, network, schedule):
    state = get_current_state(env, schedule=schedule,day=int(copy(env.sim_time)))
    value_estimate = network.predict(state).item()
    return value_estimate

# def estimate_schedule_value(env, network, schedule, 
#         value_estimate_array):
#     current_day = int(copy(env.sim_time))
#     planning_limit = env.sim_time + env.fixed_planning_horizon
    
#     while current_day < planning_limit and current_day < env.n_steps:
#         # Get the value estimate for each day from the current_day to the
#         # planning horizon. Use predictions for future days as that is the
#         # information the agent is making its decisions on.
#         state = get_current_state(env, schedule=schedule, 
#             day=current_day)
#         value_estimate_array[current_day] = network.predict(state).item()
    
#         current_day += 1 

#     return value_estimate_array

# Generate schedule from policy network
def q_scheduler(env, network, schedule, epsilon):
    '''
    Inputs
    =========================================================================
    env: productionFacility object
    network: policy network object
    schedule: numpy array containing schedule
    confidence_level: float or None. If the max probability that comes 
        from the policy network is below the confidence_level, then the 
        schedule defaults to the heuristic.
    '''
    q_vals = []
    random_selection = []
    predicted_state = []

    try:
        if schedule.shape[0] < 1:
            schedule = None
    except AttributeError:
        pass
    
    # Get last scheduled day
    if schedule is None:
        planning_day = env.sim_time
    else:
        planning_day = np.max(schedule[:, 
            env.sched_indices['prod_end_time']])

    planning_limit = env.sim_time + env.planning_horizon
    
    while planning_day < planning_limit:
        state = env.get_current_state(schedule=schedule, 
            day=planning_day)
        qvals = network.predict(state)
        if np.random.random() < epsilon:
            action = np.random.choice(env.action_list)
            random_selection.append([planning_day, 1])
        else:
            action = np.argmax(qvals, dim=-1)
            random_selection.append([planning_day, 0])
        

        # nan should not appear in action probabilities
        if any(np.isnan(action_probs)):
            print(action_probs)
            print("Output from last layer")
            # TODO: Change output from TF to PT code
            # print(network.sess.run(
            #     [network.hidden_dict[network.n_hidden_layers]],
            #      feed_dict={
            #         network.state: state
            #                   })[0])
            raise ValueError("nan found in state-action output.")

        q_vals.append(qvals)
        predicted_state.append(state)

        schedule = env.append_schedule(schedule, action)

        # Log actions if within simulation horizon to avoid dimension
        # mismatch when updating network
        if planning_day < env.n_steps:
            if planning_day >= len(env.containers.actions):
                env.containers.actions.append(int(action))
                env.containers.predicted_state.append(state)
            else:
                env.containers.actions[int(planning_day)] = int(action)
                env.containers.predicted_state[int(planning_day)] = state
        
        sched_end = np.max(schedule[:,env.sched_indices['prod_end_time']])
        planning_day = sched_end 

    # Reshape probs and heuristics
    if len(random_selection) == 0 or len(q_vals) == 0:
        # Occurs if schedule is unchanged
        planning_data = None
    else:
        random_selection = np.vstack(random_selection)
        q_vals = np.vstack(q_vals)
        predicted_state_ = np.vstack(predicted_state)
        planning_data = np.hstack([random_selection, q_vals, 
            predicted_state])

    return schedule, planning_data, predicted_state_
