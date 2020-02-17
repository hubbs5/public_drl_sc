# This file contains functions to enable random schedule generation in 
# conjunction with the production_env environment

import numpy as np

# build_schedule is the main function to call when generating the random 
# schedule. It combines the other functions in this file. This contains
# an error catching feature to serve as a wrapper for the build_schedule_
# function.
def build_schedule(env, schedule=None):
    try:
        schedule = _build_schedule(env, schedule)
    except ValueError:
        # ValueError can occur if schedule is not None and
        # the array is empty. This happens when a production
        # interruption occurs during the first time step
        schedule=None
        schedule = _build_schedule(env, schedule)
        
    return schedule

# random_scheduler checks to see if a schedule exists. If not, then it
# it selects a suitable start-up grade based on the env.transition_matrix.
# It then proceeds to randomly select actions based on the available actions.
def random_scheduler(env, schedule=None):
    # Develop a schedule to randomly select items and maintain the production horizon
    
    # Check to see if coming from start-up to select possible start-up grade
    if env.current_prod == 0 and schedule is None:
        start_up_actions = np.where(env.transition_matrix[0][1:]==
                                    env.transition_matrix[0][1:].min())[0]
        # Because actions are indexed and we remove the 'do nothing' option
        # by selecting [1:], we need to add 1 to the actions to map them
        # to the proper product
        action = np.random.choice(start_up_actions + 1)
    else:
        # Select action
        action = np.random.choice(env.action_list)

    return action

# build_schedule_ to plan out to planning horizon and keep that up to date
# for each time step
def _build_schedule(env, schedule=None):
    # Check to see if a schedule exists, if not, initialize one
    if schedule is None:
        action = random_scheduler(env)
        schedule = append_schedule(schedule, action, env)
        # Get last scheduled day
        planning_day = env.sim_time

    else:
        planning_day = np.max(schedule[:, 
            env.sched_indices['prod_start_time']])

    planning_limit = env.sim_time + env.fixed_planning_horizon
    
    while planning_day < planning_limit:
        # Get last action for schedule
        prev_action = schedule[-1, env.sched_indices["inventory_index"]].astype(int)
        action = random_scheduler(env, schedule)
        schedule = append_schedule(schedule, action, env)

         # Log actions if within simulation horizon to avoid dimension
        # mismatch when updating network
        if planning_day < env.n_days:
            if planning_day >= len(env.containers.actions):
                env.containers.actions.append(int(action))
            else:
                env.containers.actions[int(planning_day)] = int(action)
        planning_day += 1
        
    return schedule

# The append schedule function will take the selected next action
# from the network and fill this data into the schedule. The off_grade
# value is used to determine whether or not an action is allowed.
# To save repetition, the action is vetted before going to the schedule
# and thus the off-grade amount is passed as an argument.
def append_schedule(schedule, action, env):
    # Look up off-grade production value
    off_grade = env.transition_matrix[0, int(action)]

    #########################################################################
    # Placeholder in case 0 is entered
    #########################################################################    
    if action == 0:
        next_sched_entry = np.array([
                env.sched_indices["batch_num"], # Batch number
                action, # Product
                0, # Production rate
                0, # Silo size
                24, # Silo fill time
                env.sim_time, # Start day
                env.stop_hours + 24, # End time
                env.curing_time, # Curing time 
                0, # Curing completion time placeholder
                0, # Curing Completion flag
                0, # Inventory index
                0, # Off-grade production
                0  # Actual production quantity
            ])
        # Calculate curing completion time
        next_sched_entry[env.sched_indices["cure_end_time"]] = (
                next_sched_entry[env.sched_indices["prod_end_day"]] + 
            next_sched_entry[env.sched_indices["cure_time"]])
        schedule = next_sched_entry.reshape(1,-1)
    #########################################################################
    # End placeholder in case 0 is entered
    #########################################################################

    if schedule is None:
        # Initialize schedule
        next_sched_entry = np.array([
                env.sched_indices["batch_num"], # Batch number
                action, # Product
                env.action_dict[action][1], # Production rate
                (env.action_dict[action][0] *
                 env.action_dict[action][1]), # Silo size
                env.action_dict[action][0], # Silo fill time
                env.sim_time, # Start day
                env.stop_hours + env.action_dict[action][0], # End time
                env.curing_time, # Curing time 
                0, # Curing completion time placeholder
                0, # Curing Completion flag
                env.action_dict[action][2], # Inventory index
                off_grade, # Off-grade production
                (env.action_dict[action][0] *
                 env.action_dict[action][1] -
                 off_grade) # Actual production quantity
            ])

        # Calculate curing completion time
        next_sched_entry[env.sched_indices["cure_end_time"]] = (
                next_sched_entry[env.sched_indices["prod_end_day"]] + 
            next_sched_entry[env.sched_indices["cure_time"]])
        schedule = next_sched_entry.reshape(1,-1)

    else:            
        # Look up off-grade production value
        off_grade = env.transition_matrix[
            int(schedule[-1,env.sched_indices["prod_num"]]), int(action)]

        next_sched_entry = np.array([
                schedule[-1,env.sched_indices["batch_num"]] + 1, # Batch Nmber
                action, # Product
                env.action_dict[action][1], # Production rate
                (env.action_dict[action][0] *
                 env.action_dict[action][1]), # Silo size
                env.action_dict[action][0], # Silo fill time
                env.stop_hours + schedule[-1, env.sched_indices["prod_end_day"]], # Start time
                0,  # End time placeholder
                env.curing_time, # Curing time
                0, # Curing completion time placeholder
                0, # Curing Completion flag
                env.action_dict[action][2], # Inventory index
                off_grade, # Off-grade production
                (env.action_dict[action][0] * 
                 env.action_dict[action][1] - 
                 off_grade) # Actual production quantity
            ])
        # Calculate silo end time
        next_sched_entry[env.sched_indices["prod_end_day"]] = (
            next_sched_entry[env.sched_indices["prod_start_time"]] + 
            next_sched_entry[env.sched_indices["prod_time"]])
        # Calculate curing completion time
        next_sched_entry[env.sched_indices["cure_end_time"]] = (
                next_sched_entry[env.sched_indices["prod_end_day"]] + 
                next_sched_entry[env.sched_indices["cure_time"]])
        schedule = np.vstack([schedule, next_sched_entry.reshape(1,-1)])
        
        # Remove stop hours from calculation
        env.stop_hours = 0
        
    return schedule