# schedule_utils: Contains common functions used across scheduling files
# Christian Hubbs
# 18.06.2018

import numpy as np 

# build_schedule is the main function to call when generating the random 
# schedule. It combines the other functions in this file. This contains
# an error catching feature to serve as a wrapper for the build_schedule_
# function.
def build_schedule(env, scheduler_function, schedule=None, *args, **kwargs):
    if kwargs is not None:
        if 'model' in kwargs:
            model = kwargs['model']
    try:
        schedule = _build_schedule(env, scheduler_function, schedule, model=model)
    except ValueError:
        # ValueError can occur if schedule is not None and
        # the array is empty. This happens when a production
        # interruption occurs during the first time step
        schedule=None
        schedule = _build_schedule(env, scheduler_function, schedule, model=model)
        
    return schedule

# _build_schedule to plan out to planning horizon and keep that up to date
# for each time step
def _build_schedule(env, scheduler_function, schedule=None, *args, **kwargs):
    if kwargs is not None:
        if 'model' in kwargs:
            model = kwargs['model']

    # Check to see if a schedule exists, if not, initialize one
    if schedule is None:
        planning_time = env.sim_time

    else:
        # Get last scheduled day
        planning_time = np.max(schedule[:, 
            env.sched_indices['prod_end_time']])

    planning_limit = env.sim_time + env.fixed_planning_horizon
    
    while planning_time < planning_limit:
        # Get last action for schedule
        action = scheduler_function(env, schedule=schedule, 
            model=model, planning_time=planning_time)
        # TODO: Catch error in cases where opt schedule function times out
        # and None is scheduled as a result. Now, selects random value, 
        # update to employ heuristic or other result.
        try:        
            schedule = env.append_schedule(schedule, action)
        except TypeError:
            action = np.random.choice(env.action_list)
            schedule = env.append_schedule(schedule, action)

        # Log actions if within simulation horizon to avoid dimension
        # mismatch when updating network
        if planning_time <= env.n_days:
            if planning_time >= len(env.containers.actions):
                env.containers.actions.append(int(action))
            else:
                env.containers.actions[int(planning_time)] = int(action)
        planning_time += 1
        
    return schedule

# The append schedule function will take the selected next action
# from the network and fill this data into the schedule. The off_grade
# value is used to determine whether or not an action is allowed.
# To save repetition, the action is vetted before going to the schedule
# and thus the off-grade amount is passed as an argument.
# def append_schedule(schedule, action, env):
#     # Look up off-grade production value
#     off_grade = env.transition_matrix[0, int(action)]

#     #########################################################################
#     # Placeholder in case 0 is entered
#     #########################################################################    
#     if action == 0:
#         next_sched_entry = np.array([
#                 env.sched_indices["batch_num"], # Batch number
#                 action, # Product
#                 0, # Production rate
#                 0, # Silo size
#                 24, # Silo fill time
#                 env.sim_time, # Start day
#                 env.stop_hours + 24, # End time
#                 env.curing_time, # Curing time 
#                 0, # Curing completion time placeholder
#                 0, # Curing Completion flag
#                 0, # Inventory index
#                 0, # Off-grade production
#                 0  # Actual production quantity
#             ])
#         # Calculate curing completion time
#         next_sched_entry[env.sched_indices["cure_end_time"]] = (
#                 next_sched_entry[env.sched_indices["prod_end_time"]] + 
#             next_sched_entry[env.sched_indices["cure_time"]])
#         schedule = next_sched_entry.reshape(1,-1)
#     #########################################################################
#     # End placeholder in case 0 is entered
#     #########################################################################

#     if schedule is None:
#         # Initialize schedule
#         next_sched_entry = np.array([
#                 env.sched_indices["batch_num"], # Batch number
#                 action, # Product
#                 env.action_dict[action][1], # Production rate
#                 (env.action_dict[action][0] *
#                  env.action_dict[action][1]), # Silo size
#                 env.action_dict[action][0], # Silo fill time
#                 env.sim_time, # Start day
#                 env.stop_hours + env.action_dict[action][0], # End time
#                 env.curing_time, # Curing time 
#                 0, # Curing completion time placeholder
#                 0, # Curing Completion flag
#                 env.action_dict[action][2], # Inventory index
#                 off_grade, # Off-grade production
#                 (env.action_dict[action][0] *
#                  env.action_dict[action][1] -
#                  off_grade) # Actual production quantity
#             ])

#         # Calculate curing completion time
#         next_sched_entry[env.sched_indices["cure_end_time"]] = (
#                 next_sched_entry[env.sched_indices["prod_end_time"]] + 
#             next_sched_entry[env.sched_indices["cure_time"]])
#         schedule = next_sched_entry.reshape(1,-1)

#     else:            
#         # Look up off-grade production value
#         off_grade = env.transition_matrix[
#             int(schedule[-1,env.sched_indices["gmid"]]), int(action)]

#         next_sched_entry = np.array([
#                 schedule[-1,env.sched_indices["batch_num"]] + 1, # Batch Nmber
#                 action, # Product
#                 env.action_dict[action][1], # Production rate
#                 (env.action_dict[action][0] *
#                  env.action_dict[action][1]), # Silo size
#                 env.action_dict[action][0], # Silo fill time
#                 env.stop_hours + schedule[-1, env.sched_indices["prod_end_time"]], # Start time
#                 0,  # End time placeholder
#                 env.curing_time, # Curing time
#                 0, # Curing completion time placeholder
#                 0, # Curing Completion flag
#                 env.action_dict[action][2], # Inventory index
#                 off_grade, # Off-grade production
#                 (env.action_dict[action][0] * 
#                  env.action_dict[action][1] - 
#                  off_grade) # Actual production quantity
#             ])
#         # Calculate silo end time
#         next_sched_entry[env.sched_indices["prod_end_time"]] = (
#             next_sched_entry[env.sched_indices["prod_start_time"]] + 
#             next_sched_entry[env.sched_indices["prod_time"]])
#         # Calculate curing completion time
#         next_sched_entry[env.sched_indices["cure_end_time"]] = (
#                 next_sched_entry[env.sched_indices["prod_end_time"]] + 
#                 next_sched_entry[env.sched_indices["cure_time"]])
#         schedule = np.vstack([schedule, next_sched_entry.reshape(1,-1)])
        
#         # Remove stop hours from calculation
#         env.stop_hours = 0
        
#     return schedule