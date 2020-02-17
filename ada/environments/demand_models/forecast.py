# forecast models
# Author: Christian Hubbs
# Email: christiandhubbs@gmail.com
# Date: 03.05.2019

import numpy as np 
from calendar import monthrange
import warnings
from .demand_utils import *

# List of forecast commands that can be called
# forecast_options = [False, 'UNIFORM', 'UNIFORM_HORIZON',
#  'AGGREGATE_HORIZON', 'STOCHASTIC_AVERAGE_DEMAND',
#  'DETERMINISTIC_AVERAGE_DEMAND']
forecast_options = [False, 
 'UNIFORM', 
 'STOCHASTIC_AVERAGE_DEMAND',
 'DETERMINISTIC_AVERAGE_DEMAND']

def select_forecast(env):
    # Build monthly demand forecast if it does not exist and
    # if training. If testing, forecast will be provided and loaded.
    if env.settings['TRAIN']:
        try:
            _ = env.monthly_forecast
        except AttributeError:
            env.monthly_forecast = build_monthly_forecast(env)
    try:
        _ = env.vsm_mean
    except AttributeError:
        env.vsm_mean = [env.order_book[np.where(
            env.order_book[:,env.ob_indices['gmid']]==g)[0], 
            env.ob_indices['var_std_margin']].mean()
            for g in env.gmids]
    if env.forecast_model == 'UNIFORM' or env.forecast_model == True:
        build_uniform_forecast(env)
    elif env.forecast_model == 'AGGREGATE_HORIZON':
        build_aggregated_horizon_forecast(env)
    elif env.forecast_model == 'UNIFORM_HORIZON':
        build_uniform_horizon_forecast(env)
    elif env.forecast_model == 'STOCHASTIC_AVERAGE_DEMAND':
        build_stochastic_average_forecast(env)
    elif env.forecast_model == 'DETERMINISTIC_AVERAGE_DEMAND':
        build_deterministic_average_forecast(env)

# Build forecast volumes for the entire simulation horizon
def build_monthly_forecast(env):
    monthly_demand = np.zeros((len(env.shipping_dict), env.n_products))
    current_month = env.sim_day_to_date[env.sim_time][0]
    for i, month in enumerate(env.shipping_dict):
        if month < current_month:
            continue
        monthly_orders = env.order_book[
            np.where(
                (env.order_book[:,env.ob_indices['planned_gi_month']]==month)
                )[0]].copy()
        # Get total demand for each gmid each month
        unique_gmid, unique_gmid_id = np.unique(
            monthly_orders[:, env.ob_indices['gmid']], return_inverse=True)
        demand = np.bincount(unique_gmid_id,
                monthly_orders[:,env.ob_indices['order_qty']])
        try:
            monthly_demand[i] = demand.copy()
        except ValueError:
            # Insert missing gmids, if any
            missing_gmid = [j for j, k in enumerate(env.gmids) if k not in unique_gmid]
            for g_idx in missing_gmid:
                demand = np.insert(demand, g_idx, 0)
            monthly_demand[i] = demand.copy()
        
    monthly_demand_forecast = monthly_demand * (1 + env.settings['FORECAST_ACCURACY']/100)
    return monthly_demand_forecast

# Sums forecast at the end of the planning horizon
def build_aggregated_horizon_forecast(env):
    agg_qty, orders_in_view = env.aggregate_monthly_visible_demand()
    months = list(env.shipping_dict.keys())
    current_month = env.sim_day_to_date[env.sim_time][0]
    # Generate dummy orders with the average daily quantity for days where no orders
    # currently exist
    forecasted_orders = None
    for i, month in enumerate(months):
        agg_time = env.sim_time + env.lookahead_planning_horizon
        for j, g in enumerate(env.gmids):
            if month < current_month:
                # No forecast
                continue
            elif month == current_month:
                # If agg_time falls into the next month, reduce current month forecast it to 0.
                # New forecast will be based on the following month's forecast
                try:
                    if env.sim_day_to_date[agg_time][0] > current_month:
                        continue
                except KeyError:
                    continue
            # If agg_time straddles a month, place month t+1 forecast at horizon, 
            # otherwise place it at first day of month t+n
            else:
                if env.sim_day_to_date[agg_time][0] < month:
                    agg_time = env.month_to_sim_day[month][0]
            net_forecast = max(env.monthly_forecast[i, j] - agg_qty[i, j], 0)
            # Split by min order quantity so that one, large order doesn't dominate
            n_forecast_orders = int(net_forecast/25)
            dummy_orders = np.zeros((n_forecast_orders, 
                                     env.order_book.shape[1]))
            dummy_orders[:, env.ob_indices['planned_gi_time']] = int(agg_time)
            dummy_orders[:, env.ob_indices['planned_gi_month']] = month
            dummy_orders[:, env.ob_indices['gmid']] = g
            dummy_orders[:, env.ob_indices['order_qty']] = 25
            dummy_orders[:, env.ob_indices['var_std_margin']] = env.vsm_mean[
                env.gmid_index_map[g]].astype(int)

            # Ensure values are consistent
            tot = dummy_orders[:, env.ob_indices['order_qty']].sum() + agg_qty[i, j]
            diff = tot - env.monthly_forecast[i, j]
            assert diff >= -31 and diff <= 31, "Volume Mismatch for Month: {}\tGMID: {}\n".format(month, g) + \
                "Diff: {:.0f}\n".format(tot-env.monthly_forecast[i, j]) + \
                "Actual: {:.0f}\tForecast: {:.0f}\tMonthly Forecast: {:.0f}".format(
                    agg_qty[i, j], dummy_orders[:, env.ob_indices['order_qty']].sum(), 
                    env.monthly_forecast[i, j])

            # Stack with forecast
            forecasted_orders = np.vstack([forecasted_orders, dummy_orders.copy()]) \
                if forecasted_orders is not None else dummy_orders.copy()
    
    if forecasted_orders is not None:
        # Add unique order numbers to forecast
        forecasted_orders[:, env.ob_indices['doc_num']] = np.arange(-len(forecasted_orders), 0)
        env.order_book = np.vstack([env.order_book, forecasted_orders.astype(int)])

# Pushes uniform forecast to the edge of the planning horizon
def build_uniform_horizon_forecast(env):
    agg_qty, orders_in_view = env.aggregate_monthly_visible_demand()
    months = list(env.shipping_dict.keys())
    # Generate dummy orders with the average daily quantity for days where no orders
    # currently exist
    forecasted_orders = None
    for i, month in enumerate(months):
        for j, g in enumerate(env.gmids):
            # Get days without demand
            un_days_in_view = np.unique(orders_in_view[
                orders_in_view[:,env.ob_indices['planned_gi_month']]==month,
                env.ob_indices['planned_gi_time']])
            days_remaining = env.month_to_sim_day[month][env.month_to_sim_day[month]> \
                env.sim_time]
            if len(days_remaining) < 1:
                continue
            days_without_demand = days_remaining[
                np.array([i not in un_days_in_view for i in days_remaining])]
            n_days_remaining = len(days_remaining)

            # Get avg daily forecast based on the net forecast for each month-gmid combination
            avg_daily_forecast_gmid = (
                env.monthly_forecast[i, j] - agg_qty[i, j])/len(days_without_demand) if \
                len(days_without_demand) > 0 else 0
            #print(avg_daily_forecast_gmid)
            avg_daily_forecast_gmid = 0 if avg_daily_forecast_gmid < 0 else \
                avg_daily_forecast_gmid
            
            # Generate orders for days where no demand exists
            dummy_orders = np.zeros((len(days_without_demand), 
                env.order_book.shape[1]), dtype='int32')
            dummy_orders[:, env.ob_indices['planned_gi_time']] = days_without_demand
            dummy_orders[:, env.ob_indices['planned_gi_month']] = month
            dummy_orders[:, env.ob_indices['gmid']] = g
            dummy_orders[:, env.ob_indices['order_qty']] = avg_daily_forecast_gmid
            dummy_orders[:, env.ob_indices['var_std_margin']] = env.vsm_mean[
                env.gmid_index_map[g]].astype(int)
            
            # Ensure values are consistent
            tot = dummy_orders[:, env.ob_indices['order_qty']].sum() + agg_qty[i, j]
            diff = tot - env.monthly_forecast[i, j]
            assert diff >= -31 and diff <= 31, "Volume Mismatch for Month: {}\tGMID: {}\n".format(month, g) + \
                "Diff: {:.0f}\n".format(tot-env.monthly_forecast[i, j]) + \
                "Actual: {:.0f}\tForecast: {:.0f}\tMonthly Forecast: {:.0f}".format(
                    agg_qty[i, j], dummy_orders[:, env.ob_indices['order_qty']].sum(), 
                    env.monthly_forecast[i, j])

            # Stack with forecast
            forecasted_orders = np.vstack([forecasted_orders, dummy_orders.copy()]) \
                if forecasted_orders is not None else dummy_orders.copy()
    
    if forecasted_orders is not None:
        # Add unique order numbers to forecast
        forecasted_orders[:, env.ob_indices['doc_num']] = np.arange(-len(forecasted_orders), 0)
        env.order_book = np.vstack([env.order_book, forecasted_orders.astype(int)])

# build_uniform_forecast takes the order book and generates dummy orders.
# It then uniformly distributes those dummy orders over the planning 
# horizon and adds them to the order book.
def build_uniform_forecast(env):
    agg_qty, orders_in_view = env.aggregate_monthly_visible_demand()

    # Get number of days remaining in each month
    current_month = env.sim_day_to_date[env.sim_time][0]
    months = list(env.shipping_dict.keys())
    forecasted_orders = None
    for i, month in enumerate(months):
        days_in_month = env.month_to_sim_day[month]
        days_remaining = days_in_month[env.month_to_sim_day[month]>env.sim_time]
        n_days_remaining = len(days_remaining)
        if month < current_month or n_days_remaining <= 1:
            continue
        for j, g in enumerate(env.gmids):
            net_forecast = max(env.monthly_forecast[i, j] - agg_qty[i, j], 0)
            if net_forecast <= 0:
                continue
            daily_avg_forecast = net_forecast / n_days_remaining if n_days_remaining > 0 else 0
            n_orders_per_day = daily_avg_forecast / env.order_size
            full_orders_per_day = int(n_orders_per_day)
            remainder = n_orders_per_day % 1
            n_orders = int(n_orders_per_day) * n_days_remaining
            order_days = np.repeat(days_remaining, full_orders_per_day)
            dummy_orders = np.zeros((n_orders + n_days_remaining,
                                    env.order_book.shape[1]))
            
            dummy_orders[:n_orders, env.ob_indices['planned_gi_time']] = order_days
            dummy_orders[:, env.ob_indices['planned_gi_month']] = month
            dummy_orders[:, env.ob_indices['gmid']] = g
            dummy_orders[:n_orders, env.ob_indices['order_qty']] = env.order_size
            dummy_orders[:, env.ob_indices['var_std_margin']] = env.vsm_mean[
                env.gmid_index_map[g]].astype(int)
            # Add Partial Orders
            dummy_orders[n_orders:, env.ob_indices['planned_gi_time']] = days_remaining
            dummy_orders[n_orders:, env.ob_indices['order_qty']] = int(remainder * 
                env.order_size)
            
            # Ensure values are consistent
            tot = dummy_orders[:, env.ob_indices['order_qty']].sum() + agg_qty[i, j]
            diff = tot - env.monthly_forecast[i, j]
            assert diff >= -31 and diff <= 31, "Volume Mismatch for Month: {}\tGMID: {}\n".format(month, g) + \
                "Diff: {:.0f}\n".format(tot-env.monthly_forecast[i, j]) + \
                "Actual: {:.0f}\tForecast: {:.0f}\tMonthly Forecast: {:.0f}".format(
                    agg_qty[i, j], dummy_orders[:, env.ob_indices['order_qty']].sum(), 
                    env.monthly_forecast[i, j])

            forecasted_orders = np.vstack([forecasted_orders, dummy_orders.copy()]) \
                if forecasted_orders is not None else dummy_orders.copy()
    
    if forecasted_orders is not None:
        forecasted_orders = forecasted_orders[
            np.where(forecasted_orders[:,env.ob_indices['order_qty']]!=0)[0]]
        # Add unique order numbers to forecast
        forecasted_orders[:, env.ob_indices['doc_num']] = np.arange(-len(forecasted_orders), 0)
        env.order_book = np.vstack([env.order_book, forecasted_orders.astype(int)])

def build_stochastic_average_forecast(env):
    agg_qty, orders_in_view = env.aggregate_monthly_visible_demand()
    months = list(env.shipping_dict.keys())
    forecasted_orders = None
    current_month = env.sim_day_to_date[env.sim_time][0]
    # Get days remaining in month
    for i, month in enumerate(months):
        days_in_month = env.month_to_sim_day[month]
        days_remaining = days_in_month[env.month_to_sim_day[month]>(
        	env.sim_time)] #+env.fixed_planning_horizon)]
        n_days_remaining = len(days_remaining)
        if month < current_month or n_days_remaining <= 1:
            continue
        for j, g in enumerate(env.gmids):
            net_forecast = max(env.monthly_forecast[i, j] - agg_qty[i, j], 0)
            if net_forecast <= 0:
                continue
            daily_avg_forecast = net_forecast / n_days_remaining if n_days_remaining > 0 else 0
            orders_in_view_gmid = orders_in_view[np.where(
                (orders_in_view[:,env.ob_indices['gmid']]==g) &
                (orders_in_view[:,env.ob_indices['planned_gi_month']]==month))[0]].copy()
            un_days, un_idx = np.unique(
                orders_in_view_gmid[:,env.ob_indices['planned_gi_time']], return_inverse=True)
            daily_demand = np.zeros(max(days_in_month) + 1)
            try:
                daily_demand[un_days.astype(int)] = np.bincount(un_idx,
                    orders_in_view_gmid[:,env.ob_indices['order_qty']])
            except IndexError:
                daily_demand = np.zeros(max(
                    env.order_book[:,env.ob_indices['planned_gi_time']]) + 1)
                daily_demand[un_days.astype(int)] = np.bincount(un_idx,
                    orders_in_view_gmid[:,env.ob_indices['order_qty']])
            
            
            daily_demand = daily_demand[-n_days_remaining:]
            comp = max(daily_demand.max(), daily_avg_forecast)
            den = (comp - daily_demand).sum()
            if den == 0:
                # Avoid division by zero by reverting to uniform distribution
                prob = np.repeat(1/len(daily_demand), len(daily_demand))
            else:
                prob = (comp - daily_demand) / (comp - daily_demand).sum()
            if np.isnan(prob).any():
                # Avoid NaN's by reverting to uniform distribution (very rare)
                prob = np.repeat(1/len(daily_demand), len(daily_demand))
        
            # Build order book
            n_orders = np.round(net_forecast/env.order_size).astype(int)
            order_days = np.random.choice(days_remaining, p=prob, size=n_orders)
            dummy_orders = np.zeros((n_orders, env.order_book.shape[1]))
            dummy_orders[:, env.ob_indices['planned_gi_time']] = order_days
            dummy_orders[:, env.ob_indices['planned_gi_month']] = month
            dummy_orders[:, env.ob_indices['gmid']] = g
            dummy_orders[:, env.ob_indices['order_qty']] = env.order_size
            dummy_orders[:, env.ob_indices['var_std_margin']] = env.vsm_mean[
                env.gmid_index_map[g]].astype(int)
            
            # Ensure values are consistent
            tot = dummy_orders[:, env.ob_indices['order_qty']].sum() + agg_qty[i, j]
            diff = tot - env.monthly_forecast[i, j]
            assert diff >= -31 and diff <= 31, "Volume Mismatch for Month: {}\tGMID: {}\n".format(month, g) + \
                "Diff: {:.0f}\n".format(tot-env.monthly_forecast[i, j]) + \
                "Actual: {:.0f}\tForecast: {:.0f}\tMonthly Forecast: {:.0f}".format(
                    agg_qty[i, j], dummy_orders[:, env.ob_indices['order_qty']].sum(), 
                    env.monthly_forecast[i, j])

            # Stack with forecast
            forecasted_orders = np.vstack([forecasted_orders, dummy_orders.copy()]) \
                if forecasted_orders is not None else dummy_orders.copy()
                 
    if forecasted_orders is not None:
        forecasted_orders = forecasted_orders[
            np.where(forecasted_orders[:,env.ob_indices['order_qty']]!=0)[0]]
        # Add unique order numbers to forecast
        forecasted_orders[:, env.ob_indices['doc_num']] = np.arange(-len(forecasted_orders), 0)
        env.order_book = np.vstack([env.order_book, forecasted_orders.astype(int)])

# def build_stochastic_average_forecast(env):
#     agg_qty, orders_in_view = env.aggregate_monthly_visible_demand()
#     months = list(env.shipping_dict.keys())
#     current_month = env.sim_day_to_date[env.sim_time][0]
#     count = 0
#     forecasted_orders = None
#     agg_time = env.sim_time + env.lookahead_planning_horizon
#     for i, month in enumerate(months):
#         days_in_month = env.month_to_sim_day[month]
#         days_remaining = days_in_month[env.month_to_sim_day[month]>env.sim_time]
#         n_days_remaining = len(days_remaining)
#         for j, g in enumerate(env.gmids):
#             orders_in_view_gmid = orders_in_view[np.where(
#                 (orders_in_view[:,env.ob_indices['gmid']]==g) &
#                 (orders_in_view[:,env.ob_indices['planned_gi_month']]==month))[0]].copy()
#             if month < current_month or n_days_remaining <= 1:
#                 # No forecast for prior months
#                 continue
#     #         elif month == current_month:
#     #             # If agg_time falls into the following month, then reduce
#     #             # the current month's forecast to 0.
#     #             if env.sim_day_to_date[agg_time][0] > current_month:
#     #                 continue
            
#             net_forecast = max(env.monthly_forecast[i, j] - agg_qty[i, j], 0)
#             if net_forecast == 0:
#                 continue
#             # Aggregate actual demand
#             un_days, un_index = np.unique(
#                 orders_in_view_gmid[:,env.ob_indices['planned_gi_time']], 
#                 return_inverse=True)
#             _daily_demand = np.bincount(un_index,
#                 orders_in_view_gmid[:,env.ob_indices['order_qty']])
#             daily_demand = np.zeros(max(days_in_month)+1)
#             daily_demand[un_days] += _daily_demand
#             daily_demand = daily_demand[days_in_month]
#             daily_avg_forecast = net_forecast / n_days_remaining
#             _forecast = np.zeros(max(days_in_month)+1)
#             _forecast[days_remaining] = daily_avg_forecast
#             _forecast[days_in_month] = np.where(
#                 days_in_month<=env.sim_time,0,daily_avg_forecast-daily_demand)
#             k = int(np.round(net_forecast)/env.order_size)
#             forecast_days = []
#             for sample in range(k):
#                 order_prob = softmax(_forecast[days_in_month][days_in_month>env.sim_time])
#                 selected_day = np.random.choice(
#                     days_in_month[days_in_month>env.sim_time], p=order_prob)
#                 _forecast[selected_day] -= 25
#                 forecast_days.append(selected_day)
#             # Ensure that forecast days are in days remaining. If not, resample until they are
#             # Very low probability that this would occur, but just to be safe.
#             while all(elem in days_remaining for elem in forecast_days) == False:
#                 count += 1
#                 forecast_days = np.random.choice(env.month_to_sim_day[month], p=order_prob, size=k)
#                 if count > 10:
#                     print(env.sim_time, count)
#                     break
                
#             # Build dummy order book
#             dummy_orders = np.zeros((len(forecast_days), 
#                                      env.order_book.shape[1]))
#             dummy_orders[:, env.ob_indices['planned_gi_time']] = np.array(forecast_days)
#             dummy_orders[:, env.ob_indices['planned_gi_month']] = month
#             dummy_orders[:, env.ob_indices['gmid']] = g
#             dummy_orders[:, env.ob_indices['order_qty']] = env.order_size
#             dummy_orders[:, env.ob_indices['var_std_margin']] = env.vsm_mean[
#                 env.gmid_index_map[g]].astype(int)

#             # Stack with forecast
#             forecasted_orders = np.vstack([forecasted_orders, dummy_orders.copy()]) \
#                 if forecasted_orders is not None else dummy_orders.copy()

#     if forecasted_orders is not None:
#         env.order_book = np.vstack([env.order_book, forecasted_orders.astype(int)])

def build_deterministic_average_forecast(env):
    agg_qty, orders_in_view = env.aggregate_monthly_visible_demand()
    months = list(env.shipping_dict.keys())
    forecasted_orders = None
    current_month = env.sim_day_to_date[env.sim_time][0]
    # Get days remaining in month
    for i, month in enumerate(months):
        days_in_month = env.month_to_sim_day[month]
        days_remaining = days_in_month[env.month_to_sim_day[month]>env.sim_time]
        n_days_remaining = len(days_remaining)
        if month < current_month or n_days_remaining < 1:
            continue
        for j, g in enumerate(env.gmids):
            net_forecast = max(env.monthly_forecast[i, j] - agg_qty[i, j], 0)
            if net_forecast <= 0:
                continue
            orders_in_view_gmid = orders_in_view[np.where(
                (orders_in_view[:,env.ob_indices['gmid']]==g) &
                (orders_in_view[:,env.ob_indices['planned_gi_month']]==month))[0]].copy()
            un_days, un_idx = np.unique(
                orders_in_view_gmid[:,env.ob_indices['planned_gi_time']], 
                return_inverse=True)
            daily_demand = np.zeros(max(days_in_month)+1)
            try:
                daily_demand[un_days.astype(int)] = np.bincount(un_idx,
                    orders_in_view_gmid[:,env.ob_indices['order_qty']])
            except IndexError:
                daily_demand = np.zeros(max(
                    env.order_book[:,env.ob_indices['planned_gi_time']]) + 1)
                daily_demand[un_days.astype(int)] = np.bincount(un_idx,
                    orders_in_view_gmid[:,env.ob_indices['order_qty']])
            _daily_demand = daily_demand[-n_days_remaining:].copy()
            n_full_orders = int(net_forecast / env.order_size)
            forecast_remainder = np.round((net_forecast % 1) * \
                env.order_size).astype(int)
            day_index = []
            for v in range(n_full_orders + 1):
                day = np.argmin(_daily_demand)
                day_index.append(day)
                _daily_demand[day] += env.order_size

            dummy_orders = np.zeros((n_full_orders+1, env.order_book.shape[1]))
            dummy_orders[:, env.ob_indices['planned_gi_time']] = days_remaining[day_index]
            dummy_orders[:, env.ob_indices['planned_gi_month']] = month
            dummy_orders[:, env.ob_indices['gmid']] = g
            dummy_orders[:n_full_orders, env.ob_indices['order_qty']] = env.order_size
            dummy_orders[:, env.ob_indices['var_std_margin']] = env.vsm_mean[
                env.gmid_index_map[g]].astype(int)
            # Add partial orders
            dummy_orders[n_full_orders:, env.ob_indices['order_qty']] = forecast_remainder
            
            # Ensure values are consistent
            tot = dummy_orders[:, env.ob_indices['order_qty']].sum() + agg_qty[i, j]
            diff = tot - env.monthly_forecast[i, j]
            assert diff >= -31 and diff <= 31, "Volume Mismatch for Month: {}\tGMID: {}\n".format(month, g) + \
                "Diff: {:.0f}\n".format(tot-env.monthly_forecast[i, j]) + \
                "Actual: {:.0f}\tForecast: {:.0f}\tMonthly Forecast: {:.0f}".format(
                    agg_qty[i, j], dummy_orders[:, env.ob_indices['order_qty']].sum(), 
                    env.monthly_forecast[i, j])

            # Stack with forecast
            forecasted_orders = np.vstack([forecasted_orders, dummy_orders.copy()]) \
                if forecasted_orders is not None else dummy_orders.copy()
                 
    if forecasted_orders is not None:
        forecasted_orders = forecasted_orders[
            np.where(forecasted_orders[:,env.ob_indices['order_qty']]!=0)[0]]
        # Add unique order numbers to forecast
        forecasted_orders[:, env.ob_indices['doc_num']] = np.arange(-len(forecasted_orders), 0)
        env.order_book = np.vstack([env.order_book, forecasted_orders.astype(int)])