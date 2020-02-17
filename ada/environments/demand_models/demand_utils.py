import numpy as np 
import sys
import os
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from ada.Loggermixin import *

# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger=Loggermixin.get_default_logger()

def softmax(x, axis=0, temp=1):
    if axis == 1:
        x = x.T
    probs = np.exp((x - np.max(x))/temp) / np.sum(
        np.exp((x - np.max(x))/temp), axis=0)
    return probs

# Check forecast values
# Run this periodically during testing to ensure forecasts
# are properly calculated. Because of integer conversion in 
# Order Books (for calculation speed) error of +/- 31 is accepted.
def check_forecast_consistency(env):
    passed = True
    last_month = env.end_time.month
    for i, month in enumerate(env.shipping_dict.keys()):
        if month >= last_month:
            continue
        last_week = env.month_to_sim_day[month][-7]
        for j, g in enumerate(env.gmids):
            act_demand = env.order_book[np.where(
                (env.order_book[:,env.ob_indices['doc_create_time']]<=env.sim_time) &
                (env.order_book[:,env.ob_indices['doc_num']]>0) &
                (env.order_book[:,env.ob_indices['planned_gi_month']]==month) &
                (env.order_book[:,env.ob_indices['gmid']]==g)
                )[0], env.ob_indices['order_qty']].sum()
            dummy_demand = env.order_book[np.where(
                (env.order_book[:,env.ob_indices['doc_create_time']]<=env.sim_time) &
                (env.order_book[:,env.ob_indices['doc_num']]<=0) &
                (env.order_book[:,env.ob_indices['planned_gi_month']]==month) &
                (env.order_book[:,env.ob_indices['gmid']]==g)
                )[0], env.ob_indices['order_qty']].sum()
            total_demand = act_demand + dummy_demand
            diff = np.abs(total_demand-env.monthly_forecast[i,j])
            current_month = env.sim_day_to_date[env.sim_time][0]
            if env.settings['FORECAST'] == False or env.settings['FORECAST'] == 'False':
                if dummy_demand > 0:
                    h_logger.debug("Run on:\t{}-{}".format(current_month, env.sim_day_to_date[env.sim_time][1]))
                    h_logger.debug("Month: {}\tGMID: {}\tForecast: {}".format(month, g, env.settings['FORECAST']))
                    h_logger.debug("Actual Demand:\t{:.0f}\tDummy Demand:\t{:.0f}\t".format(
                        act_demand, dummy_demand))
                    h_logger.debug("Total Demand:\t{:.0f}\tForecasted Demand:\t{:.0f}\tDiff={:.0f}".format(
                        total_demand, env.monthly_forecast[i,j], total_demand-env.monthly_forecast[i,j]))
                    passed = False
            elif env.settings['FORECAST']:
                if diff > 31 and month >= current_month and env.sim_time < last_week: # Diff could be 31 given rounding errors over 31 days in a month
                    h_logger.debug("Run on:\t{}-{}".format(current_month, env.sim_day_to_date[env.sim_time][1]))
                    h_logger.debug("Month: {}\tGMID: {}\tForecast: {}".format(month, g, env.settings['FORECAST']))
                    h_logger.debug("Actual Demand:\t{:.0f}\tDummy Demand:\t{:.0f}\t".format(
                        act_demand, dummy_demand))
                    h_logger.debug("Total Demand:\t{:.0f}\tForecasted Demand:\t{:.0f}\tDiff={:.0f}".format(
                        total_demand, env.monthly_forecast[i,j], total_demand-env.monthly_forecast[i,j]))
                    passed = False
    return passed

# Visualize forecasts
def plot_net_forecast(env, gmid, month):
    v_adj = 2 # Adjust placement of vertical text
    plt.style.use('ggplot')
    #https://coolors.co/345995-f0c808-ff6700-34d1bf-4cb944
    colors = ['#345995',
              '#ff6700',
              '#4cb944',
              '#34d1bf',
              '#f0c808']
    plt.rcParams.update({'font.size': 8})
    days = env.month_to_sim_day[month]
    df = env.get_orders()
    # TODO: Update to match on forecast flag
    p_demand = df.loc[(df['gmid']==gmid) & 
                      (df['doc_num']>0) & 
                      (df['doc_create_time']<=env.sim_time) & 
                      (df['planned_gi_month']==month)]
    p_fcast = df.loc[(df['gmid']==gmid) &
                     (df['doc_num']<=0) & 
                     (df['planned_gi_month']==month)]

    # Visible demand at Day = 1
    act_agg = p_demand.groupby('planned_gi_time')['order_qty'].sum()
    if act_agg.values.size == 0:
        act_agg = pd.Series(np.zeros(len(days)), index=days)
    fcast_agg = p_fcast.groupby('planned_gi_time')['order_qty'].sum()

    gs.GridSpec(4, 4)
    fig = plt.figure(1, figsize=(12,15))
    # Actual Demand Plot
    plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    plt.bar(act_agg.index, act_agg, color=colors[0])
    plt.xlim([min(days) - 1, max(days) + 1])
    plt.xlabel('Day')
    plt.title('Actual Demand Visible at Day {} for Product {}'.format(
        env.sim_time, gmid))
    plt.ylabel('Demand (MT)')
    if env.fixed_planning_horizon + env.sim_time in days:
        plt.axvline(x=env.fixed_planning_horizon + env.sim_time,
                    c='k')
        plt.text(s='Fixed Schedule Horizon', 
                 x=env.fixed_planning_horizon + env.sim_time,
                 y=max(act_agg)/v_adj, rotation=270, fontsize=12)
        plt.axvline(x=env.lookahead_planning_horizon + env.sim_time, 
                    c='k')
        plt.text(s='Planning Schedule Horizon', 
                 x=env.lookahead_planning_horizon + env.sim_time, 
                 y=max(act_agg)/v_adj, rotation=270, fontsize=12)

    # Forecasted Demand Plot
    plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    plt.bar(fcast_agg.index, fcast_agg, color=colors[-1])
    plt.xlim([min(days) - 1, max(days) + 1])
    plt.xlabel('Day')
    plt.title('Forecasted Demand from Day {} for Product {}'.format(
        env.sim_time, gmid))
    if env.fixed_planning_horizon + env.sim_time in days:
        plt.axvline(x=env.fixed_planning_horizon + env.sim_time, 
                    c='k')
        plt.text(s='Fixed Schedule Horizon', 
                 x=env.fixed_planning_horizon + env.sim_time,
                 y=max(fcast_agg)/v_adj, rotation=270, fontsize=12)
        plt.axvline(x=env.lookahead_planning_horizon + env.sim_time,
                    c='k')
        plt.text(s='Planning Schedule Horizon',
                 x=env.lookahead_planning_horizon + env.sim_time,
                 y=max(fcast_agg)/v_adj, rotation=270, fontsize=12)

    # Net forecast
    net_agg = np.zeros(max(days) + 1)
    net_agg[fcast_agg.index] += fcast_agg.values
    bottoms = np.zeros(max(days) + 1)
    bottoms[act_agg.index] += act_agg.values

    plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=2)
    plt.bar(act_agg.index, act_agg, color=colors[0])
    plt.bar(days, net_agg[min(days):], color=colors[1], 
            bottom=bottoms[min(days):])
    plt.title('Net Forecast Total ({} MT)'.format(
        int(bottoms.sum() + net_agg.sum())))
    plt.xlim([min(days) - 1, max(days) + 1])
    plt.xlabel('Day')
    plt.ylabel('Demand (MT)')
    if env.fixed_planning_horizon + env.sim_time in days:
        plt.axvline(x=env.fixed_planning_horizon + env.sim_time, 
                    c='k')
        plt.text(s='Fixed Schedule Horizon', 
                 x=env.fixed_planning_horizon + env.sim_time,
                 y=max(net_agg + bottoms)/v_adj, rotation=270, fontsize=12)
        plt.axvline(x=env.lookahead_planning_horizon + env.sim_time, 
                    c='k')
        plt.text(s='Planning Schedule Horizon',
                 x=env.lookahead_planning_horizon + env.sim_time,
                 y=max(net_agg + bottoms)/v_adj, rotation=270, fontsize=12)
    plt.legend(prop={'size': 20})
    plt.tight_layout()
    plt.show()

def load_excel_demand_file(env):    
    file_name = "{0}.xlsx".format(env.settings["EXCEL_DEMAND_FILE"])
    ob_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"demand_files",file_name)
    order_book = pd.read_excel(ob_file)
    order_book = order_book.values 
    return order_book