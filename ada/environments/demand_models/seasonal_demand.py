
# Generate order book

# Orders are generated randomly each month of the year to reflect seasonality
# For training, the orders will keep the same probability distribution for
# each training episode, but be stochastically regenerated for each new
# episode. This will keep the statistics of each product consistent.

import numpy as np
import calendar
import string
import datetime
from .demand_utils import softmax
from argparse import ArgumentTypeError

def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False 
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_default_seasonal_demand_settings(settings):
    '''
    Inputs
    =========================================================================
    settings: dictionary of experimental settings

    Outputs
    =========================================================================
    settings: dictionary of experimental settings which includes demand model
    '''
    lead_time = 7 # Days
    std_lead_time = 2 # Days
    vsm_mean = 40 # Dollars
    vsm_std = 2 # Dollars
    load_level = 1 # Sets ratio between total demand and total theoretical supply
    defaults = {
    'ORDER_SIZE': 25, # MT
    'MEAN_LEAD_TIME': lead_time*24 if settings['BASE_TIME_INTERVAL']=='HOUR'
        else lead_time,
    'STD_LEAD_TIME': std_lead_time*24 if settings['BASE_TIME_INTERVAL']=='HOUR'
        else std_lead_time,
    'VAR_STD_MARGIN_MEAN': vsm_mean,
    'VAR_STD_MARGIN_STD': vsm_std,
    'LOAD_LEVEL': 1,
    'FORECAST': 'UNIFORM',
    'FORECAST_ACCURACY': 3
    }

    for key in defaults.keys():
        if key not in settings.keys():
            settings[key] = defaults[key]
        # elif key == 'FORECAST':
        #     settings[key] = str2bool(str(settings[key]))
        elif defaults[key] is not None:
            settings[key] = type(defaults[key])(settings[key])

    return settings

def generate_seasonal_orders(env):

    # Stochastically generate order book

    # Check to see if order statistics already exist, if not
    # generate them
    if env.order_statistics is None:
        # Average daily production
        avg_daily_prod = env.product_data[
            :,env.product_data_cols.index('min_run_time')].astype(float).mean() * env.product_data[
            :,env.product_data_cols.index('run_rate')].astype(float).mean()
        # Average yearly production
        avg_yearly_prod = avg_daily_prod * env.n_days
        # Adjust to order load level to account for higher demand years or lower
        yearly_demand_volume = env.settings['LOAD_LEVEL'] * avg_yearly_prod
        # Generate volume shares for each product using softmax
        # on standard normal draws
        x = np.random.standard_normal(env.n_products)
        prod_volume_shares = softmax(x)
        # Add some seasonality to each product
        seasonality_offset = np.random.standard_normal(1)
        # Take absolute value to ensure that products follow a similar 
        # demand pattern
        amplitude = np.abs(np.random.standard_normal(env.n_products))
        months = np.linspace(0, 2 * np.pi, 12)[:env.n_months]
        demand = np.sin(months + seasonality_offset).reshape(-1, 1) * amplitude
        monthly_prod_volume_shares = (1 + demand) * prod_volume_shares

        # Normalize everything so that each month sums to 1
        # total_monthly_demand determines how much total volume is to be shipped
        # in that month
        monthly_demand_share = softmax(monthly_prod_volume_shares.sum(axis=1))
        total_monthly_demand = monthly_demand_share * yearly_demand_volume
        # monthly_product_prob determines the probability that an order will be
        # shipped for a given product in a given month
        monthly_product_prob = softmax(demand, axis=1).T

        # Generate orders for each month
        num_orders_per_month = np.round(total_monthly_demand / 
            env.settings['ORDER_SIZE'], 0).astype(int)

        env.order_statistics = [monthly_product_prob, num_orders_per_month]

    # Build order book
    # doc_num = 0 reserved for forecasted orders
    doc_num = 1
    gen_order_placeholder = np.zeros((1, 8), dtype="int32")
    start_date = env.start_time
    year = str(start_date.year)
    start_date_np = np.datetime64(start_date, "D")
    products = env.product_data[:,2].astype(int)
    # Loop through each month to generate orders
    # TODO: this model relies on specific order statistic size.
    # e.g. if the sim starts in Feb, we'll get a key error because
    # the month number doesn't match the order_statistics index.
    for i in range(env.order_statistics[0].shape[0]):
        # Number of orders per month
        n_orders = env.order_statistics[1][i]
        if n_orders != 0:
            # Get the month
            month = i + 1
            # Convert the month to a string
            if month < 10:
                month_str = "0" + str(int(month))
            else:
                month_str = str(int(month))
            
            # Get the first of the month
            month_start = np.datetime64(year + "-" + month_str + "-01", "D")
            # Generate order numbers
            doc_nums = np.array([num for num in range(doc_num, int(doc_num + n_orders))])

            # Calculate doc creation dates and planned gi dates
            # TODO: KeyError arises during implementation runs. Calling continue because
            # the demand generated here should not matter as we'll be loading demand from
            # a demand file. May cause other issues, however.
            try:
                ship_dates = np.array(env.shipping_dict[month])
            except KeyError:
                if env.settings['TRAIN']:
                    raise KeyError("month {} not found in env.shipping_dict".format(month))
                else:
                    continue
            
            # Sample over available shipment dates for the full month to avoid
            # large bunches of orders for partial months
            if min(ship_dates) > 4 or max(ship_dates) < 26:
                last_day = calendar.monthrange(int(year), month)[1]
                ship_dates = np.arange(1, last_day + 1)
                for d in ship_dates:
                    sim_day = (d+month_start-start_date_np).astype(int)
                    if sim_day not in env.sim_day_to_date:
                        env.sim_day_to_date[sim_day] = [month, d]
                
            if env.settings['WEEKEND_SHIPMENTS'] == False:
                # Filter weekends
                weekends = [i[5:7] for i in calendar.monthcalendar(int(year), month)]
                weekends = np.array([i for k in weekends for i in k])
                ship_dates = ship_dates[np.in1d(ship_dates, weekends, 
                                                invert=True)] - 1
            
            planned_gi_dates = np.random.choice(ship_dates - 1, size=n_orders, 
                                                replace=True).astype("int")
            # Add 1 to ensure no lead times = 0 days
            lead_times = np.abs(np.random.normal(
                    env.settings['MEAN_LEAD_TIME'] - 1, 
                    env.settings['STD_LEAD_TIME'], 
                    size=n_orders)).round(0).astype("int") + 1
            # Back calculate the doc create dates based on the lead times
            doc_create_dates = planned_gi_dates - lead_times
            # Get dates in the year
            planned_gi_dates = np.array([(j + month_start) for j in planned_gi_dates])
            doc_create_dates = np.array([(j + month_start) for j in doc_create_dates])
            # Convert dates into days from start date
            planned_gi_dates = (planned_gi_dates - start_date_np).astype("int")
            doc_create_dates = (doc_create_dates - start_date_np).astype("int")
            planned_gi_month = np.repeat(month, len(planned_gi_dates))

            # Variable standard margins
            # TODO: I should also consider generating statistics for
            # price and other data for the orders.
            # TODO: Fix this VSM hardcoding
            vsm_standard = [28, 39, 40, 54]
            if env.settings['N_PRODUCTS'] == 4:
                try:
                    vsms = np.abs(np.random.normal(
                        vsm_standard[i],
                        env.settings['VAR_STD_MARGIN_STD'],
                        size=n_orders)).round(0).astype('int')
                except IndexError:
                    vsms = np.abs(np.random.normal(
                        env.settings['VAR_STD_MARGIN_MEAN'], 
                        env.settings['VAR_STD_MARGIN_STD'], 
                        size=n_orders)).round(0).astype("int")
            else:
                vsms = np.abs(np.random.normal(
                        env.settings['VAR_STD_MARGIN_MEAN'], 
                        env.settings['VAR_STD_MARGIN_STD'], 
                        size=n_orders)).round(0).astype("int")

            # Assign products
            mat_codes = np.random.choice(env.product_data[:,2].astype("int"), 
                                         size=n_orders,
                                         replace=True,
                                         p=env.order_statistics[0][i])
            # Get order quantities
            order_qtys = np.repeat(env.settings['ORDER_SIZE'], n_orders)

            # Combine values
            gen_orders = np.vstack([doc_nums, doc_create_dates, 
                planned_gi_dates, 
                planned_gi_month, 
                mat_codes, mat_codes, # One for ZEMI and one for ZFIN
                order_qtys, vsms]).T.astype("int")
            gen_order_placeholder = np.vstack([gen_order_placeholder, 
                                               gen_orders])
            doc_num = doc_nums.max() + 1

    # Add placeholder values for additional order book values
    zer = np.zeros((gen_order_placeholder.shape[0], 5), dtype="int")
    generated_orders = np.hstack([gen_order_placeholder, zer])
    # Set customer segment
    generated_orders[:, env.ob_indices['cust_segment']] = 1
    # Delete first placeholder value
    generated_orders = np.delete(generated_orders, 0, 0)

    return generated_orders