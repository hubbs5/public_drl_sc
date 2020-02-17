# Create a list of days where shipments are possible 
# Should also get a list of Dutch holidays
# Shipping dictionary provides shipment days available
# for each month in the year
from datetime import datetime, timedelta
from calendar import monthrange
import numpy as np

def create_shipping_calendar(start_day, n_days, weekends=False):
    '''
    Generate calendar of days which allows shipping

    Inputs
    =========================================================================
    n_days: int, number of simulation days
    start_day: datetime object, first day of simulation
    weekends: boolean, value which determines whether or not it is permissible
        to ship on a weekend. False forbids weekend shipments; True permits 
        such shipments.

    Outputs
    =========================================================================
    shipping_calendar: list, provides list of days where shipping is 
        is permitted (e.g. only weekdays).
    shipping_dict: dictionary, keys are the months and the values are the
        corresponding days where shipping is permitted.
    '''
    shipping_calendar = []
    shipping_dict = {} # Maps months to days
    sim_day_to_date = {} # Maps shipping days to month
    for i in range(366): # Need complete calendar for training purposes # range(n_days+1):
        date = (timedelta(i) + start_day)
        day_of_month = date.day
        month = date.month
        sim_day_to_date[i] = [month, day_of_month]
        # Weekends are marked by 4 and 5
        if weekends == False and (date.weekday() == 4 or date.weekday() == 5):
            continue
        shipping_calendar.append(i+1)
        if month not in shipping_dict:
            shipping_dict[month] = [day_of_month]
        else:
            shipping_dict[month].append(day_of_month)

    return shipping_calendar, shipping_dict, sim_day_to_date

def get_month_to_sim_day_dict(sim_day_to_date):
    month_to_sim_day = {}
    for i in sim_day_to_date.items():
        if i[1][0] not in month_to_sim_day.keys():
            month_to_sim_day[i[1][0]] = [i[0]]
        else:
            month_to_sim_day[i[1][0]].append(i[0])
    return {m: np.array(month_to_sim_day[m]) for m in month_to_sim_day.keys()}