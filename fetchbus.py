# Author: Yuwen Chang, NYU CUSP
# Last Updated: 2018/04/09
##############################
# Code written for Bus Simulator
# https://github.com/ywnch/BusSimulator
##############################
# put MTA API key, bus route, and duration as input arguments:
# i.e. run the code as:
# 	python fetchbus.py <MTA_KEY> <BUS_LINE> <DURATION>
##############################

# initialize
from __future__ import print_function, division
__author__ = 'Yuwen Chang (ywnch)'

# import packages
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import calendar
import collections
from datetime import datetime

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

# function for dictionary flattening
# code by Imran adopted and modified from
# https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
def flatten(d, parent_key='', sep='_'):
    """
    Flatten the data of a nested dictionary.
    
    PARAMETERS
    ----------
    d: dictionary
        a nested dictionary
        
    RETURNS
    -------
    dict(items): dictionary
        a dictionary with all items unpacked from their nests
    """
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# function for fetching bus data
def bus_data(apikey, route, duration=5):
    """
    Fetch MTA real-time bus location data for specified route and direction
    in a given duration.

    PARAMETERS
    ----------
    apikey: string
        API key for MTA data
    route: string
        route reference (e.g., B54)
    duration: integer
        minutes of data to fetch (by 30-second intervals)
        
    RETURNS
    -------
    df: pd.DataFrame
        a dataframe of all SIRI variables for real-time bus trajectories
    filename.csv: csv
        a csv file containing the same data saved at local folder
    """
    
    # name the output csv file
    ts = datetime.now()
    dow = calendar.day_name[ts.weekday()][:3]
    filename = '%s-%s-%s-%s.csv'%(route, dow, ts.strftime('%y%m%d-%H%M%S'), duration)
    
    # set up parameters
    t_elapsed = 0 # timer
    duration = int(duration) * 60 # minutes to seconds
    url = "http://bustime.mta.info/api/siri/vehicle-monitoring.json?key=%s&VehicleMonitoringDetailLevel=calls&LineRef=%s"%(apikey, route)
    df = pd.DataFrame() # empty dataframe
    
    # main block for fetching data
    while True:
        # fetch data through MTA API
        response = urllib.urlopen(url)
        data = response.read().decode("utf-8")
        data = json.loads(data)

        # check if bus route exists
        try:
            data2 = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['VehicleActivity']
        # print error if bus route not found
        except:
            error = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['ErrorCondition']
            print(error['Description'])
            sys.exit()

        # print info of the current query request
        print("\nTime Elapsed: " + str(t_elapsed/60) + " min(s)")
        print("Bus Line: " + route)
        print("Number of Active Buses: " + str(len(data2)))

        # parse the data of each active vehicle
        for i, v in enumerate(data2):
            #if 'OnwardCall' in v['MonitoredVehicleJourney']['OnwardCalls']:
            try:
                # map variables
                dict1 = flatten(v['MonitoredVehicleJourney'])
                dict1['RecordedAtTime'] = v['RecordedAtTime']
                #dict1['SituationSimpleRef'] = dict1['SituationRef'][0]['SituationSimpleRef']
                dict1.pop('SituationRef')
                dict1.pop('OnwardCall')

                # print info of the vehicle
                print("Bus %s (#%s) is at latitude %s and longitude %s"%(i+1, dict1['VehicleRef'], dict1['Latitude'], dict1['Longitude']))

                # write data to dictionary
                df = pd.concat([df, pd.DataFrame(dict1, index=[i])])
        
            except:
                e = sys.exc_info()[0]
                print("Error: %s"%e)

        # write/update data to csv
        df.to_csv(filename)

        # check and update timer
        #UPDATE THIS TO WHILE LOOP IN FUTURE VERSION
        if t_elapsed < duration:
            t_elapsed += 30
            time.sleep(30)
        else:
            return df

# function for plotting time-space diagram
def plot_tsd(df, dir_ref=0, start_min=None, end_min=None, save=False, fname='TSD'):
    """
    Plot the time-space diagram for a given dataframe containing
    real-time MTA bus data (as generated from fetchbus.py)
    PARAMETERS
    ----------
    df: pd.DataFrame
        Input dataframe containing required columns for plotting time-space diagram.
    dir_ref: integer (0 or 1)
        The direction to be plotted.
    start_min: numeric
        Plot from this given minute (time elapsed).
    end_min: numeric
        Plot until this given minute (time elapsed).
    save: boolean
        Save TSD to .png at current directory.
    fname: string
        Assign a filename for saved TSD (otherwise may be overwritten).
        
    RETURNS
    -------
    fig:
    ax:
    filename.png: png
        a saved TSD file (optional)
    """
    # determine time interval to be plotted
    try:
        s = start_min * 2 # * 60 sec / 30 sec interval
        e = end_min * 2
    except:
        s = start_min
        e = end_min
    
    # subset df for given direction
    df = df[df['DirectionRef'] == dir_ref]
    
    # convert time format
    df['RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime'])

    # plot figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    
    # calculate vehicle distance along the route
    df['VehDistAlongRoute'] = df['CallDistanceAlongRoute'] - df['DistanceFromCall']
    
    # plot the trajectory for each vehicle
    for i, v in enumerate(df['VehicleRef'].unique()):
        # subset data for single vehicle
        veh_df = df[df['VehicleRef'] == v]
        # subset within specified time window
        veh_df = veh_df.iloc[s:e,:]
        
        # plot CallDistanceAlongRoute (bus stops)
        [ax.plot([df['RecordedAtTime'].min(), df['RecordedAtTime'].max()], [i, i], color='gray', alpha=0.1) for i in df['CallDistanceAlongRoute'].unique()]
        
        ax.plot(veh_df['RecordedAtTime'], veh_df['VehDistAlongRoute'], marker='.')
        ax.annotate('%s'%v.split("_")[1], (list(veh_df['RecordedAtTime'])[0],list(veh_df['VehDistAlongRoute'])[0]))
        
        ax.grid()
        ax.set_xlabel("time", fontsize=14)
        ax.set_ylabel("distance along route (m)", fontsize=14)
        ax.set_title("Time-space Diagram", fontsize=18)
    
    plt.tight_layout()
    
    # save figure locally if specified
    if save:
        plt.savefig("%s.png"%(fname), dpi=300)
    else:
        pass
    plt.show()
    
    return fig, ax

# check input args
if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print ("Invalid number of arguments. Run as: python fetchbus.py <MTA_KEY> <BUS_LINE> <DURATION>")
        sys.exit()

    # read args and fetch data
    df = bus_data(sys.argv[1], sys.argv[2], sys.argv[3])

    # # name the output png file (!!!SHOULD UPDATE THIS PART!!!)
    # ts = datetime.now()
    # dow = calendar.day_name[ts.weekday()][:3]
    # filename = '%s-%s-%s-%s'%(sys.argv[2], dow, ts.strftime('%y%m%d-%H%M%S'), sys.argv[3])

    # # plot time-space diagram
    # plot_tsd(df, save=True, fname=filename)
