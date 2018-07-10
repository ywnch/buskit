# Author: Yuwen Chang, NYU CUSP
# Last Updated: 2018/05/16
##############################
# Code written for Bus Simulator
# https://github.com/ywnch/BusSimulator
##############################
# put MTA API key, bus route, and duration as input arguments:
# i.e. run the code as:
# 	python busdata.py <MTA_KEY> <BUS_LINE> <DURATION>
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
from datetime import datetime, timedelta

from .dashboard import *

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

# function for streaming bus data
def stream_bus(apikey, linename, duration=5):
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
    filename = '%s-%s-%s-%s.csv'%(linename, ts.strftime('%y%m%d-%H%M%S'), round(duration), dow)
    
    # set up parameters
    t_elapsed = 0 # timer
    duration = duration * 60 # minutes to seconds
    url = "http://bustime.mta.info/api/siri/vehicle-monitoring.json?key=%s&VehicleMonitoringDetailLevel=calls&LineRef=%s"%(apikey, linename)
    df = pd.DataFrame() # empty dataframe
    
    # main block for fetching data
    while t_elapsed <= duration:
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
        print("Bus Line: " + linename)
        print("Number of Active Buses: " + str(len(data2)))

        # parse the data of each active vehicle
        for i, v in enumerate(data2):
            #if 'OnwardCall' in v['MonitoredVehicleJourney']['OnwardCalls']:
            ### ^ seems like a better way to avoid null calls
            try:
                # map variables
                dict1 = flatten(v['MonitoredVehicleJourney'])
                dict1['RecordedAtTime'] = v['RecordedAtTime']
                #dict1['SituationSimpleRef'] = dict1['SituationRef'][0]['SituationSimpleRef']
                dict1.pop('SituationRef')
                dict1.pop('OnwardCall')

                # print info of the vehicle
                print("Bus %s (#%s) is at latitude %s and longitude %s, heading for %s (direction: %s)"%(i+1,
                      dict1['VehicleRef'], dict1['Latitude'], dict1['Longitude'], dict1['DestinationName'], dict1['DirectionRef']))

                # write data to dictionary
                df = pd.concat([df, pd.DataFrame(dict1, index=[i])])
        
            except:
                e = sys.exc_info()[0]
                print("Error: %s"%e)

        ### preprocessing ###
        # calculate vehicle distance along the route
        df.loc[:,'VehDistAlongRoute'] = df['CallDistanceAlongRoute'] - df['DistanceFromCall']

        # write/update data to csv
        df.to_csv(filename)

        # sleep and update timer
        if t_elapsed < duration:
            time.sleep(30)
            
        t_elapsed += 30

    return df

# function for splitting multiple trips of same vehicles
def split_trips(df):
    """
    Split different trips made by the same vehicle within
    a given dataframe containing real-time MTA bus data

    PARAMETERS
    ----------
    df: pd.DataFrame
        Input dataframe containing standard MTA SIRI variables.
        
    RETURNS
    -------
    df_all: pd.DataFrame
        Output dataframe with NewVehicleRef that is split
    """

    def split_oneway(df):
        """
        The core function for splitting, works for one direction at a time.
        ### This may be slow for bigger datasets and requires optimization in the future ###
        """
        dfs = []
        for v in df['VehicleRef'].unique():       
            
            trip = 1 # start with trip no. 1
            NewVehicleRef = [] # include initial one for the first record?: v + '_' + str(trip)
            
            test = df[df['VehicleRef'] == v].sort_index()
            
            for boo in list(test['CallDistanceAlongRoute'].diff() < -2000):
                if boo: # if this is a new trip (which jumps back more than 2 km)
                    trip += 1 # assign new trip no.
                NewVehicleRef.append(v + '_' + str(trip)) # each iteration, append new vehicle ref
            
            test.loc[:,'NewVehicleRef'] = NewVehicleRef
            dfs.append(test)
        df_all = pd.concat(dfs)
        return df_all

    # split by different directions
    dfs = []
    for d in df['DirectionRef'].unique():
        df_dir = df[df['DirectionRef'] == d]
        df_dir = split_oneway(df_dir)
        dfs.append(df_dir)
    df_all = pd.concat(dfs)
    df_all.sort_values("RecordedAtTime", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    return df_all

def df_process(df, dir_ref):
    # subset df for given direction
    df = df[df['DirectionRef'] == dir_ref]

    # calculate vehicle distance along the route
    df.loc[:,'VehDistAlongRoute'] = df['CallDistanceAlongRoute'] - df['DistanceFromCall']
    
    # convert time format
    df.loc[:,'RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime']) \
                                   .dt.tz_localize('UTC') \
                                   .dt.tz_convert('America/New_York')

    return df

### this function is to be removed in future ###
def df_addts(df):
    """ this bloc generate a timestep sequence for psuedo real-time simulation """
    mod = list(df['Unnamed: 0'])
    tiempo = 0
    ts = []
    for i,v in enumerate(mod[:-1]):
        ts.append(tiempo)
        if mod[i+1] < mod[i]:
            tiempo += 30
        else:
            continue
    ts.append(ts[-1])
    df.loc[:,'ts'] = ts
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
    ### MORE PLOTTING KWARGS TO BE ADDED ###

    # pre-process the data if necessary
    try:
        df = df_process(df, dir_ref)
    except:
        pass

    # determine time interval to be plotted
    start = df["RecordedAtTime"].min()
    end = df["RecordedAtTime"].max()

    s = start if start_min == None else start + timedelta(minutes=start_min)
    e = end if end_min == None else start + timedelta(minutes=end_min)

    bool1 = np.array(df['RecordedAtTime'] > s)
    bool2 = np.array(df['RecordedAtTime'] < e)
    df = df[bool1 & bool2]
    
    # check if trips are split already
    try:
        vehref = df['NewVehicleRef'] # use split vehicles if available
    except:
        vehref = df['VehicleRef']

    # plot figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    # plot CallDistanceAlongRoute (bus stops)
    stops = df['CallDistanceAlongRoute'].unique()
    left = [df['RecordedAtTime'].min()] * len(stops)
    right = [df['RecordedAtTime'].max()] * len(stops)
    ax.plot([left, right], [stops, stops], color='gray', alpha=0.2);

    # plot the trajectory for each vehicle
    for i, v in enumerate(vehref.unique()):
        # subset data for single vehicle
        veh_df = df[vehref == v]
        
        ax.plot(veh_df['RecordedAtTime'], veh_df['VehDistAlongRoute'], marker='.')
        ax.annotate('%s'%v.split("_")[1],
                    (list(veh_df['RecordedAtTime'])[0], list(veh_df['VehDistAlongRoute'])[0]))
        
    ax.grid()
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Distance along route (meter)", fontsize=14)
    ax.set_title("Time-space Diagram of Bus %s (direction: %s) from %s to %s"%(
                 df['PublishedLineName'].unique()[0], df['DirectionRef'].unique()[0],
                 str(s.time())[:-3], str(e.time())[:-3]), fontsize=18)
    
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
        print ("Invalid number of arguments. Run as: python busdata.py <MTA_KEY> <BUS_LINE> <DURATION>")
        sys.exit()

    # read args and fetch data
    df = stream_bus(sys.argv[1], sys.argv[2], sys.argv[3])
