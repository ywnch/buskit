# Author: Yuwen Chang, NYU CUSP
# Last Updated: 2018/03/17
##############################
# Code written for Bus Simulator
# https://github.com/ywnch/BusSimulator
##############################
# put MTA API key, bus route, bus direction, and duration as input arguments:
# i.e. run the code as:
# 	python fetchbus.py <MTA_KEY> <BUS_LINE> <BUS_DIRECTION> <DURATION>
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
from datetime import datetime

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

# function for fetching bus data
def bus_data(apikey, route, direction=0, duration=5):
    """
    Fetch MTA real-time bus location data for specified route and direction
    in a given duration.

    ARGUMENTS
    ----------
    apikey: API key for MTA data
    route: route reference (e.g., B54)
    direction: 0 or 1
    duration: minutes of data to fetch (by 30-second intervals)
    
    RETURN
    ----------
    - a dataframe of vehicle location data
    - a csv file containing the same data
    """
    
    ts = datetime.now()
    dow = calendar.day_name[ts.weekday()][:3]
    ts_date = str(ts.year) + str(ts.month) + str(ts.day)
    ts_time = str(ts.hour) + str(ts.minute) + str(ts.second)
    filename = '%s-%s-%s-%s-%s-%s.csv'%(route, direction, dow, ts_date, ts_time, duration)
    
    duration = int(duration) * 60 # minutes to seconds
    t_elapsed = 0
    
    url = "http://bustime.mta.info/api/siri/vehicle-monitoring.json?key=%s&VehicleMonitoringDetailLevel=calls&LineRef=%s&DirectionRef=%s"%(apikey, route, direction)
    
    df = pd.DataFrame(columns=['VehicleRef', 'Latitude', 'Longitude',
                               'StopName', 'StopStatus', 'VehDistAlongRoute'])
    while True:
        
        # fetch data
        response = urllib.urlopen(url)
        data = response.read().decode("utf-8")
        data = json.loads(data)

        # check if bus route exists
        try:
            data2 = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['VehicleActivity']
        except:
            error = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['ErrorCondition']
            print(error['Description'])
            sys.exit()

        # parse route info: line number, active vehicles
        print("Time Elapsed: " + str(t_elapsed/60) + " min(s)")
        print("Bus Line: " + route)
        print("Number of Active Buses: " + str(len(data2)))

        # parse info
        vrf = []
        lat = []
        lng = []
        stop = []
        status = []
        vdars = []
        tiempos = []

        # parse vehicle info: location (lat, long)
        for i, v in enumerate(data2):
            # parse time
            tiempo = v['RecordedAtTime']
            # parse location
            vr = v['MonitoredVehicleJourney']['VehicleRef'].split("_")[1]
            lt = v['MonitoredVehicleJourney']['VehicleLocation']['Latitude']
            lg = v['MonitoredVehicleJourney']['VehicleLocation']['Longitude']
            vrf.append(vr)
            lat.append(lt)
            lng.append(lg)

            # parse next stop info
            if 'OnwardCall' in v['MonitoredVehicleJourney']['OnwardCalls']:
                s = v['MonitoredVehicleJourney']['OnwardCalls']['OnwardCall'][0]
                stt = s['Extensions']['Distances']['PresentableDistance']
                stp = s['StopPointName']
                cdar = s['Extensions']['Distances']['CallDistanceAlongRoute']
                dfc = s['Extensions']['Distances']['DistanceFromCall']
                vdar = cdar - dfc
            else:
                stt = 'N/A'
                stp = 'N/A'
            status.append(stt)
            stop.append(stp)
            vdars.append(vdar)
            tiempos.append(tiempo)

            # print info
            print("Bus %s (#%s) is at latitude %s and longitude %s"%(i+1, vr, lt, lg))

        # write data to dictionary, indexing from 1
        d = {'VehicleRef': vrf, 'Latitude': lat, 'Longitude': lng,
             'StopName': stop, 'StopStatus': status, 'VehDistAlongRoute': vdars,
             'RecordedAtTime': tiempos}
        df = pd.concat([df, pd.DataFrame(data=d)])        
        df.to_csv(filename)
        
        if t_elapsed < duration:
            t_elapsed += 30
            time.sleep(30)
        else:
            return(df)

# function for plotting time-space diagram
def plot_tsd(df, start_min=None, end_min=None, save=False, fname='TSD'):
    """
    Plot the time-space diagram for a given dataframe containing
    real-time MTA bus data (as generated from fetchbus.py)
    
    ARGUMENTS
    ----------
    df: input dataframe containing required columns for plotting time-space diagram
    start_min: plot from this given minute (time elapsed)
    end_min: plot until this given minute (time elapsed)
    save: save TSD to .png at current directory
    fname: assign a filename for saved TSD (otherwise may be overwritten)
    
    RETURN
    ----------
    - fig
    - ax
    - a saved TSD .png file (optional)
    """
    # determine time interval to be plotted
    try:
        s = start_min * 2 # * 60 sec / 30 sec interval
        e = end_min * 2
    except:
        s = start_min
        e = end_min
    
    # convert time format
    df['RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime'])

    # plot
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    
    for i, v in enumerate(df['VehicleRef'].unique()):
        veh_df = df[df['VehicleRef'] == v]
        veh_df = veh_df.iloc[s:e,:]
        
        ax.plot(veh_df['RecordedAtTime'], veh_df['VehDistAlongRoute'], marker='.')
        ax.annotate('%s'%v, (list(veh_df['RecordedAtTime'])[0],list(veh_df['VehDistAlongRoute'])[0]))
        
        ax.set_xlabel("time (sec)", fontsize=14)
        ax.set_ylabel("distance along route (m)", fontsize=14)
        ax.set_title("Time-space Diagram", fontsize=18)
    
    plt.tight_layout()
    if save:
        plt.savefig("%s.png"%(fname), dpi=300)
    else:
        pass
    plt.show()
    
    return(fig, ax)

# check input args
if __name__ == '__main__':
    if not len(sys.argv) == 5:
        print ("Invalid number of arguments. Run as: python show_bus_locations_ywc249.py <MTA_KEY> <BUS_LINE>")
        sys.exit()

    # read args and fetch data
    df = bus_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
