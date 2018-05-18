# Author: Yuwen Chang, NYU CUSP
# Last Updated: 2018/05/16
##############################
# Code written for Bus Simulator
# https://github.com/ywnch/BusSimulator
##############################

# import packages
from __future__ import print_function, division
from IPython.display import display, clear_output, Image

import os
import sys
import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fiona
import folium
import geopandas as gpd
import mplleaflet as mlf
from shapely.geometry import Point

import time
import calendar
from datetime import datetime

import collections
from collections import defaultdict

import scipy.stats as ss
from .busdata import *

try:
    import urllib2 as urllib
    from urllib2 import HTTPError
    from urllib2 import urlopen
    from urllib import urlencode
    from StringIO import StringIO as io

except ImportError:
    import urllib.request as urllib
    from urllib.error import HTTPError
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
    from io import BytesIO as io


def df_process(df, direction):
    """ Pre-process data for plotting"""
    df = df[df['DirectionRef'] == direction]
    df['RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime'])
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
    df['ts'] = ts
    return df

def dash_hist(df, route_shp):
    time_coef = 100

    # plot figure
    fig = plt.figure(figsize=(18,11))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)

    # plot CallDistanceAlongRoute (bus stops)
    stops = df['CallDistanceAlongRoute'].unique()
    left = [df['RecordedAtTime'].min()] * 12
    right = [df['RecordedAtTime'].max()] * 12
    ax.plot([left, right], [stops, stops], color='gray', alpha=0.2);

    p0, = ax1.plot([], [], '-', color='steelblue')

    ax1.grid()
    ax1.set_xlabel("time", fontsize=14)
    ax1.set_ylabel("Distance along route (m)", fontsize=14)
    ax1.set_title("Time-space Diagram", fontsize=16)

    # plot route shape on map (2-D)
    route_shp.plot(ax=ax2)
    p1, = ax2.plot([], [], 'o', color='lawngreen')
    p2, = ax2.plot([], [], 'o', color='indianred')

    ax2.set_ylabel("Latitude", fontsize=14)
    ax2.set_xlabel("Longitude", fontsize=14)
    ax2.set_title("Active Vehicles on Route (Map)", fontsize=16)

    # plot dynamic route line (1-D)
    ax3.plot(df['CallDistanceAlongRoute'], [0]*len(df), '.-', color='steelblue')
    p3, = ax3.plot([], [], 'o', color='lawngreen')
    p4, = ax3.plot([], [], 'o', color='indianred')

    ax3.set_yticks([])
    ax3.set_xlabel("Distance along route (m)", fontsize=14)
    ax3.set_title("Active Vehicles on Route (1-D)", fontsize=16)

    plt.suptitle("Streaming Bus Trajectories for Route M1", fontsize=22)

    # update
    for i in range(0,df['ts'].max(),30):
    #for i in range(0,1500,30): # just do partial for demo
        df3 = df[df['ts'] == i]
        df1 = df[df['ts'] <= i]

        # mark vehicles that are bunching 
        df3.sort_values(['VehDistAlongRoute'], inplace=True)
        spacing = np.diff(df3['VehDistAlongRoute'])
        bunch = spacing < 100 # set threshold (meters) to be identified as BB
        bunch_a = np.array([False] + list(bunch))
        bunch_b = np.array(list(bunch) + [False])
        bunch = bunch_a + bunch_b
        bb_df = df3[bunch]

        # plot TSD for each vehicle
        for i, v in enumerate(df1['VehicleRef'].unique()):
            # subset data for single vehicle
            veh_df = df1[df1['VehicleRef'] == v]
            ax1.plot(veh_df['RecordedAtTime'], veh_df['VehDistAlongRoute'], '-', color='steelblue', alpha=0.5)
            ax1.plot(bb_df['RecordedAtTime'], bb_df['VehDistAlongRoute'], 'o', color='indianred', alpha=0.5)
            #ax1.annotate('%s'%v.split("_")[1], (list(veh_df['RecordedAtTime'])[0],list(veh_df['VehDistAlongRoute'])[0]))

        p1.set_data(df3['Longitude'], df3['Latitude'])
        p2.set_data(bb_df['Longitude'], bb_df['Latitude'])
        p3.set_data(df3['VehDistAlongRoute'], [0]*len(df3))
        p4.set_data(bb_df['VehDistAlongRoute'], [0]*len(bb_df))
        clear_output(wait=True)
        display(fig)
        print("Timestep: %s"%(i))
        time.sleep(1/time_coef)
        
def plot_headway(df, lineref, stop_no):
    df2 = df.sort_values(['CallDistanceAlongRoute', 'RecordedAtTime'])
    
    stops = df2['CallDistanceAlongRoute'].unique() # a list stops (their 1-D distance along route)
    stopname = df2[df2['CallDistanceAlongRoute'] == stops[stop_no]]['StopPointName'].iloc[0]
    
    # subset every vehicle that passed the stop
    stop = df2[df2['CallDistanceAlongRoute'] == stops[stop_no]].drop_duplicates('VehicleRef', keep='last')
    # calculate differences between each vehicle (in seconds)
    hws = stop['RecordedAtTime'].diff().astype('timedelta64[m]') + (stop['RecordedAtTime'].diff().astype('timedelta64[s]') % 60) / 60

    # plot headways
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    ax.bar(range(-1,len(hws)-1), hws, align='edge')
    ax.set_xticks(range(len(hws)))
    
    plt.title("Headways at Stop %s on Route %s"%(stopname, lineref), fontsize=18)
    plt.xlabel("No. of Vehicle (Chronological)", fontsize=14)
    plt.ylabel("Headway (minutes)", fontsize=14)
    plt.show()

SQL_SOURCE = 'https://ywc249.carto.com/api/v2/sql?q='
def queryCartoDB(query, format='CSV', source=SQL_SOURCE):
    """
    Queries carto datasets from a given carto account
    
    ARGUMENTS
    ----------
    query: a valid sql query string
    format: outlut format (default: CSV)
    source: a valid sql api endpoint OPTIONAL (default: Carto ywc249 account)

    RETURN
    ----------
    - the return of the sql query AS A STRING
    
    NOTES
    ----------
    designed for the carto API, tested only with CSV return format
    """
    # set SQL source
    
    data = urlencode({'format':format, 'q':query}).encode('utf-8')
    try:
        response = urlopen(source, data)
    except HTTPError as e:
        raise ValueError('\n'.join(ast.literal_eval(e.readline())['error']))
    except Exception:
        raise
    return response.read()