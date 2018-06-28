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

def plot_1D(df, figsize=(20,6), save=False, fname='1D'):
    stops_x = df['CallDistanceAlongRoute'].unique()
    stops_y = [0] * len(stops_x)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(stops_x, stops_y, 'o-', color='steelblue', markersize=5)

    # save figure locally if specified
    if save:
        plt.savefig("%s.png"%(fname), dpi=300)
    else:
        pass
    plt.show()

    return fig, ax

def plot_2D(df, basemap=False, figsize=(10,10), save=False, fname='2D'):
    # specify line and direction query
    linename = df['PublishedLineName'].unique()[0]
    direction = df['DirectionRef'].unique()[0]

    # read route shapefile
    gdf = gpd.read_file("MTA_shp/bus_routes_nyc_aug2017.shp")
    gdf.to_crs(epsg=4326, inplace=True)

    # plot route
    route_shp = gdf[gdf['route_dir'] == '%s_%s'%(linename, direction)]
    route_shp.plot(color='red', figsize=figsize)

    # save figure locally if specified
    if save:
        plt.savefig("%s.png"%(fname), dpi=300)
    else:
        pass

    # plot on a basemap or not
    # requires "mplleaflet" package
    if basemap:
        mlf.display()
    else:
        plt.show()

def dash_hist(df, time_coef=100):
    try:
        # specify line and direction query
        linename = df['PublishedLineName'].unique()[0]
        direction = df['DirectionRef'].unique()[0]

        # read route shapefile
        gdf = gpd.read_file("MTA_shp/bus_routes_nyc_aug2017.shp")
        gdf.to_crs(epsg=4326, inplace=True)

        # plot route
        route_shp = gdf[gdf['route_dir'] == '%s_%s'%(linename, direction)]

        # plot figure
        fig = plt.figure(figsize=(18,11))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(212)

        # plot CallDistanceAlongRoute (bus stops)
        stops = df['CallDistanceAlongRoute'].unique()
        left = [df['RecordedAtTime'].min()] * 12
        right = [df['RecordedAtTime'].max()] * 12
        ax1.plot([left, right], [stops, stops], color='gray', alpha=0.2);

        p0, = ax1.plot([], [], '-', color='steelblue') ##### redundant? #####

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
    except KeyboardInterrupt:
        print("Interrupted")
        
def plot_headway(df, stop_no):
    df = df.sort_values(['CallDistanceAlongRoute', 'RecordedAtTime'])
    linename = df['PublishedLineName'].unique()[0]
    
    stops = df['CallDistanceAlongRoute'].unique() # a list stops (their 1-D distance along route)
    stopname = df[df['CallDistanceAlongRoute'] == stops[stop_no]]['StopPointName'].iloc[0]
    
    # subset every vehicle that passed the stop
    stop = df[df['CallDistanceAlongRoute'] == stops[stop_no]].drop_duplicates('VehicleRef', keep='last')
    # calculate differences between each vehicle (in seconds)
    hws = stop['RecordedAtTime'].diff().astype('timedelta64[m]') + (stop['RecordedAtTime'].diff().astype('timedelta64[s]') % 60) / 60

    # plot headways
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    ax.bar(range(-1,len(hws)-1), hws, align='edge')
    ax.set_xticks(range(len(hws)))
    
    plt.title("Headways at Stop %s on Route %s"%(stopname, linename), fontsize=18)
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