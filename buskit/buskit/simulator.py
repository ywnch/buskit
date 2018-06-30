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
import csv
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

def remove_inactive(batch, live_bus):
    """
    remove buses from live_bus absent in the new stream batch
    """
    new_refs = [bus['VehicleRef'] for bus in batch]
    for ref in list(live_bus.keys()):
        if ref not in new_refs:
            del live_bus[ref]
    return live_bus

def update_bus(new, ref, pos, live_bus):
    """
    DEVELOPING
    """
    for bus in live_bus.values():
        if bus.ref == ref:
            bus.pos = pos

def stream_next(data, live_bus):
    batch = next(data)
    live_bus = remove_inactive(batch, live_bus)
    for bus in batch:
        ref = bus['VehicleRef']
        pos = float(bus['VehDistAlongRoute'])
#         bus_ref = [bus.ref for bus in live_bus]
        bus_ref = live_bus.keys()

        if ref not in bus_ref:
            live_bus[ref] = Bus(ref, pos)
        else:
            update_bus(bus, ref, pos, live_bus)
    return live_bus

def load_archive(filename, dir_ref):
    """
    stream in data batch by batch from given archive data
    """
    with open(filename, 'r') as fi:
        reader = csv.DictReader(fi)
        batch = []
        item = 0
        
        for row in reader:
            # seq number retrieved within each 30-sec query
            new_item = int(row[''])
            
            # new batch
            if new_item < item:
                yield batch
                batch = []

            # collect pings from the same batch
            if row['DirectionRef'] == str(dir_ref):
                batch.append(row)
                
            # update item number
            item = new_item

def read_data(filename, direction=0):
    """
    read stop and speed from data created by fetchbus.py
    """
    data = pd.read_csv(filename)
    
    # subset data for non-repeating stops
    data = data[data['DirectionRef'] == direction]
    data.drop_duplicates(['StopPointRef'], inplace=True) ##### may be problematic #####
    data.sort_values(['CallDistanceAlongRoute'], inplace=True)
    
    # read 1-D stop distances (in meters) and names
    stop_ref = np.array(data['StopPointRef'])
    stop_pos = np.array(data['CallDistanceAlongRoute'])
    stop_name = np.array(data['StopPointName'])
    
    stop_pos -= stop_pos[0] # reset first stop to 0 ### TEMPORARY MEASURE ###
    
    # read LTT or speed
    ### CURRENTLY UNAVAILABLE ###
    
    # read dwelling time
    ### CURRENTLY UNAVAILABLE ###
    
    return stop_ref, stop_pos, stop_name

def bus_tsd(bus):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(bus.log_pos)), bus.log_pos)
    plt.title("Time-space Diagram of Bus Ref. %s"%(bus.ref), fontsize=18)
    plt.xlabel("Timestep (second)", fontsize=14)
    plt.ylabel("Distance (meter)", fontsize=14)
    plt.show()
    
def stop_pax(stop):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(stop.log_pax)), stop.log_pax)
    plt.title("Passengers at Stop Ref. %s"%(stop.ref), fontsize=18)
    plt.xlabel("Timestep (second)", fontsize=14)
    plt.ylabel("Waiting Passengers", fontsize=14)
    plt.show()



############ THIS PART IS TEMPORARY ############

# specify path to archive AVL file
archive_path = '/Users/Yuwen/Dropbox/work_BusSimulator/MTA_data/B15-180625-235941-44650-Mon.csv'

# determine data source
beta = False

time_coef = 100000 # simulation time is __ times faster than the reality
avg_door_t = 5 # assume opening and closing the door take 5 seconds in total
avg_board_t = 3 # assume each boarding takes 3 sec
avg_alight_t = 2 # assume each alight takes 2 sec

if beta:
    # artificial data   ### make this part automatized with given number of stop
    stop_ref = np.array([1, 2, 3, 4, 5, 6, 7])
    stop_pos = np.array([0, 100, 200, 300, 400, 500, 600])
    stop_name = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
else:
    # historical data
    stop_ref, stop_pos, stop_name = read_data(archive_path, 1)

# speed and travel time data are currently artificial
link_vel = 1.5 * np.random.randn(len(stop_pos)) + 7 # make sure the unit is m/sec
#dwell_t = 7 * np.random.randn(len(stop_pos)) + 20 # make sure the unit is sec

# pax distribution
stop_pos_next = np.append(stop_pos, stop_pos[-1])[1:]

pos_mu = stop_pos.mean()
pos_std = stop_pos.std()
pax_norm = ss.norm(loc=pos_mu, scale=pos_std)
pax_perc = np.array([pax_norm.cdf(stop_pos_next[i]) - pax_norm.cdf(stop_pos[i]) for i in range(len(stop_pos))]) ### a temporary measure ###

pax_hr_route = 5000
pax_hr_stop = pax_hr_route * pax_perc
pax_at_stop = np.zeros(len(stop_pos))


# Bus class
class Bus(object):
    
    capacity = 60
    seat = 40
    
    def __init__(self, ref, pos=0):
        self.ref = ref # vehicle reference
        self.pos = pos # vehicle location (1-D)
        
        ############### fix this: stop_pos, link, vel, next_stop should not require pre-specify, how? ##############
        
        self.link = np.sum(self.pos >= stop_pos) - 1 # link index starts from 0  ### unified with the formula in Stop Class
        self.vel = link_vel[self.link] # speed at current link
        self.next_stop = stop_pos[self.link + 1] # position of next stop
        self.dwell_t = 0
        self.pax = 0
        self.clock = 0
        self.operate = True
        self.atstop = False
        
        self.log_pos = [self.pos]
        self.log_vel = [self.vel]
        self.log_pax = [0]
        self.log_dwell = [0]

    def terminal(self):
        print("The bus has reached the terminal")
        self.operate = False
        self.vel = 0
        self.pax = 0
        
    def stop(self):
        print("Bus %s is making a stop at %s (position %i)"%(self.ref, stop_name[self.link + 1], self.next_stop))
        self.atstop = True
        self.pax_to_board = pax_at_stop[self.link + 1] # check how many pax at stop
        self.board_t = self.pax * avg_board_t
        self.alight_t = 0 * avg_alight_t  #### TO DEVELOP
        self.dwell_t = avg_door_t + self.alight_t + self.board_t # supposed to dwell for this long
        self.clock += 1

#         self.vel = 0
#         self.pos += self.vel
        self.record()

    def move(self):
        pax_at_stop[self.link + 1] = 0 # clear all pax at stop
        self.log_dwell.append(self.dwell_t)
        # move on!
        self.atstop = False
        self.dwell_t = 0
        self.clock = 0
        self.link += 1
        self.pax = 0 # update pax onboard ###################
        self.record()
        self.vel = link_vel[self.link] # new link speed
        self.next_stop = stop_pos[self.link + 1] # new next stop

    def record(self):
        self.log_pos.append(self.pos)
        self.log_pax.append(self.pax)
        
    def proceed(self):
        if self.operate:
            if self.pos + self.vel >= stop_pos[-1]:
                self.terminal()
            elif self.pos + self.vel >= self.next_stop:  ### this judgement restricts from recording vel as 0 at stop, change to sth else
                self.stop()
                if self.clock >= self.dwell_t:
                    self.move()
            else:
                print("Current position of bus %s: %i"%(self.ref, self.pos))
                self.pos += self.vel
                self.record()
        else:
            print("Bus %s is not operating."%(self.ref))

############ THIS PART IS TEMPORARY ############