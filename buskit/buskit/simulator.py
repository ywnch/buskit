# Author: Yuwen Chang, NYU CUSP
# Last Updated: 2018/06/30
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
import dateutil
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
    FUTURE: also remove ones that reaches the last stop
    """
    new_refs = [bus['VehicleRef'] for bus in batch]
    for ref in list(live_bus.keys()):
        if ref not in new_refs:
            del live_bus[ref]
    return live_bus

def update_bus(new, ref, pos, time, live_bus, stop_pos):
    """
    DEVELOPING
    """
    for bus in live_bus.values():
        if bus.ref == ref:
            bus.pos = pos

def stream_next(data, live_bus, stop_pos):
    """
    major function for stream update
    data: streaming data, i.e., archive generator through load_archive() or live feed
    live_bus: currently active dictionary of bus objects
    """
    # read next batch of streaming data
    batch = next(data)
    # remove buses that did not show up in new batch
    live_bus = remove_inactive(batch, live_bus)

    for bus in batch:
        ref = bus['VehicleRef']
        pos = float(bus['VehDistAlongRoute'])
        time = dateutil.parser.parse(bus['RecordedAtTime'])
        new = Bus(ref, pos, time, stop_pos)
#         bus_ref = [bus.ref for bus in live_bus]
        bus_ref = live_bus.keys()

        # add buses that did not show up in existing set
        if ref not in bus_ref:
            live_bus[ref] = new
        # update bus and link information between two pings
        else:
            update_bus(new, live_bus)
    return live_bus

def load_archive(filename, direction=0):
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
            if row['DirectionRef'] == str(direction):
                batch.append(row)
                
            # update item number
            item = new_item

def load_stops(filename, direction=0):
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
    
    # read LTT or speed
    ### CURRENTLY UNAVAILABLE ###
    
    # read dwelling time
    ### CURRENTLY UNAVAILABLE ###
    
    return stop_ref, stop_pos, stop_name

###############
# ROUTE SETUP #
###############

def set_stops(filename, direction=0):
    stops = {}
    stop_ref, stop_pos, stop_name = load_stops(filename, direction)
    for i in np.arange(len(stop_ref)):
        stops[i] = Stop(i, stop_ref[i], stop_pos[i], stop_name[i])
    return stops, stop_pos

def set_links(stops):
    links = {}
    for k, v in stops.items():
        links[k] = Link(k, v.ref)
    return links

def set_route(filename, direction=0):
    """
    Extract stop information from archive and generate stops and links.
    FUTURE: specify route reference.

    stops: a dictionary of Stop objects along the route
    links: a dictionary of Link objects along the route
    stop_pos: an array of stop distances along the route
    """
    stops, stop_pos = set_stops(filename, direction)
    links = set_links(stops)
    return stops, links, stop_pos


################
# OBJECT SETUP #
################

class Stop(object):

    def __init__(self, idx, ref, pos, name):
        self.idx = idx # link index
        self.ref = ref # stop reference
        self.pos = pos # stop location (1-D)
        self.name = name # stop name
#         self.link = list(stop_ref).index(self.ref) # link index starts from 0
        self.clock = 0

        self.log_pax = [0]
        self.log_wait_t = [0]

    def update_dwell(self):
        pass

class Link(object):
    
    def __init__(self, idx, origin):
        self.idx = idx # link index
        self.origin = origin # origin stop of this link
        self.speed_window = [7, 7, 7, 7, 7]
        self.speed = 7
        
    def update_speed(self, new):
        self.speed_window.pop(0)
        self.speed_window.append(new)
        self.speed = np.mean(self.speed_window)

# Bus class
class Bus(object):
    
    capacity = 60
    seat = 40
    stop_range = 100
    
    def __init__(self, ref, pos, time, stop_pos):
        self.ref = ref # vehicle reference
        self.pos = pos # vehicle location (1-D)
        self.time = time # timestamp
        self.stop_pos = stop_pos ##### IS IT POSSIBLE TO MAKE THIS GLOBAL INSTEAD? #####
        self.link = sum(self.pos > self.stop_pos) - 1 # link index the bus is at
        # at stop if within specified range of a stop
        # at_prev if pos is ahead of the closest stop (in range)
        self.at_prev = self.pos - self.stop_pos[self.link] <= stop_range
        # at_next if pos is before the closest stop (in range)
        self.at_next = self.pos - self.stop_pos[self.link+1] >= -stop_range
        # at stop if either at_prev or at_next
        self.at_stop = at_prev or at_next
        ##### CONSIDER SIMPLIFYING THE ABOVE AS A SINGLE "NONE" or "INDEX" ATTRIBUTE #####

        # assign stop index that the bus is at
        ##### CURRENTLY IGNORING OVERLAPPING CASES #####
        if at_stop:
            if at_next:
                self.at_stop_idx = link + 1
            else:
                self.at_stop_idx = link

#############
# ANALYTICS #
#############

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
