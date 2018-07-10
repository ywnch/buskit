# Author: Yuwen Chang, NYU CUSP
# Last Updated: 2018/07/10
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
from datetime import timedelta

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

#############
# STREAMING #
#############

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

def stream_next(data, live_bus, stops, links, stop_pos):
    """
    major function for stream update
    data: streaming data, i.e., archive generator through load_archive() or live feed
    live_bus: currently active dictionary of Bus objects
    """
    # read next batch of streaming data
    batch = next(data)

    # remove buses that did not show up in new batch
    ############### live_bus = remove_inactive(batch, live_bus, sim=False) ###############

    bus_ref = live_bus.keys()

    for bus in batch:
        # ignore layover pings
        if bus['ProgressStatus'] == 'layover':
            continue

        # set local vars
        ref = bus['VehicleRef'] # naive reference (w/o runs)
        pos = float(bus['VehDistAlongRoute'])
        time = dateutil.parser.parse(bus['RecordedAtTime'])

        # create buffer Bus before updating
        new = Bus(ref, 1, pos, time, stop_pos)

        # check for run number
        runs = len([br for br in bus_ref if ref in br])

        ##### COMPRESS CODES BELOW #####
        # add buses that did not show up in existing set
        # if ref not in bus_ref:
        if runs == 0:
            live_bus[ref] = new
            continue
        # else vehicle exist, update reference for following query and update
        # if more than one run, add suffix
        elif runs > 1:
            nref = '%s_%s'%(ref, runs)
            new.ref = nref
            new.run = runs
        else:
            nref = ref

        # now "new" should have the latest ref
        # then check if this is a new run by the same vehicle
        # if so (position drops 2+ KM), split reference
        if live_bus[nref].pos - pos > 2000:
            # update ref and run, and treat as a new bus
            nref = '%s_%s'%(ref, runs+1)
            new.ref = nref
            new.run = runs+1
            live_bus[nref] = new
        # normal progress, update bus and link information between two pings
        else:
            update_bus(new, live_bus, stops, links)

    return live_bus

def sim_next(live_bus, active_bus, stops, links, stop_pos):
    """
    major function for stream update IN A SIMULATION
    data: streaming data, i.e., archive generator through load_archive() or live feed
    active_bus: currently active dictionary of SimBus objects
    """

    # remove buses that did not show up in new live_bus
    ##### active_bus = remove_inactive(live_bus, active_bus, sim=True)
    bus_ref = active_bus.keys()

    for bus in live_bus.values():
        ref = bus.ref
        pos = bus.pos
        time = bus.time
        new = SimBus(ref, pos, time, stops, links, stop_pos)

        # add buses that did not show up in existing set
        if ref not in bus_ref:
            active_bus[ref] = new
        # update bus and link information between two pings
        else:
            active_bus[ref].update_info(stops, links)

    return active_bus

def remove_inactive(batch, live_bus, sim=False):
    ##### CHANGE ARGS NAMING CONVENTION #####
    """
    remove buses from live_bus absent in the new stream batch
    sim: stream_batch -> live_bus [FALSE]
         live_bus -> active_bus [TRUE]
    FUTURE: also remove ones that reaches the last stop
    """
    if sim:
        new_refs = [bus for bus in batch.keys()]
    else:
        new_refs = [bus['VehicleRef'] for bus in batch]

    for ref in list(live_bus.keys()):
        if ref not in new_refs:
            del live_bus[ref]
    return live_bus

def update_bus(new, live_bus, stops, links, default_dwell=5):
    """
    major function for updating all the buffer information adopted by the simulator
    default_dwell: default assumed dwelling time if the bus did not make a stop ping at stop
    """
    old = live_bus[new.ref]
    elapsed_t = new.time - old.time
    distance = new.pos - old.pos
    # SCENARIO 1: consecutive pings at the same stop, update:
    # arrival time
    # dwell time
    if (new.at_stop and old.at_stop) and (new.at_stop_idx == old.at_stop_idx):
        new_arr = old.time
        dwell_time = elapsed_t.total_seconds()
        stops[new.at_stop_idx].update_dwell(new.ref, new_arr, dwell_time, at_stop=True)

    # SCENARIO 2: other situations, update:
    # speed for every link traveled
    # arrival time for each stop passed
    # minimal dwell time for each stop passed
    else:
        # every stop that is passed by assumed to have 5 sec dwelling time
        dwell_time = default_dwell
        # traveling speed in m/s
        try:
            speed = distance / elapsed_t.total_seconds()
        except ZeroDivisionError:
            speed = 0
            # print("Elapsed = 0:", new.ref, new.time, new.pos, elapsed_t, distance) # ERROR REPORT
        # the origin link
        lower = old.link
        # upper requires link + 1 in np.arange
        # but don't include current link if new ping is at a stop
        # therefore, when at_stop, no need to + 1
        if new.at_stop:
            upper = new.link
        else:
            upper = new.link + 1

        for l in list(np.arange(lower, upper)):
            # time of arrival at stop
            if distance > 0:
                new_arr = old.time + elapsed_t * (stops[l].pos - old.pos) / distance
            else:
                new_arr = old.time
                # print("Distance <= 0:", new.ref, new.time, new.pos, elapsed_t, distance) # ERROR REPORT
            stops[l].update_dwell(new.ref, new_arr, dwell_time, at_stop=False)
            links[l].update_speed(speed)

    # overwrite live_bus
    live_bus[new.ref] = new

###############
# ROUTE SETUP #
###############

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

######################
# OBJECTS: streaming #
######################

class Stop(object):

    def __init__(self, idx, ref, pos, name):
        self.idx = idx # link index, starting from 0
        self.ref = ref # stop reference
        self.pos = pos # stop location (1-D)
        self.name = name # stop name

        self.prev_bus = None # latest bus_ref that pass the stop
        self.prev_arr = None # latest bus arrival time
        self.dwell_window = [10, 10, 10] # in seconds
        self.dwell = np.mean(self.dwell_window)

        # log information
        self.log_bus_ref = []
        self.log_arr_t = []
        self.log_wait_t = [] # headway
        self.log_dwell_t = []
        
    def record(self, new_bus, new_arr, dwell_time):
        try:
            wait_t = new_arr - self.prev_arr
        except TypeError:
            wait_t = 0

        self.dwell_window.pop(0)
        self.dwell_window.append(dwell_time)

        self.log_bus_ref.append(new_bus)
        self.log_arr_t.append(new_arr)
        self.log_wait_t.append(wait_t)
        self.log_dwell_t.append(dwell_time)

        self.prev_arr = new_arr
        self.prev_bus = new_bus
        
    def update_dwell(self, new_bus, new_arr, dwell_time, at_stop):
        # same bus making a stop for 2+ pings
        if at_stop and (new_bus == self.prev_bus):
            self.dwell_window[-1] += dwell_time
            self.log_dwell_t[-1] += dwell_time

        # two pings of the new bus: roll the window; OR,
        # bus passing through a stop
        else:
            self.record(new_bus, new_arr, dwell_time)

        self.dwell = np.mean(self.dwell_window)

class Link(object):
    
    def __init__(self, idx, origin):
        self.idx = idx # link index
        self.origin = origin # origin stop of this link
        self.speed_window = [6, 6, 6]
        self.speed = np.mean(self.speed_window)

        # log information
        self.log_speed = []
        
    def update_speed(self, new):
        self.speed_window.pop(0)
        self.speed_window.append(new)
        self.speed = np.mean(self.speed_window)

        # log records
        self.log_speed.append(new)

class Bus(object):
    
    # capacity = 60
    # seat = 40
    stop_range = 100
    
    def __init__(self, ref, run, pos, time, stop_pos, hold=False):
        self.ref = ref # vehicle reference
        self.run = run # trip runs made by this vehicle (to distinguish)
        self.pos = pos # vehicle location (1-D)
        self.time = time # timestamp
        self.stop_pos = stop_pos ##### IS IT POSSIBLE TO MAKE THIS GLOBAL INSTEAD? #####
        self.hold = hold
        self.link = sum(self.pos >= self.stop_pos) - 1 # link index the bus is at
        # at stop if within specified range of a stop
        # at_prev if pos is ahead of the closest stop (in range)
        self.at_prev = self.pos - self.stop_pos[self.link] <= self.stop_range
        # at_next if pos is before the closest stop (in range)
        self.at_next = self.pos - self.stop_pos[self.link+1] >= -self.stop_range
        # at stop if either at_prev or at_next
        self.at_stop = self.at_prev or self.at_next
        ##### CONSIDER SIMPLIFYING THE ABOVE AS A SINGLE "NONE" or "INDEX" ATTRIBUTE #####

        # assign stop index that the bus is at
        ##### CURRENTLY IGNORING OVERLAPPING CASES #####
        if self.at_stop:
            if self.at_next:
                self.at_stop_idx = self.link + 1
            else:
                self.at_stop_idx = self.link

class SimBus(object):

    # capacity = 60
    # seat = 40
    
    def __init__(self, ref, pos, time, stops, links, stop_pos):
        self.ref = ref # vehicle reference
        self.pos = pos # vehicle location (1-D)
        self.time = time # timestamp
        self.stops = stops
        self.links = links
        self.stop_pos = stop_pos ##### IS IT POSSIBLE TO MAKE THIS GLOBAL INSTEAD? #####
        
        self.link = sum(self.pos >= self.stop_pos) - 1 # link index the bus is at
        self.next_stop = self.stop_pos[self.link + 1] # position of next stop
        self.speed = self.links[self.link].speed # speed at current link (m/s)
        self.dwell = 0
        self.headway = None
        # self.pax = 0 # vehicle load
        
        self.clock = 0 # dwell time counter
        self.operate = True
        self.atstop = False
        # self.headway = None # headway with the vehicle ahead
        
        self.log_pos = [self.pos]
        self.log_time = [self.time]
        self.log_speed = [self.speed]
        self.log_dwell = []
        self.log_status = []
        # self.log_pax = [0]

    def terminal(self):
        print("The bus has reached the terminal.")
        self.operate = False
        self.speed = 0
        self.record(self.speed, "terminal")
        # self.pax = 0
        
    def reach_stop(self):
        print("Bus %s arrives a stop at %s (position %i)"%(self.ref, self.stops[self.link + 1].name, self.next_stop))
        self.atstop = True
        # self.pax_to_board = pax_at_stop[self.link + 1] # check how many pax at stop
        # self.board_t = self.pax * avg_board_t
        # self.alight_t = 0 * avg_alight_t  #### TO DEVELOP
        # self.dwell = avg_door_t + self.alight_t + self.board_t # supposed to dwell for this long

        # self.speed = 0
        self.dwell = self.stops[self.link + 1].dwell
        self.prev_arr = self.stops[self.link + 1].prev_arr
        try:
            self.headway = self.time - (self.prev_arr + self.dwell) # a_{i} - d_{i-1}
        except TypeError:
            self.headway = None

        self.record(0, "reaching")
        self.clock += 1

    def dwell_stop(self):
        print("Bus %s is still making a stop at %s (position %i)"%(self.ref, self.stops[self.link + 1].name, self.next_stop))
        self.record(0, "dwelling")
        self.clock += 1
    
    def leave_stop(self):
        print("Bus %s departs a stop at %s (position %i)"%(self.ref, self.stops[self.link + 1].name, self.next_stop))
        self.atstop = False # move on!
        # pax_at_stop[self.link + 1] = 0 # clear all pax at stop
        self.log_dwell.append(self.dwell)
        
        # reset vars
        self.dwell = 0
        self.clock = 0
        self.link += 1
        self.next_stop = self.stop_pos[self.link + 1] # new next stop
        self.speed = self.links[self.link].speed # new link speed
        # self.pax = 0 # update pax onboard
        
        self.record(self.speed, "leaving")

    def record(self, speed, status):
        self.log_pos.append(self.pos)
        self.log_time.append(self.time)
        self.log_speed.append(speed)
        self.log_status.append(status)
        self.time += timedelta(seconds=1)
        
    def update_info(self, stops, links):
        ##### use global for other functions instead? #####
        self.stops = stops
        self.links = links
        
    def proceed(self):
        if self.operate:
            # SCENARIO 1: reach terminal
            if self.pos + self.speed >= self.stop_pos[-1]:
                self.terminal()
                
            # SCENARIO 2: reach a stop
            ### this judgement restricts from changing speed to 0 at stop, consider modification
            ### because if speed = 0, this condition wont be fulfilled
            elif self.pos + self.speed >= self.next_stop:
                # first reach
                if not self.atstop:
                    self.reach_stop()
                # still dwelling
                else:
                    self.dwell_stop()
                # check if dwelling is complete
                if self.clock >= self.dwell:
                    self.leave_stop()
            
            # SCENARIO 3: reach nothing, keep moving
            else:
                print("Current position of bus %s: %i"%(self.ref, self.pos))
                self.pos += self.speed
                self.record(self.speed, "traveling")
        else:
            print("Bus %s is not operating."%(self.ref))
            pass

#################
# VISUALIZATION #
#################

def plot_stream(filename, direction, live_bus, stops, links, stop_pos, stream_time=10, rate=1):
    """
    visualize a streaming from an archive
    stream_time: minutes to stream
    rate: streams a new batch (30-sec interval) per "X" second
    """
    data = load_archive(filename, direction)
    bus_pos = [bus.pos for bus in live_bus.values()]

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)

    ax.plot(stop_pos, np.ones(len(stop_pos)), '.-')
    vehs, = ax.plot(bus_pos, np.ones(len(bus_pos)), '*', markersize=10)
    ax.set_xlabel('Distance along the route', fontsize=14)

    clock = 0 # minutes

    while clock <= stream_time:
        stream_next(data, live_bus, stops, links, stop_pos)
        bus_pos = [bus.pos for bus in live_bus.values()]
        vehs.set_data(bus_pos, np.ones(len(bus_pos)))

        clear_output(wait=True)
        display(fig)
        print("Time elapsed: %s minutes"%(clock))

        clock += 0.5
        time.sleep(rate) # set a global time equivalent parameter

def plot_sim(filename, direction, live_bus, active_bus, stops, links, stop_pos, sim_time=10, rate=0.1):
    """
    visualize a simulation streaming from an archive
    sim_time: minutes to simulate
    rate: simulate a second per "X" real-world second
    """
    data = load_archive(filename, direction)
    bus_pos = [bus.pos for bus in active_bus.values()]

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)

    ax.plot(stop_pos, np.ones(len(stop_pos)), '.-')
    vehs, = ax.plot(bus_pos, np.ones(len(bus_pos)), '*', markersize=10)
    ax.set_xlabel('Distance along the route', fontsize=14)

    clock = 0 # seconds

    # while bus1.operate or bus2.operate or bus3.operate:
    while clock <= sim_time * 60:
        # stream next batch and update sim info per 30 seconds
        if clock % 30 == 0:
            print("Fetching new feeds...")
            stream_next(data, live_bus, stops, links, stop_pos)
            print("Updating simulator...")
            sim_next(live_bus, active_bus, stops, links, stop_pos)
        
        # run each SimBus
        if len(active_bus) > 0:
            [bus.proceed() for bus in active_bus.values()]

        bus_pos = [bus.pos for bus in active_bus.values()]
        vehs.set_data(bus_pos, np.ones(len(bus_pos)))
        
        clear_output(wait=True)
        display(fig)
        print("Time elapsed: %i seconds"%(clock))

        clock += 1
        time.sleep(rate)

##############
# SIMULATION #
##############

def infer(filename, direction, live_bus, stops, links, stop_pos, runtime=60):
    """
    infer link traveling speed and dwelling information from archive
    runtime: run the archive stream for "X" minutes

    """
    data = load_archive(filename, direction)

    for i in np.arange(runtime * 2):
        stream_next(data, live_bus, stops, links, stop_pos)

def simulate(filename, direction, live_bus, active_bus, stops, links, stop_pos, sim_time=10):
    """
    simply simulate to get data (i.e., plot_sim w/o plot)
    sim_time: minutes to simulate
    """
    data = load_archive(filename, direction)
    
    clock = 0

    while clock <= sim_time * 60:
        # stream next batch and update sim info per 30 seconds
        if clock % 30 == 0:
            stream_next(data, live_bus, stops, links, stop_pos)
            sim_next(live_bus, active_bus, stops, links, stop_pos)
        
        # run each SimBus
        if len(active_bus) > 0:
            [bus.proceed() for bus in active_bus.values()]

        clock += 1

#############
# ANALYTICS #
#############

def sim_tsd(active_bus, stops, archive_path, dir_ref, start_min=None, end_min=None, save=False, fname='sim_TSD'):
    
    ### TO BE MODIFIED: NOT USEFUL IN LIVE STREAM ###
    df = pd.read_csv(archive_path)
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
    
    # plot figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    # plot CallDistanceAlongRoute (bus stops)
    stops = [stop.pos for stop in stops.values()]
    left = [df['RecordedAtTime'].min()] * len(stops)
    right = [df['RecordedAtTime'].max()] * len(stops)
    ax.plot([left, right], [stops, stops], color='gray', alpha=0.2);

    # plot the trajectory for each vehicle
    for bus in active_bus.values():
        
        ax.plot(bus.log_time, bus.log_pos)
        ax.annotate(bus.ref.split("_")[1], (bus.log_time[0], bus.log_pos[0]))
        
    ax.grid()
    ax.set_xlabel("Timestep (second)", fontsize=14)
    ax.set_ylabel("Distance along route (meter)", fontsize=14)
    ax.set_title("Time-space Diagram of Bus Simulation")
    
    plt.tight_layout()
    
    # save figure locally if specified
    if save:
        plt.savefig("%s.png"%(fname), dpi=300)
    else:
        pass
    plt.show()
    
    return fig, ax

def bus_tsd(bus):
    plt.figure(figsize=(12,8))
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

#################
# END OF SCRIPT #
#################