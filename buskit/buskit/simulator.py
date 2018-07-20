# Author: Yuwen Chang, NYU CUSP
# Last Updated: 2018/07/17
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import dateutil
from datetime import timedelta

import random
import scipy.stats as ss
from .busdata import *

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
        try:
            new = Bus(ref, 1, pos, time, stop_pos)
        except IndexError: ###### if next stop is terminal?? #####
            continue

        # check for run number
        runs = len([br for br in bus_ref if ref in br])

        ##### COMPRESS CODES BELOW #####
        # add buses that did not show up in existing set
        # if ref not in bus_ref:
        if runs == 0:
            nref = '%s_%s'%(ref, 1)
            new.ref = nref
            live_bus[nref] = new
            continue
        # else vehicle exist, update reference for following query and update
        # if more than one run, add suffix
        elif runs >= 1:
            nref = '%s_%s'%(ref, runs)
            new.ref = nref
        else:
            print("run count error")

        # now "new" should have the latest ref
        # then check if this is a new run by the same vehicle
        # if so (position drops 2+ KM), split reference
        if live_bus[nref].pos - pos > 2000:
            # update ref & run, and treat as a new bus
            nref = '%s_%s'%(ref, runs+1)
            new.ref = nref
            live_bus[nref] = new
        # normal progress, update bus and link information between two pings
        else:
            update_bus(new, live_bus, stops, links)

    return live_bus

def sim_next(live_bus, active_bus, stops, links, stop_pos, control=0):
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

        # add buses that did not show up in existing set
        if ref not in bus_ref:
            active_bus[ref] = SimBus(ref, pos, time, stops, links, stop_pos, control)
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

def infer_dwell(distance, elapsed_t, n_links):
    # traveling speed in m/s
    try:
        speed = distance / elapsed_t.total_seconds()
    except ZeroDivisionError:
        speed = 0
        # print("Elapsed = 0:", new.ref, new.time, new.pos, elapsed_t, distance) # ERROR REPORT
    
    # interpolate dwelling time for all covered stops in 4 levels:
    # LEVEL 1: if bus speed is high enough, assume passing stops w/o stopping
    if speed >= 11:
        dwell_time = 0
    # LEVEL 2: bus driving fast, assume a minimal stop
    elif speed >= 8:
        dwell_time = 11
    # LEVEL 3: the average baseline case
    # avg mere dwelling time 16 + (avg stopping/leaving/door time 6)
    elif speed >= 5:
        dwell_time = 22
    # LEVEL 4: very slow bus
    # assume avg speed = 7 m/s, the redundant time is the assumed total dwelling time
    ##### TEMPORARILY ATTEMPTED TO COVER "link-stop-link" ping case #####
    else:
        dwell_time = elapsed_t.total_seconds() - distance / 7

    # recalculate speed
    ##### TEMPORARY #####
    try:
        speed = distance / (elapsed_t.total_seconds() - dwell_time * 0.5)
    except ZeroDivisionError:
        speed = 0

    # the average dwell_time for each stop covered
    dwell_time /= max(n_links - 1, 1) ##### neglecting the case when "lower" starts at a stop

    return dwell_time, speed

def update_bus(new, live_bus, stops, links):
    ##### CHANGE NAME TO update_stops/links TO AVOID CONFUSION? #####
    """
    major function for updating all the buffer information adopted by the simulator
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
        stops[old.at_stop_idx].update_dwell(new.ref, new_arr, dwell_time, at_stop=True)
        stops[old.at_stop_idx].prev_stopped_bus = new.ref # update bus ref of latest stop-stop scenario at stop

    # SCENARIO 2: stop-link pings (both S-S-L and L-S-L)
    # consider bus is still on the same link, meaning that the bus has not left too far
    # also meaning that part of this interval is likely to be dwelling
    elif ((not new.at_stop) and old.at_stop) and (new.link == old.link):
        new_arr = old.time # arrival time at stop
        dwell_time, speed = infer_dwell(distance, elapsed_t, 2)
        # SCENARIO 2-1 S-S-L: fall back to scenario 1
        if new.ref == stops[old.at_stop_idx].prev_stopped_bus:
            stops[new.link].update_dwell(new.ref, new_arr, dwell_time, at_stop=True)
        # SCENARIO 2-2 L-S-L: follow scenario 3
        ##### DOES NOT HANDLE SITUATION WHEN DWELL HAPPENS IN EARLIER PING PAIR (L-S)
        else:
            links[new.link].update_speed(speed)
            stops[new.link].update_dwell(new.ref, new_arr, dwell_time, at_stop=False)

    # SCENARIO 3: other situations, update:
    # speed for every link traveled
    # arrival time for each stop passed
    # minimal dwell time for each stop passed
    else:
        # the origin link
        lower = old.link
        # upper requires link + 1 in np.arange
        # but don't include current link if new ping is at a stop
        # therefore, when at_stop, no need to + 1
        if new.at_stop:
            upper = new.link
        else:
            upper = new.link + 1

        dwell_time, speed = infer_dwell(distance, elapsed_t, upper-lower)

        for l in list(np.arange(lower, upper)):
            # infer time of arrival at stop
            if distance > 0:
                new_arr = old.time + elapsed_t * (stops[l].pos - old.pos) / distance
            else:
                new_arr = old.time
                # print("Distance <= 0:", new.ref, new.time, new.pos, elapsed_t, distance) # ERROR REPORT
            ##### FOR stop-stop-link pings, the latter interval should NOT update lower/old stop
            ##### FOR link-stop-link pings, the latter interval should update
            ##### TEMPORARY: dont update old stop no matter what to avoid overwrite
            links[l].update_speed(speed)
            if l != lower:
                stops[l].update_dwell(new.ref, new_arr, dwell_time, at_stop=False)

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
        self.prev_stopped_bus = None # the latest bus ref that made a stop-stop ping scenario
        self.prev_arr = None # latest bus arrival time
        self.prev_dep = None
        self.dwell_window = [22, 22, 22] # in seconds (k + dwell)
        self.dwell = np.mean(self.dwell_window)
        self.q_window = [0.05, 0.05, 0.05, 0.05] # dwell time parameter for simulator
        self.q = np.mean(self.q_window)

        # log information
        self.log_bus_ref = []
        self.log_arr_t = []
        self.log_dep_t = []
        self.log_headway = []
        self.log_dwell_t = []
        self.log_q = []

        # simulation
        self.sim_prev_dep = None
        self.nbus_at_stop = 0 # number of buses at stop

        # log information for simulation
        self.sim_arr_t = []
        self.sim_dep_t = []
        self.sim_headway = []
        
    def record(self, new_bus, new_arr, dwell_time):

        try:
            headway = (new_arr - self.prev_dep).total_seconds()
            if headway > 0:
                q = min(dwell_time / headway, 0.2) # keep q small
            else:
                q = np.mean(self.q_window)

        # if prev_dep = None (no prev call yet), just succeed previous q
        except TypeError:
            headway = None
            q = np.mean(self.q_window)

        # calculate departure time
        new_dep = new_arr + timedelta(seconds=dwell_time)

        self.dwell_window.pop(0)
        self.dwell_window.append(dwell_time)

        self.q_window.pop(0)
        self.q_window.append(q)

        self.log_bus_ref.append(new_bus)
        self.log_arr_t.append(new_arr)
        self.log_dep_t.append(new_dep)
        self.log_headway.append(headway)
        self.log_dwell_t.append(dwell_time)
        self.log_q.append(q)

        self.prev_bus = new_bus
        self.prev_arr = new_arr
        self.prev_dep = new_dep
        
    def update_dwell(self, new_bus, new_arr, dwell_time, at_stop):
        # same bus making a stop for 2+ pings
        if at_stop and (new_bus == self.prev_bus):
            self.dwell_window[-1] += dwell_time
            self.log_dwell_t[-1] += dwell_time

            # update q
            ##### MERGE SIMILAR CODE FROM ABOVE #####
            try:
                headway = (new_arr - self.prev_dep).total_seconds()
                if headway > 0:
                    q = min(self.log_dwell_t[-1] / headway, 0.2) # keep q small
                else:
                    q = np.mean(self.q_window)

            ### SUPPOSEDLY SHOULDNT HAPPEN ###
            except TypeError:
                headway = None
                q = np.mean(self.q_window)

            ##### requires record ??? #####
        # two pings of the new bus: roll the window; OR,
        # bus passing through a stop
        else:
            self.record(new_bus, new_arr, dwell_time)

        ##### may have to adjust depending on how stop-stop-link pings update stops #####
        self.dwell = np.mean(self.dwell_window)
        self.q = np.mean(self.q_window)

    # for simulation
    ### requires simplification ###
    def update_sim_arr(self, new_arr):
        self.sim_prev_arr = new_arr
        self.sim_arr_t.append(new_arr)

        # if no bus at stop: update headway (max wait time)
        if self.nbus_at_stop == 0 and self.sim_prev_dep != None:
            headway = (new_arr - self.sim_prev_dep).total_seconds()
            self.sim_headway.append(headway)

        self.nbus_at_stop += 1

    def update_sim_dep(self, new_dep):
        self.sim_prev_dep = new_dep
        self.sim_dep_t.append(new_dep)
        self.nbus_at_stop -= 1

class Link(object):
    
    def __init__(self, idx, origin):
        self.idx = idx # link index
        self.origin = origin # origin stop of this link
        self.speed_window = [7, 7, 7]
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
    
    def __init__(self, ref, run, pos, time, stop_pos):
        self.ref = ref # vehicle reference
        # self.run = run # trip runs made by this vehicle (to distinguish)
        self.pos = pos # vehicle location (1-D); ENSURE > 0???
        self.time = time # timestamp
        self.stop_pos = stop_pos ##### IS IT POSSIBLE TO MAKE THIS GLOBAL INSTEAD? #####
        self.link = sum(self.pos >= self.stop_pos) - 1 # link index the bus is at
        # at stop if within specified range of a stop
        # at_prev if pos is ahead of the closest stop (in range)
        self.at_prev = self.pos - self.stop_pos[self.link] <= self.stop_range
        # at_next if pos is before the closest stop (in range)
        self.at_next = self.pos - self.stop_pos[self.link + 1] >= -self.stop_range
            
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
    
    def __init__(self, ref, pos, time, stops, links, stop_pos, control=0):
        self.ref = ref # vehicle reference
        self.pos = pos # vehicle location (1-D)
        self.time = time # timestamp
        self.stops = stops
        self.links = links
        self.stop_pos = stop_pos ##### IS IT POSSIBLE TO MAKE THIS GLOBAL INSTEAD? #####
        self.control = control # whether to trigger holding strategies
        
        ##### TEMPORARY MEASURE TO RESOLVE KEYERROR (-1)
        self.link = max(sum(self.pos >= self.stop_pos) - 1, 0) # link index the bus is at
        self.next_stop = self.stop_pos[self.link + 1] # position of next stop
        self.speed = self.links[self.link].speed # speed at current link (m/s)
        self.dwell = 0
        self.hold = 0
        self.headway = None # seconds
        # self.pax = 0 # vehicle load
        
        self.clock = 0 # dwell time counter
        self.operate = True
        self.atstop = False
        self.reaching = False
        self.leaving = False
        
        self.log_pos = [self.pos]
        self.log_time = [self.time]
        self.log_status = ["initiating"]
        self.log_speed = [self.speed]
        self.log_dwell = [None]
        self.log_hold = [None]
        self.log_stop = [None]
        # self.log_pax = [0]

    def terminal(self):
        # print("The bus has reached the terminal.")
        self.operate = False
        self.speed = 0
        self.record("terminal")
        # self.pax = 0
        
    def reach_stop(self):
        # print("Bus %s arrives a stop at %s (position %i)"%(self.ref, self.stops[self.link + 1].name, self.next_stop))
        self.atstop = True
        self.reaching = True # for update_sim_stops

        ##### currently disabled utilities #####
        # self.pax_to_board = pax_at_stop[self.link + 1] # check how many pax at stop
        # self.board_t = self.pax * avg_board_t
        # self.alight_t = 0 * avg_alight_t  #### TO DEVELOP
        # self.dwell = avg_door_t + self.alight_t + self.board_t # supposed to dwell for this long
        # self.speed = 0

        # calculate headway
        try:
            # a_{i} - d_{i-1}
            self.headway = (self.time - self.stops[self.link + 1].sim_prev_dep).total_seconds()
        except TypeError:
            # if no sim_prev_dep yet
            self.headway = None

        # calculate dwelling time
        ##### Q VALVE HERE: CURRENTLY SHUT OFF DUE TO BAD PERFORMANCE #####
        self.dwell = self.calc_dwell(q=self.stops[self.link + 1].q)
        # self.dwell = self.calc_dwell() # use default q

        # calculate holding time
        self.hold = self.calc_hold()

        self.record("reaching", 0, self.dwell, self.hold, self.stops[self.link + 1].idx)
        self.clock += 1

    def dwell_stop(self):
        # print("Bus %s is still making a stop at %s (position %i)"%(self.ref, self.stops[self.link + 1].name, self.next_stop))
        self.record("dwelling")
        self.clock += 1
    
    def hold_stop(self):
        # print("Bus %s is being held at stop %s (position %i)"%(self.ref, self.stops[self.link + 1].name, self.next_stop))
        self.record("holding")
        self.clock += 1

    def leave_stop(self):
        # print("Bus %s departs a stop at %s (position %i)"%(self.ref, self.stops[self.link + 1].name, self.next_stop))
        self.atstop = False # move on!
        self.leaving = True # for update_sim_stops
        ## self.pos += self.speed + 0.1 # make sure to push the bus out of the stop (loop)!
        self.record("leaving")
        # pax_at_stop[self.link + 1] = 0 # clear all pax at stop
        
        # reset and update vars
        self.clock = 0
        ## self.link = sum(self.pos >= self.stop_pos) - 1
        self.link += 1
        self.next_stop = self.stop_pos[self.link + 1] # new next stop
        self.speed = self.links[self.link].speed # new link speed
        self.dwell = 0 # reset dwelling time
        self.hold = 0 # reset holding time
        # self.pax = 0 # update pax onboard

    def record(self, status, speed=0, dwell=None, hold=None, stop=None):
        self.time += timedelta(seconds=1)
        self.log_pos.append(self.pos)
        self.log_time.append(self.time)
        self.log_status.append(status)
        self.log_speed.append(speed)
        self.log_dwell.append(dwell)
        self.log_hold.append(hold)
        self.log_stop.append(stop) # record stop index
        
    def calc_dwell(self, k=5, q=0.03):
        """
        This is the algorithm for calculating a dwelling time.
        """
        # determine base dwell_time depending on whether headway (sim_prev_dep) exists
        if self.headway != None:
            dwell = k + 0.2 * q * self.headway ##### adding 0.5 for temporary adjustment #####
        else:
            dwell = self.stops[self.link + 1].dwell ##### TEMPORARY #####

        # randomize based on bunching or not
        # to escape simulation convergence of multiple buses
        if self.headway != None and self.headway < 90:
            ##### ADJUST (REDUCE) DWELL_TIME WHEN HW=0 TO REFLECT BUNCHING #####
            ##### THIS METHOD MAY LEAD TO A TANGLING SPEED UP CYLE !!! #####
            # pass over prevbus, each share half load
            dwell = random.choice([0, dwell/3])
        else:
            dwell = max(8 * np.random.randn() + dwell, 0)

        return dwell

    def calc_hold(self):
        """
        This is the algorithm for determining hold time.
        """
        # check if bunching in place
        if self.headway != None and self.headway < 600: # seconds
            # method 0: no control
            if self.control == 0:
                hold = 0
            # method 1: fixed holding
            elif self.control == 1:
                hold = 60
            # method 2: naive headway
            elif self.control == 2:
                hold = 600 - self.headway
            else:
                hold = 0

        else:
            hold = 0

        return hold

    def update_info(self, stops, links=None):
        ##### use global for other functions instead? #####
        self.stops = stops
        if links != None:
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
                # 2.1: first reach
                if not self.atstop:
                    self.reach_stop()
                # 2.2: still dwelling
                elif self.clock < self.dwell:
                    self.dwell_stop()
                # 2.3: still holding
                elif self.clock < self.dwell + self.hold:
                    self.hold_stop()
                # 2.4: dwell and hold complete, leave
                elif self.clock >= self.dwell + self.hold:
                    self.leave_stop()
                # safety check
                else:
                    ##### Include in error log #####
                    print("Unexpected condition during proceed. (Bus: %s at Pos: %s on %s)"%(
                          self.ref, self.pos, self.time))
            
            # SCENARIO 3: reach nothing, keep moving
            else:
                # print("Current position of bus %s: %i"%(self.ref, self.pos))
                self.pos += self.speed
                self.record("traveling", self.speed)
        else:
            # print("Bus %s is not operating."%(self.ref))
            pass

    def report(self, filename=None):
        """
        create a DataFrame report of the vehicle
        filename: when specified (str), export as csv
        """
        df = pd.DataFrame({'vehref': self.ref,
                           'position': self.log_pos,
                           'time': self.log_time,
                           'status': self.log_status,
                           'speed': self.log_speed,
                           'dwell': self.log_dwell,
                           'hold': self.log_hold,
                           'stop': self.log_stop})
        if filename != None:
            df.to_csv(filename)

        return df

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

### MERGE THIS WITH simulate ###
def plot_sim(filename, direction, live_bus, active_bus, stops, links, stop_pos, control=0, sim_time=10, rate=0.1):
    """
    visualize a simulation streaming from an archive
    sim_time: minutes to simulate
    rate: simulate a second per "X" real-world second
    please reset route before each sim ##### to incorporate auto-check
    """
    data = load_archive(filename, direction)
    bus_pos = [bus.pos for bus in active_bus.values()]

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)

    ax.plot(stop_pos, np.ones(len(stop_pos)), '.-')
    vehs, = ax.plot(bus_pos, np.ones(len(bus_pos)), '*', markersize=10)
    ax.set_xlabel('Distance along the route', fontsize=14)

    clock = 0 # seconds
    bunch = 0 # vehicle-interval-second

    # while bus1.operate or bus2.operate or bus3.operate:
    while clock <= sim_time * 60:
        # stream next batch and update sim info per 30 seconds
        if clock % 30 == 0:
            print("Fetching new feeds...")
            stream_next(data, live_bus, stops, links, stop_pos)
            print("Updating simulator...")
            sim_next(live_bus, active_bus, stops, links, stop_pos, control)
        
        # run each SimBus
        if len(active_bus) > 0:
            [bus.proceed() for bus in active_bus.values()]
            update_sim_stops(active_bus, stops)
            bunch += count_bunch(active_bus)

        bus_pos = [bus.pos for bus in active_bus.values()]
        vehs.set_data(bus_pos, np.ones(len(bus_pos)))
        
        clear_output(wait=True)
        display(fig)
        print("Time elapsed: %i seconds"%(clock))

        clock += 1
        time.sleep(rate)

    hold = eval_hold(active_bus)
    hw_avg, hw_std = eval_headway(stops)
    report = {'bunch':bunch,
              'sumHold':hold,
              'avgHeadway':hw_avg,
              'stdHeadway':hw_std}

    return report ##### adjust how to record this when a simulator grand class is built #####

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

def simulate(filename, direction, live_bus, active_bus, stops, links, stop_pos, control=0, sim_time=10):
    """
    simply simulate to get data (i.e., plot_sim w/o plot)
    sim_time: minutes to simulate
    please reset route before each sim ##### to incorporate auto-check
    """
    data = load_archive(filename, direction)
    
    clock = 0 # seconds
    bunch = 0 # vehicle-interval-second

    while clock <= sim_time * 60:
        # stream next batch and update sim info per 30 seconds
        if clock % 30 == 0:
            stream_next(data, live_bus, stops, links, stop_pos)
            sim_next(live_bus, active_bus, stops, links, stop_pos, control)
        
        # run each SimBus
        if len(active_bus) > 0:
            [bus.proceed() for bus in active_bus.values()]
            update_sim_stops(active_bus, stops)
            bunch += count_bunch(active_bus)

        clock += 1

    hold = eval_hold(active_bus)
    hw_avg, hw_std = eval_headway(stops)
    report = {'bunch':bunch,
              'sumHold':hold,
              'avgHeadway':hw_avg,
              'stdHeadway':hw_std}

    return report ##### adjust how to record this when a simulator grand class is built #####

def update_sim_stops(active_bus, stops):
    """
    Update all stops with simulation info of SimBus (active_bus)
    """
    # find all arriving/departing buses
    arr_bus = [bus for bus in active_bus.values() if bus.reaching]
    dep_bus = [bus for bus in active_bus.values() if bus.leaving]

    # update each stop that has a bus arrival with latest arr time
    for bus in arr_bus:
        stops[bus.link + 1].update_sim_arr(bus.time) # NOTICE: +1 b/c link number not updated to the next yet
        bus.reaching = False # turn off indicator after update is done
    
    # update each stop that has a bus departure with latest dep time
    for bus in dep_bus:
        stops[bus.link].update_sim_dep(bus.time)
        bus.leaving = False # turn off indicator after update is done
    
    # update stops to all active buses
    for bus in active_bus.values():
        bus.update_info(stops)

##############
# EVALUATION #
##############

def count_bunch(bus_dict, threshold=300):
    """
    determines spacing-based bunching for each sim run (second)
    each unit is a (vehicle) interval-second of bunching
    threshold for bunching: default 300 meter
    """
    bus_pos = [bus.pos for bus in bus_dict.values() if bus.operate]
    result = sum(np.abs(np.diff(bus_pos)) < threshold)

    return result

def eval_hold(bus_dict):
    all_log = [bus.log_hold for bus in bus_dict.values()]
    all_hold = [hold for log in all_log for hold in log if hold != None]
    total_hold = sum(all_hold)
    
    return total_hold

def eval_headway(stop_dict):
    all_log = [stop.sim_headway for stop in stop_dict.values()]
    all_hw = [hw for log in all_log for hw in log if hw != None]
    mean_hw = np.mean(all_hw) / 60 # minutes
    std_hw = np.std(all_hw) / 60 # minutes
    
    return mean_hw, std_hw

def sim_report(reports):
    """
    merge all simulation reports into a single dataframe
    reports: a list of sim reports returned by each simulate()
    """
    report_df = pd.DataFrame({'bunch':[r['bunch'] for r in reports],
                              'hold':[r['sumHold'] for r in reports],
                              'avghw':[r['avgHeadway'] for r in reports],
                              'stdhw':[r['stdHeadway'] for r in reports]})
    report_df = report_df[['bunch', 'hold', 'avghw', 'stdhw']]
    # report_df.index = []
    
    return report_df

def eval_sim(data, active_bus, direction):
    """
    evaluates how well the simulation is compared to the original archive
    returns distance (meter) MSE per ping pair (between data and simulation at the same timestamp)
    data: archive data for simulation
    active_bus: SimBus dictionary from simulation
    direction: the route direction that the simulation is performed
    """
    # read data
    # df = archive_reader(data, dir_ref, start_min, end_min) ##### INCOMPLETE #####
    df = pd.read_csv(data)
    df = df[df['DirectionRef'] == direction] # subset direction
    df = split_trips(df)
    df = df[df['ProgressStatus'] != 'layover'] # remove layover

    # extract mutual vehicle refs
    buses1 = set(df['NewVehicleRef'])
    buses2 = set(active_bus.keys())
    mutual_ref = buses1.intersection(buses2)
    
    # mse from all runs
    sim_mse = []
    weights = []
    
    # iterate through all vehicles
    for ref in mutual_ref:
        time1 = set(map(dateutil.parser.parse, df[df['NewVehicleRef'] == ref]['RecordedAtTime']))
        time2 = set(active_bus[ref].log_time) ### subset [:-1] as a shortcut to exclude potential terminal records ###
        mutual_time = list(time1.intersection(time2))

        # extract positions from archive data
        tmpdf = df[df['NewVehicleRef'] == ref]
        tlist = list(map(dateutil.parser.parse, tmpdf['RecordedAtTime']))
        pos1 = [[t, i] for i, t in enumerate(tlist) if t in mutual_time] # extract time and index
        pos1 = np.transpose(pos1)
        pos1[1] = tmpdf.iloc[pos1[1],:]['VehDistAlongRoute'] # use index to subset positions
        pos1 = dict(zip(pos1[0], pos1[1])) # auto-remove duplicate
        pos1 = np.array(list(pos1.values()))
        pos1.sort()

        # extract positions from SimBus log
        pos2 = [[t, active_bus[ref].log_pos[i]] for i, t in enumerate(active_bus[ref].log_time) if t in mutual_time]
        pos2 = np.transpose(pos2)
        pos2 = dict(zip(pos2[0], pos2[1])) # auto-remove duplicate
        pos2 = np.array(list(pos2.values()))
        pos2.sort()

        # calculate the MSE of individual vehicle
        err = np.abs(pos1 - pos2)
        mse = np.mean(err ** 2)
        sim_mse.append(mse)
        weights.append(len(err))
    
    # calculate overall MSE of the simulation
    result = np.average(sim_mse, weights=weights)
    print("MSE of simulated vehicle distance (meter) per ping pair: %.2f"%(result))
    return result # mse distance (meter) per ping pair (at the same time)

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