# Bus Analytics Toolkit for Bunching Mitigation

## A ready-to-use python toolkit for bus data streaming, analytics, and simulation with real-time, historical, and artificial bus trajectory data

Author: Yuwen Chang (M.S. at NYU CUSP 2018)

Research Project at BUILT@NYU

## Motivation

- Simple and generalized data pipeline
- Simple and flexible simulation environment
- Scalable to other cities given GTFS and SIRI

## Data Streaming Functions

### stream_bus: Get bus data

Manually download the [script](https://github.com/ywnch/BusSimulator/blob/master/fetchbus.py) or clone the repo and import the functions to your Jupyter notebook or run it on command lines as the example below:

```python
git clone https://github.com/ywnch/BusSimulator.git
# fetch bus data for route B54 for 1 minute (once per 30 secs)
python fetchbus.py $MTAAPIKEY "B54" 1
# SBS lineref example: "S79%2B", which would be interpreted as "S79+"
```

### split_trips: Split different trips made by same vehicles

```python
# new vehicle reference number will have a trip no. suffix
df = split_trips(df)
df["NewVehicleRef"]
```

### plot_tsd: Plot time-space diagram

Import the functions and simply plot the dataframe as downloaded using `bus_data`.

```python
from fetchbus import plot_tsd
# read the downloaded data
df = pd.read_csv("B54-180403-170417-5-Tue.csv")
# plot bus direction 1 in df starting from minute 10 to minute 30 and save it as TSD.png
plot_tsd(df, dir_ref=1, start_min=10, end_min=30, save=True, fname='TSD')
```

## Current Output

1. Currently, we may use the `bus_data` function to fetch data of a specified route and direction at a given time window and then plot the time-space diagram with the `plot_tsd` function, both functions can be found in `fetchbus.py`. Below is a sample time-space diagram. (the x-axis is plot by recorded time instead, not time elapsed as shown here)

2. We may also use a conceptual dashboard to monitor real-time bus trajectories.

3. A prototype simulator that treat Bus and Stop as objects is also available

4. One of the benefits of objects is that we can record activities and events in a single bus or stop and examine the log easily.
