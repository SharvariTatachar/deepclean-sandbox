from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data,find_max_corr_channel,plot_max_corr_chan

from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lowfreq', type=float, help='lower boundary for frequncy in DeepClean')
parser.add_argument('--highfreq', type=float, help='upper boundary for frequncy in DeepClean')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory in which data is saved')
parser.add_argument('--ifo', default='L1', type=str, help='Interferometer')
args = parser.parse_args()

def give_group(a):
    group = a.split('_')[1]
    return group

time_ = int(args.savedir.split("/")[-2])
print('Current time is:', time_)

print('Start loading the *.csv files and doing the plot')
vals = plot_max_corr_chan(args.savedir, 10, args.ifo)

print('Print sorted list of witness channels to be copied to DeepClean')
vals_selc = vals.loc[(vals['frequency']>args.lowfreq) & (vals['frequency']<args.highfreq) & ((vals['coherence']>0.2) | (vals['Coherence'] > 0.2))]

channels = []
channels1 = vals_selc.sort_values(['coherence'], ascending=False).drop_duplicates(['channel1'])['channel1'].to_list()
channels2 = vals_selc.sort_values(['Coherence'], ascending=False).drop_duplicates(['channel2'])['channel2'].to_list()

channels = [c for c in channels1]
for c in channels2:
    if c not in channels1:
        channels.append(c)

with open('chanlist_o4.ini','w') as f:
    f.write(args.ifo+':GDS-CALIB_STRAIN\n')
    for c in channels:
        f.write(c.split('.')[0]+'\n')
10:20
"""Utilities for coherence monitor"""

from gwpy.time import to_gps
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import gwdatafind
import numpy as np
import pandas as pd
import os
import glob
import plotly.express as px
import plotly

__author__ = 'Siddharth Soni <siddharth.soni@ligo.org>'


def give_group_v2(a):
    group = a.split(':')[1].split('-')[0]
    return group


def get_strain_data(starttime, endtime, ifo='L1'):
    ht = TimeSeries.get(f'{ifo}:GDS-CALIB_STRAIN', starttime, endtime)
    return ht


def get_frame_files(starttime, endtime, ifo='L1'):
    site_ = ifo[0]
    files = gwdatafind.find_urls(f'{site_}', f'{ifo}_R', starttime, endtime)
    return sorted(files)


def get_unsafe_channels(ifo):
    path = f'channel_files/{ifo}/{ifo}_unsafe_channels.csv'
    return pd.read_csv(path)


def get_observing_segs(t1, t2, ifo):
    tstart = to_gps(t1)
    tend = to_gps(t2)
    segs = DataQualityFlag.query(f'{ifo}:DMT-ANALYSIS_READY:1', t1, t2)
    seg_list = SegmentList()

    for seg in segs.active:
        if seg.end - seg.start > 3600:
            seg_list.append(seg)

    print("Got the segments")
    return seg_list


def get_times(seglist, duration=3600):
    times = [np.arange(i.start, i.end - duration, duration) for i in seglist]
    return [item for sublist in times for item in sublist]


def calc_coherence(channel2, frame_file, start_time, end_time, fft, overlap, strain_data, channel1=None):
    t1 = to_gps(start_time)
    t2 = to_gps(end_time)
    ts2 = TimeSeries.read(frame_file, channel=channel2, start=t1, end=t2)

    if channel1:
        TimeSeries.fetch(channel1, t1, t2)
    else:
        ts1 = strain_data

    ts1 = ts1.resample(ts2.sample_rate)
    coh = ts1.coherence(ts2, fftlength=fft, overlap=overlap)

    for i in np.where(coh.value == np.inf)[0]:
        try:
            coh.value[i] = (coh.value[i - 2] + coh.value[i - 1]) / 2
        except IndexError:
            coh.value[i] = 1e-20

    return coh


import concurrent.futures


def run_coherence(channel_list, frame_files, starttime, endtime, strain_data, savedir, coh_thresh, ifo='L1', timeout=30):
    t1, t2 = to_gps(starttime), to_gps(endtime)
    savedir = os.path.join(savedir, f'{t1}', '')

    if not os.path.exists(savedir):
        print(f"Creating the output dir {savedir}")
        os.makedirs(savedir)

    for channel in channel_list:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(calc_coherence, strain_data=strain_data, channel1=None,
                                     channel2=channel, frame_file=frame_files,
                                     start_time=t1, end_time=t2, fft=10, overlap=5)

            try:
                coh = future.result(timeout=timeout)
                #coh_ = coh[coh.value > coh_thresh]
                coh_ = coh
                if len(coh_) > 0:
                    coh_.write(f'{savedir}{channel}.csv')

            except concurrent.futures.TimeoutError:
                print(f"Calculation for channel {channel} timed out.")
                continue


def get_max_corr(output_dir, save=False):
    files = glob.glob(f'{output_dir}*.csv')
    vals = []

    for file in files:
        chan_name = file.split('/')[-1].split('DQ')[0] + 'DQ'
        fs = FrequencySeries.read(file)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(1 / n_diff), int(200 / n_diff)
        fs_ = fs[ind1:ind2]
        max_value = fs_.max().value
        max_value_frequency = fs_.frequencies[fs_.argmax()].value

        if save:
            vals.append((chan_name, max_value, max_value_frequency))
            df_vals = pd.DataFrame(vals, columns=['channel', 'max_correlation', 'frequency'])
            df_vals_ = df_vals[df_vals['max_correlation'] > 0]
        else:
            vals.append(-1)

    return df_vals_


def get_max_corr_band(output_dir, flow=10, fhigh=20, save=False):
    files = glob.glob(f'{output_dir}*.csv')
    vals = []

    for file in files:
        chan_name = file.split('/')[-1].split('DQ')[0] + 'DQ'
        fs = FrequencySeries.read(file)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(flow / n_diff), int(fhigh / n_diff)
        fs_ = fs[ind1:ind2]
        max_value = fs_.max().value
        max_value_frequency = fs_.frequencies[fs_.argmax()].value

        if save:
            vals.append((chan_name, max_value, max_value_frequency))
            df_vals = pd.DataFrame(vals, columns=['channel', 'max_correlation', 'frequency'])
            df_vals_ = df_vals[df_vals['max_correlation'] > 0]
        else:
            vals.append(-1)

    return df_vals_


def combine_csv(dir_path, ifo):
    all_files = glob.glob(f'{dir_path}*.csv')
    chan_removes = get_unsafe_channels(ifo=ifo)['channel']

    for j in chan_removes:
        all_files = [i for i in all_files if not i.startswith(f'{dir_path}{j}')]

    li = []
    fns = []

    for filename in all_files:
        fn = filename.split('/')[-1].split('_14')[0] + '_freq'
        fn_val = filename.split('/')[-1].split('_14')[0] + '_corr'
        fns.append(fn)
        fns.append(fn_val)
        df = pd.read_csv(filename, index_col=None, header=None)
        li.append(df)

    frame = pd.concat(li, axis=1, ignore_index=True)
    frame.columns = fns
    return frame


def find_max_corr_channel(path, fft=10, ifo='L1'):
    frame_ = combine_csv(path, ifo)
    max_vals = []

    for i in range(len(frame_)):
        max_val_ = frame_.iloc[i, 1::2].sort_values(ascending=False)
        chan_names = max_val_.index[:2]
        chan_names = [i.replace('_corr', '').replace('.csv','') for i in chan_names]
        max_corr_val = [max_val_.iloc[0], max_val_.iloc[1]]
        max_vals.append((i / fft, chan_names[0], max_corr_val[0], chan_names[1], max_corr_val[1]))

    df = pd.DataFrame(max_vals, columns=['frequency', 'channel1', 'coh1', 'channel2', 'coh2'])
    return df


def plot_max_corr_chan(path, fft, ifo, flow=0, fhigh=200, plot=True, savedir=os.getcwd()+'/plots'):
    time_ = int(path.split('/')[-2])
    vals = find_max_corr_channel(path=path, fft=fft, ifo=ifo)
    print("Got the data, now making plots")
    vals = vals.iloc[flow * fft:fhigh * fft + 1]
    vals['group1'] = vals['channel1'].apply(give_group_v2)
    vals['group2'] = vals['channel2'].apply(give_group_v2)
    vals = vals[(vals['coh1'] <=1) & (vals['coh2']<=1)]
    vals.rename(columns={'coh1':'coherence', 'coh2':'Coherence'},  inplace=True)

    if plot:
        fig1 = px.scatter(vals, x="frequency", y="coherence", hover_data=['channel1'], color="group1",
                          labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"})
        fig1.update_layout(
            title=dict(text=f"Highest Coherence channel at each frequency during {time_} -- {time_ + 900}",
                       font=dict(family="Courier New, monospace", size=28, color="RebeccaPurple")))

        fig2 = px.scatter(vals, x="frequency", y="coherence", hover_data=['channel2'], color="group2",
                          labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"})
        fig2.update_layout(
            title=dict(text=f"Second highest Coherence channel at each frequency during {time_} -- {time_ + 900}",
                       font=dict(family="Courier New, monospace", size=28, color="RebeccaPurple")))

        plotly.offline.plot(fig1, filename=f'{savedir}/channels_coh_{int(time_)}_a.png')
        plotly.offline.plot(fig2, filename=f'{savedir}/channels_coh_{int(time_)}_b.png')

    return vals
def check_channel_coherence(channel, ifo, t1, t2):

    files = get_frame_files(starttime=t1, endtime=t2, ifo=ifo)
    ts_aux = TimeSeries.read(files, channel=channel, start=t1, end=t2)
    ts_gds = TimeSeries.fetch('{}:GDS-CALIB_STRAIN'.format(ifo), t1, t2)

    ts_gds = ts_gds.resample(ts_aux.sample_rate)
    coh = ts_gds.coherence(ts_aux, fftlength=10, overlap=5)

    return coh