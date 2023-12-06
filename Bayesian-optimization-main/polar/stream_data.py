# import typing
from typing import List
import logging

import numpy as np
import math
import pylsl
import time, os, copy


# for processing
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt


# remove future warnings
import warnings
warnings.filterwarnings("ignore")

# Go to the directory where the script is located
os.chdir('../')
# Print the current working directory (for debugging)
print(os.getcwd())
# loading the communcation module
from utils.main_communication import MainCommunication

class Inlet:
    """Parent class
    """
    def __init__(self, info: pylsl.StreamInfo, data_length: int):
        # fill the basic information from the stream.
        self.inlet = pylsl.StreamInlet(info, max_buflen = data_length,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.new_data = False



class DataInlet(Inlet):
    """ Class to stream the data for all the channels"""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, data_length: int):
        super().__init__(info, data_length)
        buffer_size = (2 * math.ceil(info.nominal_srate() * data_length), info.channel_count())
        self.buffer = np.empty(buffer_size, dtype = self.dtypes[info.channel_format()])
        self.first_data = True
        self.store_data = np.array([])
        info  = pylsl.StreamInfo('ECG_processed',  'Marker', 1, 0, 'float32', 'myuidw43537')
        self.outlet = pylsl.StreamOutlet(info)
        self.communication = MainCommunication()
        # setting some large RMMSD value at the start.
        self.previous_HR = 1000
        self.skip_threshold = 30
        self.skip_counter = 0
        self.cleaned = []

    def get_data(self):
        # pull the data and store in the buffer
        _, ts = self.inlet.pull_chunk(timeout = 0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)

        if ts:
            ts = np.asarray(ts)
            print(self.name == 'polar ECG')
            if self.name  == 'polar ECG':
                if self.first_data:
                    self.store_data = self.buffer[0:ts.size,:]
                    self.first_data = False
                else:
                    # print(self.store_data.shape, self.buffer[0:ts.size,:].shape)
                    self.store_data = np.append(self.store_data.flatten(), self.buffer[0:ts.size].flatten(), axis = 0)
                    # quick convert the data to the pd dataframe to clean the data.
                    raw_data_pd = pd.DataFrame(self.store_data[~4000:])
                    raw_data_pd.columns = ['ecg']
                    self.cleaned = nk.ecg_clean(raw_data_pd['ecg'], sampling_rate = 133)



    # TODO test this individually
    def _indetify_no_zeros(self, a: np.array):
        a = np.array(list(a))    # convert elements to `str`
        rr = np.argwhere(a == '0').ravel()  # find out positions of `0`
        if not rr.size:  # if there are no zeros, return 0
            return 0

        full = np.arange(rr[0], rr[-1]+1)  # get the range of spread of 0s

        # get the indices where `0` was flipped to something else
        diff = np.setdiff1d(full, rr)
        if not diff.size:     # if there are no bit flips, return the
            return len(full)  # size of the full range

        # break the array into pieces wherever there's a bit flip
        # and the result is the size of the largest chunk
        pos, difs = full[0], []
        for el in diff:
            difs.append(el - pos)
            pos = el + 1

        difs.append(full[-1]+1 - pos)

        # return size of the largest chunk
        res = max(difs) if max(difs) != 1 else 0

        return res

# DONE: make a seperate module for the Realtime ECG processing
    def send_data(self):
        """
        Check the data for the number of zeros and check the data for number of skipped R-peaks.
        """


        if len(self.cleaned) == 0 :
            return 0
        # creating a seperate cleaned.
        cleaned = copy.copy(self.cleaned)

        # get the number of zeros in the data.
        no_zeros = self._indetify_no_zeros(cleaned)

        # identify the number of peaks in the data.
        peaks, info = nk.ecg_peaks(cleaned, sampling_rate = 133)

        # calculate the RMSSD.
        signal = nk.hrv_time(peaks, sampling_rate=133, show = False)

        # setting the new data
        new_rmssd = float(signal['HRV_RMSSD'])

        # check if this is the first data.
        if self.first_data:
            logging.info(f"{__name__}: added the first data")
            self.previous_HR = new_rmssd

        # if the number of zeros is more than set value (100) recollect the data.
        # or difference between the previous and the current RMSSD is more threshold. skip the data..
        if no_zeros > 50 or abs(self.previous_HR - new_rmssd) > self.skip_threshold:
            logging.warn(f"{__name__}: resetting the data collection number of zeros found {no_zeros}, { self.previous_HR < 4 * float(signal['HRV_RMSSD'])} {float(signal['HRV_RMSSD'])} ")
            self.store_data = []
            self.first_data = True
            self.previous_HR = 50
            return 0

        # if not send the data

        # sending the data to the pylsl stream.
        self.outlet.push_sample([signal['HRV_RMSSD'][0]])
        # sending the data to the BO through the communication module.
        self.communication.send_pred(float(signal['HRV_RMSSD']))
        # setting the previous HR to the current HR.
        self.previous_HR = new_rmssd
        # setting the new data to True
        self.new_data = True

        # debugging
        print(f"{__name__}: sending the data {signal['HRV_RMSSD']}")

        # return success
        return 1



class SetupStreams():
    def __init__(self):
        self.inlets: List[Inlet] = []
        self.streams = pylsl.resolve_streams()

        for info in self.streams:
            print(info.name())
            if info.name() == 'polar ECG':
                self.inlets.append(DataInlet(info, 1000))

    def run(self):
        for inlet in self.inlets:
            inlet.get_data()
            # Checking the inlet data size and send the data to the pylsl stream.
            if len(inlet.store_data) > 250:
                inlet.send_data()
            else:
                logging.warn(f"{__name__}: no data to send")

            # return success
            return 1




if __name__ == '__main__':
    stream = SetupStreams()
    while True:
        time.sleep(1)
        stream.run()






