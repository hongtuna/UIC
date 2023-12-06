# import typing
from typing import List
import logging

import numpy as np
import math
import pylsl
import time, os, copy
import sys

# for processing
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt


# send the data to the LSL stream
from cost_utils import RMSSD_stream, RMSSD_fault_stream

os.chdir('../')
from main import MainCommunication
# print(os.getcwd())

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

        # setting up the LSL stream outlets
        self.outlet = RMSSD_stream()
        self.fault_outlet = RMSSD_fault_stream()
        self.communication = MainCommunication()

        # thresholds and previous values
        # setting some large RMMSD value at the start.
        self.previous_HR = 1000
        self.SKIP_THRESHOLD = 30
        self.SKIP_COUNTER = 0


        # setting up the buffer
        buffer_size = (2 * math.ceil(info.nominal_srate() * data_length), info.channel_count())
        self.buffer = np.empty(buffer_size, dtype = self.dtypes[info.channel_format()])
        self.FIRST_DATA = True
        self.store_data = []

    def get_data(self):
        # pull the data and store in the buffer
        _, ts = self.inlet.pull_chunk(timeout = 0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)

        # check if the data is new
        if ts:
            # if the data is new, set the new data to False append the data to the store_data
            ts = np.asarray(ts)
            if self.FIRST_DATA:
                self.store_data = self.buffer[0:ts.size,:]
                self.FIRST_DATA = False
            # if the data is old then append the data to the store_data
            else:
                # print(self.store_data.shape, self.buffer[0:ts.size,:].shape)
                self.store_data = np.append(self.store_data, self.buffer[0:ts.size], axis = 0)
                # quick convert the data to the pd dataframe to clean the data.
                raw_data_pd = pd.DataFrame(self.store_data[~4000:])
                # adding data to the dataframe for the processing.
                raw_data_pd.columns = ['ecg']
                # preprocessing the data.
                self.cleaned = nk.ecg_clean(raw_data_pd['ecg'], sampling_rate = 133)



    # TODO test this individually
    def _indetify_no_zeros(self, a: np.array) -> np.array:
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
    def send_data(self) -> None:
        """
        Check the data for the number of zeros and check the data for number of skipped R-peaks.
        """
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
        if self.first_data and not self.new_data:
            logging.info(f"{__name__}: added the first data")
            self.previous_HR = new_rmssd

        # if the number of zeros is more than set value (100) recollect the data.
        # or difference between the previous and the current RMSSD is more threshold. skip the data..
        if no_zeros > 100 or (self.previous_HR - self.new_data) > self.SKIP_THRESHOLD:
            logging.info(f"{__name__}: resetting the data collection number of  \
                         zeros found {no_zeros}, {self.previous_HR}, \
                         { self.previous_HR < 4 * float(signal['HRV_RMSSD'])}, \
                         {float(signal['HRV_RMSSD'])} ")
            # update the fault counter.
            self.SKIP_COUNTER += 1
            # send the data back to the fault counter.
            self.fault_outlet.push_sample([self.SKIP_COUNTER])
            # reset the data.
            self.store_data = []
            # change the flag to first data
            self.FIRST_DATA = True
            self.previous_HR = 1000
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

        # return success
        return 1



class SetupStreams(object):
    """ Get the streams and setup the inlets and outlets"""

    def __init__(self, stream_name: str = 'polar ECG') -> None:
        """ Initialize the streams and inlets and outlets

        :param stream_name: the name of the stream to be used.

        """
        self.inlets: List[Inlet] = []
        self.streams = pylsl.resolve_streams()
        self.stream_name = stream_name

        for info in self.streams:
            # check if the stream is the one we want.
            if info.name == self.stream_name:
                logging.info(f"{__name__}: found the stream {info.name}")
                data_inlet = DataInlet(info, 1000)
                self.inlets.append(data_inlet)
            # send an error message
            else:
                print(f"{__name__}: did not find the stream {info.name}")
                logging.info(f"{__name__}: did not find the stream {info.name}")
                logging.error(f"{__name__}: could not find the stream {self.stream_name}")

    def run(self) -> int:
        """ Run the streams and inlets"""

        for inlet in self.inlets:
            inlet.get_data()
            # Checking the inlet data size and send the data to the pylsl stream.
            if inlet.store_data.size > 250:
                # send the data back to the pylsl stream.
                # send the data to the speedgoat though UDP communication.
                inlet.send_data()
            else:
                logging.info(f"{__name__}: no data to send")
                return 0

            # return success
            return 1




if __name__ == '__main__':
    stream = SetupStreams()
    while True:
        time.sleep(0.1)
        stream.run()






