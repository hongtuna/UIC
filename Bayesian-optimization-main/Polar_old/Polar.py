#  general import
import asyncio, os, signal, sys, time, math

# logger
import logging as logger

# data processing
import pandas as pd
import numpy as np

# ble realted import
from bleak import BleakClient
from bleak.uuids import uuid16_dict

# plotting
import matplotlib.pyplot as plt
import matplotlib

#lsl related improt
import pylsl

# import utils
from utils import convert_array_to_signed_int, convert_to_unsigned_long
from lslload import ECG_stream, ACC_stream


class polar(object):
    def __init__(self, address:str = "E5:55:3B:1E:63:CA", ECG:bool = True, ACC:bool = True,
                 publish_dashboard:bool = True):
        self.ADDRESS = address
        self.ECG_DATA = ECG
        self.ACC_DATA = ACC
        self.SAVE_DATA = True
        self.PUBLISH_DASHBOARD = publish_dashboard
        # place holder for client
        self.client = None
        self._setup()
        self.ECG_data = {}
        self.ECG_data['ecg'] = []
        self.ECG_data['time'] = []
        self.ACC_data = {}
        self.ACC_data['acc'] = []
        self.ACC_data['time'] = []
        self.previous_time = 0
        self.ACC_stream = ACC_stream('polar accel')
        self.ECG_stream = ECG_stream('polar ECG')

    def _setup(self):
        """ declare all the parameter necessary
        """
        uuid16_dict_polar = {v: k for k, v in uuid16_dict.items()}

        ## This is the device MAC ID, please update with your device ID

        ## UUID for model number ##
        self.MODEL_NBR_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
            uuid16_dict_polar.get("Model Number String")
            )


        ## UUID for manufacturer name ##
        self.MANUFACTURER_NAME_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
            uuid16_dict_polar.get("Manufacturer Name String")
        )

        ## UUID for battery level ##
        self.BATTERY_LEVEL_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
            uuid16_dict_polar.get("Battery Level")
        )

        ## UUID for connection establsihment with device ##
        self.PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"

        ## UUID for Request of stream settings ##
        self.PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"

        ## UUID for Request of start stream ##
        self.PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

        # load all the write data byte arrays
        self.WRITE_DATA = {'ECG':
                           bytearray([0x02, 0x00, 0x00, 0x01, 0x82, \
                                      0x00, 0x01, 0x01, 0x0E, 0x00]),
                           'ACC':bytearray(
                               [0x02,
                                0x02,
                                0x00,
                                0x01,
                                0xC8,
                                0x00,
                                0x01,
                                0x01,
                                0x10,
                                0x00,
                                0x02,
                                0x01,
                                0x08,
                                0x00])
                           }

        ## For Plolar H10  sampling frequencies ##
        self.SAMPLING_FREQ = {'ECG': 130, 'ACC': 200}


    async def main(self):
        try:
            print('first line of try statement')
            async with BleakClient(self.ADDRESS) as client:
                print('added device')
                signal.signal(signal.SIGINT, self._interrupt_handler)
                tasks = [
                    asyncio.ensure_future(self._run(client)),
                ]

                await asyncio.gather(*tasks)
        except:
            pass

    ## Bit conversion of the Hexadecimal stream




    ## Aynchronous task to start the data stream for ECG ##
    async def _run(self, client, debug:bool =False):

        ## Writing chracterstic description to control point for request of UUID (defined above) ##

        await client.is_connected()
        print("---------Device connected--------------")

        model_number = await client.read_gatt_char(self.MODEL_NBR_UUID)
        print("Model Number: {0}".format("".join(map(chr, model_number))))

        manufacturer_name = await client.read_gatt_char(self.MANUFACTURER_NAME_UUID)
        print("Manufacturer Name: {0}".format("".join(map(chr, manufacturer_name))))

        battery_level = await client.read_gatt_char(self.BATTERY_LEVEL_UUID)
        print("Battery Level: {0}%".format(int(battery_level[0])))

        att_read = await client.read_gatt_char(self.PMD_CONTROL)

        if self.ACC_DATA:
            print('In the acc write condition')
            await client.write_gatt_char(self.PMD_CONTROL, self.WRITE_DATA['ACC'], response  = True)
            print('finished the ACC')
        if self.ECG_DATA:
            print('In the ecg write condition')
            await client.write_gatt_char(self.PMD_CONTROL, self.WRITE_DATA['ECG'], response = True)

        ## ECG stream started
        await client.start_notify(self.PMD_DATA, self.data_conv)

        print("Collecting ECG data...")

        # ## Plot configurations
        # plt.style.use("ggplot")
        # fig = plt.figure(figsize=(15, 6))
        # ax = fig.add_subplot()
        # fig.show()


        while True:

            ## Collecting ECG data for 1 second
            await asyncio.sleep(1)
            print(len(self.ACC_data['time']))
            # ax.plot(ecg_session_data, color="r")


    def _interrupt_handler(self, signnum, frame):
        if self.client is None:
            logger.info('no ble client found closing the device')
            sys.exit()
        else:
            logger.info('found ble client stopping')
            # close the connection properly
            self.client.disconnect()
            print('self')
            sys.exit()

    def data_conv(self, sender, data):
        save_data = np.copy(data)
        # ecg data
        print(sender)
        # print(data)
        if data[0] == 0x00:
            timestamp = convert_to_unsigned_long(data, 1, 8)
            step = 3
            samples = data[10:]
            offset = 0
            i = 0
            time_diff = time.time() - self.previous_time
            ECG_ = []
            while offset < len(samples):
                i += 1
                ecg = convert_array_to_signed_int(samples, offset, step)
                offset += step
                self.ECG_data['ecg'].extend([ecg])
                self.ECG_data['time'].extend([timestamp])
                ECG_.append(ecg)
            print('ecg', ecg, i)
            self.previous_time_ecg = time.time()
            self.ECG_stream.push_chunk(ECG_, pylsl.local_clock() - time_diff)
        # ACC data
        if data[0] == 0x02:
            print('data')
            timestamp = convert_to_unsigned_long(data, 1, 8)
            frame_type = data[9]
            resolution = (frame_type + 1) * 8
            step = math.ceil(resolution / 8.0)
            samples = data[10:]
            offset = 0
            i = 0
            time_diff = time.time() - self.previous_time
            ACC_data = []
            while offset < len(samples):
                i += 1
                x = convert_array_to_signed_int(samples, offset, step)
                offset += step
                y = convert_array_to_signed_int(samples, offset, step)
                offset += step
                z = convert_array_to_signed_int(samples, offset, step)
                offset += step
                ACC_data.append([x,y,z])
                self.ACC_data['acc'].extend([[x, y, z]])
                self.ACC_data['time'].extend([timestamp])
            print('x',x, 'y ', y, 'z ', z, i)
            self.ACC_stream.push_chunk(ACC_data, pylsl.local_clock() - time_diff)
            self.previous_time = time.time()
