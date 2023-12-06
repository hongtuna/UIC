from  pylsl import StreamInfo, StreamOutlet
import pylsl

def RMSSD_stream():
    info  = pylsl.StreamInfo('RMSSD',  'Marker', 1, 0, 'float32', 'myuidw43537')
    return StreamOutlet(info)

def RMSSD_fault_stream():
    info  = pylsl.StreamInfo('RMSSD_fault_counter',  'Marker', 1, 0, 'float32', 'myuidw43537')
    return StreamOutlet(info)

