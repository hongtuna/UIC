from  pylsl import StreamInfo, StreamOutlet

def ECG_stream(name:str, sampling_frq:int = 130):
    info = StreamInfo(name, 'ECG', 1, sampling_frq, 'float32', 'myuid2424')
    info.desc().append_child_value("manufacturer", "Polar")
    channels = info.desc().append_child("channels")
    for c in ["ECG"]:
        channels.append_child("channel")\
            .append_child_value("name", c)\
            .append_child_value("unit", "microvolts")\
            .append_child_value("type", "ECG")
    return StreamOutlet(info, 74)

def ACC_stream(name:str, sampling_frq:int  = 200):
    info = StreamInfo(name, 'ACC', 3, sampling_frq, 'float32', 'myuid2425')
    info.desc().append_child_value("manufacturer", "Polar")
    channels = info.desc().append_child("channels")
    for c in ['X', 'Y', 'Z']:
        channels.append_child("channel")\
            .append_child_value("name", c)\
            .append_child_value("unit", "mg")\
            .append_child_value("type", "ACC")
    return StreamOutlet(info, 32)
    
