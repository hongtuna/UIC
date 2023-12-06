import pyxdf
import matplotlib.pyplot as plt
import numpy as np

data, header = pyxdf.load_xdf('test_data/sub-P001_ses-S001_task-Default_run-002_eeg.xdf')

for stream in data:
    plt.figure()
    y = stream['time_series']
    plt.plot(y)
    plt.show()
