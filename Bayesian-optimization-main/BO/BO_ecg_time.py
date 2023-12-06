"""Example program to demonstrate how to read string-valued markers from LSL."""

from pylsl import StreamInlet, resolve_stream
import logging, socket

# data processing
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec

# BO import 
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

# kernel hyperparameter tuning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn import preprocessing
import time
import struct
import sys

class BO():
    def __init__(self):
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds={'x': (1, 100), 'y': (-2, 2)},
            verbose=2,
            random_state=1,
        )

        self.grid = np.linspace(1, 100, 100).reshape(-1, 1)
        self.tmin_data = []
        self.rm_arr = []
        self.stif_arr = [30,50,90]
        
        kernel = ConstantKernel(1, (0.01, 100)) * RBF(length_scale=50, length_scale_bounds=(0.01, 100.0)) + WhiteKernel(noise_level = 1, noise_level_bounds=(0.01, 100.0))
        self.optimizer._gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
        
    
    def posterior(self,optimizer, x_obs, y_obs, grid):
        self.optimizer._gp.fit(x_obs, y_obs)
        mu, sigma = self.optimizer._gp.predict(grid, return_std=True)
        return mu, sigma

    def make_plot(self, x_obs, y_obs, grid, sigma, mu, utility):
        print('generate plot')

        fig = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        axis = plt.subplot(gs[0])
        acq = plt.subplot(gs[1])
        
        axis.plot(x_obs.flatten(), y_obs, 'o', markersize=6, label=u'Observations', color='r')
        axis.plot(grid, mu, '--', color='k', label='Prediction')

        axis.fill(np.concatenate([grid, grid[::-1]]), 
                np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
            alpha=.6, fc='gray', ec='None', label='95% confidence interval')
        
        axis.set_xlim((1, 100))
        axis.set_ylim((-2, 2))
        axis.set_ylabel('f(x)', fontdict={'size':20})
        axis.set_xlabel('x', fontdict={'size':20})
        
        acq.plot(grid, utility, label='EI', color='purple')


        acq.plot(grid[np.argmax(utility)], np.max(utility), '*', markersize=15, 
                label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        
        acq.set_xlim((1, 100))
        acq.set_ylim((0, np.max(utility) + 0.5))


        
        acq.set_ylabel('EI', fontdict={'size':20})
        acq.set_xlabel('x', fontdict={'size':20})
        
        axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
        acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

class UDP():
    def __init__(self):
        self.serverAddressPort = ("192.168.1.35", 20001) # **client address**
        self.bufferSize = 1024
        self.UDPsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)


    def send(self, message):
        MESSAGE = struct.pack('i', int(message)) # encoding utf-8 can not work on simulink
        # print('sending parameter',i,'to client')
        self.UDPsocket.sendto(MESSAGE, self.serverAddressPort)

    def close(self):
        logging.warning('closing the socket')
        self.UDPsocket.close()



if __name__ == "__main__":
    print("looking for a marker stream...")
    streams = resolve_stream()
    bayop = BO()
    udp = UDP()
    i = 0
    
    try : 
        start_time = time.time()
        getdata = []

        for inlet in streams:
            inlet.name() == 'RMSSD'
            # create a new inlet to read from the stream
            inlet = StreamInlet(streams[0])
            
            while True:
                # get a new sample (you can also omit the timestamp part if you're not
                # interested in it)
                sample,timestamp = inlet.pull_sample()
                print("got %s at time %s" % (sample[0], timestamp))
                getdata.append(sample[0])   
                
                if int(time.time()) - int(start_time) == 60: # execute Bayesian optimization every 1 minute 
                    
                    bayop.rm_arr.append(np.mean(getdata)) 
                    i = i + 1
                    
                    if i >= 3 : # BO start 
                    
                        x_obs = np.array(bayop.stif_arr).reshape(-1,1)
                        y_obs = np.array(bayop.rm_arr)
                       
                        normalized_arr = preprocessing.normalize([y_obs])
                        y_obs = normalized_arr[0]
                        
                        print('x_obs',x_obs)
                        print('y_normalize',y_obs)
                        
                        utility_function = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
                        utility = utility_function.utility(bayop.grid, bayop.optimizer._gp, 0)

                        print('next stiffness parameter:',bayop.grid[np.argmax(utility)])

                        mu, sigma = bayop.posterior(bayop.optimizer, x_obs, y_obs, bayop.grid)
                        next_ = bayop.grid[np.argmax(utility)][0]
                        
                        bayop.make_plot(x_obs, y_obs, bayop.grid, sigma, mu, utility)
                        plt.savefig('{}_iteration.jpg'.format(i))

                        bayop.stif_arr.append(next_)
                        
                        # Send parameter
                        udp.send(next_)
                        print('send')

                        getdata =[]
                    start_time = time.time()
                
                else:
                    continue

    except KeyboardInterrupt:
        print ('keyboard Interrupted')
        udp.close()
        sys.exit(0)


