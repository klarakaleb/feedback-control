"""
Some useful extra functions.
"""

import sklearn.metrics as met
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(output,tid,target):
    try:
        # setup
        cols = [plt.cm.magma(i) for i in np.linspace(0.1,0.9,8)]
        dt = 0.01
        
        # helper functions
        def get_pos(vel):
            pos = np.zeros(vel.shape)
            for j in range(vel.shape[1]):
                if j==0:
                    pos[:,j] = dt*vel[:,j]
                else:
                    pos[:,j] = pos[:,j-1] + dt*vel[:,j]
            return pos
        
        pos = get_pos(output)
        posT = get_pos(target)
        for j in range(pos.shape[0]):
            plt.plot(pos[j,:,0],pos[j,:,1],color=cols[tid[j]],alpha=0.2)
        # plot also targets
        for j in range(8):
            tmp = posT[tid==j,-1][0]
            plt.scatter(tmp[0],tmp[1],edgecolor=cols[j],facecolor='None',marker='s')
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        plt.axis('off')
    except:
        pass
    try:
        score = met.explained_variance_score(output.reshape(-1,2),target.reshape(-1,2))
        plt.title('VAF=%.2f'%score)
    except:
        plt.title('VAF=nan')
    