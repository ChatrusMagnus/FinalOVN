import random
import numpy as np
import copy
from FinalOVN.Network import Network
from FinalOVN.Connection import Connection
from FinalOVN.Analytics import Monte_carlo
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
from mpl_toolkits.mplot3d import Axes3D


def create_traffic_matrix (nodes , rate , multiplier =1):
    s = pd. Series ( data =[0.0] * len( nodes ), index = nodes )
    df = pd. DataFrame ( float ( rate * multiplier ),index =s.index , columns =s.index , dtype =s. dtype )
    np. fill_diagonal (df. values , s)
    return df

def plot3Dbars(t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_data,y_data=np.meshgrid(np. arange (t. shape [1]) ,np. arange (t. shape [0]))
    x_data = x_data.flatten ()
    y_data = y_data.flatten ()
    z_data = t.flatten ()
    ax.bar3d( x_data,y_data,np.zeros(len(z_data)),1,1,z_data)
    plt.show()

def main():
    analitic=Monte_carlo(2,10,600,'snr','shannon',5)
    analitic.run_simulations()

if __name__ == "__main__":
    main()