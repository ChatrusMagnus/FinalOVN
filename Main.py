import random
import numpy as np
import copy
from FinalOVN.Network import Network
from FinalOVN.Connection import Connection
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
    NMC = 1
    node_pairs_realizations = []
    stream_conn_list = []
    lines_state_list = []
    for i in range(NMC):
        '''starts monte carlo realisation'''
        print('Monte - Carlo Realization #{:d}'.format(i + 1))

        network = Network('nodes.json',10)  # creates nodes and line objects
        network.connect()
        #network.draw()

        node_labels = list(network.nodes.keys())
        T = create_traffic_matrix(node_labels, 600,10)
        t = T.values
        #all seems ok
        connections = []
        node_pairs = list(filter(lambda x: x[0] != x[1], list(it.product(node_labels, node_labels))))
        random.shuffle(node_pairs)
        node_pairs_realizations.append(copy.deepcopy(node_pairs))
        connections = []

        for node_pair in node_pairs:
            connection = Connection(node_pair[0], node_pair[-1], float(T.loc[node_pair[0], node_pair[-1]]))
            connections.append(connection)  # list of connection objects
        streamed_connections = network.stream(connections, best='snr', transceiver='shannon')
        stream_conn_list.append(streamed_connections)
        lines_state_list.append(network.lines)
        snrs = []
        [snrs.extend(connection.snr) for connection in streamed_connections]
        rbl = []

        for connection in streamed_connections:
            for lightpath in connection.lightpaths:
                rbl.append(lightpath.bitrate)
        # Plot
        plt.hist(snrs, bins=10)
        plt.title('SNR Distribution [dB]')
        plt.show()
        rbc = [connection.calculate_capacity() for connection in streamed_connections]
        plt.hist(rbc, bins=10)
        plt.title('Connection Capacity Distribution [ Gbps ]')
        plt.show()
        plt.hist(rbl, bins=10)
        plt.title('Lightpaths Capacity Distribution [ Gbps ]')
        plt.show()
        s = pd.Series(data=[0.0] * len(node_labels), index=node_labels)
        df = pd.DataFrame(0.0, index=s.index, columns=s.index, dtype=s.dtype)
        print(df)

        for connection in streamed_connections:
            df.loc[connection.start_node, connection.end_node] = connection.bitrate
        print(df)
        plot3Dbars(t)
        plot3Dbars(df.values)
        print('Avg SNR: {:.2f} dB '.format(np.mean(list(filter(lambda x: x != 0, snrs)))))
        print('Total Capacity Connections : {:.2f} Tbps '.format(np.sum(rbc) * 1e-3))
        print('Total Capacity Lightpaths : {:.2f} Tbps '.format(np.sum(rbl) * 1e-3))
        print('Avg Capacity : {:.2f} Gbps '.format(np.mean(rbc)))

    # Get MC stats
    snr_conns = []
    rbl_conns = []
    rbc_conns = []

    for streamed_conn in stream_conn_list:
        snrs = []
        rbl = []
        [snrs.extend(connection.snr) for connection in streamed_conn]
        for connection in streamed_conn:
            for lightpath in connection.lightpaths:
                rbl.append(lightpath.bitrate)

        rbc = [connection.calculate_capacity() for connection in streamed_conn]

        snr_conns.append(snrs)
        rbl_conns.append(rbl)
        rbc_conns.append(rbc)

        avg_snr_list = []
        avg_rbl_list = []
        traffic_list = []
        [traffic_list.append(np.sum(rbl_list)) for rbl_list in rbl_conns]
        [avg_rbl_list.append(np.mean(rbl_list)) for rbl_list in rbl_conns]
        [avg_snr_list.append(np.mean(list(filter(lambda x: x != 0, snr_list)))) for snr_list in snr_conns]

        # Congestion
        lines_labels = list(lines_state_list[0].keys())
        congestions = {label: [] for label in lines_labels}
        for line_state in lines_state_list:
            for line_label, line in line_state.items():
                cong = line.state.count('occupied') / len(line.state)
                congestions[line_label].append(cong)
        avg_congestion = {label: [] for label in lines_labels}
        for line_label, cong in congestions.items():
            avg_congestion[line_label] = np.mean(cong)
            print(avg_congestion[line_label])
        plt.bar(range(len(avg_congestion)), list(avg_congestion.values()), align='center')
        plt.xticks(range(len(avg_congestion)), list(avg_congestion.keys()))
        plt.show()


        print('\n')
        print('Line to upgrade : {} '.format(max(avg_congestion, key=avg_congestion.get)))
        print('Avg Total Traffic : {:.2f} Tbps '.format(np.mean(traffic_list) * 1e-3))
        print('Avg Lighpath Bitrate : {:.2f} Gbps '.format(np.mean(avg_rbl_list)))
        print('Avg Lighpath SNR: {:.2f} dB '.format(np.mean(avg_snr_list)))



        '''
        network = Network('nodes.json')
        network.connect()
        network.draw()
        node_labels = list(network.nodes.keys())
        connections = []
        for i in range(100):
            random.shuffle(node_labels)
            connection = Connection(node_labels[0], node_labels[-1],100)
            connections.append(connection)
        streamed_connections = network.stream(connections, best='snr')
        snrs = [connection.snr for connection in streamed_connections]
        plt.hist(snrs, bins=10)
        plt.title('SNR Distribution')
        plt.show()
        df = network.route_space
        df.to_html('free.html')
        rbs = [connection.calculate_capacity() for connection in streamed_connections]
        plt.hist(rbs, bins=10)
        plt.title('Bitrate Distribution [Gbps]')
        plt.show()
    
        print('Total Capacity: {:.2f} Tbps '.format(np.sum(rbs) * 1e-3))
        print('Avg Capacity: {:.2f} Gbps '.format(np.mean(rbs)))
    '''
if __name__ == "__main__":
    main()