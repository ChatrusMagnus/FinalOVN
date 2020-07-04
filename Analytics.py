from FinalOVN.Network import Network
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import copy
from FinalOVN.Connection import Connection


class Monte_carlo(object):
    def __init__(self,simulation_number=1,channel=10,request_rate=600,best='snr',transceiver='shannon',multiplier=1):
        self._number_simulations=simulation_number
        self._channel=channel
        self._request_rate=request_rate
        self._best=best
        self._multiplier=multiplier
        self._transceiver=transceiver
        self._list_snr=[]
        self._list_bitrate_lightpaths=[]
        self._list_bitrate_connections=[]
        self._request_matrix=None
        self._dataframe_connection_bitrate=None
        self._stream_conn_list=[]
        self._lines_state_list=[]
        self._avg_snr_list=[]
        self._avg_rbl_list=[]
        self._traffic_list=[]

    @property
    def avg_snr_list(self):
        return self._avg_snr_list

    @property
    def avg_rbl_list(self):
        return self._avg_rbl_list

    @property
    def traffic_list(self):
        return self._traffic_list


    @property
    def stream_conn_list(self):
        return self._stream_conn_list

    @property
    def lines_state_list(self):
        return self._lines_state_list

    @property
    def multiplier(self):
        return self._multiplier

    @property
    def number_simulations(self):
        return self._number_simulations

    @property
    def channel(self):
        return self._channel

    @property
    def request_rate(self):
        return self._request_rate

    @property
    def best(self):
        return self._best

    @property
    def transceiver(self):
        return self._transceiver

    @property
    def list_snr(self):
        return self._list_snr

    @list_snr.setter
    def list_snr(self,snrs):
        self._list_snr=snrs

    @property
    def list_bitrate_lightpaths(self):
        return self._list_bitrate_lightpaths

    @list_bitrate_lightpaths.setter
    def list_bitrate_lightpaths(self, lbl):
        self._list_bitrate_lightpaths=lbl

    @property
    def list_bitrate_connections(self):
        return self._list_bitrate_connections

    @list_bitrate_connections.setter
    def list_bitrate_connections(self, lbc):
        self._list_bitrate_connections=lbc

    @property
    def request_matrix(self):
        return self._list_bitrate_connections

    @request_matrix.setter
    def request_matrix(self, rm):
        self._request_matrix=rm

    @property
    def dataframe_connection_bitrate(self):
        return self._dataframe_connection_bitrate

    @dataframe_connection_bitrate.setter
    def dataframe_connection_bitrate(self, dfcb):
        self._dataframe_connection_bitrate=dfcb

    def create_traffic_matrix (self,nodes,rate,multiplier =1):
        s = pd. Series (data=[0.0]*len(nodes), index = nodes )
        df = pd. DataFrame (float ( rate * multiplier ),index =s.index , columns =s.index , dtype =s. dtype )
        np. fill_diagonal (df.values,s)
        return df

    def plot3Dbars(self,t,title='',type='console',mcn=0):
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111, projection='3d')
        x_data,y_data=np.meshgrid(np. arange (t. shape [1]) ,np. arange (t. shape [0]))
        x_data = x_data.flatten ()
        y_data = y_data.flatten ()
        z_data = t.flatten ()
        ax.bar3d( x_data,y_data,np.zeros(len(z_data)),1,1,z_data)
        if type == 'console':
            plt.show()
        elif type == 'pdf':
            plt.savefig('plot_'+title+'_' + str(mcn) + '.pdf')

    def plot_list_snr(self,type='console',precision=10,mcn=0):
        fig=plt.figure()
        fig.suptitle('SNR Distribution [dB]')
        plt.hist(self.list_snr, bins=precision)
        if type == 'console':
            plt.show()
        elif type == 'pdf':
            plt.savefig('plot_list_snr_'+str(mcn)+'.pdf')

    def plot_list_bitrate_connections(self,type='console',precision=10,mcn=0):
        fig=plt.figure()
        fig.suptitle('Connection Capacity Distribution [Gbps]')
        plt.hist(self.list_bitrate_connections, bins=precision)
        if type == 'console':
            plt.show()
        elif type == 'pdf':
            plt.savefig('plot_list_bitrate_connections_'+str(mcn)+'.pdf')

    def plot_list_bitrate_lightpaths(self,type='console',precision=10,mcn=0):
        fig=plt.figure()
        fig.suptitle('Lightpaths Capacity Distribution [Gbps]')
        plt.hist(self.list_bitrate_lightpaths, bins=precision)
        if type == 'console':
            plt.show()
        elif type == 'pdf':
            plt.savefig('plot_list_bitrate_lightpaths_'+str(mcn)+'.pdf')

    def plot_dataframe_connection_bitrate(self, type='console', mcn=0):
        self.plot3Dbars(self.dataframe_connection_bitrate.values,'Connection Bitrate',type,mcn)

    def general_statistics_Monte_Carlo(self):
        print('Avg SNR: {:.2f} dB '.format(np.mean(list(filter(lambda x: x != 0, self.list_snr)))))
        print('Total Capacity Connections : {:.2f} Tbps '.format(np.sum(self.list_bitrate_connections) * 1e-3))
        print('Total Capacity Lightpaths : {:.2f} Tbps '.format(np.sum(self.list_bitrate_lightpaths) * 1e-3))
        print('Avg Capacity : {:.2f} Gbps '.format(np.mean(self.list_bitrate_connections)))

    def mc_stats(self,type='console'):
        fig=plt.figure()
        fig.suptitle('Average Congestion')
        lines_labels=list(self.lines_state_list[0].keys())
        congestions={label: [] for label in lines_labels}
        for line_state in self.lines_state_list:
            for line_label, line in line_state.items():
                cong=line.state.count('occupied') / len(line.state)
                congestions[line_label].append(cong)
        avg_congestion={label: [] for label in lines_labels}
        for line_label, cong in congestions.items():
            avg_congestion[line_label]=np.mean(cong)
        plt.bar(range(len(avg_congestion)), list(avg_congestion.values()), align='center')
        plt.xticks(range(len(avg_congestion)), list(avg_congestion.keys()))
        if type == 'console':
            plt.show()
        elif type == 'pdf':
            plt.savefig('plot_congestion_.pdf')

        print('\n')
        print('Line to upgrade : {} '.format(max(avg_congestion, key=avg_congestion.get)))
        print('Avg Total Traffic : {:.2f} Tbps '.format(np.mean(self.traffic_list) * 1e-3))
        print('Avg Lighpath Bitrate : {:.2f} Gbps '.format(np.mean(self.avg_rbl_list)))
        print('Avg Lighpath SNR: {:.2f} dB '.format(np.mean(self.avg_snr_list)))


    def run_simulations(self):
        node_pairs_realizations = []
        stream_conn_list = []
        lines_state_list = []
        for mc in range(self._number_simulations):
            print('Monte - Carlo Realization #{:d}'.format(mc + 1))
            network = Network('nodes.json', self.channel)  # creates nodes and line objects
            network.connect()

            node_labels=list(network.nodes.keys())
            T=self.create_traffic_matrix(node_labels, self.request_rate, self.multiplier)
            t=T.values
            node_pairs=list(filter(lambda x: x[0] != x[1], list(it.product(node_labels, node_labels))))
            unsorted_node_pairs=node_pairs
            random.shuffle(node_pairs)
            node_pairs_realizations.append(copy.deepcopy(node_pairs))
            connections=[]


            for node_pair in node_pairs:
                connection=Connection(node_pair[0], node_pair[-1], float(T.loc[node_pair[0], node_pair[-1]]))
                connections.append(connection)  # list of connection objects
            streamed_connections=network.stream(connections, best='snr', transceiver='shannon')
            self.stream_conn_list.append(streamed_connections)
            self.lines_state_list.append(network.lines)
            #populate list_snrs
            [self.list_snr.extend(connection.snr) for connection in streamed_connections]

            #populate list_bitrate_lightpath
            for connection in streamed_connections:
                for lightpath in connection.lightpaths:
                    self.list_bitrate_lightpaths.append(lightpath.bitrate)
            # populate list_bitrate_connections
            self.list_bitrate_connections = [connection.calculate_capacity() for connection in streamed_connections]
             #populate dataframe_connection_bitrate
            s=pd.Series(data=[0.0]*len(node_labels), index=node_labels)
            self.dataframe_connection_bitrate=pd.DataFrame(0.0, index=s.index, columns=s.index, dtype=s.dtype)
            for connection in streamed_connections:
                self.dataframe_connection_bitrate.loc[connection.start_node, connection.end_node]=connection.bitrate
            #channel occupancy bitrate
            columns=[]
            index=[]
            for line in network.lines:
                index.append(line)
            channels=[]
            for i in range(self.channel):
                columns.append(str(i))
                channels.append([])
            df=pd.DataFrame(index=index, columns=columns)
            for connection in streamed_connections:
                for lightpath in connection.lightpaths:
                    tmp_channel=lightpath.channel
                    path=lightpath.perma_path
                    for i in range(len(path) - 1):
                        df[str(tmp_channel)][path[i] + path[i + 1]]=lightpath.bitrate
            df.to_html('channel_occupancy_bitrate'+str(mc)+'.html')

            self.plot_list_snr('pdf',10,mc)
            self.plot_list_bitrate_lightpaths('pdf', 10, mc)
            self.plot_list_bitrate_connections('pdf', 10, mc)
            self.plot_dataframe_connection_bitrate('pdf', mc)


        # congestion
        snr_conns=[]
        rbl_conns=[]
        rbc_conns=[]

        for streamed_conn in self._stream_conn_list:
            snrs=[]
            rbl=[]
            [snrs.extend(connection.snr) for connection in streamed_conn]
            for connection in streamed_conn:
                for lightpath in connection.lightpaths:
                    rbl.append(lightpath.bitrate)

            rbc=[connection.calculate_capacity() for connection in streamed_conn]

            snr_conns.append(snrs)
            rbl_conns.append(rbl)
            rbc_conns.append(rbc)


            [self.traffic_list.append(np.sum(rbl_list)) for rbl_list in rbl_conns]
            [self.avg_rbl_list.append(np.mean(rbl_list)) for rbl_list in rbl_conns]
            [self.avg_snr_list.append(np.mean(list(filter(lambda x: x != 0, snr_list)))) for snr_list in snr_conns]
        self.mc_stats('pdf')

