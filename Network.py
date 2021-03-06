import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfcinv

from FinalOVN.Line import Line
from FinalOVN.Node import Node
from FinalOVN.Lightpath import Lightpath

class Network(object):
    def __init__(self, json_path,Nch=10, upgrade_line ='',fiber_type='SMF'):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._Nch=Nch
        self._weighted_paths = None
        self._connected = False
        self._route_space = None
        self._is_route = False
        # node label is a number??
        for node_label in node_json:
            # copy node form json to dictionary form
            node_dict = node_json[node_label]
            # extract label
            node_dict['label'] = node_label
            # creation of node with label
            node = Node(node_dict)
            # add node instance to network dictionary
            self._nodes[node_label] = node

            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                # A B -> AB
                line_label = node_label + connected_node_label
                # mannaia print(line_label)
                # add labal required for creating instance
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                # calculate lenght required for creating instance
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))
                if fiber_type == 'SMF':
                    line=Line(line_dict, self.Nch,6,80e3,21.27e-27)
                elif fiber_type == 'LEAF':
                    line=Line(line_dict, self.Nch,6,80e3,6.58e-27)
                # adds new line to dictionary
                self._lines[line_label] = line

        # if specified , upgrade line
        if not upgrade_line == '':

            self.lines[upgrade_line].noise_figure = self.lines[upgrade_line].noise_figure - 3

    @property
    def nodes(self):
        return self._nodes

    @property
    def Nch(self):
        return self._Nch

    @property
    def lines(self):
        return self._lines

    @property
    def connected(self):
        return self._connected

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space

    # return line that connects start_node with end_node
    def single_line_free(self, start_node, end_node):
        channel=self.Nch
        label = start_node + end_node
        for linep in self._lines.keys():
            for i in range(channel):
                if (linep == label and self._lines[label].state[i] == 'free'):
                    return linep
        print('Error did not find a line whit this label or free ', label)
        # problem here !!!
        return None

    def optimization(self,lightpath):
        path= lightpath.path
        start_node=self.nodes[path[0]]
        optimized_lightpath=start_node.optimize(lightpath)
        optimized_lightpath.path=path
        return optimized_lightpath

    def available_paths(self, input_node, output_node):
        if self.weighted_paths is None:
            self.set_weighted_paths(1)
        '''
        #checks if input_node belongs to the networks
        if not input_node in self._nodes.values():
            print("Error input_node is not valid or is not in the network",input_node)
        #checks if input_node belongs to the networks
        if not output_node in self._nodes.values():
            print("Error output_node is not valid or is not in the network",output_node)
        '''
        all_paths = self.find_paths(input_node, output_node)
        free_paths = []
        for path in all_paths:
            path = path.replace("", "->")[2:-2]
            # checks if path is free
            path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]

            if 'free' in path_occupancy:
                free_paths.append(path)

        return free_paths

    ## added end of lab 3 here in lab 4
    # tried to used setter but didn't work
    def set_weighted_paths(self):
        channel = self.Nch

        # checks the network is connected
        if not self.connected:
            self.connect()
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)
        columns = ['path', 'latency', 'noise', 'snr']
        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []
        weighted_paths = []
        for pair in pairs:
            """
            major modification
            """
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                # Propagation
                lightpath = Lightpath(path,channel)
                lightpath = self.optimization(lightpath)

                lightpath = self.propagate(lightpath, occupation=False)
                # mistake in the solutions here we need to put false because otherwise only with set all the channel
                # are occupied, channel should be occupied only when fast connection or reliable connection is established
                latencies.append(lightpath.latency)
                noises.append(lightpath.noise_power)
                snrs.append(10*np.log10(lightpath.snr))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths = df
        if not self._is_route:
            route_space = pd.DataFrame()
            route_space['path'] = paths
            for i in range(channel):
                route_space[str(i)] = ['free'] * len(paths)
            self._route_space = route_space
            self._is_route = True

    # let's draw it
    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go ', markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {}
        inner_paths['0'] = label1
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node for cross_node in cross_nodes if (
                            (inner_path[-1] + cross_node in cross_lines) & (cross_node not in inner_path))]
        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
        self._connected = False

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(lightpath, occupation)
        return propagated_signal_information


    # version 2
    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:

            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        return best_path


        # verion 1
        def find_best_latency(self, input_node, output_node):
            available_paths = self.available_paths(input_node, output_node)

            if available_paths:
                inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
                best_latency = np.min(inout_df.latency.values)
                best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0].replace('->', '')
            else:
                best_path = None
            return best_path

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    def stream(self, connections, best='latency',transceiver ='shannon'):
        channel = self._Nch
        streamed_connections = []
        for connection in connections:
            start_node = connection.start_node
            end_node = connection.end_node
            self.set_weighted_paths()
            if best == 'latency':
                path = self.find_best_latency(start_node, end_node)
            elif best == 'snr':
                path = self.find_best_snr(start_node, end_node)
            else:
                print('ERROR : best input does not coindice with either latency or snr, best.')
                continue

            if path:
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                tmp_channel = 0
                new_path = path.replace('->', '')

                for i in path_occupancy:
                    if (i == 'free'):
                        break
                    tmp_channel += 1

                if (tmp_channel >= 0 and tmp_channel < channel):
                    start_lightpath = Lightpath(new_path, tmp_channel,transceiver = transceiver)
                    start_lightpath = self.optimization(start_lightpath)
                    end_lightpath = self.propagate(start_lightpath, True)
                    #
                    self.calculate_bitrate(end_lightpath)
                    #
                    if end_lightpath.bitrate == 0.0:
                        [self.update_route_space(lp.path, lp.channel, 'free') for lp in connection.lightpaths]
                        connection.block_connection()
                    else:
                        connection.set_connection(end_lightpath)
                        #before new_path there was end_lightpath
                        self.update_route_space(new_path, end_lightpath.channel, 'occupied')
                        #self.route_space.to_html('test1.html')
                        if connection.residual_rate_request > 0:
                            self.stream([connection], best, transceiver)
                            #if residual it allocates on the same channel because is not set occupied
                            #, type='iter'?????
            else:
                [self.update_route_space(lp.path, lp.channel, 'free ') for lp in connection.lightpaths]
                connection.block_connection()
            streamed_connections.append(connection)
        return streamed_connections

    def calculate_bitrate(self, lightpath, bert=1e-3,bn=12.5e9):
        snr = 10*np.log10(lightpath.snr)
        Rs=lightpath.rs

        if lightpath.transceiver.lower() == 'fixed-rate':
            snrt = 2 * erfcinv(2*bert) * (Rs/bn)
            rb=np.piecewise(snr, [snr < snrt, snr >= snrt], [0, 100])

        elif lightpath.transceiver.lower() == 'flex-rate':
            snrt1= 2*erfcinv(2*bert)**2 * (Rs/bn)
            snrt2= (14/3)* erfcinv(3/2*bert)**2 * (Rs/bn)
            snrt3= (10)* erfcinv(8/3*bert)**2 * (Rs/bn)

            cond1 = (snr < snrt1)
            cond2 = (snr >= snrt1 and snr < snrt2)
            cond3 = (snr >= snrt2 and snr <snrt3)
            cond4 = (snr >= snrt3)

            rb=np.piecewise(snr, [cond1, cond2, cond3, cond4], [0, 100, 200, 400])
        elif lightpath.transceiver.lower() == 'shannon':
            rb=2*Rs*np.log2(1+snr*(Rs/bn)) * 1e-9

        lightpath.bitrate= float(rb)
        return float(rb)

    def update_route_space(self, path, channel,state):
        all_paths = [self.path_to_line_set(p) for p in self.route_space.path.values]
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = state
        self.route_space[str(channel)] = states

    def optimization(self, lightpath):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        optimized_lightpath = start_node.optimize(lightpath)
        optimized_lightpath.path = path
        return optimized_lightpath


