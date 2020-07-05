import numpy as np
class Node(object):
    """
        wle<
    """
    def __init__(self, node_dict):
        """
        :param node_dict: ciao
        """
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, lightpath, occupation=False):
        """
        :param lightpath: A lightpath object
        :param occupation: Boolean
        :return:
        """
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath.signal_power = lightpath.optimized_powers[line_label]
            lightpath = line.propagate(lightpath, occupation)
        return lightpath

    def optimize(self,lightpath):
        path = lightpath.path
        if len(path) > 1:
            #first 2 elements
            line_label = path[:2]
            line = self.successive[line_label]

            ase = line.ase_generation()
            eta = line.nli_generation(1,lightpath.rs,lightpath.df)
            po= (ase/(2*eta))**(1/3)
            lightpath.optimized_powers.update({line_label: po})

            lightpath.next()
            node= line.successive[lightpath.path[0]]
            lightpath = node.optimize(lightpath)
        return lightpath