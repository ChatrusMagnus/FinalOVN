import numpy as np
class Connection(object):
    def __init__(self, start_node, end_node,rate_request=0):
        self._start_node = start_node
        self._end_node = end_node
        self._signal_power = None
        self._latency = 0
        self._snr = []
        self._bitrate = None
        self._rate_request = float(rate_request)
        self._residual_rate_request = float(rate_request)
        self._lightpaths = []

    @property
    def lightpaths(self):
        return self._lightpaths

    @lightpaths.setter
    def lightpaths(self, lightpath):
        self._lightpaths.append(lightpath)

    def clear_lightpaths(self):
        self._lightpaths = []

    def calculate_capacity(self):
        self.bitrate = sum([lightpath.bitrate for lightpath in self.lightpaths])
        return self.bitrate

    @property
    def rate_request(self):
        return self._rate_request

    @property
    def residual_rate_request(self):
        return self._residual_rate_request

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    @property
    def start_node(self):
        return self._start_node

    @property
    def end_node(self):
        return self._end_node

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self,signal_power):
        self._signal_power = signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr.append(snr)

    def set_connection(self, lightpath):
        self.signal_power = lightpath.signal_power
        self.latency = max(self.latency, lightpath.latency)
        self.snr = 10*np.log10(lightpath.snr)
        self.lightpaths = lightpath
        self._residual_rate_request = self._residual_rate_request - lightpath.bitrate
        return self

    def block_connection(self):
        self.latency = None
        self.snr = 0
        self.bitrate = 0
        self.clear_lightpaths()
        return self