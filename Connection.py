class Connection(object):
    def __init__(self, start_node, end_node):
        self._start_node = start_node
        self._end_node = end_node
        self._signal_power = None
        self._latency = 0
        self._snr = 0
        self._bitrate = None

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
        self._snr = snr

    def calculate_capacity(self):
        return self.bitrate

