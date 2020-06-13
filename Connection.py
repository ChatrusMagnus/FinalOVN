class Connection(object):
    def __init__(self, start_node, end_node):
        self._start_node = start_node
        self._end_node = end_node
        self._signal_power = None
        self._latency = 0
        self._snr = 0

    @property
    def start_node(self):
        return self._start_node

    @property
    def end_node(self):
        return self._end_node

    @property
    def signal_power(self):
        return self._signal_power

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

