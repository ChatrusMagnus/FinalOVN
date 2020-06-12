class Lightpath(object):
    """
    A class used to virtualized lightpaths

    Attributes
    ----------
    signal_power : float64 W o dBm
        a float that represents the signal power on that lightpath

    path : str
        a string that represents the path of the lightpath
        this path A->B->C->D is formatted in this way "ABCD" where every letter is a node

    latency : float64 s
        represent the time in which the signal goes throw the path defined

    noise_power : float64 dB o dBm
        the noise power of the path with this lightpath

    channel : int
        the number of channels of the lightpath

    rs : int
        symbol rate

    df : float64 or int
        Wavelength Division Multiplexing channel spacing

    Methods
    -------
    add_noise(self, noise):
        adds noise_power to the current noise_power

    def add_latency(self, latency):
        adds latency to the current latency

    def next(self):
        used for recursive functions based on path
    """
    def __init__(self, path, channel=10, rs=32e9, df=50e9):
        """
        Parameters
        ----------
            path : str
                the path of the lightpath

            channel : int
                the number of channels of the lightpath formatted to 10

            rs : int
                symbol rate formatted to 32e9

            df : float64 or int
                Wavelength Division Multiplexing channel spacing formatted to 50e9
        """
        self._signal_power = None
        self._path = path
        self._latency = 0
        self._noise_power = 0
        self._channel = channel
        self._rs = rs
        self._df = df

    @property
    def channel(self):
        # was different in solution i don't know why
        return self._channel

    @property
    def rs(self):
        return self._rs

    @property
    def df(self):
        return self._df

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):
        """
        Parameters
        ----------
            noise_power : float64 dB o dBm
                the noise power of the path with this lightpath
        """
        self._noise_power += noise

    def add_latency(self, latency):
        """
        Parameters
        ----------
            latency : float64 s
                represent the time in which the signal goes throw the path defined
         """
        self._latency += latency

    def next(self):
        self._path = self._path[1:]