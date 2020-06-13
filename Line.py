import numpy as np
from numpy import pi
from scipy.constants import h
class Line(object):
    def __init__(self, line_dictionary,channel = 10, noise_figure=6,distance_amp=80e3):
        """
        :param line_dictionary:
        """
        self._label = line_dictionary['label']
        self._length = line_dictionary['length'] *1e3 #conversion in m
        self._state = ['free'] * channel
        self._amplifier = int(np.ceil(self._length / distance_amp))
        self._b2 = 21.27e-27
        self._span_length = self._length / self._amplifier
        self._alpha = 4.6e-5 #linear 0.20 db/m
        self._gamma = 1.27e-3
        # Set Gain to transparency
        self._gain = self.transparency()
        self._noise_figure = noise_figure
        self._successive = {}
        # Physical parameter of the fiber
        # Km-1

        # ps^2/K
        # (W*Km)^-1
    @property
    def span_length(self):
            return self._span_length

    @property
    def alpha(self):
        """
        alpha is linear
        """
        return self._alpha

    @property
    def b2(self):
        return self._b2

    @property
    def gamma(self):
        return self._gamma

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def amplifier(self):
        return self._amplifier

    @amplifier.setter
    def amplifier(self, amplifier):
        self._amplifier = amplifier

    @property
    def noise_figure(self):
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self, noise_figure):
        self._noise_figure = noise_figure

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def state(self):
        return self._state

    # in the solution is different
    @state.setter
    def state(self, state, channel=10):
        state = state.lower().strip()
        if (channel < 0 or channel > channel - 1):
            print('Error this channel is not in the range 0-', channel);
            exit(1);
        if (state == 'free'):
            self._state[channel] = state;
        elif (state == 'occupied'):
            self._state[channel] = state;
        else:
            print('Error assigning state value', state)

    def latency_generation(self):
        return (self.length / 300000000) * (3 / 2)

    def noise_generation(self, lightpath):
        noise = self.ase_generation() + self.nli_generation(lightpath.signal_power, lightpath.Rs, lightpath.df)

        return noise

    def propagate(self, lightpath, occupation=False):
        # updates latency
        lightpath.add_latency(self.latency_generation())

        # update noise
        noise = self.noise_generation(lightpath)

        lightpath.add_noise(noise)

        # Update line state
        if occupation:
            channel = lightpath.channel

            self._state[channel] = 'occupied'

        node = self.successive[lightpath.path[0]]
        signal_information = node.propagate(lightpath, occupation)
        return signal_information

    def ase_generation(self,Bn = 12.5e9):
        # THz
        frequency = 193.4e12
        # noise Bandwith GHz
        N = self.amplifier
        NF = self.noise_figure
        G = self.gain
        ase = N * (h * frequency * Bn * 10**(NF/10) * (G - 1))
        return ase

    def eta_nli(self,Rs,df,channel=10):
        Nch = channel;
        eta = 16 / (27 * pi) * \
              np.log(pi ** 2 * self.b2 * Rs ** 2 * Nch ** (2 * Rs / df) / (2 * self.alpha)) * \
              self.gamma ** 2 / (4 * self.alpha * self.b2 * Rs ** 3) * BN
        return eta

    def nli_generation(self, signal_power, Rs, df,Bn=12.5e9,channel=10):
        # Power of the channel
        Pch = signal_power;
        Nch = channel;
        loss = np.exp(-self.alpha * self.span_length)
        Na = self.amplifier
        eta= self.eta_nli(Rs,df,channel)
         # now I can calculate the Non Linear Interferance
        nli = Na*(Pch **3 * loss * self . gain * eta * Bn)

        return nli

    def transparency(self):
        gain = (np.exp(self.alpha * self.span_length))
        return gain
