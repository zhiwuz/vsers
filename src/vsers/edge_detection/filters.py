import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz


class Filter(object):
    def filter(self, data):
        raise NotImplementedError


class LowPassFilter(Filter):
    def __init__(self, fs=10.0):
        self.filtered_y = None
        self.order = 6
        self.fs = fs
        self.cutoff = 3.667
        self.b, self.a = self.butter_low_pass(self.cutoff, self.fs, self.order)

    def butter_low_pass(self, cutoff, fs, order=6):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def filter(self, data):
        filtered_y = lfilter(self.b, self.a, data)
        self.filtered_y = filtered_y
        return filtered_y

    def plot_frequency_response(self):
        w, h = freqz(self.b, self.a, worN=8000)
        plt.plot(0.5 * self.fs * w / np.pi, np.abs(h), 'b')
        plt.plot(self.cutoff, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(self.cutoff, color='k')


class ContinuousFilter(Filter):
    def __init__(self, gap_threshold=0.01):
        self.gap_threshold = gap_threshold

    def filter(self, coordinates):
        filteredCoordinates = []
        for i in range(coordinates.shape[0]):
            if i == 0:
                filteredCoordinates.append([coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]])
            if abs(coordinates[i, 1] - filteredCoordinates[-1][1]) < self.gap_threshold:
                filteredCoordinates.append([coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]])

        filteredCoordinates = np.array(filteredCoordinates)
        return filteredCoordinates

class RangeFilter(Filter):
    def __init__(self, minimum=None, maximum=None):
        self.minimum = minimum
        self.maximum = maximum

    def filter(self, data):
        if self.minimum is not None:
            data = data[data[:, 1] > self.minimum]
        if self.maximum is not None:
            data = data[data[:, 1] < self.maximum]
        return data
