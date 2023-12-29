import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.signal import filtfilt, butter, hilbert, convolve2d
from scipy.signal.windows import parzen
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass


def prepare_coordinates(x):
    is_negative = x < 0
    is_negative = is_negative.astype(np.int32)
    neg_diff = np.diff(is_negative)
    start_idx = int(np.argwhere(neg_diff == -1).ravel()[0])
    start_idx = start_idx + 1

    end_idx = int(np.argwhere(neg_diff == 1).ravel()[-1])
    end_idx = end_idx + 1

    x = fill_vals_by_interpol(x[start_idx:end_idx])

    return x, (start_idx, end_idx)


def fill_vals_by_interpol(x, val=-1.0):
    """
    :param x: array
    :param val: value, which need to fill
    :return: filled array
    """
    is_fill = x == val
    fill_idx = np.argwhere(is_fill).ravel()
    right_idx = np.argwhere(~is_fill).ravel()


    x[is_fill] = np.interp(fill_idx, right_idx, x[right_idx])

    return x
def get_gaussian_kernel(length=100, sigma=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    xy = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    gauss = np.exp(-0.5 * (xy/sigma)**2 )
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def fourie_transform(x, dt):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, dt)[1:]

    X = 2.0 * np.abs(X[1:])

    return X, freqs


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def __clear_articacts(lfp, win, threshold):
    lfp = lfp - np.mean(lfp)
    lfp_std = np.std(lfp)
    is_large = np.logical_and( (lfp > 10*lfp_std), (lfp < 10*lfp_std) )
    is_large = is_large.astype(np.float64)
    is_large = np.convolve(is_large, win)
    is_large = is_large[win.size // 2:-win.size // 2 + 1]
    is_large = is_large > threshold
    lfp[is_large] = np.random.normal(0, 0.001*lfp_std, np.sum(is_large) )
    return lfp


def clear_articacts(lfp, win_size=101, threshold=0.1):
    win = parzen(win_size)
    lfp = __clear_articacts(lfp, win, threshold)
    return lfp

def select_channel_for_gamma(channels, gamma_freqs, fs):

    gamma_max = {}
    for gamma in gamma_freqs.keys():
        gamma_max[gamma] = {
            'channel_name' : "",
            'mean_gamma_power' : 0,
        }


    for ch_idx, (channel_name, channel_data) in enumerate(channels.items()):


        lfp = 0.001 * channel_data[:].astype(np.float32)
        lfp = clear_articacts(lfp)

        lfp_spectrum, lfp_freqs = fourie_transform(lfp, 1/fs)

        for gamma, freqs_range in gamma_freqs.items():
            range4sum = np.logical_and((lfp_freqs > freqs_range[0]), (lfp_freqs <  freqs_range[1]))
            gamma_power = np.mean( lfp_spectrum[range4sum])

            #gamma_max[channel_name][gamma] = gamma_power

            if gamma_power > gamma_max[gamma]['mean_gamma_power']:
                gamma_max[gamma] = {
                    'channel_name': channel_name,
                    'mean_gamma_power': gamma_power,
                }


    return gamma_max


class Map:
    def __init__(self, smoothing=2.5, box_size=None, bin_size=0.5, kernel_size=50):
        self.box_size = box_size
        self.kernel_size = kernel_size
        self.bin_size = bin_size

        self.smoothing = smoothing / self.bin_size

        self.gaussian_kernel = get_gaussian_kernel(self.kernel_size, self.smoothing)

    def get_map(self, x_an, y_an, Zt, isfillnan=True):
        """
        :param x_an: position of animal by x axis in cm
        :param y_an:  position of animal by y axis in cm
        :param Zt:  parameter for map
        :param isfillnan : fill by zeros non visited places
        :return: map of Zt by coordinates
        """
        if self.box_size is None:
            xmin = np.min(x_an[x_an > 0])
            xmax = np.max(x_an)

            ymin = np.min(y_an[y_an > 0])
            ymax = np.max(y_an)

            xrange = xmax - xmin
            yrange = ymax - ymin
            #box_size = [xrange, yrange]

            nbins = int(xrange / self.bin_size)
            print("Hello")

        else:
            nbins = int(self.box_size / self.bin_size)


        map, xbins, ybins = np.histogram2d(x_an, y_an, bins=[nbins,nbins] , weights=Zt, range=[[xmin, xmax], [ymin, ymax]])
        occupacy_map, _, _ = np.histogram2d(x_an, y_an, bins=[nbins,nbins], range=[[xmin, xmax], [ymin, ymax]])

        map = convolve2d(map, self.gaussian_kernel, mode='same')
        occupacy_map = convolve2d(occupacy_map, self.gaussian_kernel, mode='same')

        map = map / occupacy_map
        if isfillnan:
            map[np.isnan(map)] = 0

        return map, xbins, ybins





