import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.signal import filtfilt, butter, hilbert
from scipy.signal.windows import parzen
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass


def fourie_transform(x, dt):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, dt)[1:]

    X = 2.0 * np.abs(X[1:])

    return X, freqs


def butter_bandpass(lowcut, highcut, fs, order=5):
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