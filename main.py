import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import h5py
import lib
from pprint import pprint

rhythms_freqs_range = { # Гц
    "theta": [4, 12],
    "slow_gamma": [25, 45],
    "middle_gamma": [45, 80],
    "fast_gamma": [80, 150],
}

def select_files(sourses_path):
    SelectedFiles = []

    for path, _, files in os.walk(sourses_path):

        for file in files:
            if file.find(".hdf5") == -1:
                continue
            if file == 'ec013.459.hdf5':
                continue
            if file == 'ec013.248.hdf5':
                continue

            sourse_hdf5 = h5py.File(path + '/' + file, "r")
            for ele_key, electrode in sourse_hdf5.items():
                try:
                    if electrode.attrs['brainZone'] != 'CA1':
                        continue
                    #if 'xOfFirstLed' not in sourse_hdf5["animalPosition/"] and 'yOfFirstLed' not in sourse_hdf5[
                    #    "animalPosition/"]:
                    #    continue
                    if sourse_hdf5.attrs["behavior_test"] != "bigSquare":
                        continue
                except KeyError:
                    continue

                for cluster in electrode['spikes'].values():
                    try:
                        if cluster.attrs['type'] != 'Pyr' or cluster.attrs['quality'] != 'Nice':
                            continue
                    except KeyError:
                        continue

                    SelectedFiles.append(path + '/' + file)

            sourse_hdf5.close()
    return SelectedFiles



def get_rhytms_and_maps(filepath):
    hdffile = h5py.File(filepath, 'r')
    fs_coords = hdffile['animalPosition'].attrs['coordinatesSampleRate']
    x = hdffile['animalPosition/xOfFirstLed'][:]
    y = hdffile['animalPosition/yOfFirstLed'][:]

    x, cut_idxes = lib.prepare_coordinates(x)
    y, _ = lib.prepare_coordinates(y)



    map_object = lib.Map()

    for electrode_name, electrode_values in hdffile.items():
        if electrode_name.find('electrode') == -1:
            continue

        if electrode_values.attrs['brainZone'] != 'CA1' :
            continue

        if len(electrode_values['lfp'].keys()) < 2:
            continue

        fs_signal = electrode_values['lfp'].attrs['lfpSamplingRate']
        fs_for_map = int(fs_signal//fs_coords)

        gamma_max_channel_names = lib.select_channel_for_gamma(electrode_values['lfp'], rhythms_freqs_range, fs_signal)


        fig, axes = plt.subplots(ncols=len(rhythms_freqs_range), figsize=(16, 4))

        for rhythm_idx, (rhythm_name, rhythm_range) in enumerate(rhythms_freqs_range.items()):
            lfp = electrode_values['lfp'].get(  gamma_max_channel_names[rhythm_name]['channel_name'] )
            lfp = np.asarray(lfp).astype(np.float32)

            range_lfp = lib.butter_bandpass_filter(lfp, rhythm_range[0], rhythm_range[1], fs_signal, 3)
            range_lfp = lib.hilbert(range_lfp)

            ampls_lfp = np.abs(range_lfp)
            ampls_lfp = np.reshape(ampls_lfp, (int(lfp.size//fs_for_map), int(fs_for_map)))
            ampls_lfp = np.mean(ampls_lfp, axis=1)

            ampls_lfp = ampls_lfp[cut_idxes[0] : cut_idxes[1]]

            map, xbins, ybins = map_object.get_map(x, y, ampls_lfp)


            axes[rhythm_idx].pcolor(xbins, ybins, map, cmap='rainbow')
            axes[rhythm_idx].set_title(rhythm_name)


        plt.show()

    hdffile.close()

def main():
    sourses_path = '/home/ivan/Data/Data_from_CRCNS/'
    # SelectedFiles = select_files(sourses_path)
    SelectedFiles = [sourses_path + 'ec013.439.hdf5', ]

    for filepath in SelectedFiles:
        get_rhytms_and_maps(filepath)
        print(filepath, " is processed")






if __name__ == '__main__':
    main()


