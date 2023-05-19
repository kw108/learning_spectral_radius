import numpy as np
from scipy.io import loadmat
from scipy.linalg import eig
import pickle


sampling_rate = '1e5'
channel_gains = loadmat('channel_gains_' + sampling_rate + '.mat')

with open('eigentriples_' + sampling_rate + '.pkl', 'wb') as file:
    for fading_type in ['rayleigh_gain', 'rician_gain']:
        gain = abs(channel_gains[fading_type])
        w = np.zeros(gain.shape[0])
        lv = np.zeros((gain.shape[0], int(np.sqrt(gain.shape[1]))))
        rv = np.zeros((gain.shape[0], int(np.sqrt(gain.shape[1]))))

        for timespec in range(gain.shape[0]):
            m = gain[timespec].reshape((8, 8))
            w1, lv1, rv1 = eig(m, left=True)
            pf_index = np.argmax(abs(w1))
            w[timespec] = abs(w1[pf_index])
            lv[timespec] = abs(lv1[:, pf_index])
            rv[timespec] = abs(rv1[:, pf_index])

        pickle.dump(gain, file)
        pickle.dump(w, file)
        pickle.dump(lv, file)
        pickle.dump(rv, file)


with open('eigentriples_' + sampling_rate + '.pkl', 'rb') as file:
    for fading_type in ['rayleigh_gain', 'rician_gain']:
        gain = pickle.load(file)
        w = pickle.load(file)
        lv = pickle.load(file)
        rv = pickle.load(file)
