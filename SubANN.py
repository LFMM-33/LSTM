# Contains all the subfunctions / small function that help the other modules.

# Imports
import matplotlib.pyplot as plt
import numpy as np
import random

def print_hist_parm(history, parm):
    plt.figure(parm)
    plt.plot(history.history[parm])
    plt.plot(history.history['val_'+parm])
    plt.title('model '+parm)
    plt.ylabel(parm)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')


def center_signals(inputs):
    for inx in range(inputs.shape[0]):
        signal = inputs[inx]
        s_unmasked = unmask_row(signal)
        s_mean = np.mean(s_unmasked[:5])
        meaned_s = s_unmasked - s_mean
        length = len(meaned_s)
        meaned_s = meaned_s.reshape(1,length)
        inputs[inx,0:length] = meaned_s
    return inputs


def amplify_signals(inputs, amp_v):
    for inx in range(inputs.shape[0]):
        signal = inputs[inx]
        s_unmasked = unmask_row(signal)
        amp_s = s_unmasked*amp_v
        length = len(amp_s)
        amp_s = amp_s.reshape(1,length)
        inputs[inx,0:length] = amp_s
    return inputs