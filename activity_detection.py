import scikits.audiolab as al
import numpy as np
import os
import re

audio_path = "audio_assignment/"

def log_aad():
    pass

def log_energy(signal):
    return np.sum(np.log(signal**2))

def energy(signal):
    a.split(10)

def frames(signal, samplerate=8000, frame_ms=10):
    samples = 1000.0/samplerate
    frame_size = frame_ms/samples
    padding = frame_size-(len(a)%frame_size)
    a = np.append(a, np.zeroes(padding)+0.0001)
    return np.reshape(a, (-1, frame_size))

def filelist():
    audio_files = []
    for fn in os.listdir(audio_path):
        if re.search("(wav|ogg|flac|mp3)$", fn):
            audio_files.append(audio_path+fn)
    return audio_files

def read_file(fn):
    return al.Sndfile(fn, 'r')
