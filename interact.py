from __future__ import print_function
import activity_detection as ad
from matplotlib.pyplot import *
from scipy import signal
import numpy as np
try:
    import audiolab as al
except ImportError:
    al = None
    from scipy.io import wavfile

sounds = []
for path in ad.filelist(ad.audio_path):
    sound = ad.read_file(path)
    if sound != None:
        sounds.append(sound)
