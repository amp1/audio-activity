from __future__ import print_function
import activity_detection as ad
from matplotlib.pyplot import *
from scipy import signal
import sigproc as sigutil
import numpy as np
from aubio import source, pitch, freqtomidi

try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    from scipy.io import wavfile

sounds = []
for path in ad.filelist(ad.audio_path):
    sound = ad.read_file(path)
    if sound != None:
        sounds.append(sound)
