import activity_detection as ad
import scikits.audiolab as al
from matplotlib.pyplot import *
from scipy import signal
import numpy as np

sounds = []
for path in ad.filelist(ad.audio_path):
    sounds.append(ad.read_file(path))