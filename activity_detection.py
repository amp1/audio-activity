import scikits.audiolab as al
import numpy as np
import matplotlib.pyplot as plt
import math, os, re
from scipy import signal

audio_path = "audio_assignment/"
segment_file = "audio_assignment/segments.txt"

def aad():
    files = filelist(audio_path)
    sounds = []
    thresholds = []
    for path in files:
        sounds.append(read_file(path))
    for sound in sounds:
        thresholds.append(threshods(sound))

def thresholds(x, samplerate=8000, db_ceil=0.25, db_scale=0.5):
    window_len = 10
    frame_ms = 10
    x, frame_size = frames(x, samplerate, frame_ms)
    x = log_energy(x, frame_size)
    smooth = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.hamming(window_len)
    smooth = np.convolve(w/w.sum(),smooth,mode='valid')
    W_ms = 1.0e4 # 1*10^4
    W_len = int(math.floor(W_ms/frame_ms*(1000.0/samplerate)))
    t = np.zeros_like(smooth)
    m = signal.argrelmin(smooth, order=W_len)[0]
    y = np.ones
    w = np.ones(W_len*2)
    #w = np.hamming(W_len*2)
    a = np.convolve(w/w.sum(),smooth,mode='same')
    j=0
    for i in range(len(smooth)):
        if i > m[j] and j < len(m)-1:
            j += 1
        if j == 0 or j == len(m)-1:
            lmin = smooth[m[j]]
        elif abs(m[j]-i) < abs(m[j-1]-i):
            lmin = smooth[m[j]]
        else:
            lmin = smooth[m[j-1]]
        #t[i] = lmin+max(db_ceil, 0.7*(a[i]-lmin))
        t[i] = lmin+max(db_ceil,db_scale*(a[i]-lmin))#+0.5*(a[i]-lmin)
    return x, smooth, a, t

def classify(x, t):
    x = np.vstack((x,t))
    return np.apply_along_axis(lambda y: int(y[0] >= y[1]), 0, x)

def plot(x, smooth, a, t):
    #plt.plot(x)
    #plt.plot(a, color='lightblue')
    plt.plot(smooth, color='blue')
    with open(segment_file, 'r') as f:
        lines = []
        for l in f.readlines():
            l = l.split('\t')
            st, en, i = float(l[0])*100, float(l[1])*100, l[2]
            lines.append(st)
            lines.append(en)
        print(lines)
        plt.vlines(lines, -5, -11, color='grey')
    plt.plot(t, color='red')
    plt.show()

def log_energy(signal, frame_size=80):
    return np.sum(np.log10(signal**2), 1)/frame_size

def frames(sound, samplerate=8000, frame_ms=10):
    """ Frames as 2-D numpy.array, no windowing """
    samples = 1000.0/samplerate
    frame_size = frame_ms/samples
    padding = frame_size-(len(sound)%frame_size)
    sound = np.append(sound, np.zeros(padding)).clip(min=1e-06)
    return np.reshape(sound, (-1, frame_size)), frame_size

def filelist(path):
    audio_files = []
    for fn in os.listdir(path):
        if re.search("(wav|ogg|flac|mp3)$", fn):
            audio_files.append(path+fn)
    return audio_files

def read_file(path):
    soundfile = al.Sndfile(path, 'r')
    return soundfile.read_frames(soundfile.nframes)
