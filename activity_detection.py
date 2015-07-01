from __future__ import print_function
import scikits.audiolab as al
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math, os, re
from scipy import signal

audio_path = "audio_assignment/"
segment_path = "audio_assignment/segments.txt"
result_path = "audio_assignment/generated_segments.txt"

def aad():
    files = filelist(audio_path)
    sounds = []
    thresholds = []
    for path in files:
        sounds.append(read_file(path))
    for sound in sounds:
        thresholds.append(threshods(sound))

def highpass(x, N=4, samplerate=8000, cutoff=100):
    Wn = cutoff/(samplerate/2)
    b, a = signal.butter(N, Wn, 'high')
    return signal.filtfilt(b, a, x)

def thresholds(x, samplerate=8000, noise_dist=0.3, a_scale=0.6, min_scale=0.1, W_ms=10, smoothed_min = False):
    window_len = 10
    frame_ms = 10
    #x = highpass(x)
    x, frame_size = frames(x, samplerate, frame_ms)
    x = log_energy(x, frame_size)
    smooth = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.hamming(window_len)
    smooth = np.convolve(w/w.sum(),smooth,mode='valid')
    W_len = int(math.floor(W_ms*1e3/frame_ms*(1000.0/samplerate)))
    w = np.ones(W_len*2,'d')
    padd_y = np.abs(np.amin(smooth))
    a = np.convolve(w/w.sum(),smooth+padd_y,mode='same')
    a = a-padd_y
    t = np.zeros_like(smooth)
    m = signal.argrelmin(smooth, order=W_len)[0]
    j=0
    lmin = np.zeros_like(smooth)
    for i in range(len(smooth)):
        if i > m[j] and j<len(m)-1:
            j += 1
        if j == 0 or j == (len(m)-1):
            min_pos = m[j]
        elif abs(m[j]-i) < abs(m[j-1]-i):
            min_pos = m[j]
        else:
            min_pos = m[j-1]
        lmin[i] = smooth[min_pos]
    window_len = W_len*3
    w = np.hamming(window_len)
    lm2 = np.r_[lmin[window_len-1:0:-1], lmin, lmin[-1:-window_len:-1]]
    min_smooth = np.convolve(w/w.sum(),lm2,mode='valid')
    if smoothed_min == True:
        lmin = min_smooth
    for i in range(len(t)):
        #t[i] = (np.average(lmin)+min_scale*lmin[i])+max(noise_dist, db_scale*(a[i]-lmin[i]))
        #t[i] = min_smooth[i]+max(noise_dist, a_scale*(a[i]-min_smooth[i]))
        min_a=np.average(lmin)
        t[i] = min_a-(min_scale*min_a)+(min_scale*lmin[i])+max(noise_dist, a_scale*(a[i]-lmin[i]))
    return x, smooth, a, t, lmin

def get_segment_indexes(x, t, min_len=30):
    indexes = [0]
    segment = [0]
    #segment = []
    for i in range(1, len(x)-1):
        if x[i] >= t[i] and x[i-1] < t[i-1]:
            segment.append(i)
            segment, indexes = validate_segment(x, t, segment, indexes)
        elif x[i] < t[i] and x[i-1] >= t[i-1]:
            segment.append(i)
            segment, indexes = validate_segment(x, t, segment, indexes)
    if len(segment) == 1:
        segment.append(len(x)-1)
    elif len(segment) == 0:
        indexes.append(len(x)-1)
    if(len(segment)==2):
        segment, indexes = validate_segment(x, t, segment, indexes)
    if len(segment) >0:
        for i in segment:
            if i > indexes[-1]:
                indexes.append(i)
    return indexes

def validate_segment(x, t, segment, indexes, min_len=30):
    if(len(segment)==2):
        if segment[1]-segment[0] > min_len :
            if segment[0] != indexes[-1]:
                indexes.append(segment[0])
            indexes.append(segment[1])
            segment=[]
        else:
            segment=[indexes[-1]]
    return segment, indexes

def get_voice_segments(x, t, indexes):
    segments = []
    for i in range(1,len(indexes)):
        segment = x[indexes[i-1]:indexes[i]]
        thresh = t[indexes[i-1]:indexes[i]]
        if np.average(segment) >= np.average(thresh):
            segments.append((indexes[i-1], indexes[i]))
    return segments

def write_segments(segments, path=None):
    if path == None:
        path = result_path
    with open(path, "w") as f:
        for i in range(len(segments)):
            print("\t".join((str(segments[i][0]/100.0), str(segments[i][1]/100.0), str(i))), file=f)

def plot(x, smooth, a, t, lmin, min_len=30, y0=-2, y1=-8):
    my_segments = get_segment_indexes(smooth, t, min_len)
    plt.vlines(my_segments, y0, y1, color='orange')
    segments = []
    with open(segment_path) as sf:
        for l in sf.readlines():
            l = l.split('\t')
            segments.append(float(l[0])*100)
            segments.append(float(l[1])*100)
        plt.vlines(segments, y0, y1, color='black', linestyles='dashed')
    plt.plot(smooth, color='blue')
    plt.plot(t, color='red')
    plt.plot(lmin, color='pink')
    plt.plot(a, color='lightgreen')

def log_energy(signal, frame_size=80):
    return np.sum(np.log10((signal**2).clip(1e-30)), 1)/frame_size

def frames(sound, samplerate=8000, frame_ms=10):
    """ Frames as 2-D numpy.array, no windowing """
    samples = 1000.0/samplerate
    frame_size = frame_ms/samples
    padding = frame_size-(len(sound)%frame_size)
    zerolevel = np.average(sound)
    #margin = np.zeros(100*frame_size).clip(min=cliplevel)
    sound = np.append(sound, np.repeat(zerolevel, padding))
    #sound = np.append(margin, sound)
    #sound = np.append(sound, margin)
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
