from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math, os, re
from scipy import signal
from sys import argv
try:
    import scikits.audiolab as al
except ImportError:
    al = None
    from scipy.io import wavfile

audio_path = "audio_assignment/"
segment_path = "audio_assignment/segments.txt"
result_path = "audio_assignment/generated_segments.txt"

def thresholds(x, samplerate=8000, noise_dist=0.3, a_scale=0.6, min_scale=0.1, W_ms=2000, smoothed_min = False):
    """ """
    window_len = 10 # size of window used for initial smoothing, n of frames
    #x = highpass(x)
    # split signal to 2-D array with a frame on each row
    x, frame_size = frames(x, samplerate, 10)
    x = log_energy(x, frame_size)
    smooth = smooth_signal(x, 10)
    W_len = int(math.floor(W_ms*(1000.0/samplerate)))
    a = average(smooth, W_len)
    lmin, min_smooth = local_min_array(smooth, W_len)
    if smoothed_min == True:
        lmin = min_smooth
    min_a=np.average(lmin)
    t = np.zeros_like(smooth)
    print(len(t), len(x), len(smooth))
    for i in range(len(t)):
        #t[i] = min_smooth[i]+max(noise_dist, a_scale*(a[i]-min_smooth[i]))
        t[i] = min_a-(min_scale*min_a)+(min_scale*lmin[i])+max(noise_dist, a_scale*(a[i]-lmin[i]))
    return x, smooth, a, t, lmin

def smooth_signal(x, window_len):
    """ Smooth (average) signal with a hamming window """
    w = np.hamming(window_len)
    padd_y = np.amin(x)
    smooth = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    smooth = np.convolve(w/w.sum(),smooth+padd_y,mode='same')
    return smooth - padd_y

def average(x, W_len):
    """ Get moving average of signal """
    #frame_ms = 10 #how many ms is one frame 
    w = np.ones(W_len,'d')
    padd_y = np.abs(np.amin(x))
    a = np.convolve(w/w.sum(),x+padd_y,mode='same')
    a = a-padd_y
    return a

def local_min_array(x, W_len):
    m = signal.argrelmin(x, order=W_len/2)[0]
    j=0
    lmin = np.zeros_like(x)
    for i in range(len(x)):
        if i > m[j] and j<len(m)-1:
            j += 1
        if j == 0 or j == (len(m)-1):
            min_pos = m[j]
        elif abs(m[j]-i) < abs(m[j-1]-i):
            min_pos = m[j]
        else:
            min_pos = m[j-1]
        lmin[i] = x[min_pos]
    window_len = W_len*2
    w = np.hamming(window_len)
    lm2 = np.r_[lmin[window_len-1:0:-1], lmin, lmin[-1:-window_len:-1]]
    min_smooth = np.convolve(w/w.sum(),lm2,mode='valid')
    return lmin, min_smooth

def highpass(x, N=4, samplerate=8000, cutoff=100):
    Wn = cutoff/(samplerate/2)
    b, a = signal.butter(N, Wn, 'high')
    return signal.filtfilt(b, a, x)

def get_segment_indexes(x, t, min_len=30):
    indexes = [0]
    segment = [0]
    #segment = []
    for i in range(1, len(x)-1):
        if x[i] >= t[i] and x[i-1] < t[i-1]:
            segment.append(i)
            segment, indexes = validate_segment(x, t, segment, indexes, min_len)
        elif x[i] < t[i] and x[i-1] >= t[i-1]:
            segment.append(i)
            segment, indexes = validate_segment(x, t, segment, indexes, min_len)
    if len(segment) == 1:
        segment.append(len(x)-1)
    elif len(segment) == 0:
        indexes.append(len(x)-1)
    if(len(segment)==2):
        segment, indexes = validate_segment(x, t, segment, indexes, min_len)
    if len(segment) >0:
        for i in segment:
            if i > indexes[-1]:
                indexes.append(i)
    return indexes

def validate_segment(x, t, segment, indexes, min_len=30, validate=True):
    if validate:
        if(len(segment)==2):
            if segment[1]-segment[0] >= min_len :
                if segment[0] != indexes[-1]:
                    indexes.append(segment[0])
                indexes.append(segment[1])
                segment=[]
            else:
                segment=[indexes[-1]]
    else:
        for i in segment:
            indexes.append(i)
        return [], indexes
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

def plot(x, smooth, a, t, lmin, min_len=30, y0=-2, y1=-8, ax=None):
    if ax == None:
        p = plt
    else:
        p = ax
    my_segments = get_segment_indexes(smooth, t, min_len)
    p.vlines(my_segments, y0, y1, color='orange')
    segments = []
    with open(segment_path) as sf:
        for l in sf.readlines():
            l = l.split('\t')
            segments.append(float(l[0])*100)
            segments.append(float(l[1])*100)
        p.vlines(segments, y0, y1, color='black', linestyles='dashed')
    p.plot(smooth, color='blue')
    p.plot(t, color='red')
    p.plot(lmin, color='pink')
    p.plot(a, color='lightgreen')

def log_energy(signal, frame_size=80):
    return np.sum(np.log10((signal**2).clip(1e-30)), 1)/frame_size

def frames(sound, samplerate=8000, frame_ms=10):
    """ Frames as 2-D numpy.array, no windowing """
    period = 1000.0/samplerate
    frame_size = int(frame_ms/period)
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
    if al != None:
        soundfile = al.Sndfile(path, 'r')
        return soundfile.read_frames(soundfile.nframes)
    else:
        try:
            wav = scipy.io.wavfile.read(path)
            wav = np.float64(wav)/np.iinfo(np.int16).max
            return wav
        except ValueError:
            return None

def multiplot(sounds):
    min_len = 30
    min_distance = 1
    W_ms = 5e3
    a_scale = 0.6
    fig = plt.figure()
    ax = fig.add_subplot(321)
    ax.axis([0, 4870, -8, -2])
    plot(*thresholds(sounds[0], noise_dist=min_distance, a_scale=a_scale, min_scale=0.25, W_ms=W_ms), min_len=min_len, ax=ax)
    ax = fig.add_subplot(322)
    ax.axis([0, 4870, -8, -2])
    plot(*thresholds(sounds[1], noise_dist=min_distance, a_scale=a_scale, min_scale=0.25, W_ms=W_ms), min_len=min_len, ax=ax)
    ax = fig.add_subplot(323)
    ax.axis([0, 4870, -8, -2])
    plot(*thresholds(sounds[0], noise_dist=min_distance, a_scale=a_scale, min_scale=0, W_ms=W_ms), min_len=min_len, ax=ax)
    ax = fig.add_subplot(324)
    ax.axis([0, 4870, -8, -2])
    plot(*thresholds(sounds[1], noise_dist=min_distance, a_scale=a_scale, min_scale=0, W_ms=W_ms), min_len=min_len, ax=ax)
    ax = fig.add_subplot(325)
    ax.axis([3500, 4500, -8, -2])
    plot(*thresholds(sounds[0], noise_dist=min_distance, a_scale=a_scale, min_scale=0.25, W_ms=W_ms), min_len=min_len, ax=ax)
    ax = fig.add_subplot(326)
    ax.axis([3500, 4500, -8, -2])
    plot(*thresholds(sounds[1], noise_dist=min_distance, a_scale=a_scale, min_scale=0.25, W_ms=W_ms), min_len=min_len, ax=ax)
    plt.show()

if __name__ == "__main__":
    if len(argv) == 2:
        path = argv[2]
    else:
        path = audio_path
    sounds = []
    for f in filelist(path):
        sound = ad.read_file(f)
        if sound != None:
            sounds.append(sound)
    multiplot(sounds)
