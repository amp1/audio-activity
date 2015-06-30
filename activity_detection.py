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
    import wavfile from scipy.io

audio_path = "audio_assignment/"
segment_path = "audio_assignment/segments.txt"

def aad(path):
    files = filelist(path)
    sounds = []
    t = []
    for f in files:
        sounds.append(read_file(f))
    for sound in sounds:
        t.append(thresholds(soundpath))
    return t

def thresholds(x, samplerate=8000, noise_diff=0.3, db_scale=0.6, min_scale=1):
    window_len = 10
    frame_ms = 10
    x, frame_size = frames(x, samplerate, frame_ms)
    x = log_energy(x, frame_size)
    smooth = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.hamming(window_len)
    smooth = np.convolve(w/w.sum(),smooth,mode='valid')
    W_ms = 5.0e3 # 1*10^4
    W_len = int(math.floor(W_ms/frame_ms*(1000.0/samplerate)))
    t = np.zeros_like(smooth)
    m = signal.argrelmin(smooth, order=W_len)[0]
    w = np.ones(W_len*4)
    a = np.convolve(w/w.sum(),smooth,mode='same')
    j=0
    lmin = np.zeros_like(smooth)
    for i in range(len(smooth)):
        if i > m[j] and j<len(m)-1:
            j += 1
        if j == 0 or j == (len(m)-1):
            min_pos = m[j]
        elif abs(m[j]-i) <= abs(m[j-1]-i):
            min_pos = m[j]
        else:
            min_pos = m[j-1]
        lmin[i] = smooth[min_pos]
    window_len = W_len*3
    w = np.hamming(window_len)
    lm2 = np.r_[lmin[window_len-1:0:-1], lmin, lmin[-1:-window_len:-1]]
    min_smooth = np.convolve(w/w.sum(),lm2,mode='valid')
    lmin = min_smooth
    for i in range(len(t)):
        #t[i] = (np.average(lmin)+min_scale*lmin[i])+max(noise_diff, db_scale*(a[i]-lmin[i]))
        #t[i] = min_smooth[i]+max(noise_diff, db_scale*(a[i]-min_smooth[i]))
        t[i] = min_scale*(np.average(min_smooth)+min_smooth[i])+max(noise_diff, db_scale*(a[i]-min_smooth[i]))
    return x, smooth, a, t, lmin

def get_segments(x, t, min_len=40):
    indexes = []
    segment = []
    for i in range(1, len(x)):
        if x[i] >= t[i] and x[i-1] < t[i-1]:
            segment.append(i)
            if(len(segment)==2):
                if segment[1]-segment[0] > min_len :
                    indexes.append(segment[0])
                    indexes.append(segment[1])
                segment=[]
        elif x[i] < t[i] and x[i-1] >= t[i-1]:
            segment.append(i)
            if(len(segment)==2):
                if segment[1]-segment[0] > min_len :
                    indexes.append(segment[0])
                    indexes.append(segment[1])
                segment=[]
    return indexes

def plot(x, smooth, a, t, lmin, x0=0, x1=4900, y0=-7, y1=-12):
    my_segments = get_segments(smooth, t)
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
    plt.plot(a, color='yellow')
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
    if al != None:
        soundfile = al.Sndfile(path, 'r')
        return soundfile.read_frames(soundfile.nframes)
    else:
        wav = scipy.io.wavfile.read(path)
        wav = np.float64(wav)/np.iinfo(np.int16).max
        return wav

def multiplot(t):
    plt.figure()
    plt.add_subplot(321)
    plot(t[0])

if __name__ == "__main__":
    if len(argv) == 2:
        path = argv[2]
    else:
        path = audio_path
    for f in path
        sounds.append(ad.read_file(f))
