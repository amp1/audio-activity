from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math, os, re
from scipy import signal, arange
from sys import argv
try:
    import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

audio_path = "audio_assignment/"
segment_path = "audio_assignment/segments.txt"
result_path = "audio_assignment/"

#Relative spectral entropy, using running average for mean spectrum
#Todo: units: dB, temporal
def RSE_soundsense(signal, samplerate=8000, frame_ms=25, tail=0.8):
    signal = smooth_signal(signal, 10) #initial smoothing of signal with a 10 frame win
    y, frame_size = frames(signal, samplerate, frame_ms)
    m = np.zeros(len(y))
    p = np.zeros(len(y))
    rse = np.zeros(len(y))
    n = len(y[0])
    p[0] = np.linalg.norm(spect_power(y[0], samplerate, n))
    m[0] = p[0]
    for t in range(1,len(y)):
        p[t] = np.linalg.norm(spect_power(y[t], samplerate, n))
        m[t] = m[t-1]*tail + p[t] * (1-tail)
    for t in range(1,len(rse)):
        rse[t] = np.sum(p[t]*np.log(m[t-1]/p[t]))
    return rse,p,m

def segment_rse(rse, threshold=0.0001, samplerate=8000, frame_ms=25, admission_delay=10):
    indexes = [0]
    segments= []
    rse_max = -1
    last_voice = admission_delay * -1
    current_voiced=False
    x_scale = samplerate/(2*frame_ms)
    for i in range(len(rse)):
        if rse[i] > rse_max:
            rse_max = rse[i]
        if rse[i] < threshold and not current_voiced:
            if i-last_voice > admission_delay:
                indexes.append(i*x_scale)
            current_voiced = True
            last_voice = i
        elif rse[i] > threshold and current_voiced:
            if i-last_voice > admission_delay:
                segments.append([indexes[-1], i*x_scale])
                current_voiced = False
                indexes.append(i*x_scale)
                #last_voice = i
    return indexes, segments

def segment_rse_adaptive(rse, threshold=0.0001, samplerate=8000, frame_ms=25, admission_delay=10):
    indexes = [0]
    segments= []
    rse_max = -1
    last_voice = admission_delay * -1
    current_voiced=False
    x_scale = samplerate/(2*frame_ms)
    for i in range(len(rse)):
        if rse[i] > rse_max:
            rse_max = rse[i]
        if rse[i] < rse_max+threshold and not current_voiced:
            if i-last_voice > admission_delay:
                indexes.append(i*x_scale)
            current_voiced = True
            last_voice = i
        elif rse[i] > threshold and current_voiced:
            if i-last_voice > admission_delay:
                segments.append([indexes[-1], i*x_scale])
                current_voiced = False
                indexes.append(i*x_scale)
                #last_voice = i
    return indexes, segments

def f0_acf(frames, samplerate=8000):
    acpeaks = [signal.argrelmax((np.correlate(f,f,mode='full')[len(f)-1:])) for f in frames]
    f0s = [samplerate/float(x[0][0]) for x in acpeaks if len(x[0])>0]
    return f0s

def f0_yin(frames, samplerate=8000):
    pass

def f0_test():
    pass

#Relative spectral entropy, using mean of 500 preceding frames 
#wraps around signal endpoints
def RSE_basu(signal, samplerate=8000, frame_ms=25):
    signal = smooth_signal(signal, 10) #initial smoothing of signal with a 10 frame win
    y, frame_size = frames(signal, samplerate, frame_ms)
    m = np.zeros(len(y))
    for t in range(len(y)):
        m[t] = np.average(fft[t-500:t])
    for t in range(len(smooth)):
        H[t] = np.sum(fft[t]*np.log(fft[t]/m[t]))
    return H

def spect_power(frame, rate, size): #size=len(frame)
    k = arange(size)
    T = float(size)/rate
    frq = k/T
    frq = frq[range(size/2)]

    Y = np.fft.fft(frame)/size
    Y = Y[range(size/2)]
    return abs(Y)

#Long term spectral divergence
def LTSD(signal, samplerate=8000):
    w_len = 10
    signal, frame_size = frames(signal, samplerate, 10)

def energy_thresholds(x, samplerate=8000, noise_dist=0.9, a_scale=0.5, min_scale=0.25, W_ms=2000, smoothed_min = False):
    """ Generate log energies, smoothed log energies, their thresholds, local averages and minimums"""
    #x = highpass(x)
    window_len = 10
    x, frame_size = frames(x, samplerate, 10)
    x = log_energy(x, frame_size)
    smooth = smooth_signal(x, 10) #initial smoothing of signal with ~10ms window
    W_len = int(math.floor(W_ms*(1000.0/samplerate)))
    a = average(smooth, W_len)
    lmin, min_smooth = local_min_array(smooth, W_len)
    if smoothed_min == True:
        lmin = min_smooth
    min_a=np.average(lmin)
    t = np.zeros_like(smooth)
    for i in range(len(t)):
        #t[i] = min_smooth[i]+max(noise_dist, a_scale*(a[i]-min_smooth[i]))
        t[i] = min_a-(min_scale*min_a)+(min_scale*lmin[i])+max(noise_dist, a_scale*(a[i]-lmin[i]))
    return x, smooth, a, t, lmin

def frames(sound, samplerate=8000, frame_ms=10):
    """ Frames as 2-D numpy.array, no window function """
    period = samplerate/1000
    frame_size = int(period*frame_ms)
    padding = frame_size-(len(sound)%frame_size)
    zerolevel = np.average(sound)
    #margin = np.zeros(100*frame_size).clip(min=cliplevel)
    sound = np.append(sound, np.repeat(zerolevel, padding))
    #sound = np.append(margin, sound)
    #sound = np.append(sound, margin)
    return np.reshape(sound, (-1, frame_size)), frame_size

def log_energy(signal, frame_size=80):
    return np.sum(np.log10((signal**2).clip(1e-30)), 1)/frame_size

def smooth_signal(x, window_len=10):
    """ Smooth (average) signal with a hanning window """
    w = np.hanning(window_len)
    #padd_y = np.abs(np.amin(x))
    #smooth = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    #smooth = np.convolve(w/w.sum(),smooth+padd_y,mode='same')
    smooth = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    smooth = np.convolve(w/w.sum(),smooth,mode='valid')
    #return smooth - padd_y
    return smooth

def average(x, W_len):
    """ Get moving average of signal """
    #frame_ms = 10 #how many ms is one frame 
    w = np.ones(W_len,'d')
    padd_y = np.abs(np.amin(x))
    a = np.convolve(w/w.sum(),x+padd_y,mode='same')
    return a-padd_y

def local_min_array(x, W_len):
    """ local minimums collected into a numpy array, plus a smoothed version """
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

def write_segments(segments, samplerate=8000, path=None):
    if path == None:
        path = result_path+"generated_segments.txt"
    with open(path, "w") as f:
        for i in range(len(segments)):
            print("\t".join((str(segments[i][0]/float(samplerate)), str(segments[i][1]/float(samplerate)), str(i))))
            print("\t".join((str(segments[i][0]/float(samplerate)), str(segments[i][1]/float(samplerate)), str(i))), file=f)
        #f.close()

def write_segments_logen(segments, path=None):
    if path == None:
        path = result_path+"generated_segments.txt"
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

def filelist(path):
    audio_files = []
    for fn in os.listdir(path):
        if re.search(".(wav|ogg|flac|mp3)$", fn):
            audio_files.append(path+fn)
    return audio_files

def read_file(path):
    if al != None:
        soundfile = al.Sndfile(path, 'r')
        return soundfile.read_frames(soundfile.nframes)
    else:
        try:
            print("Warning: no audiolab. Trying to read WAV: "+path)
            wav = wavfile.read(path)[1]
            wav = np.float64(wav)/np.iinfo(np.int16).max
            return wav
        except ValueError:
            return None

def run_batch(sounds, min_scale = 0.3, min_distance = 1, a_scale = 0.5, W_ms = 2000, min_len = 30):
    fig = plt.figure()
    ax = fig.add_subplot(321)
    name0 = filelist(audio_path)[0]
    name1 = filelist(audio_path)[1]

    ax.axis([0, 4870, -8, -2])
    ax.set_title(name0+", local_min*0.3")
    x, smooth, a, t, lmin = energy_thresholds(sounds[0], noise_dist=min_distance, a_scale=a_scale, min_scale=min_scale, W_ms=W_ms)
    plot(x, smooth, a, t, lmin, min_len=min_len, ax=ax)
    write_segments(get_voice_segments(smooth, t, get_segment_indexes(smooth, t, min_len=10)),
        name0+".txt")
    ax = fig.add_subplot(322)

    ax.axis([0, 4870, -8, -2])
    ax.set_title(name1+", local_min*0.3")
    x, smooth, a, t, lmin = energy_thresholds(sounds[1], noise_dist=min_distance, a_scale=a_scale, min_scale=min_scale, W_ms=W_ms)
    plot(x, smooth, a, t, lmin, min_len=min_len, ax=ax)
    write_segments(
        get_voice_segments(smooth, t, get_segment_indexes(smooth, t, min_len=10)),
        name1+".txt")
    ax = fig.add_subplot(323)

    ax.axis([0, 4870, -8, -2])
    ax.set_title(filelist(audio_path)[0]+", local_min*0")
    x, smooth, a, t, lmin = energy_thresholds(sounds[0], noise_dist=min_distance, a_scale=a_scale, min_scale=0, W_ms=W_ms)
    plot(x, smooth, a, t, lmin, min_len=min_len, ax=ax)
    ax = fig.add_subplot(324)

    ax.axis([0, 4870, -8, -2])
    ax.set_title(filelist(audio_path)[1]+", local_min*0")
    x, smooth, a, t, lmin = energy_thresholds(sounds[1], noise_dist=min_distance, a_scale=a_scale, min_scale=0, W_ms=W_ms)
    plot(x, smooth, a, t, lmin, min_len=min_len, ax=ax)
    ax = fig.add_subplot(325)

    ax.axis([3500, 4500, -8, -2])
    ax.set_title(filelist(audio_path)[0]+", zoomed in, local_min*0.3")
    x, smooth, a, t, lmin = energy_thresholds(sounds[0], noise_dist=min_distance, a_scale=a_scale, min_scale=min_scale, W_ms=W_ms)
    plot(x, smooth, a, t, lmin, min_len=min_len, ax=ax)
    ax = fig.add_subplot(326)

    ax.axis([3500, 4500, -8, -2])
    ax.set_title(filelist(audio_path)[1]+", zoomed in, local_min*0.3")
    x, smooth, a, t, lmin = energy_thresholds(sounds[1], noise_dist=min_distance, a_scale=a_scale, min_scale=min_scale, W_ms=W_ms)
    plot(x, smooth, a, t, lmin, min_len=min_len, ax=ax)
    plt.show()

if __name__ == "__main__":
    if len(argv) == 2:
        audio_path = argv[1]
    sounds = []
    for f in filelist(audio_path):
        sound = read_file(f)
        if sound != None:
            sounds.append(sound)
    run_batch(sounds)
