import scikits.audiolab as al
import numpy as np
import os, re

audio_path = "audio_assignment/"

def aad():
    files = filelist(audio_path)
    sounds = []
    thresholds = []
    for path in files:
        sounds.append(read_file(path))
    for sound in sounds:
        thresholds.append(threshods(sound))
        
def thresholds(x):
    window_len = 10
    frame_ms = 10
    x, frame_size = frames(x)
    x = log_energy(x, frame_size)
    smooth = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.hamming(window_len)
    smooth = np.convolve(w/w.sum(),smooth,mode='valid')
    W_ms = 1.0e4 # 1*10^4
    W_len = W_ms/frame_ms*(1000.0/samplerate)
    t = np.zeros_like(x)
    for i in range(len(smooth)):
        e = np.r_[smooth[W_len-1:0:-1], smooth, smooth[-1:-W_len:-1]]
        t[i] = min(e)+max(2, 0.7*sum(e)/(len(e)-min(e)))
    return smooth, t

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
