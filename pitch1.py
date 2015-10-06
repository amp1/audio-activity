import sys
from aubio import source, pitch, freqtomidi

#from aubio examples
def argv_soundfile():
    if len(sys.argv) < 2:
        print "Usage: %s <filename> [samplerate]" % sys.argv[0]
        sys.exit(1)
    return sys.argv[1]

#yin pitch contours and confidences, 4096 window
def pitches(filename, samplerate=None, downsample=1):
    if samplerate is None:
        samplerate = 44100 / downsample

    win_s = 4096 / downsample # fft size
    hop_s = 512  / downsample # hop size

    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    tolerance = 0.8

    pitch_o = pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []

# total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        #pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        #if confidence < 0.8: pitch = 0.
        #print "%f %f %f" % (total_frames / float(samplerate), pitch, confidence)
        pitches += [pitch]
        confidences += [confidence]
        total_frames += read
        if read < hop_s: break

    return samples, pitches, confidences, total_frames

from numpy import array, ma
import matplotlib.pyplot as plt
from demo_waveform_plot import get_waveform_plot, set_xlabels_sample2time


