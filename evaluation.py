import activity_detection as ad
import aubio as aubio


#rse = ad.RSE_soundsense(x, samplerate)


def find_voice_segments(soundfile):
    pass

def find_pitches(soundfile):
    pass

def evaluate_soundfiles(path):
    files = ad.filelist(path)
    for f in files.iteritems():
        evaluate_file(f)

def evaluate_file(filename):
    samplerate = samplerate(filename)
    with ad.read_file(filename) as f:
        frames = ad.frames(x, samplerate, 64)
        ac_peaks = ad.ac_peaks(frames)
        energy = ad.energies(frames, ...)
        normalized_spectra = ad.normalized_spectra(frames, samplerate)
        energy_t = ad.thresholds(frames)
        entropy_t = ad.entropy_t(frames)
        energy_indexes = ad.get_segment_indexes(x, t, min_len=30)
        #TODO:
        entropy_indexes = ad.get_entropy_indexes(x, t, min_len=30)
        #TODO:
        combination_indexes = ad.get_combined_indexes(x, t, min_len=30)
    with aubio.source(filename, samplerate) as s:
        #TODO
        pitches, pitch_confidences = aubio.pitches(f)
