import activity_detection as ad
import numpy as np
import scikits.samplerate as srate
try:
    try:
        import scikits.audiolab as al
    except ImportError:
        import audiolab as al
except ImportError:
    al = None
    print("Warning: scikits.audiolab not found! Using scipy.io.wavfile")
    from scipy.io import wavfile

#rse = ad.RSE_soundsense(x, samplerate)
datapath = "g/8k"

def open_file(filename, resample_to=8000):
    soundfile = al.Sndfile(filename, 'r')
    samplerate=soundfile.samplerate
    sig = ad.read_file(filename)
    return sig, samplerate

def analyze(sig, samplerate=8000, resample_to=8000):
    e_min_scale = 0.3
    e_min_distance = 1
    e_a_scale = 0.5
    e_W_ms = 2000
    min_len = 30
    if samplerate != resample_to:
        print("resampling")
        sig = srate.resample(sig, resample_to/samplerate, 'sinc_best')

    frames,frame_size = ad.frames(sig, samplerate, 64)
    #ac_peaks = [ad.ac_peaks(frame) for frame in frames]
    #energy = ad.energies(frames, ...)
    #energy = log_energy
    print("getting normalized spectra")
    normalized_spectrum = ad.normalized_spectrum(frames, samplerate)
    print("spectral entropy")
    spectral_entropy = np.fromiter(ad.entropy(frame) for frame in normalized_spectrum)
    entopy_t = np.percentile(spectral_entropy, 80)
    se_segments = ad.segments_to_seconds(ad.entropy_segment_indexes(spectral_entropy, entropy_t))
    print("energy based computations")
    energy, smooth, en_a, en_t, en_lmin = ad.energy_thresholds(sig, noise_dist=e_min_distance, a_scale=e_a_scale, min_scale=e_min_scale, W_ms=e_W_ms)
    energy_segments = ad.get_voice_segments(smooth, en_t, ad.get_segment_indexes(smooth, en_t, min_len=10))
    #energy_t = ad.thresholds(frames)
    #entropy_t = ad.entropy_t(frames)
    #energy_indexes = ad.get_segment_indexes(x, t, min_len=30)
    #entropy_indexes = ad.get_entropy_indexes(x, t, min_len=30)
    #combination_indexes = ad.get_combined_indexes(x, t, min_len=30)
    return spectral_entropy, se_t, se_segments, energy, en_t, energy_segments

def compare_labels(predicted, truth):
    timeline = []
    for p in predicted:
        timeline.append([p[0], "p", "st"])
        timeline.append([p[1], "p", "en"])
    for t in truth:
        timeline.append([t[0], "t", "st"])
        timeline.append([t[1], "t", "en"])
    timeline = sorted(timeline)
    for i in timeline:
        print(i)
    x = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    state = False
    prediction = False
    for i in timeline:
        if i[1] == "p" and i[2] == "st":
            prediction = True
            if state:
                fn += i[0]-x
            else:
                tn += i[0]-x
            x = i[0]
        elif i[1] == "p" and i[2] == "en":
            prediction = False
            if state:
                tp += i[0]-x
            else:
                fp += i[0]-x
            x = i[0]
        elif i[1] == "t" and i[2] == "st":
            state = True
            if prediction:
                fp += i[0]-x
            else:
                tn += i[0]-x
            x = i[0]
        elif i[1] == "t" and i[2] == "en":
            state = False
            if prediction:
                tp += i[0]-x
            else:
                fn += i[0]-x
            x = i[0]
    return tp/x,tn/x,fp/x,fn/x

def analyze_soundfiles(path, resample_to=8000):
    results = {}
    for filename in ad.filelist(path):
        f, samplerate = open_file(filename)
        #se, se_t, se_sgments, energy, en_t, energy_segments = analyze(f, samplerate)
        results[filename] = analyze(f, samplerate)
    return results

def features(win_s, hop_s):
    win_s = 4096
    hop_s = 4096
    pitch_tolerance = 0.8
    pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("hz")
    pitch_o.set_tolerance(tolerance)
    pitches = []
    confidences = []
    with aubio.source(filename, samplerate, hop_s) as s:
        samplerate = s.samplerate
        while True:
            samples, read = s()
            pitch, pitch_confidence = pitches(pitch_o, tolerance, samples)
            pitches += [pitch]
            pitch_confidences += [pitch_confidence]
            total_frames += read
            if read < hop_s: break
    return samples, pitches, pitch_confidences

def pitches(pitch_o, tolerance, samples):
    pitch = pitch_o(samples)[0]
    confidence = pitch_o.get_confidence()
    return pitch, confidence

def xspace(sig, unit_w):
    x_timebased = np.linspace(0,len(frames)*frame_len, len(frames))

def plot(n):
    n += 1
    fig = plt.figure()
    ax = fig.add_subplot(n)
    ax.axis([0, 4870, 0, 0])

def plot_vlines():
    pass

def plot_line():
    pass
