import activity_detection as aed #audiodetect

class service(Flask):
    def post_soundfile(tmpname):
        fileid = store_soundfile(tmpname)
        samplerate = None# libsamplerate, aubio?
        self.fileid = fileid
        self.samplerate = samplerate
        self.x = aed.frames(fileid)
        self.rse = aed.RSE_soundsense(signal, samplerate)
        self.acpeaks = aed.acpeaks(signal)
        self.noisypeaks = aed.acpeaks(aed.noise(signal, samplerate))
        self.pitches = aed.pitches(signal, samplerate)
        self.voiced, self.indexes = aed.segment(signal, rse, acpeaks
        return render_soundfile_page(self)

    def post_segment(tmpname)
        fileid = store_soundfile(tmpname)
        samplerate = None# libsamplerate, aubio?
        self.x = aed.frames(fileid)
        self.rse = aed.RSE_soundsense(signal, samplerate)
        self.acpeaks = aed.acpeaks(signal)
        self.noisypeaks = aed.acpeaks(signal+aed.noise)
        self.pitches = aed.pitches(signal, samplerate)
        self.voiced, self.indexes = aed.segment(signal, rse, acpeaks
        return render_segment_info_page(self)

    def segment(x, samplerate):
        frames = sigutil.frame(x)
        for f in frames:
            rse = aed.RSE_soundsense(signal, samplerate)
            acpeaks = aed.acpeaks(signal)
            pitches = aed.pitches(signal, samplerate)
            voiced, self.indexes = aed.segment(signal, rse, acpeaks
    
