import essentia
from essentia import FrameGenerator

pool = essentia.Pool()

for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    pool.add('lowlevel.mfcc', mfcc_coeffs)
    pool.add('lowlevel.mfcc_bands', mfcc_bands)

imshow(pool['lowlevel.mfcc'].T[1:,:], aspect = 'auto')
show() # unnecessary if you started "ipython --pylab"
figure()
imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', interpolation = 'nearest')
