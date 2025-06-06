"""
Implements Automatic Gain Control (AGC) for audio signals, as described in http://labrosa.ee.columbia.edu/matlab/tf_agc/
"""

import numpy as np
import scipy.signal as signal


def fft2melmx(
    nfft,
    sr=8000.0,
    nfilts=None,
    width=1.0,
    minfrq=0.0,
    maxfrq=None,
    htkmel=False,
    constamp=False,
):
    """
    Generate a matrix of weights to combine FFT bins into Mel
    bins.  nfft defines the source FFT size at sampling rate sr.
    Optional nfilts specifies the number of output bands required
    (else one per "mel/width"), and width is the constant width of each
    band relative to standard Mel (default 1).
    While wts has nfft columns, the second half are all zero.
    Hence, Mel spectrum is fft2melmx(nfft,sr)*abs(fft(xincols,nfft));
    minfrq is the frequency (in Hz) of the lowest band edge;
    default is 0, but 133.33 is a common standard (to skip LF).
    maxfrq is frequency in Hz of upper edge; default sr/2.
    You can exactly duplicate the mel matrix in Slaney's mfcc.m
    as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    htkmel=1 means use HTK's version of the mel curve, not Slaney's.
    constamp=1 means make integration windows peak at 1, not sum to 1.
    frqs returns bin center frqs.
    """

    if maxfrq is None:
        maxfrq = sr / 2.0

    if nfilts is None:
        nfilts = int(np.ceil(hz2mel(maxfrq, htkmel) / 2.0))

    wts = np.zeros((nfilts, nfft))

    # Center freqs of each FFT bin
    fftfrqs = np.arange(nfft / 2 + 1, dtype=float) / nfft * sr

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfrq, htkmel)
    maxmel = hz2mel(maxfrq, htkmel)
    binfrqs = mel2hz(
        minmel + np.arange(nfilts + 2, dtype=float) / (nfilts + 1) * (maxmel - minmel),
        htkmel,
    )

    # binbin = round(binfrqs / sr * (nfft - 1))

    for i in range(nfilts):
        fs = binfrqs[i + np.array([0, 1, 2])]
        # scale by width
        fs = fs[1] + width * (fs - fs[1])
        # lower and upper slopes for all bin
        loslope = (fftfrqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs) / (fs[2] - fs[1])
        # .. then intersect them with each other and zero
        wts[i, : int(nfft / 2) + 1] = np.maximum(0, np.minimum(loslope, hislope))

    if not constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = np.dot(np.diag(2.0 / (binfrqs[2 : nfilts + 2] - binfrqs[:nfilts])), wts)

    # Make sure 2nd half of FFT is zero
    wts[:, int(nfft / 2 + 2) :] = 0
    # seems like a good idea to avoid aliasing

    return (wts, binfrqs)


def mel2hz(z, htk=False):
    """
    Convert 'mel scale' frequencies into Hz.
    Optional htk=True means use the HTK formula; else use the formula from Malcolm Slaney's mfcc.m
    """

    if htk:
        f = 700.0 * (10.0 ** (z / 2595.0) - 1)
    else:
        f_0 = 0  # 133.33333
        f_sp = 200.0 / 3.0  # 66.66667
        brkfrq = 1000.0
        brkpt = (brkfrq - f_0) / f_sp  # starting mel value for log region
        logstep = np.exp(
            np.log(6.4) / 27.0
        )  # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        linpts = z < brkpt

        f = 0 * z

        if np.isscalar(z):
            f = (
                f_0 + f_sp * z
                if linpts
                else brkfrq * np.exp(np.log(logstep) * (z - brkpt))
            )
        else:
            # fill in parts separately
            f[linpts] = f_0 + f_sp * z[linpts]
            f[~linpts] = brkfrq * np.exp(np.log(logstep) * (z[~linpts] - brkpt))

    return f


def hz2mel(f, htk=False):
    """
    Convert frequencies f (in Hz) to mel 'scale'.
    Optional htk=True uses the mel axis defined in the HTKBook; otherwise use Malcolm Slaney's formula.
    """

    if htk:
        z = 2595.0 * np.log10(1.0 + f / 700.0)
    else:
        # pass
        f_0 = 0  # 133.33333;
        f_sp = 200.0 / 3.0  # 66.66667;
        brkfrq = 1000.0
        brkpt = (brkfrq - f_0) / f_sp  # starting mel value for log region
        logstep = np.exp(
            np.log(6.4) / 27.0
        )  # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        linpts = f < brkfrq

        z = 0 * f

        if np.isscalar(f):
            z = (
                (f - f_0) / f_sp
                if linpts
                else brkpt + (np.log(f / brkfrq)) / np.log(logstep)
            )
        else:
            # fill in parts separately
            z[linpts] = (f[linpts] - f_0) / f_sp
            z[~linpts] = brkpt + (np.log(f[~linpts] / brkfrq)) / np.log(logstep)

    return z


def stft(x, frame_size, hop_size=None, window=None, N=None, only_positive_freqs=True):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal x.

    x:          signal (for now, only mono is supported)
    frame_size: in samples
    hop_size:   in samples (default 25% of frame size)
    window:     numpy array with the window to be used
    N:          number of FFT points to compute
    only_positive_freqs:
                if True, only the positive FFT bins (including DC) are returned
    """

    # set defaults and sanity check
    assert isinstance(frame_size, int)
    if hop_size is None:
        hop_size = (
            frame_size / 4
        )  # the default windows (stft & isftf) are designed to work with 25% overlap
    assert isinstance(hop_size, int)
    if window is None:
        # window = 0.5 * (1. - np.cos(2. * np.pi * np.arange(frame_size) / frame_size))
        window = np.hanning(frame_size + 1)[:-1]
    assert window.size == frame_size
    if N is None:
        N = frame_size
    assert isinstance(N, int)

    # compute the stft
    X = np.array(
        [
            np.fft.fft(window * x[i : i + frame_size], N)
            for i in range(0, len(x) - frame_size + 1, hop_size)
        ]
    ).T

    # if requested, remove the "negative frequencies"
    if only_positive_freqs:
        X = X[: int(N / 2) + 1, :]

    return X


def istft(X, frame_size, hop_size=None, window=None, only_positive_freqs=True):
    """
    Compute the Inverse Short-Time Fourier Transform (ISTFT) of a stft X.

    X:          Complex STFT (columns encode time and rows encode frequency)
    frame_size: time-domain frame size to use, in sample (each column in X correspond to these many samples in time)
    hop_size:   in samples (default 25% of frame size)
    only_positive_freqs:
                if True, only the positive bins (including DC) are considered to be included in X
    """

    # set defaults and sanity check
    assert isinstance(frame_size, int)
    if hop_size is None:
        hop_size = (
            frame_size / 4
        )  # the default windows (stft & isftf) are designed to work with 25% overlap
    assert isinstance(hop_size, int)
    if window is None:
        # window = 0.5 * (1. - np.cos(2. * np.pi * np.arange(frame_size) / frame_size))
        window = np.hanning(frame_size + 1)[:-1]
        # make it COLA for 25% overlap when using the above window
        window = window * 2.0 / 3.0
    assert window.size == frame_size

    # if required, construct the full spectrogram
    if only_positive_freqs:
        X = np.vstack((X, np.flipud(np.conj(X[1:-1, :]))))

    # allocate output array
    x = np.zeros(X.shape[1] * hop_size + frame_size - hop_size)

    # compute istft by: IFFT + OLA
    for n, i in enumerate(range(0, len(x) - frame_size + 1, hop_size)):
        x[i : i + frame_size] += window * np.real(np.fft.ifft(X[:, n], n=frame_size))

    return x


def tf_agc(d, sr, t_scale=0.5, f_scale=1.0, causal_tracking=True):
    """
    Perform frequency-dependent automatic gain control on an auditory
    frequency axis.
    d is the input waveform (at sampling rate sr);
    y is the output waveform with approximately constant
    energy in each time-frequency patch.
    t_scale is the "scale" for smoothing in time (default 0.5 sec).
    f_scale is the frequency "scale" (default 1.0 "mel").
    causal_tracking == 0 selects traditional infinite-attack, exponential release.
    causal_tracking == 1 selects symmetric, non-causal Gaussian-window smoothing.
    D returns actual STFT used in analysis.  E returns the
    smoothed amplitude envelope divided out of D to get gain control.
    """

    hop_size = 0.032  # in seconds

    # Make STFT on ~32 ms grid
    ftlen = int(2 ** np.round(np.log(hop_size * sr) / np.log(2.0)))
    winlen = ftlen
    hoplen = int(winlen / 2)
    D = stft(d, winlen, hoplen)  # using my code
    ftsr = sr / hoplen
    ndcols = D.shape[1]

    # Smooth in frequency on ~ mel resolution
    # Width of mel filters depends on how many you ask for,
    # so ask for fewer for larger f_scales
    nbands = max(10, 20 / f_scale)  # 10 bands, or more for very fine f_scale
    mwidth = f_scale * nbands / 10  # will be 2.0 for small f_scale
    (f2a_tmp, _) = fft2melmx(ftlen, sr, int(nbands), mwidth)
    f2a = f2a_tmp[:, : int(ftlen / 2) + 1]
    audgram = np.dot(f2a, np.abs(D))

    if causal_tracking:
        # traditional attack/decay smoothing
        fbg = np.zeros(audgram.shape)
        # state = zeros(size(audgram,1),1);
        state = np.zeros(audgram.shape[0])
        alpha = np.exp(-(1.0 / ftsr) / t_scale)
        for i in range(audgram.shape[1]):
            state = np.maximum(alpha * state, audgram[:, i])
            fbg[:, i] = state

    else:
        # noncausal, time-symmetric smoothing
        # Smooth in time with tapered window of duration ~ t_scale
        tsd = np.round(t_scale * ftsr) / 2
        htlen = 6 * tsd  # Go out to 6 sigma
        twin = np.exp(-0.5 * (((np.arange(-htlen, htlen + 1)) / tsd) ** 2)).T

        # reflect ends to get smooth stuff
        AD = audgram
        x = np.hstack(
            (
                np.fliplr(AD[:, :htlen]),
                AD,
                np.fliplr(AD[:, -htlen:]),
                np.zeros((AD.shape[0], htlen)),
            )
        )
        fbg = signal.lfilter(twin, 1, x, 1)

        # strip "warm up" points
        fbg = fbg[:, twin.size + np.arange(ndcols)]

    # map back to FFT grid, flatten bark loop gain
    sf2a = np.sum(f2a, 0)
    sf2a_fix = sf2a
    sf2a_fix[sf2a == 0] = 1.0
    E = np.dot(np.dot(np.diag(1.0 / sf2a_fix), f2a.T), fbg)
    # Remove any zeros in E (shouldn't be any, but who knows?)
    E[E <= 0] = np.min(E[E > 0])

    # invert back to waveform
    y = istft(D / E, winlen, hoplen, window=np.ones(winlen))  # using my code

    return (y, D, E)
