import numpy as np
from embedia.native.signals.spectrum_base import SpectrogramBase


class MelSpectrum(SpectrogramBase):
    '''
    Class that implements Mel Spectrogram processing.
    '''

    def __init__(self, frame_length, n_mels=40, overlap_length=None, window_type='hann', input_length=16000, input_fs=16000, fmin=0, fmax=None):
        '''
        MelSpectrum constructor.

        Parameters
        ----------
        frame_length : int
            The number of data points used in each frame/block for the DFT.
        n_mels : int, optional
            Number of mel bins. Default is 40.
        overlap_length : int, optional
            The number of points of overlap between blocks.
        window_type : str or tuple or array_like
            Desired window to use. Default is 'hann'.
        input_fs : int, optional
            Sampling rate of the audio signal. Default is 16000.
        fmin : float, optional
            Minimum frequency for mel bands. Default is 0.
        fmax : float, optional
            Maximum frequency for mel bands. Default is sr/2.
        '''
        super().__init__(frame_length, overlap_length, n_channels, window_type, input_length, input_fs, padding=False)
        #super().__init__(n_fft, window_type=window_type, overlap_length=overlap_length)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr // 2
        self._mel_filterbank = self._create_mel_filterbank()

    def _hz_to_mel(self, hz):
        """Convert frequency from Hz to mel scale"""
        return 2595 * np.log10(1 + hz / 700.0)

    def _mel_to_hz(self, mel):
        """Convert frequency from mel scale to Hz"""
        return 700 * (10 ** (mel / 2595.0) - 1)

    def _create_mel_filterbank(self):
        """Create a Mel filterbank"""
        # Calculate frequencies for each FFT bin
        fft_freqs = np.linspace(0, self.sr // 2, self.n_fft // 2 + 1)

        # Calculate mel frequencies
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Convert hz_points to fft bin indices
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)

        # Create filterbank
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for i in range(1, self.n_mels + 1):
            left = bin_indices[i - 1]
            center = bin_indices[i]
            right = bin_indices[i + 1]

            # Rising slope
            if left < center:
                filterbank[i - 1, left:center] = np.linspace(0, 1, center - left)

            # Falling slope
            if center < right:
                filterbank[i - 1, center:right] = np.linspace(1, 0, right - center)

        # Normalize filters by their area
        filterbank = filterbank / (filterbank.sum(axis=1, keepdims=True) + 1e-10)

        return filterbank[:, :self.n_fft // 2]  # We only need the first half

    def compute_spectrum(self, frame):
        """Calculate mel spectrum for a frame"""
        # Compute STFT magnitude spectrum
        spectrum = np.fft.fft(frame)
        magnitude = np.abs(spectrum[:len(spectrum) // 2])

        # Apply mel filterbank
        mel_spectrum = np.dot(self._mel_filterbank, magnitude)

        return mel_spectrum