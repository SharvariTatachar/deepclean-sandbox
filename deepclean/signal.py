import math
import warnings

import numpy as np
import scipy.signal as sig

def _parse_window(nperseg, noverlap, window):
    """Parse window string to numpy array
    
    Parameters
    ----------
    nperseg: int
        Length of each segment
    noverlap: int
        Number of overlapping samples
    window: str
        Window type (e.g., 'hann', 'hamming', 'blackman')
    
    Returns
    -------
    numpy.ndarray
        Window function array
    """
    try:
        return sig.get_window(window, nperseg)
    except ValueError:
        # If window name not recognized, default to Hann
        warnings.warn(f"Window '{window}' not recognized, using 'hann'")
        return sig.get_window('hann', nperseg)

def bandpass(data, fs, fl, fh, order=None, axis=-1):
    """ Apply Butterworth bandpass filter using scipy.signal.sosfiltfilt method
    Parameters
    ----------
    data: array
    fs: sampling frequency
    fl, fh: low and high frequency for bandpass
    axis: axis to apply the filter on 
    
    Returns:
    --------
    data_filt: filtered array 
    """
    if order is None:
        order = 8
        
    # Make filter
    nyq = fs/2.
    low, high = fl/nyq, fh/nyq  # normalize frequency
    z, p, k = sig.butter(order, [low, high], btype='bandpass', output='zpk')
    sos = sig.zpk2sos(z, p, k)

    # Apply filter and return output
    data_filt = sig.sosfiltfilt(sos, data, axis=axis)
    return data_filt

def overlap_add(data, noverlap, window, verbose=True):
    """ Concatenate timeseries using the overlap-add method 
    Parameters
    -----------
    data: `numpy.ndarray` of shape (N, nperseg)
        array of timeseries segments to be concatenate
    noverlap: `int`
        number of overlapping samples between each segment in `data`
    window: `str`, `numpy.ndarray`
    
    Returns
    --------
    `numpy.ndarray`
        concatenated timeseries
    """
    # Get dimension
    N, nperseg = data.shape
    stride = nperseg - noverlap
    
    # Get window function
    if isinstance(window, str):
        window = _parse_window(nperseg, noverlap, window)
        
    # Concatenate timeseries
    nsamp = int((N - 1) * stride + nperseg)
    new = np.zeros(nsamp)    
    for i in range(N):
        new[i * stride: i * stride + nperseg] += data[i] * window
    return new
