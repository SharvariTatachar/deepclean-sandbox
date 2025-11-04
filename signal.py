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
