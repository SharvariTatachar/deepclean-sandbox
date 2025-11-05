import numpy as np

import torch
import torch.nn as nn
import torch.fft


def _torch_welch(x, fs=1.0, nperseg=None, noverlap=None, device='cpu'):
    """
    Compute Welch's power spectral density estimate using PyTorch.
    
    Parameters
    ----------
    x : torch.Tensor
        Input time series data (batch_size, sequence_length)
    fs : float
        Sample rate
    nperseg : int
        Length of each segment
    noverlap : int, optional
        Number of points to overlap between segments
    device : str
        Device to run computation on
    
    Returns
    -------
    torch.Tensor
        PSD estimate (batch_size, nfreq)
    """
    if nperseg is None:
        nperseg = x.shape[-1]
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Move to device
    x = x.to(device)
    
    # Calculate number of segments
    nstep = nperseg - noverlap
    nseg = (x.shape[-1] - noverlap) // nstep
    
    # Window function (Hanning)
    window = torch.hann_window(nperseg, device=device)
    norm = (window ** 2).sum()
    
    # Compute FFT for each segment
    psd_list = []
    for i in range(nseg):
        start = i * nstep
        stop = start + nperseg
        segment = x[..., start:stop] * window
        fft_seg = torch.fft.rfft(segment, n=nperseg)
        psd_seg = torch.abs(fft_seg) ** 2
        psd_list.append(psd_seg)
    
    # Average over segments
    psd = torch.stack(psd_list, dim=0).mean(dim=0)
    
    # Normalize
    psd = psd / (fs * norm)
    
    # Return only positive frequencies
    return psd


class PSDLoss(nn.Module):
    ''' Compute the power spectrum density (PSD) loss, defined 
    as the average over frequency of the PSD ratio '''
    
    
    def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None, 
                 asd=False, average='mean', reduction='mean', device='cpu'):
        super().__init__()
        
        if isinstance(fl, (int, float)):
            fl = (fl, )
        if isinstance(fh, (int, float)):
            fh = (fh, )
        
        # Initialize attributes
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        self.fs = fs
        self.average = average
        self.device = device
        self.asd = asd
        
        nperseg = int(fftlength * self.fs)
        if overlap is not None:
            noverlap = int(overlap * self.fs)
        else:
            noverlap = None
        self.welch = lambda x: _torch_welch(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap, device=device)
        
        # Get scaling and masking
        freq = torch.linspace(0., fs/2., nperseg//2 + 1, device=device)
        self.dfreq = freq[1] - freq[0]
        self.mask = torch.zeros(nperseg//2 + 1, dtype=torch.bool, device=device)
        self.scale = 0.
        for l, h in zip(fl, fh):
            self.mask = self.mask | ((l < freq) & (freq < h))
            self.scale += (h - l)
    
    def forward(self, pred, target):
        
        # Calculate the PSD of the residual and the target
        psd_res = self.welch(target - pred)
        psd_target = self.welch(target)
        psd_res[:, ~self.mask] = 0.

        # psd loss is the integration over all frequencies
        psd_ratio = psd_res/psd_target
        asd_ratio = torch.sqrt(psd_ratio)
            
        if self.asd:
            loss = torch.sum(asd_ratio, 1)*self.dfreq/self.scale
        else:
            loss = torch.sum(psd_ratio, 1)*self.dfreq/self.scale
        
        # Averaging over batch
        if self.reduction == 'mean':
            loss = torch.sum(loss)/len(psd_res)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss     
    
