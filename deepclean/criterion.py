

import numpy as np

import torch
import torch.nn as nn
import logging.config

# Debug flag - set to True to enable detailed debugging output
DEBUG_FFT = False  # Set to False to disable debugging


def _torch_welch(data, fs=1.0, nperseg=256, noverlap=None, average='mean', 
                 device='cpu'):
    """ Compute PSD using Welch's method. 
    NOTE: The function is off by a constant factor from scipy.signal.welch 
    Because we will be taking the ratio, this is not important (for now) 
    """
    if len(data.shape) > 2:
        data = data.view(data.shape[0], -1)
    N, nsample = data.shape
    
    # Get parameters
    if noverlap is None:
        noverlap = nperseg//2
    nstride = nperseg - noverlap
    nseg = int(np.ceil((nsample-nperseg)/nstride)) + 1
    nfreq = nperseg // 2 + 1
    T = nsample*fs
   
    # Calculate the PSD
    starts = list(range(0, max(0, nsample - nperseg + 1), nstride))
    nseg = len(starts)
    psd = torch.zeros((nseg, N, nfreq), device=device)
    window = torch.hann_window(nperseg, device=device) * 2

    for i, s in enumerate(starts):
        seg_ts = data[:, s : s + nperseg]  # exactly nperseg long due to starts
        seg_fd = torch.fft.rfft(seg_ts * window, dim=1)
        psd[i] = (seg_fd.abs() ** 2)

    # average/median over segments
    if average == 'mean':
        psd = psd.mean(dim=0)              # true mean
    elif average == 'median':
        psd = psd.median(dim=0).values
    else:
        raise ValueError('average must be "mean" or "median"')

    # Normalize — constant factor differences don’t break ratios, but keep it stable
    psd /= max(T, 1.0)  # T = nsample * fs
    return psd


class MSELoss(nn.Module):
    """ Mean-squared error loss """
    
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, pred, target):
        loss = (target - pred) ** 2
        loss = torch.mean(loss, 1)
        
        # Averaging over patch
        if self.reduction == 'mean':
            loss = torch.sum(loss) / len(pred)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss

    
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
        self.mask = torch.zeros(nperseg//2 +1, dtype=torch.bool, device=device)
        self.scale = 0.0
        for l, h in zip(fl, fh):
            self.mask = self.mask | (l < freq) & (freq < h)
            self.scale += (h - l)
        self.mask = self.mask.to(device)
        self.scale = float(self.scale) if self.scale > 0 else float(self.dfreq.item())
      
    
    def forward(self, pred, target, eps: float = 1e-20):
        # pred, target: (B, T)
        psd_res    = self.welch(target - pred)   # (B, F)
        psd_target = self.welch(target)          # (B, F)

        # select only masked freqs
        psd_res_band    = psd_res[:, self.mask]
        psd_target_band = psd_target[:, self.mask]

        # safe denominator to avoid 0/0 and Inf
        denom = psd_target_band.clamp_min(eps)
        psd_ratio = psd_res_band / denom

        if self.asd:
            # sqrt can create NaN if ratio is negative or inf; clamp first
            psd_ratio = psd_ratio.clamp_min(0)
            band_vals = torch.sqrt(psd_ratio)
        else:
            band_vals = psd_ratio

        # Integrate over band (approx): sum * df / total_bandwidth
        # Use bin count * df for more precise normalization than (h-l) sum
        band_width = self.mask.sum().to(band_vals.dtype) * self.dfreq
        loss_per_sample = band_vals.sum(dim=1) * self.dfreq / band_width

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample
    
class CrossPSDLoss(nn.Module):
    ''' Compute the power spectrum density (PSD) loss, defined 
    as the average over frequency of the PSD ratio 
    Unlike the other one, here the prediction from multiple segs in a batch 
    are combined to weigh more on the edges
    '''
    
    
    def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None, 
                 asd=False, average='mean', train_kernel = 4, batch_size=32, train_stride = 0.25,
                 reduction='mean', device='cpu'):
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
        self.train_stride = train_stride
        self.train_kernel = train_kernel
        self.batch_size = batch_size
        
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
        self.mask = torch.zeros(nperseg//2 +1, dtype=torch.bool, device=device)
        self.scale = 0.0
        for l, h in zip(fl, fh):
            self.mask = self.mask | (l < freq) & (freq < h)
            self.scale += (h - l)
        self.mask = self.mask.to(device)
        self.scale = float(self.scale) if self.scale > 0 else float(self.dfreq.item())
        
    def forward(self, pred, target, eps: float = 1e-20):
        res = target - pred
        shape_x = int(self.train_kernel / self.train_stride)
        shape_y = int(self.batch_size * self.train_stride * self.fs)

        cross_res    = torch.zeros([shape_x, shape_y], device=self.device)
        cross_target = torch.zeros([shape_x, shape_y], device=self.device)
        for i in range(cross_res.shape[0]):
            s = int(i * int(self.train_stride * self.fs))
            e = int((i + 1) * int(self.train_stride * self.fs))
            cross_res[i, :]    = res[:, s:e].flatten()
            cross_target[i, :] = target[:, s:e].flatten()

        psd_res    = self.welch(cross_res)
        psd_target = self.welch(cross_target)

        psd_res_band    = psd_res[:, self.mask]
        psd_target_band = psd_target[:, self.mask]

        denom = psd_target_band.clamp_min(eps)
        psd_ratio = psd_res_band / denom

        band_width = self.mask.sum().to(psd_ratio.dtype) * self.dfreq
        loss_per_seg = psd_ratio.sum(dim=1) * self.dfreq / band_width

        # your “edges” averaging policy; keep as-is but now it’s finite
        edge_idx = torch.tensor([-8,-7,-6,-5,-4,-3,-2,-1], device=loss_per_seg.device)
        loss = loss_per_seg.index_select(0, edge_idx).mean() * len(loss_per_seg)
        return loss

    
class EdgeMSELoss(nn.Module):
    """ Mean-squared error loss at the edges of segments"""
    
    def __init__(self, reduction='mean', edge_frac=0.1, eps=1e-8):
        super().__init__()
        
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        self.eps = eps
        self.edge_frac = edge_frac
        
    def forward(self, pred, target):
        
        ## compute number of samples per one edge (left/right) - 0.5 is for 'one-side'
        nsamp_edge = round(pred.shape[1] * self.edge_frac)
        # indices of left edge
        #idx_edge2 = np.arange(pred.shape[1]-nsamp_one_edge, pred.shape[1])
        # indices of right edge
        idx_edge1 = np.arange(0,nsamp_edge)
        idx_edge = list(idx_edge1)

        residual = target - pred
        residual_edge = residual[:,idx_edge]
        ## mean squared residual
        loss = torch.mean(residual_edge**2, 1)

        # Averaging over patch
        if self.reduction == 'mean':
            loss = torch.sum(loss) / len(residual_edge)

        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

    
    
    
class CompositePSDLoss(nn.Module):
    ''' PSD + MSE Loss with weight '''
    
    def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None, 
                 asd=False, average='mean', reduction='mean', psd_weight=0.5, 
                 mse_weight=0.5, edge_weight=0.0, edge_frac = 0.1, cross_psd_weight = 0.0,
                 train_kernel = 4, batch_size=32, train_stride = 0.25, device='cpu'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        
        self.psd_loss = PSDLoss(
            fs=fs, fl=fl, fh=fh, fftlength=fftlength, overlap=overlap, asd=asd, 
            average=average, reduction=reduction, device=device)
        self.cross_psd_loss = CrossPSDLoss(
            fs=fs, fl=fl, fh=fh, fftlength=fftlength, overlap=overlap, asd=asd, 
            average=average, 
            train_kernel = train_kernel, batch_size = batch_size, train_stride = train_stride,
            reduction=reduction, device=device)
        self.mse_loss = MSELoss(reduction=reduction)
        self.edge_loss = EdgeMSELoss(reduction=reduction, edge_frac=edge_frac)
        
        self.psd_weight  = psd_weight
        self.cross_psd_weight  = cross_psd_weight
        self.mse_weight  = mse_weight
        self.edge_weight = edge_weight
                
    def forward(self, pred, target):

        if self.psd_weight == 0:
            psd_loss = 0
        else:
            psd_loss  = self.psd_weight  * self.psd_loss  (pred, target)

        if self.cross_psd_weight == 0:
            cross_psd_loss = 0
        else:
            cross_psd_loss  = self.cross_psd_weight  * self.cross_psd_loss  (pred, target)

            
        if self.mse_weight == 0:
            mse_loss = 0
        else:
            mse_loss  = self.mse_weight  * self.mse_loss  (pred, target)

        if self.edge_weight == 0:
            edge_loss = 0
        else:
            edge_loss  = self.edge_weight  * self.edge_loss  (pred, target)
        
        return (psd_loss + cross_psd_loss + mse_loss + edge_loss)
