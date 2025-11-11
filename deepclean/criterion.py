

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
    psd = torch.zeros((nseg, N, nfreq)).to(device)
    window =  torch.hann_window(nperseg).to(device)*2
    
    # calculate the FFT amplitude of each segment
    for i in range(nseg):
        seg_ts = data[:, i*nstride:i*nstride+nperseg]*window
        seg_fd = torch.fft.rfft(seg_ts, dim=1)
        
        # Debug: Print FFT information
        # if DEBUG_FFT and i == 0:  # Only print for first segment to avoid spam
        #     print(f"\n[DEBUG _torch_welch] FFT Analysis:")
        #     print(f"  Input seg_ts shape: {seg_ts.shape}, dtype: {seg_ts.dtype}")
        #     print(f"  FFT output seg_fd shape: {seg_fd.shape}, dtype: {seg_fd.dtype}")
        #     print(f"  FFT output is complex: {torch.is_complex(seg_fd)}")
        #     # Show first few frequency bins
        #     if seg_fd.numel() > 0:
        #         print(f"  First 5 FFT values (complex): {seg_fd[0, :5]}")
        #         print(f"  First 5 FFT magnitudes: {torch.abs(seg_fd[0, :5])}")
        #         print(f"  First 5 FFT magnitudes squared: {torch.abs(seg_fd[0, :5])**2}")
        
        seg_fd_abs = torch.abs(seg_fd)**2
        
        # Debug: Verify magnitude calculation
        # if DEBUG_FFT and i == 0:
        #     print(f"  seg_fd_abs shape: {seg_fd_abs.shape}, dtype: {seg_fd_abs.dtype}")
        #     print(f"  First 5 PSD values: {seg_fd_abs[0, :5]}")
        #     print(f"  PSD min: {seg_fd_abs.min().item():.6e}, max: {seg_fd_abs.max().item():.6e}, mean: {seg_fd_abs.mean().item():.6e}")
        
        psd[i] = seg_fd_abs
    
    # taking the average
    if average == 'mean':
        psd = torch.sum(psd, 0)
    elif average == 'median':
        psd = torch.median(psd, 0)[0]*nseg
    else:
        raise ValueError(f'average must be "mean" or "median", got {average} instead')

    # Normalize
    psd /= T
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
        freq = torch.linspace(0., fs/2., nperseg//2 + 1)
        self.dfreq = freq[1] - freq[0]
        self.mask = torch.zeros(nperseg//2 +1, dtype=torch.bool)
        self.scale = 0.
        for l, h in zip(fl, fh):
            self.mask = self.mask | (l < freq) & (freq < h)
            self.scale += (h - l)
        self.mask = self.mask.to(device)
        
        # Debug: Print mask information
        # if DEBUG_FFT:
        #     print(f"\n[DEBUG PSDLoss.__init__] Mask Configuration:")
        #     print(f"  Frequency range: {freq.min().item():.2f} - {freq.max().item():.2f} Hz")
        #     print(f"  Number of frequency bins: {len(self.mask)}")
        #     print(f"  Mask dtype: {self.mask.dtype}")
        #     # Check if mask is boolean or ByteTensor
        #     if self.mask.dtype == torch.bool:
        #         print(f"  ✓ Mask is boolean (correct for PyTorch >= 1.2)")
        #     else:
        #         print(f"  ⚠ WARNING: Mask is {self.mask.dtype} (not boolean)!")
        #         print(f"    This will cause RuntimeError when used for indexing!")
        #         print(f"    PyTorch requires boolean masks for operations like psd_res[:, ~mask] = 0")
        #     print(f"  Mask True count: {self.mask.sum().item()} (frequencies kept)")
        #     print(f"  Mask False count: {(~self.mask).sum().item()} (frequencies masked)")
        #     print(f"  Frequency ranges to keep: {list(zip(fl, fh))}")
        #     # Show which frequencies are kept
        #     kept_freqs = freq[self.mask]
        #     if len(kept_freqs) > 0:
        #         print(f"  Kept frequency range: {kept_freqs.min().item():.2f} - {kept_freqs.max().item():.2f} Hz")
        #     print(f"  Scale factor: {self.scale}")
        #     print(f"  Frequency resolution (dfreq): {self.dfreq:.6f} Hz")
        #     # Show sample mask values
        #     print(f"  First 10 mask values: {self.mask[:10]}")
        #     print(f"  Last 10 mask values: {self.mask[-10:]}")
    
    def forward(self, pred, target):
        
        # Calculate the PSD of the residual and the target
        psd_res = self.welch(target - pred)
        psd_target = self.welch(target)
        
        # Debug: Print PSD values before masking
        # if DEBUG_FFT:
        #     print(f"\n[DEBUG PSDLoss.forward] PSD Calculation:")
        #     print(f"  psd_res shape: {psd_res.shape}, dtype: {psd_res.dtype}")
        #     print(f"  psd_target shape: {psd_target.shape}, dtype: {psd_target.dtype}")
        #     print(f"  psd_res before masking - min: {psd_res.min().item():.6e}, max: {psd_res.max().item():.6e}, mean: {psd_res.mean().item():.6e}")
        #     print(f"  psd_res sum before masking: {psd_res.sum().item():.6e}")
        #     # Show PSD values at a few frequency bins
        #     if psd_res.shape[0] > 0 and psd_res.shape[1] > 0:
        #         print(f"  psd_res[0, :5] (first 5 freq bins): {psd_res[0, :5]}")
            # Check mask before using it
            # print(f"  About to apply mask - mask dtype: {self.mask.dtype}")
            # print(f"  Mask shape: {self.mask.shape}, psd_res shape: {psd_res.shape}")
            # if self.mask.dtype != torch.bool:
            #     print(f"  ⚠ ERROR: Mask dtype is {self.mask.dtype}, but PyTorch requires torch.bool!")
            #     print(f"    The next line will fail: psd_res[:, ~self.mask] = 0.")
            #     print(f"    Fix: Change mask creation to use dtype=torch.bool")
        
        # Apply mask - this will fail if mask is not boolean
        psd_res[:, ~self.mask] = 0.
        # try:
        #     psd_res[:, ~self.mask] = 0.
        # except RuntimeError as e:
        #     if DEBUG_FFT:
        #         print(f"  ❌ ERROR during masking: {e}")
        #         print(f"    Mask dtype: {self.mask.dtype}")
        #         print(f"    This error occurs because PyTorch requires boolean masks for indexing.")
        #         print(f"    Solution: Change line 134 from .type(torch.ByteTensor) to dtype=torch.bool")
        #     raise
        
        # Debug: Print PSD values after masking
        # if DEBUG_FFT:
        #     print(f"  psd_res after masking - min: {psd_res.min().item():.6e}, max: {psd_res.max().item():.6e}, mean: {psd_res.mean().item():.6e}")
        #     print(f"  psd_res sum after masking: {psd_res.sum().item():.6e}")
        #     print(f"  psd_res[0, :5] after masking: {psd_res[0, :5]}")
        #     # Verify masking worked correctly
        #     masked_bins = (~self.mask).sum().item()
        #     zero_bins = (psd_res[0] == 0).sum().item()
        #     print(f"  Expected zero bins (masked): {masked_bins}, Actual zero bins: {zero_bins}")

        # psd loss is the integration over all frequencies
        psd_ratio = psd_res/psd_target
        asd_ratio = torch.sqrt(psd_ratio)
        
        # Debug: Print ratio information
        # if DEBUG_FFT:
        #     print(f"  psd_ratio shape: {psd_ratio.shape}")
        #     print(f"  psd_ratio min: {psd_ratio.min().item():.6e}, max: {psd_ratio.max().item():.6e}, mean: {psd_ratio.mean().item():.6e}")
        #     print(f"  psd_ratio sum (per sample): {psd_ratio.sum(dim=1)[:3] if len(psd_ratio) > 0 else 'N/A'}")
            
        if self.asd:
            loss = torch.sum(asd_ratio, 1)*self.dfreq/self.scale
        else:
            loss = torch.sum(psd_ratio, 1)*self.dfreq/self.scale
        
        # Debug: Print loss information
        # if DEBUG_FFT:
        #     print(f"  Loss per sample (before reduction): {loss[:3] if len(loss) > 0 else 'N/A'}")
        #     print(f"  Loss shape: {loss.shape}")
        #     print(f"  dfreq: {self.dfreq:.6e}, scale: {self.scale:.2f}")
        
        # Averaging over batch
        if self.reduction == 'mean':
            loss = torch.sum(loss)/len(psd_res)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        # if DEBUG_FFT:
        #     print(f"  Final loss value: {loss.item():.6e}\n")
        
        return loss     
    
    
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
        freq = torch.linspace(0., fs/2., nperseg//2 + 1)
        self.dfreq = freq[1] - freq[0]
        self.mask = torch.zeros(nperseg//2 +1, dtype=torch.bool)
        self.scale = 0.
        for l, h in zip(fl, fh):
            self.mask = self.mask | (l < freq) & (freq < h)
            self.scale += (h - l)
        self.mask = self.mask.to(device)
        
        # Debug: Print mask information
        # if DEBUG_FFT:
        #     print(f"\n[DEBUG CrossPSDLoss.__init__] Mask Configuration:")
        #     print(f"  Frequency range: {freq.min().item():.2f} - {freq.max().item():.2f} Hz")
        #     print(f"  Number of frequency bins: {len(self.mask)}")
        #     print(f"  Mask dtype: {self.mask.dtype}")
        #     print(f"  Mask True count: {self.mask.sum().item()} (frequencies kept)")
        #     print(f"  Mask False count: {(~self.mask).sum().item()} (frequencies masked)")
        #     print(f"  Frequency ranges to keep: {list(zip(fl, fh))}")
        #     # Show which frequencies are kept
        #     kept_freqs = freq[self.mask]
        #     if len(kept_freqs) > 0:
        #         print(f"  Kept frequency range: {kept_freqs.min().item():.2f} - {kept_freqs.max().item():.2f} Hz")
        #     print(f"  Scale factor: {self.scale}")
        #     print(f"  Frequency resolution (dfreq): {self.dfreq:.6f} Hz")
    
    def forward(self, pred, target):
        
        res = target - pred
        shape_x = int(self.train_kernel/self.train_stride)
        shape_y = int(self.batch_size*self.train_stride*self.fs)
        
        #cross_res = torch.zeros(res.shape).to(self.device)
        cross_res = torch.zeros([shape_x, shape_y]).to(self.device)
        cross_target = torch.zeros([shape_x, shape_y]).to(self.device)
        for i in range(cross_res.shape[0]):
            slice_i_begin = int(i*int(self.train_stride*self.fs))
            slice_i_end   = int((i+1)*int(self.train_stride*self.fs))
            slice_res_i  = res[:,slice_i_begin:slice_i_end].flatten()
            slice_target_i  = target[:,slice_i_begin:slice_i_end].flatten()
            #print (slice_i.shape, " and ", cross_res.shape)
            cross_res[i,:]  = slice_res_i
            cross_target[i,:]  = slice_target_i
        
        
        # Calculate the PSD of the residual and the target
        psd_res    = self.welch(cross_res)
        psd_target = self.welch(cross_target)
        
        # Debug: Print PSD values before masking
        # if DEBUG_FFT:
        #     print(f"\n[DEBUG CrossPSDLoss.forward] PSD Calculation:")
        #     print(f"  cross_res shape: {cross_res.shape}")
        #     print(f"  psd_res shape: {psd_res.shape}, dtype: {psd_res.dtype}")
        #     print(f"  psd_target shape: {psd_target.shape}, dtype: {psd_target.dtype}")
        #     print(f"  psd_res before masking - min: {psd_res.min().item():.6e}, max: {psd_res.max().item():.6e}, mean: {psd_res.mean().item():.6e}")
        
        psd_res[:, ~self.mask] = 0.
        
        # Debug: Print PSD values after masking
        # if DEBUG_FFT:
        #     print(f"  psd_res after masking - min: {psd_res.min().item():.6e}, max: {psd_res.max().item():.6e}, mean: {psd_res.mean().item():.6e}")

        # psd loss is the integration over all frequencies
        psd_ratio = psd_res/psd_target
        
        # if DEBUG_FFT:
        #     print(f"  psd_ratio shape: {psd_ratio.shape}")
        #     print(f"  psd_ratio min: {psd_ratio.min().item():.6e}, max: {psd_ratio.max().item():.6e}, mean: {psd_ratio.mean().item():.6e}")
        
        loss = torch.sum(psd_ratio, 1)*self.dfreq/self.scale
        
        # if DEBUG_FFT:
        #     print(f"  Loss per segment (before edge selection): {loss}")
        #     print(f"  Selecting edges: indices [-8,-7,-6,-5,-4,-3,-2,-1]")
        
        ## we average over just the 3 timeseries from each edges
        loss = torch.mean(loss[[-8,-7,-6,-5,-4,-3,-2,-1]]) * len(loss)
        
        # if DEBUG_FFT:
        #     print(f"  Final CrossPSD loss value: {loss.item():.6e}\n")
        
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
