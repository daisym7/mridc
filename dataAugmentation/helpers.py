"""
Utility and helper functions for MRAugment.
"""
import numpy as np
import torch

def to_repeated_list(a, length):
    if isinstance(a, list):
        return a
    elif isinstance(a, tuple):
        return list(a)
    else:
        a = [a] * length
        return a


def ifft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def fft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def fftshift(x: torch.Tensor, dim=None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

    
def pad_if_needed(im, min_shape, mode):
    min_shape = _to_repeated_list(min_shape, 2)
    if im.shape[-2] >= min_shape[0] and im.shape[-1] >= min_shape[1]:
        return im
    else:
        pad = [0, 0]
        if im.shape[-2] < min_shape[0]:
            p = (min_shape[0] - im.shape[-2])//2 + 1
            pad[0] = p
        if im.shape[-1] < min_shape[1]:
            p = (min_shape[1] - im.shape[-1])//2 + 1
            pad[1] = p
        if len(im.shape) == 2:
            pad = ((pad[0], pad[0]), (pad[1], pad[1]))
        else:
            assert len(im.shape) == 3
            pad = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]))

        padded = np.pad(im, pad_width=pad, mode=mode)
        return padded


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift,
    dim,
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def crop_if_needed(im, max_shape):
    assert len(max_shape) == 2
    if im.shape[-2] >= max_shape[0]:
        h_diff = im.shape[-2] - max_shape[0]
        h_crop_before = h_diff // 2
        h_interval = max_shape[0]
    else:
        h_crop_before = 0
        h_interval = im.shape[-2]

    if im.shape[-1] >= max_shape[1]:
        w_diff = im.shape[-1] - max_shape[1]
        w_crop_before = w_diff // 2
        w_interval = max_shape[1]
    else:
        w_crop_before = 0
        w_interval = im.shape[-1]

    return im[...,h_crop_before:h_crop_before+h_interval, w_crop_before:w_crop_before+w_interval]

def complex_crop_if_needed(im, max_shape):
    assert len(max_shape) == 2
    if im.shape[-3] >= max_shape[0]:
        h_diff = im.shape[-3] - max_shape[0]
        h_crop_before = h_diff // 2
        h_interval = max_shape[0]
    else:
        h_crop_before = 0
        h_interval = im.shape[-3]

    if im.shape[-2] >= max_shape[1]:
        w_diff = im.shape[-2] - max_shape[1]
        w_crop_before = w_diff // 2
        w_interval = max_shape[1]
    else:
        w_crop_before = 0
        w_interval = im.shape[-2]

    return im[...,h_crop_before:h_crop_before+h_interval, w_crop_before:w_crop_before+w_interval, :]
    
    
def ifft2_np(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x.astype(np.complex64), axes=[-2, -1]), norm='ortho'), axes=[-2, -1]).astype(np.complex64)


def fft2_np(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x.astype(np.complex64), axes=[-2, -1]), norm='ortho'), axes=[-2, -1]).astype(np.complex64)


def complex_channel_first(x):
    assert x.shape[-1] == 2
    if len(x.shape) == 3:
        # Single-coil (H, W, 2) -> (2, H, W)
        x = x.permute(2, 0, 1)
    else:
        # Multi-coil (C, H, W, 2) -> (2, C, H, W)
        assert len(x.shape) == 4
        x = x.permute(3, 0, 1, 2)
    return x

def complex_channel_last(x):
    assert x.shape[0] == 2
    if len(x.shape) == 3:
        # Single-coil (2, H, W) -> (H, W, 2)
        x = x.permute(1, 2, 0)
    else:
        # Multi-coil (2, C, H, W) -> (C, H, W, 2)
        assert len(x.shape) == 4
        x = x.permute(1, 2, 3, 0)
    return x
