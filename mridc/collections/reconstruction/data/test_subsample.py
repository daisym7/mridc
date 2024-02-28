# coding=utf-8
# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI
__author__ = "Dimitrios Karkalousos"

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numba as nb
import numpy as np
import torch

import matplotlib.pyplot as plt
import Esaote_powerlaw_mask as Esaote_func


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    """
    Temporarily sets the seed of the given random number generator.

    Parameters
    ----------
    rng: The random number generator.
    seed: The seed to set.

    Returns
    -------
    A context manager.
    """
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """A class that defines a mask function."""

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Initialize the mask function.

        Parameters
        ----------
        center_fractions: Fraction of low-frequency columns to be retained. If multiple values are provided, then \
        one of these numbers is chosen uniformly each time. For 2D setting this value corresponds to setting the \
        Full-Width-Half-Maximum.
        accelerations: Amount of under-sampling. This should have the same length as center_fractions. If multiple \
        values are provided, then one of these is chosen uniformly each time.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError("Number of center fractions should match number of accelerations")

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """

        Parameters
        ----------
        shape: Shape of the input tensor.
        seed: Seed for the random number generator.
        half_scan_percentage: Percentage of the low-frequency columns to be retained.
        scale: Scale of the mask.

        Returns
        -------
        A tuple of the mask and the number of low-frequency columns retained.
        """
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N columns, the mask \
    picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a probability equal to: \
        prob = (N / acceleration - N_low_freqs) /  (N - N_low_freqs). This ensures that the expected number of \
        columns selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which case one possible (center_fraction, \
    acceleration) is chosen uniformly at random each time the RandomMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there is a 50% probability that \
    4-fold acceleration with 8% center  fraction is selected and a 50% probability that 8-fold acceleration with 4% \
    center fraction is selected.
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time \
        for the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: Optional; Defines the scale of the center of the mask.

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            ##########################
            # print("CHECK", center_fraction, num_low_freqs)
            # print("CHECK ACCELERATION", acceleration)
            ###############################
            prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
            mask = self.rng.uniform(size=num_cols) < prob  # type: ignore
            pad = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad : pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

            #############################
            # print("CHECK MASK", mask.shape, mask[:,:,147:173,:])
            #########################################

        return mask, acceleration


class Equispaced1DMaskFunc(MaskFunc):
    """
    Equispaced1DMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N columns, the mask \
    picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion that reaches the desired acceleration \
        rate taking into consideration the number of low frequencies. This ensures that the expected number of \
        columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible (center_fraction, \
    acceleration) is chosen uniformly at random each time the Equispaced1DMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in \
    https://github.com/facebookresearch/fastMRI/issues/54), which will require modifications to standard GRAPPA \
    approaches. Nonetheless, this aspect of the function has been preserved to match the public multicoil data.
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time for \
        the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: Optional; Defines the scale of the center of the mask.

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad : pad + num_low_freqs] = True  # type: ignore

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask, acceleration


class Equispaced2DMaskFunc(MaskFunc):
    """Same as Equispaced1DMaskFunc, but for 2D k-space data."""

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time for \
        the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: Optional; Defines the scale of the center of the mask.

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()

            acceleration = acceleration / 2
            center_fraction = center_fraction / 2

            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            num_rows = shape[-3]
            num_high_freqs = int(round(num_rows * center_fraction))

            # create the mask
            mask = np.zeros([num_rows, num_cols], dtype=np.float32)

            pad_cols = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            pad_rows = torch.div((num_rows - num_high_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad_rows : pad_rows + num_high_freqs, pad_cols : pad_cols + num_low_freqs] = True  # type: ignore

            for i in np.arange(0, num_rows, acceleration):
                for j in np.arange(0, num_cols, acceleration):
                    mask[int(i), int(j)] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask_shape[-3] = num_rows
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask, acceleration * 2


class Powerlaw1DMaskFunc(MaskFunc):
    def __call__(
            self,
            shape: Union[Sequence[int], np.ndarray],
            seed: Optional[Union[int, Tuple[int, ...]]] = None,
            half_scan_percentage: Optional[float] = 0.0,
            scale: Optional[float] = 0.1,
    ) -> Tuple[torch.Tensor, int]:
        dims = [1 for _ in shape]
        self.shape = tuple(shape[-3:-1])
        dims[-2] = self.shape[-1]

        center_fraction, wanted_acc = self.choose_acceleration()
        phase_lines = shape[-2]
        center_lines = round(phase_lines * center_fraction)
        undersample_lines = phase_lines - center_lines
        if undersample_lines % 2 != 0:
            center_lines += 1
        # get part that needs undersampling, so leave out center lines
        n = int((phase_lines - center_lines) / 2)
        x = np.linspace(0, 1, n)
        y = x ** (-scale)
        y[np.isinf(y)] = np.max(y[~np.isinf(y)])
        y = y / np.max(y)
        y_reversed = list(reversed(y))
        y = np.hstack((y_reversed, y))

        # compute offset to reach desired acceleration factor
        b = (n * 2 / wanted_acc - np.sum(y)) / (n * 2 - np.sum(y))
        z = b + (1 - b) * y

        # plot powerlaw
        plt.figure()
        plt.plot(z)
        plt.ylim([0, 1.2])
        plt.show()

        count = 0
        acc_realization = 0
        while np.abs(acc_realization - wanted_acc) > 0.05:
            mask1 = np.where(np.random.rand(1, n * 2) < z, 1, 0)[0]
            mask2 = np.ones(phase_lines)
            mask2[:n] = mask1[:n]
            mask2[(n+center_lines):] = mask1[n:]
            acc_realization = phase_lines / np.sum(mask2)
            count += 1
            if count > 1000:
                raise ValueError(f'Generating mask failed in while loop. Try again.')
        # print("Zo lang in While loop:", count)
        acc_realization = phase_lines / np.sum(mask2)
        return torch.from_numpy(mask2.reshape(dims).astype(np.float32)), acceleration


class Powerlaw1D_Esaote_MaskFunc(MaskFunc):
    def __call__(
            self,
            shape: Union[Sequence[int], np.ndarray],
            seed: Optional[Union[int, Tuple[int, ...]]] = None,
            half_scan_percentage: Optional[float] = 0.0,
            scale: Optional[float] = 0.1,
    ) -> Tuple[torch.Tensor, int]:
        dims = [1 for _ in shape]
        self.shape = tuple(shape[-3:-1])
        dims[-2] = self.shape[-1]

        _, acceleration = self.choose_acceleration()
        print(self.shape)
        number_lines = int(self.shape[-1]/acceleration)
        # how many iterations for the point spread function
        iterations = 1000
        # tolerance of not getting the exact number of lines
        tolerance = 0.01
        # tolerance for asymmetry
        AsymTolerance = 1
        # get probability density function of powerlaw
        pdf = Esaote_func.GenPDF(self.shape[-1], acceleration, number_lines)
        # create mask using the pdf
        mask = Esaote_func.GenMask(pdf, iterations, tolerance, number_lines, AsymTolerance)
        return torch.from_numpy(mask.reshape(dims).astype(np.float32)), acceleration


class Gaussian1DMaskFunc(MaskFunc):
    """
    Creates a 1D sub-sampling mask of a given shape.

    For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse of \
    which the half-axes will set to the set scale % of the fully sampled region. The remaining points will be sampled \
    according to a Gaussian distribution.

    The center fractions here act as Full-Width at Half-Maximum (FWHM) values.
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time \
        for the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: For autocalibration purposes, data points near the k-space center will be fully sampled within an \
        ellipse of which the half-axes will set to the set scale % of the fully sampled region

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        # print(shape)
        # print(scale)
        dims = [1 for _ in shape]
        self.shape = tuple(shape[-3:-1])
        dims[-2] = self.shape[-1]
        full_width_half_maximum, acceleration = self.choose_acceleration()

        # Why make it into a list?
        # if not isinstance(full_width_half_maximum, list):
        #     full_width_half_maximum = [full_width_half_maximum] * 2

        self.full_width_half_maximum = full_width_half_maximum
        self.acceleration = acceleration
        self.scale = scale

        count = 0
        acc_realization = 100
        # if acceleration > 3:
        #     wanted_acc = 3
        # else:
        wanted_acc = acceleration
        while np.abs(acc_realization - wanted_acc) > 0.05:
            count += 1
            mask = self.gaussian_kspace()
            mask[tuple(self.gaussian_coordinates())] = 1.0

            # why is this shift here?
            # mask = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(mask, axes=0), axes=0), axes=(0, 1))

            if half_scan_percentage != 0:
                mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

            acc_realization = len(mask[0]) / np.sum(mask[0])
            print(acc_realization)
            # print(acc_realization)
            # print(acc_realization)
            if count > 1000:
                raise ValueError(f'Generating mask failed in while loop. Try again.')
        return torch.from_numpy(mask[0].reshape(dims).astype(np.float32)), acceleration

    def gaussian_kspace(self):
        """Creates a Gaussian sampled k-space center."""
        scaled = int(self.shape[1] * self.scale)
        center = np.ones((scaled, self.shape[0]))
        top_scaled = torch.div((self.shape[1] - scaled), 2, rounding_mode="trunc").item()
        bottom_scaled = self.shape[1] - scaled - top_scaled
        top = np.zeros((top_scaled, self.shape[0]))
        btm = np.zeros((bottom_scaled, self.shape[0]))
        return np.concatenate((top, center, btm)).T

    def gaussian_coordinates(self):
        """Creates a Gaussian sampled k-space coordinates."""
        n_sample = int(self.shape[1] / self.acceleration)
        kernel = self.gaussian_kernel()
        idxs = np.random.choice(range(self.shape[1]), size=n_sample, replace=False, p=kernel)
        ysamples = np.concatenate([np.tile(i, self.shape[0]) for i in idxs])
        xsamples = np.concatenate([range(self.shape[0]) for _ in idxs])
        return xsamples, ysamples

    def gaussian_kernel(self):
        """Creates a Gaussian sampled k-space kernel."""
        # removed the for loop with a break after one loop, why?
        sigma = self.full_width_half_maximum / np.sqrt(8 * np.log(2))
        x = np.linspace(-1.0, 1.0, self.shape[-1])
        kernel = np.exp(-(x**2 / (2 * sigma**2)))
        kernel = kernel / kernel.sum()
        return kernel


# class Gaussian1DMaskFunc(MaskFunc):
#     """
#     Creates a 1D sub-sampling mask of a given shape.
#
#     For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse of \
#     which the half-axes will set to the set scale % of the fully sampled region. The remaining points will be sampled \
#     according to a Gaussian distribution.
#
#     The center fractions here act as Full-Width at Half-Maximum (FWHM) values.
#     """
#
#     def __call__(
#         self,
#         shape: Union[Sequence[int], np.ndarray],
#         seed: Optional[Union[int, Tuple[int, ...]]] = None,
#         half_scan_percentage: Optional[float] = 0.0,
#         scale: Optional[float] = 0.02,
#     ) -> Tuple[torch.Tensor, int]:
#         """
#         Parameters
#         ----------
#         shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
#         along the second last dimension.
#         seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time \
#         for the same shape. The random state is reset afterwards.
#         half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
#         scale: For autocalibration purposes, data points near the k-space center will be fully sampled within an \
#         ellipse of which the half-axes will set to the set scale % of the fully sampled region
#
#         Returns
#         -------
#         A tuple of the mask and the number of columns selected.
#         """
#         dims = [1 for _ in shape]
#         self.shape = tuple(shape[-3:-1])
#         # self.shape = tuple(shape[-1])
#         dims[-2] = self.shape[-1]
#         full_width_half_maximum, acceleration = self.choose_acceleration()
#         if not isinstance(full_width_half_maximum, list):
#             full_width_half_maximum = [full_width_half_maximum] * 2
#         self.full_width_half_maximum = full_width_half_maximum
#         self.acceleration = acceleration
#
#         self.scale = scale
#
#         mask = self.gaussian_kspace()
#         ###########################
#         plt.imshow(mask, cmap='gray')
#         plt.show()
#         # print(mask.shape)
#         # print(len(self.gaussian_coordinates()[0]))
#         # print(np.unique(self.gaussian_coordinates()[0]))
#         # print(len(self.gaussian_coordinates()[1]))
#         # print(np.unique(self.gaussian_coordinates()[1]))
#         ############################
#
#         mask[tuple(self.gaussian_coordinates())] = 1.0
#         ##############
#         print("HIER 2")
#
#         plt.imshow(mask, cmap='gray')
#         plt.show()
#         ###################
#         # mask = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(mask, axes=0), axes=0), axes=(0, 1))
#
#         ################
#         plt.imshow(mask, cmap='gray')
#         plt.show()
#         ################
#
#         if half_scan_percentage != 0:
#             mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0
#
#         ###############################
#         # plt.imshow(mask, cmap='gray')
#         # plt.show()
#         # print("SHAPE", mask.shape)
#         # print("CHECK", center_fraction, num_low_freqs)
#         # print("CHECK ACCELERATION", acceleration)
#         print(torch.from_numpy(mask[0].reshape(dims).astype(np.float32)))
#         # mask2 = np.ones(kspace[0, :, :, 0].shape) * np.array(np.squeeze(mask[0]))
#         # print("HIER")
#         # mask = torch.from_numpy(mask[0].reshape(dims).astype(np.float32))
#         # mask2 = np.ones(kspace[0, :, :, 0].shape) * np.array(np.squeeze(mask))
#         # plt.imshow(mask, cmap='gray')
#         # plt.show()
#         #################################
#         return torch.from_numpy(mask[0].reshape(dims).astype(np.float32)), acceleration
#
#     def gaussian_kspace(self):
#         """Creates a Gaussian sampled k-space center."""
#         scaled = int(self.shape[0] * self.scale)
#         center = np.ones((scaled, self.shape[1]))
#         top_scaled = torch.div((self.shape[0] - scaled), 2, rounding_mode="trunc").item()
#         bottom_scaled = self.shape[0] - scaled - top_scaled
#         top = np.zeros((top_scaled, self.shape[1]))
#         btm = np.zeros((bottom_scaled, self.shape[1]))
#         return np.concatenate((top, center, btm))
#
#     def gaussian_coordinates(self):
#         """Creates a Gaussian sampled k-space coordinates."""
#         n_sample = int(self.shape[0] / self.acceleration)
#         kernel = self.gaussian_kernel()
#         idxs = np.random.choice(range(self.shape[0]), size=n_sample, replace=False, p=kernel)
#         xsamples = np.concatenate([np.tile(i, self.shape[1]) for i in idxs])
#         ysamples = np.concatenate([range(self.shape[1]) for _ in idxs])
#         return xsamples, ysamples
#
#     def gaussian_kernel(self):
#         """Creates a Gaussian sampled k-space kernel."""
#         kernel = 1
#         for fwhm, kern_len in zip(self.full_width_half_maximum, self.shape):
#             sigma = fwhm / np.sqrt(8 * np.log(2))
#             x = np.linspace(-1.0, 1.0, kern_len)
#             g = np.exp(-(x**2 / (2 * sigma**2)))
#             kernel = g
#             break
#         kernel = kernel / kernel.sum()
#         print("Kernel", kernel)
#         return kernel


class Gaussian2DMaskFunc(MaskFunc):
    """
    Creates a 2D sub-sampling mask of a given shape.

    For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse of \
    which the half-axes will set to the set scale % of the fully sampled region. The remaining points will be sampled \
    according to a Gaussian distribution.

    The center fractions here act as Full-Width at Half-Maximum (FWHM) values.
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time for \
         the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: For autocalibration purposes, data points near the k-space center will be fully sampled within an \
        ellipse of which the half-axes will set to the set scale % of the fully sampled region

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        dims = [1 for _ in shape]
        self.shape = tuple(shape[-3:-1])
        dims[-3:-1] = self.shape

        full_width_half_maximum, acceleration = self.choose_acceleration()

        if not isinstance(full_width_half_maximum, list):
            full_width_half_maximum = [full_width_half_maximum] * 2
        self.full_width_half_maximum = full_width_half_maximum

        self.acceleration = acceleration
        self.scale = scale

        mask = self.gaussian_kspace()
        mask[tuple(self.gaussian_coordinates())] = 1.0

        if half_scan_percentage != 0:
            mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

        return torch.from_numpy(mask.reshape(dims).astype(np.float32)), acceleration

    def gaussian_kspace(self):
        """Creates a Gaussian sampled k-space center."""
        a, b = self.scale * self.shape[0], self.scale * self.shape[1]
        afocal, bfocal = self.shape[0] / 2, self.shape[1] / 2
        xx, yy = np.mgrid[: self.shape[0], : self.shape[1]]
        ellipse = np.power((xx - afocal) / a, 2) + np.power((yy - bfocal) / b, 2)
        return (ellipse < 1).astype(float)

    def gaussian_coordinates(self):
        """Creates a Gaussian sampled k-space coordinates."""
        n_sample = int(self.shape[0] * self.shape[1] / self.acceleration)
        cartesian_prod = list(np.ndindex(self.shape))  # type: ignore
        kernel = self.gaussian_kernel()
        idxs = np.random.choice(range(len(cartesian_prod)), size=n_sample, replace=False, p=kernel.flatten())
        return list(zip(*list(map(cartesian_prod.__getitem__, idxs))))

    def gaussian_kernel(self):
        """Creates a Gaussian kernel."""
        kernels = []
        for fwhm, kern_len in zip(self.full_width_half_maximum, self.shape):
            sigma = fwhm / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x**2 / (2 * sigma**2)))
            kernels.append(g)
        kernel = np.sqrt(np.outer(kernels[0], kernels[1]))
        kernel = kernel / kernel.sum()
        return kernel


class Poisson2DMaskFunc(MaskFunc):
    """
    Creates a 2D sub-sampling mask of a given shape.

    # Taken and adapted from: https://github.com/mikgroup/sigpy/blob/master/sigpy/mri/samp.py
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
        calib: Optional[Tuple[float, float]] = (0.0, 0.0),
        crop_corner: bool = True,
        max_attempts: int = 30,
        tol: float = 0.3,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time \
        for the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: For autocalibration purposes, data points near the k-space center will be fully sampled within an \
        ellipse of which the half-axes will set to the set scale % of the fully sampled region
        calib: Optional; Defines the size of the calibration region. The calibration region is a square region \
        in the center of k-space. The first value defines the percentage of the center that is sampled. The second \
        value defines the size of the calibration region in the center of k-space.
        crop_corner: Optional; If True, the center of the mask will be cropped to the size of the calibration region.
        max_attempts: Optional; Maximum number of attempts to generate a mask with the desired acceleration factor.
        tol: Optional; Tolerance for the acceleration factor.

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        self.shape = tuple(shape[-3:-1])
        self.scale = scale
        _, self.acceleration = self.choose_acceleration()

        ny, nx = self.shape
        y, x = np.mgrid[:ny, :nx]
        x = np.maximum(abs(x - self.shape[-1] / 2) - calib[-1] / 2, 0)  # type: ignore
        x /= x.max()
        y = np.maximum(abs(y - self.shape[-2] / 2) - calib[-2] / 2, 0)  # type: ignore
        y /= y.max()
        r = np.hypot(x, y)

        slope_max = max(nx, ny)
        slope_min = 0
        while slope_min < slope_max:
            slope = (slope_max + slope_min) / 2
            radius_x = np.clip((1 + r * slope) * nx / max(nx, ny), 1, None)
            radius_y = np.clip((1 + r * slope) * ny / max(nx, ny), 1, None)
            mask = self.generate_poisson_mask(self.shape[-1], self.shape[-2], max_attempts, radius_x, radius_y, calib)
            if crop_corner:
                mask *= r < 1

            with np.errstate(divide="ignore", invalid="ignore"):
                actual_acceleration = mask.size / np.sum(mask)

            if abs(actual_acceleration - self.acceleration) < tol:
                break
            if actual_acceleration < self.acceleration:
                slope_min = slope
            else:
                slope_max = slope

        pattern1 = mask
        pattern2 = self.centered_circle()
        mask = np.logical_or(pattern1, pattern2)

        if abs(actual_acceleration - self.acceleration) >= tol:
            raise ValueError(f"Cannot generate mask to satisfy acceleration factor of {self.acceleration}.")

        if half_scan_percentage != 0:
            mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

        return (
            torch.from_numpy(mask.reshape(self.shape).astype(np.float32)).unsqueeze(0).unsqueeze(-1),
            self.acceleration,
        )

    def centered_circle(self):
        """Creates a boolean centered circle image using the scale as a radius."""
        center_x = int((self.shape[0] - 1) / 2)
        center_y = int((self.shape[1] - 1) / 2)

        X, Y = np.indices(self.shape)
        radius = int(self.shape[0] * self.scale)
        return ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius**2

    @staticmethod
    @nb.jit(nopython=True, cache=True)  # pragma: no cover
    def generate_poisson_mask(
        nx: int,
        ny: int,
        max_attempts: int,
        radius_x: np.ndarray,
        radius_y: np.ndarray,
        calib: Tuple[float, float],
    ):
        """
        Generates a Poisson mask.

        Parameters
        ----------
        nx: Number of columns.
        ny: Number of rows.
        max_attempts: Maximum number of attempts to generate a mask with the desired acceleration factor.
        radius_x: Radius of the Poisson distribution in the x-direction.
        radius_y: Radius of the Poisson distribution in the y-direction.
        calib: Defines the size of the calibration region. The calibration region is a square region in the center of \
        k-space. The first value defines the percentage of the center that is sampled. The second value defines the \
        size of the calibration region in the center of k-space.
        """
        mask = np.zeros((ny, nx))

        # Add calibration region
        mask[
            int(ny / 2 - calib[-2] / 2) : int(ny / 2 + calib[-2] / 2),
            int(nx / 2 - calib[-1] / 2) : int(nx / 2 + calib[-1] / 2),
        ] = 1

        # initialize active list
        pxs = np.empty(nx * ny, np.int32)
        pys = np.empty(nx * ny, np.int32)
        pxs[0] = np.random.randint(0, nx)
        pys[0] = np.random.randint(0, ny)
        num_actives = 1
        while num_actives > 0:
            i = np.random.randint(0, num_actives)
            px = pxs[i]
            py = pys[i]
            rx = radius_x[py, px]
            ry = radius_y[py, px]

            # Attempt to generate point
            done = False
            k = 0
            while not done and k < max_attempts:
                # Generate point randomly from r and 2 * r
                v = (np.random.random() * 3 + 1) ** 0.5
                t = 2 * np.pi * np.random.random()
                qx = px + v * rx * np.cos(t)
                qy = py + v * ry * np.sin(t)

                # Reject if outside grid or close to other points
                if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
                    startx = max(int(qx - rx), 0)
                    endx = min(int(qx + rx + 1), nx)
                    starty = max(int(qy - ry), 0)
                    endy = min(int(qy + ry + 1), ny)

                    done = True
                    for x in range(startx, endx):
                        for y in range(starty, endy):
                            if mask[y, x] == 1 and (
                                ((qx - x) / radius_x[y, x]) ** 2 + ((qy - y) / (radius_y[y, x])) ** 2 < 1
                            ):
                                done = False
                                break

                k += 1

            # Add point if done else remove from active list
            if done:
                pxs[num_actives] = qx
                pys[num_actives] = qy
                mask[int(qy), int(qx)] = 1
                num_actives += 1
            else:
                pxs[i] = pxs[num_actives - 1]
                pys[i] = pys[num_actives - 1]
                num_actives -= 1

        return mask


class Gaussian1DMaskFunc2(MaskFunc):
    """Same as Gaussian2DMaskFunc, but for 1D k-space data.

    .. note::
        See ..class::`atommic.collections.common.data.subsample.Gaussian2DMaskFunc` for more details.

    Examples
    --------
    >>> import torch
    >>> from atommic.collections.common.data.subsample import Gaussian1DMaskFunc
    >>> mask_func = Gaussian1DMaskFunc(center_fractions=[0.7, 0.7], accelerations=[4, 8])
    >>> mask_func.choose_acceleration()
    (0.7, 4)
    >>> kspace = torch.randn(1, 1, 640, 368)
    >>> mask, acceleration = mask_func(kspace.shape)
    >>> mask.shape
    torch.Size([1, 1, 1, 368])
    >>> acceleration
    4
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        partial_fourier_percentage: Optional[float] = 0.0,
        center_scale: Optional[float] = 0.08,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Calls :class:`Gaussian1DMaskFunc`.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the mask to be created. The shape should have at least 3 dimensions. Same as the shape of the
            input k-space data.
        seed : int or tuple of ints, optional
            Seed for the random number generator. Default is ``None``.
        partial_fourier_percentage : float, optional
            Percentage of the low-frequency columns to be retained. Default is ``0.0``.
        center_scale : float, optional
            For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse
            of which the half-axes will be set to the given `center_scale` percentage of the fully sampled region.
            Default is ``0.02``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple of the generated mask and the acceleration factor.
        """
        with temp_seed(self.rng, seed):
            dims = [1 for _ in shape]
            self.shape = tuple(shape[-3:-1])
            self.shape = (self.shape[1], self.shape[0])
            dims[-2] = self.shape[-2]

            full_width_half_maximum, acceleration = self.choose_acceleration()

            self.full_width_half_maximum = full_width_half_maximum
            self.acceleration = acceleration
            self.center_scale = center_scale

            # don't get stuck in the while loop
            counter = 0
            found_acc = 1000
            while np.abs(found_acc - acceleration) > 0.02:
                mask = self.gaussian_kspace()
                mask[tuple(self.gaussian_coordinates())] = 1.0

                # mask = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(mask, axes=0), axes=0), axes=(0, 1))

                if partial_fourier_percentage != 0:
                    mask[:, : int(np.round(mask.shape[1] * partial_fourier_percentage))] = 0.0

                mask = torch.from_numpy(np.transpose(mask, (1, 0))[0].reshape(*dims).astype(np.float32))

                check_mask = np.squeeze(mask)
                found_acc = (len(check_mask)) / np.count_nonzero(check_mask)

                # don't get stuck in the while loop
                counter += 1
                if counter > 1000:
                    raise ValueError(f'Generating mask failed in while loop. Try again.')

        return mask, acceleration

    def gaussian_kspace(self) -> np.ndarray:
        """Creates a Gaussian sampled k-space center."""
        scaled = int(self.shape[0] * self.center_scale)
        center = np.ones((scaled, self.shape[1]))
        top_scaled = torch.div((self.shape[0] - scaled), 2, rounding_mode="trunc").item()
        bottom_scaled = self.shape[0] - scaled - top_scaled
        top = np.zeros((top_scaled, self.shape[1]))
        btm = np.zeros((bottom_scaled, self.shape[1]))
        return np.concatenate((top, center, btm))

    def gaussian_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Returns Gaussian sampled k-space coordinates.

        Returns
        -------
        xsamples : np.ndarray
            A 1D numpy array of x-coordinates.
        ysamples : np.ndarray
            A 1D numpy array of y-coordinates.

        Notes
        -----
        The number of samples taken is determined by `n_sample` which is calculated as
        `self.shape[0] / self.acceleration`. The selection of the samples is based on the probabilities calculated
        from `gaussian_kernel`.
        """
        n_sample = int(self.shape[0] / self.acceleration)
        idxs = np.random.choice(range(self.shape[0]), size=n_sample, replace=False, p=self.gaussian_kernel())
        xsamples = np.concatenate([np.tile(i, self.shape[1]) for i in idxs])
        ysamples = np.concatenate([range(self.shape[1]) for _ in idxs])
        return xsamples, ysamples

    def gaussian_kernel(self) -> np.ndarray:
        r"""Creates a Gaussian sampled k-space kernel.

        .. note::
            The function calculates the Gaussian kernel by computing the sum of the exponential of the squared \
            x-values divided by 2 times the square of the standard deviation. The standard deviation is calculated \
            from the full width at half maximum (FWHM) of the Gaussian curve and is defined as the FWHM divided by \
            the square root of 8 times the natural logarithm of 2. The FWHM and the kern_len are obtained from the \
            `full_width_half_maximum` and `shape` attributes of the class respectively.

        Returns
        -------
        ndarray
            The Gaussian kernel.
        """
        kernel = 1
        for kern_len in self.shape:
            sigma = self.full_width_half_maximum / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x**2 / (2 * sigma**2)))  # noqa: F841
            kernel = g
            break
        kernel = kernel / kernel.sum()  # type: ignore
        return kernel


def create_mask_for_mask_type(
    mask_type_str: str, center_fractions: Sequence[float], accelerations: Sequence[int]
) -> MaskFunc:
    """
    Creates a MaskFunc object for the given mask type.

    Parameters
    ----------
    mask_type_str: The string representation of the mask type.
    center_fractions: The center fractions for the mask.
    accelerations: The accelerations for the mask.

    Returns
    -------
    A MaskFunc object.
    """
    if mask_type_str == "random1d":
        return RandomMaskFunc(center_fractions, accelerations)
    if mask_type_str == "equispaced1d":
        return Equispaced1DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "equispaced2d":
        return Equispaced2DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "gaussian1d":
        return Gaussian1DMaskFunc(center_fractions, accelerations)
    # Check mask van Atommic
    if mask_type_str == "gaussian1d_atommic":
        return Gaussian1DMaskFunc2(center_fractions, accelerations)
    if mask_type_str == "gaussian2d":
        return Gaussian2DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "poisson2d":
        return Poisson2DMaskFunc(center_fractions, accelerations)
    if mask_type_str =="powerlaw1d":
        return Powerlaw1DMaskFunc(center_fractions, accelerations)
    if mask_type_str =="powerlaw1d_Esaote":
        return Powerlaw1D_Esaote_MaskFunc(center_fractions, accelerations)
    raise NotImplementedError(f"{mask_type_str} not supported")


def point_spread_function(masktype, shape, iterations, center_fraction, acceleration, scale_factor):
    # apply point spread function to find best mask (as least clustered lines as possible)
    mask_func = create_mask_for_mask_type(masktype, [center_fraction], [acceleration])
    lowest_value = 100
    for i in range(iterations):
        # get mask
        best_mask, best_acc = mask_func([1, shape[0], shape[1], 2], scale=scale_factor)
        # fft mask
        mask3 = np.squeeze(best_mask[0])
        # mask3 = best_mask
        a = np.fft.fftshift(np.fft.fft(mask3))
        b = np.abs(a)
        plt.plot(b)
        plt.show()
        # remove center
        center_frac = 0.08
        center_lines = int((len(b) * center_frac) / 2)
        center = np.argmax(b)
        b[center - center_lines: center] = 0
        b[center: center + center_lines] = 0
        plt.plot(b)
        plt.show()
        # get highest value other than center
        c = np.argsort(b)
        c = b[c]
        highest_value = c[-1]
        if highest_value < lowest_value:
            lowest_value = highest_value
            mask = best_mask
            acc = best_acc
    return mask, acc


def average_acceleration(masktype, center_fraction, acceleration, scale_factor):
    average = 0
    count = 0
    for i in range(1000):
        mask_func = create_mask_for_mask_type(masktype, [center_fraction], [acceleration])
        mask = mask_func([1, 192, 192, 2], scale=scale_factor)
        # # Acceleration based on the mask
        mask3 = np.squeeze(mask[0])
        a = len(mask3) / np.count_nonzero(mask3)
        average += a
        if a > 1.9:
            count += 1
    print(count)
    print("average a:", average/1000)
    return


def find_mask_with_right_acceleration(masktype, center_fraction, acceleration, scale_factor):
    for i in range(1000):
        found_acc = 100
        count = 0
        acceleration = 3
        while np.abs(found_acc - acceleration) > 0.05:
            # print(found_acc, acceleration)
            mask_func = create_mask_for_mask_type(masktype, [center_fraction], [acceleration])
            mask, acc = mask_func([1, 192, 192, 2], scale=scale_factor)
            mask3 = np.squeeze(mask)
            found_acc = len(mask3) / np.count_nonzero(mask3)
            # print(found_acc)
            count += 1
            if count > 5000:
                raise ValueError(f'Generating mask failed in while loop. Try again.')
    print("found acc: ", found_acc)
    print(count)
    return mask, found_acc


if __name__ == "__main__":
    masktype = "gaussian1d_atommic"
    center_fraction = 0.7
    acceleration = 3
    scale_factor = 0.02
    shape = (200, 200)

    average_acceleration(masktype, center_fraction, acceleration, scale_factor)

    # use point spread function to find the best mask
    iterations = 10
    #final_mask, acc = point_spread_function(masktype, shape, iterations, center_fraction, acceleration, scale_factor)

    mask_func = create_mask_for_mask_type(masktype, [center_fraction], [acceleration])
    final_mask, acc = mask_func([1, shape[0], shape[1], 2], scale=scale_factor)

    final_mask = np.squeeze(final_mask)
    # print(final_mask.shape)
    acc = (len(final_mask)) / np.count_nonzero(final_mask)
    print("Acceleration of mask", acc)
    mask6 = np.ones((shape[0], shape[1])) * np.array(final_mask)
    plt.imshow(mask6, cmap='gray')
    plt.show()

