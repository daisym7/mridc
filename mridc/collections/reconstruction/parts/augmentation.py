"""Code adapted from the MRAugment framework: https://github.com/z-fabian/MRAugment
from the paper: Data augmentation for deep learning based accelerated MRI reconstruction with limited data"""

import numpy as np
import torchvision.transforms.functional as TF
import mridc.collections.reconstruction.parts.helpers as helpers
import torch


class AugmentationPipeline:
    """
    Describes the transformations applied to MRI data and handles
    augmentation probabilities including generating random parameters for
    each augmentation.
    """

    def __init__(self):
        self.aug_max_rotation = 180
        self.aug_max_shearing_x = 15
        self.aug_max_shearing_y = 15
        self.aug_max_scaling = 0.25
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im, sens_map):
        # Trailing dims must be image height and width (for torchvision)
        im = helpers.complex_channel_first(im)
        sens_map = helpers.complex_channel_first(sens_map)

        # ---------------------------
        # pixel preserving transforms
        # ---------------------------
        # Horizontal flip
        bin, func = self.random_apply()
        # print(func)
        if bin:
            if 'fliph' in func:
                # print("FLIPH")
                im = TF.hflip(im)
                sens_map = TF.hflip(sens_map)

            # Vertical flip
            if 'flipv' in func:
                # print("FLIPV")
                im = TF.vflip(im)
                sens_map = TF.vflip(sens_map)

            # Rotation by multiples of 90 deg
            if 'rot90' in func:
                # print('ROT90')
                k = self.rng.randint(1, 4)
                im = torch.rot90(im, k, dims=[-2, -1])
                sens_map = torch.rot90(sens_map, k, dims=[-2, -1])

            # # Translation by integer number of pixels
            if 'translation' in func:
                # print("TRANSLATION")
                h, w = im.shape[-2:]
                t_x = self.rng.uniform(-0.125, 0.125)
                t_x = int(t_x * h)
                t_y = self.rng.uniform(-0.125, 0.125)
                t_y = int(t_y * w)
                # MRI image
                pad, top, left = self._get_translate_padding_and_crop(im, (t_x, t_y))
                im = TF.pad(im, padding=pad, padding_mode='reflect')
                im = TF.crop(im, top, left, h, w)
                # Sensitivity map
                pad2, top2, left2 = self._get_translate_padding_and_crop(sens_map, (t_x, t_y))
                sens_map = TF.pad(sens_map, padding=pad2, padding_mode='reflect')
                sens_map = TF.crop(sens_map, top2, left2, h, w)

            # ------------------------
            # interpolating transforms
            # ------------------------
            interp = False

            # Rotation
            if 'rotation' in func:
                # print("Rotation")
                interp = True
                rot = self.rng.uniform(-self.aug_max_rotation, self.aug_max_rotation)
            else:
                rot = 0.

            # Shearing
            if 'shearing' in func:
                # print("SHEARING")
                interp = True
                shear_x = self.rng.uniform(-self.aug_max_shearing_x, self.aug_max_shearing_x)
                shear_y = self.rng.uniform(-self.aug_max_shearing_y, self.aug_max_shearing_y)
            else:
                shear_x, shear_y = 0., 0.

            # Scaling
            if 'scaling' in func:
                # print("Scaling")
                interp = True
                scale = self.rng.uniform(1 - self.aug_max_scaling, 1 + self.aug_max_scaling)
            else:
                scale = 1.

            # Apply interpolating transformations
            # Affine transform - if any of the affine transforms is randomly picked

            if interp:
                h, w = im.shape[-2:]
                # MRI image
                pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
                im = TF.pad(im, padding=pad, padding_mode='reflect')
                im = TF.affine(im, angle=rot, scale=scale, shear=(shear_x, shear_y), translate=[0, 0],
                               interpolation=TF.InterpolationMode.BILINEAR)
                im = TF.center_crop(im, (h, w))
                # Sensitivity map
                pad2 = self._get_affine_padding_size(sens_map, rot, scale, (shear_x, shear_y))
                sens_map = TF.pad(sens_map, padding=pad2, padding_mode='reflect')
                sens_map = TF.affine(sens_map, angle=rot, scale=scale, shear=(shear_x, shear_y), translate=[0, 0],
                               interpolation=TF.InterpolationMode.BILINEAR)
                sens_map = TF.center_crop(sens_map, (h, w))

        # Reset original channel ordering
        im = helpers.complex_channel_last(im).contiguous()
        sens_map = helpers.complex_channel_last(sens_map).contiguous()
        return im, sens_map

    def augment_from_kspace(self, kspace, sens_map):
        im = helpers.ifft2c(kspace)
        im, sens_map = self.augment_image(im, sens_map)
        target = self.im_to_target(im)
        kspace = helpers.fft2c(im)
        target = torch.view_as_complex(target)
        return kspace, target, sens_map

    def im_to_target(self, im):
        # add sense?
        # RSS
        target = np.sqrt((np.abs(im) ** 2).sum(0))
        return target

    def random_apply(self):
        if self.rng.uniform() < self.augmentation_strength:
            return True, np.random.choice(['fliph', 'flipv', 'rot90', 'translation', 'rotation', 'shearing', 'scaling'],\
                          size=2, replace=False)
        else:
            return False, []

    def set_augmentation_strength(self, p):
        self.augmentation_strength = p

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """
        Calculates the necessary padding size before applying the
        general affine transformation. The output image size is determined based on the
        input image size and the affine transformation matrix.
        """
        h, w = im.shape[-2:]
        corners = [
            [-h / 2, -w / 2, 1.],
            [-h / 2, w / 2, 1.],
            [h / 2, w / 2, 1.],
            [h / 2, -w / 2, 1.]
        ]
        mx = torch.tensor(
            TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2, 3)
        corners = torch.cat([torch.tensor(c).reshape(3, 1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h - 1)
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2), min=0.0, max=w - 1)
        return int(py.item()), int(px.item())

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        t_x, t_y = translation
        h, w = im.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1)  # pad bottom
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1)  # pad top
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1)  # pad left
            left = 0
        else:
            pad[2] = min(-t_y, w - 1)  # pad right
            left = pad[2]
        return pad, top, left


class DataAugmentor:
    """
    High-level class encompassing the augmentation pipeline and augmentation
    probability scheduling. A DataAugmentor instance can be initialized in the
    main training code and passed to the DataTransform to be applied
    to the training data.
    """

    def __init__(self, current_epoch_fn):
        """
        hparams: refer to the arguments below in add_augmentation_specific_args
        current_epoch_fn: this function has to return the current epoch as an integer
        and is used to schedule the augmentation probability.
        """
        self.current_epoch_fn = current_epoch_fn
        self.augmentation_pipeline = AugmentationPipeline()
        self.aug_on = True

    def __call__(self, kspace, target, sens_map, epoch):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [C, H, W, 2] (multi-coil) or [H, W, 2]
            where last dim is for real/imaginary channels
        target_size: [H, W] shape of the generated augmented target
        """
        # Set augmentation probability
        self.current_epoch_fn = epoch
        if self.aug_on:
            p = self.schedule_p()
            self.augmentation_pipeline.set_augmentation_strength(p)
        else:
            p = 0.0

        # Augment if needed
        if self.aug_on and p > 0.0:
            kspace, target, sens_map = self.augmentation_pipeline.augment_from_kspace(kspace, sens_map)
        return kspace, target, sens_map

    def schedule_p(self):
        D = 50
        T = 100
        t = self.current_epoch_fn
        p_max = 0.55
        if t < D:
            return 0.0
        else:
            p = (p_max / (1 - np.exp(-5))) * (1 - np.exp((-5 * t) / T))
            return p
