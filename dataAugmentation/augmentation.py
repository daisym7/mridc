import random

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import helpers
import torch
import h5py
import os


class AugmentationPipeline:
    """
    Describes the transformations applied to MRI data and handles
    augmentation probabilities including generating random parameters for
    each augmentation.
    """

    def __init__(self, hparams):
        # self.hparams = hparams
        # self.upsample_augment = hparams.aug_upsample
        # self.upsample_factor = hparams.aug_upsample_factor
        # self.upsample_order = hparams.aug_upsample_order
        # self.transform_order = hparams.aug_interpolation_order
        self.aug_max_rotation = 180
        self.aug_max_shearing_x = 15
        self.aug_max_shearing_y = 15
        self.aug_max_scaling = 0.25
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im, sens_map, max_output_size=None):
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
            #
            # # Upsample if needed (adds heavy computation)
            # upsample = interp and self.upsample_augment
            # if upsample:
            #     upsampled_shape = [im.shape[-2] * self.upsample_factor, im.shape[-1] * self.upsample_factor]
            #     original_shape = im.shape[-2:]
            #     interpolation = TF.InterpolationMode.BICUBIC if self.upsample_order == 3 else TF.InterpolationMode.BILINEAR
            #     im = TF.resize(im, size=upsampled_shape, interpolation=interpolation)
            #
            # Apply interpolating transformations
            # Affine transform - if any of the affine transforms is randomly picked

            if interp:
                h, w = im.shape[-2:]
                # MRI image
                pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
                im = TF.pad(im, padding=pad, padding_mode='reflect')
                im = TF.affine(im,
                               angle=rot,
                               scale=scale,
                               shear=(shear_x, shear_y),
                               translate=[0, 0],
                               interpolation=TF.InterpolationMode.BILINEAR
                               )
                im = TF.center_crop(im, (h, w))
                # Sensitivity map
                pad2 = self._get_affine_padding_size(sens_map, rot, scale, (shear_x, shear_y))
                sens_map = TF.pad(sens_map, padding=pad2, padding_mode='reflect')
                sens_map = TF.affine(sens_map,
                               angle=rot,
                               scale=scale,
                               shear=(shear_x, shear_y),
                               translate=[0, 0],
                               interpolation=TF.InterpolationMode.BILINEAR
                               )
                sens_map = TF.center_crop(sens_map, (h, w))
            #
            # # ---------------------------------------------------------------------
            # # Apply additional interpolating augmentations here before downsampling
            # # ---------------------------------------------------------------------
            #
            # # Downsampling
            # if upsample:
            #     im = TF.resize(im, size=original_shape, interpolation=interpolation)

            # Final cropping if augmented image is too large
            # if max_output_size is not None:
            #     im = crop_if_needed(im, max_output_size)

            # Reset original channel ordering
            im = helpers.complex_channel_last(im).contiguous()
            sens_map = helpers.complex_channel_last(sens_map).contiguous()
        return im, sens_map

    def augment_from_kspace(self, kspace, sens_map, target_size, max_train_size=None):
        kspace = torch.tensor(kspace)
        kspace = torch.view_as_real(kspace)
        sens_map = torch.tensor(sens_map)
        sens_map = torch.view_as_real(sens_map)
        im = helpers.ifft2c(kspace)
        im, sens_map = self.augment_image(im, sens_map, max_output_size=max_train_size)
        target = self.im_to_target(im, target_size)
        kspace = helpers.fft2c(im)
        target = torch.view_as_complex(target)
        sens_map = torch.view_as_complex(sens_map)

        return kspace, target, sens_map

    def im_to_target(self, im, target_size):
        # Make sure target fits in the augmented image
        # cropped_size = [min(im.shape[-3], target_size[0]),
        #                 min(im.shape[-2], target_size[1])]

        # must be Multi-coil
        # assert len(im.shape) == 4
        # target = T.center_crop(helpers.rss_complex(im), cropped_size)
        # target = helpers.rss_complex(im)
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

    def __init__(self, hparams, current_epoch_fn):
        """
        hparams: refer to the arguments below in add_augmentation_specific_args
        current_epoch_fn: this function has to return the current epoch as an integer
        and is used to schedule the augmentation probability.
        """
        self.current_epoch_fn = current_epoch_fn
        self.hparams = hparams
        self.augmentation_pipeline = AugmentationPipeline(hparams)
        # self.max_train_resolution = hparams.max_train_resolution
        self.aug_on = True
        self.max_train_resolution = None

    def __call__(self, kspace, sens_map, target_size):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [C, H, W, 2] (multi-coil) or [H, W, 2]
            where last dim is for real/imaginary channels
        target_size: [H, W] shape of the generated augmented target
        """
        # Set augmentation probability
        if self.aug_on:
            p = self.schedule_p()
            p = 1
            self.augmentation_pipeline.set_augmentation_strength(p)
        else:
            p = 0.0

        # Augment if needed
        if self.aug_on and p > 0.0:
            kspace, target, sens_map = self.augmentation_pipeline.augment_from_kspace(kspace, sens_map,
                                                                                      target_size=target_size,
                                                                            max_train_size=self.max_train_resolution)
        # else:
        #     # Crop in image space if image is too large
        #     if self.max_train_resolution is not None:
        #         if kspace.shape[-3] > self.max_train_resolution[0] or kspace.shape[-2] > self.max_train_resolution[1]:
        #             im = ifft2c(kspace)
        #             im = complex_crop_if_needed(im, self.max_train_resolution)
        #             kspace = fft2c(im)
        return kspace, target, sens_map

    def schedule_p(self):
        D = 0
        T = 20
        t = self.current_epoch_fn
        p_max = 0.55
        if t < D:
            return 0.0
        else:
            p = (p_max / (1 - np.exp(-5))) * (1 - np.exp((-5 * t) / T))
            # if self.hparams.aug_schedule == 'constant':
            #     p = p_max
            # elif self.hparams.aug_schedule == 'ramp':
            #     p = (t - D) / (T - D) * p_max
            # elif self.hparams.aug_schedule == 'exp':
            #     c = self.hparams.aug_exp_decay / (T - D)  # Decay coefficient
            #     p = p_max / (1 - torch.exp(-(T - D) * c)) * (1 - torch.exp(-(t - D) * c))
            print("probability augmentation", p)
            return p


def show_augmented_image(original, augmented, sens_map, new_sens_map):
    # plots two sensitivity maps
    fig, axes = plt.subplots(2, 2)
    # MRI image
    axes[0,0].imshow(np.abs(original), cmap='gray')
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    axes[0,1].imshow(np.abs(augmented), cmap='gray')
    axes[0,1].set_title("Scaling")
    axes[0,1].axis('off')

    # Sensitivty maps
    axes[1,0].imshow(np.abs(sens_map[0]), cmap='gray')
    axes[1,0].set_title("Original Sensitivity map")
    axes[1,0].axis('off')
    axes[1,1].imshow(np.abs(new_sens_map[0]), cmap='gray')
    axes[1,1].set_title("Augmented Sensitivity map")
    axes[1,1].axis('off')
    plt.show()
    return


if __name__ == "__main__":
    datapath = '/scratch/dmvandenberg/Esaote Trainingset/VolledigeDataset/Dataset/val/'
    for root, _, files in os.walk(datapath):
        for file in files:
            # load in the data
            data = h5py.File(datapath + file)
            print("Keys", data.keys())
            kspace = data['kspace']
            sens_map = data['sensitivity_map']
            target = data['target']
            params = []
            augm = DataAugmentor(params, 10)
            slices = kspace.shape[0]
            # for i in range(slices):
            print("Next slice")
            i = 6
            new_kspace, new_target, new_sens_map = augm(kspace[i], sens_map[i], [])
            show_augmented_image(target[i], new_target, sens_map[i], new_sens_map)
