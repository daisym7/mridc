# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.core.classes.common as common_classes
from mridc.collections.segmentation.models.idslr_base import idslr_block

__all__ = ["IDSLR"]


class IDSLR(base_segmentation_models.BaseMRIJointReconstructionSegmentationModel, ABC):
    """
    Implementation of the Image domain Deep Structured Low-Rank network, as presented in Aniket Pramanik, \
    Xiaodong Wu, and Mathews Jacob.

    References
    ----------

    ..

        Aniket Pramanik, Xiaodong Wu, and Mathews Jacob. (2021) ‘Joint Calibrationless Reconstruction and \
        Segmentation of Parallel MRI’. Available at: https://arxiv.org/abs/2105.09220

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.input_channels = cfg_dict.get("input_channels", 2)
        if self.input_channels == 0:
            raise ValueError("Segmentation module input channels cannot be 0.")
        reconstruction_out_chans = cfg_dict.get("reconstruction_module_output_channels", 2)
        self.segmentation_out_chans = cfg_dict.get("segmentation_module_output_channels", 1)
        chans = cfg_dict.get("channels", 32)
        num_pools = cfg_dict.get("num_pools", 4)
        drop_prob = cfg_dict.get("drop_prob", 0.0)
        normalize = cfg_dict.get("normalize", True)
        padding = cfg_dict.get("padding", True)
        padding_size = cfg_dict.get("padding_size", 11)
        self.norm_groups = cfg_dict.get("norm_groups", 2)
        self.num_iters = cfg_dict.get("num_iters", 5)

        self.reconstruction_encoder = idslr_block.UnetEncoder(
            chans=chans,
            num_pools=num_pools,
            in_chans=self.input_channels,
            drop_prob=drop_prob,
            normalize=normalize,
            padding=padding,
            padding_size=padding_size,
            norm_groups=self.norm_groups,
        )
        self.reconstruction_decoder = idslr_block.UnetDecoder(
            chans=chans,
            num_pools=num_pools,
            out_chans=reconstruction_out_chans,
            drop_prob=drop_prob,
            normalize=normalize,
            padding=padding,
            padding_size=padding_size,
            norm_groups=self.norm_groups,
        )
        self.segmentation_decoder = idslr_block.UnetDecoder(
            chans=chans,
            num_pools=num_pools,
            out_chans=self.segmentation_out_chans,
            drop_prob=drop_prob,
            normalize=normalize,
            padding=padding,
            padding_size=padding_size,
            norm_groups=self.norm_groups,
        )

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)
        self.magnitude_input = cfg_dict.get("magnitude_input", True)
        self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

        self.dc = idslr_block.DC()

        self.reconstruction_module_accumulate_estimates = False

    @common_classes.typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
    ) -> Tuple[Union[List, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the network.

        Parameters
        ----------
        y: Undersampled k-space data.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sub-sampling mask.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        init_reconstruction_pred: Initial reconstruction prediction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        target_reconstruction: Target reconstruction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        hx: Hidden state of the RNN.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        sigma: Noise standard deviation.
            float

        Returns
        -------
        pred_reconstruction: Predicted reconstruction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred_segmentation: Predicted segmentation.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
        """
        if self.consecutive_slices > 1:
            batch, slices = y.shape[:2]
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])  # type: ignore
            sensitivity_maps = sensitivity_maps.reshape(  # type: ignore
                sensitivity_maps.shape[0] * sensitivity_maps.shape[1], *sensitivity_maps.shape[2:]  # type: ignore
            )  # type: ignore
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])  # type: ignore

        # In case of deviating number of coils, we need to pad up to maximum number of coils == number of input \
        # channels for the reconstruction module
        num_coils = y.shape[1]
        if num_coils * 2 != self.input_channels:
            num_coils_to_add = (self.input_channels - num_coils * 2) // 2
            dummy_coil_data = torch.zeros_like(torch.movedim(y, self.coil_dim, 0)[0]).unsqueeze(self.coil_dim)
            for _ in range(num_coils_to_add):
                y = torch.cat([y, dummy_coil_data], dim=self.coil_dim)
                sensitivity_maps = torch.cat([sensitivity_maps, dummy_coil_data], dim=self.coil_dim)

        y_prediction = y.clone()
        for _ in range(self.num_iters):
            init_reconstruction_pred = fft.ifft2(
                y_prediction, self.fft_centered, self.fft_normalization, self.spatial_dims
            )
            output = self.reconstruction_encoder(init_reconstruction_pred)
            reconstruction_encoder_prediction, iscomplex, padding_size, _, _ = (
                output[0].copy(),
                output[1],
                output[2],
                output[3],
                output[4],
            )
            reconstruction_decoder_prediction = self.reconstruction_decoder(*output)
            reconstruction_decoder_prediction = reconstruction_decoder_prediction + init_reconstruction_pred
            reconstruction_decoder_prediction_kspace = fft.fft2(
                reconstruction_decoder_prediction, self.fft_centered, self.fft_normalization, self.spatial_dims
            )
            y_prediction = self.dc(reconstruction_decoder_prediction_kspace, y, mask)

        pred_reconstruction = self.process_intermediate_pred(
            y_prediction, sensitivity_maps, target_reconstruction, True
        )

        with torch.no_grad():
            pred_segmentation_input = [
                torch.nn.functional.group_norm(x, num_groups=self.norm_groups)
                for x in reconstruction_encoder_prediction
            ]
            if self.magnitude_input:
                pred_segmentation_input = [torch.abs(x) for x in pred_segmentation_input]

        pred_segmentation = self.segmentation_decoder(pred_segmentation_input, iscomplex=False, pad_sizes=padding_size)
        pred_segmentation = self.process_final_segmentation(pred_segmentation)

        if self.consecutive_slices > 1:
            pred_reconstruction = pred_reconstruction.view([batch, slices, *pred_reconstruction.shape[1:]])
            pred_segmentation = pred_segmentation.view([batch, slices, *pred_segmentation.shape[1:]])

        return pred_reconstruction, pred_segmentation

    def process_intermediate_pred(self, pred, sensitivity_maps, target, do_coil_combination=False):
        """
        Process the intermediate prediction.

        Parameters
        ----------
        pred: Intermediate prediction.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        target: Target data to crop to size.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        do_coil_combination: Whether to do coil combination.
            bool, default False

        Returns
        -------
        pred: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
        # Take the last time step of the eta
        if do_coil_combination:
            pred = fft.ifft2(
                pred, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
            )
            pred = utils.coil_combination(
                pred, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
            )
        pred = torch.view_as_complex(pred)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, pred = utils.center_crop_to_smallest(target, pred)
        return pred

    def process_final_segmentation(self, pred):
        """Process the final segmentation prediction."""
        if pred.shape[-1] == 2:
            pred = torch.view_as_complex(pred)
        if pred.shape[1] != self.segmentation_out_chans and pred.shape[1] != 2 and pred.dim() == 5:
            pred = pred.squeeze(1)
        if pred.shape[1] != self.segmentation_out_chans:
            pred = pred.permute(0, 3, 1, 2)
        pred = torch.abs(pred)
        if self.normalize_segmentation_output:
            pred = pred / torch.max(pred)
        return pred
