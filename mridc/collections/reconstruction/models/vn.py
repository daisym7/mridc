# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

import mridc.collections.common.losses.ssim as losses
import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.models.base as base_models
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.collections.reconstruction.models.varnet.vn_block as vn_block
import mridc.core.classes.common as common_classes

__all__ = ["VarNet"]


class VarNet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the End-to-end Variational Network (VN), as presented in Sriram, A. et al.

    References
    ----------

    ..

        Sriram, A. et al. (2020) ‘End-to-End Variational Networks for Accelerated MRI Reconstruction’. Available \
        at: https://github.com/facebookresearch/fastMRI.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.num_cascades = cfg_dict.get("num_cascades")

        # Cascades of VN blocks
        self.cascades = torch.nn.ModuleList(
            [
                vn_block.VarNetBlock(
                    unet_block.NormUnet(
                        chans=cfg_dict.get("channels"),
                        num_pools=cfg_dict.get("pooling_layers"),
                        padding_size=cfg_dict.get("padding_size"),
                        normalize=cfg_dict.get("normalize"),
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    no_dc=self.no_dc,
                )
                for _ in range(self.num_cascades)
            ]
        )

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        # initialize weights if not using pretrained vn
        # TODO if not cfg_dict.get("pretrained", False)

        if cfg_dict.get("train_loss_fn") == "ssim":
            self.train_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("train_loss_fn") == "l1":
            self.train_loss_fn = L1Loss()
        elif cfg_dict.get("train_loss_fn") == "mse":
            self.train_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("train_loss_fn")))
        if cfg_dict.get("val_loss_fn") == "ssim":
            self.val_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("val_loss_fn") == "l1":
            self.val_loss_fn = L1Loss()
        elif cfg_dict.get("val_loss_fn") == "mse":
            self.val_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("val_loss_fn")))

        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.accumulate_estimates = False

    @common_classes.typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        target: Target data to compute the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        pred: list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or  torch.Tensor, shape [batch_size, n_x, n_y, 2]
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        estimation = y.clone()

        for cascade in self.cascades:
            # Forward pass through the cascades
            estimation = cascade(estimation, y, sensitivity_maps, mask)

        estimation = fft.ifft2(
            estimation,
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        estimation = utils.coil_combination(
            estimation, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
        )
        estimation = torch.view_as_complex(estimation)
        _, estimation = utils.center_crop_to_smallest(target, estimation)
        return estimation
