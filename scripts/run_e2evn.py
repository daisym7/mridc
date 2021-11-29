# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import gc
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from mridc import save_reconstructions
from mridc.data.mri_data import SliceDataset
from mridc.data.subsample import create_mask_for_mask_type
from mridc.data.transforms import center_crop_to_smallest, PhysicsInformedDataTransform
from mridc.nn.e2evn import VarNet
from scripts.train_cirim import build_optim

torch.backends.cudnn.benchmark = False


def load_model(
    checkpoint_file: str, num_cascades: int, pools: int, chans: int, fft_type: str, device: str
) -> Tuple[Any, Union[torch.nn.DataParallel, VarNet], torch.optim.Optimizer]:
    """
    Loads the model from the checkpoint file.

    Args:
        checkpoint_file: Path to the checkpoint file.
        num_cascades: Number of cascades.
        pools: Number of pools.
        chans: Number of channels.
        fft_type: Type of FFT.
        device: cuda or cpu

    Returns:
        Checkpoint, E2EVN model, optimizer
    """
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

    model = VarNet(
        num_cascades=num_cascades,  # number of cascades
        pools=pools,  # number of pools
        chans=chans,  # number of channels
        normalize=True,  # normalize input
        use_sens_net=False,  # use sensitivity maps
        sens_pools=4,  # number of pools in sensitivity maps
        sens_chans=8,  # number of channels in sensitivity maps
        sens_normalize=True,  # normalize sensitivity maps
        output_type="SENSE",  # output type
        fft_type=fft_type,  # FFT type
        no_dc=False,  # remove DC component
    ).to(device)

    arguments = checkpoint["args"]

    if arguments.data_parallel:
        model = torch.nn.DataParallel(model)  # type: ignore

    model.load_state_dict(checkpoint["model"])

    optimizer = build_optim(arguments, model.parameters())  # type: ignore
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint, model, optimizer


def run_e2evn(model: VarNet, data_loader: DataLoader, device: str, progress_bar_refresh: int) -> Dict[str, np.ndarray]:
    """
    Runs the model on the data loader and returns the reconstructions.

    Args:
        model: End-to-End Variational Network
        data_loader: torch.utils.data.DataLoader
        device: cuda or cpu
        progress_bar_refresh: Refresh rate of the progress bar.

    Returns:
        dict: Dictionary containing the reconstructions.
    """
    # set the model to evaluation mode
    model.eval()
    model.to(device)

    # initialize the output dictionary
    output = defaultdict(list)

    # loop over the data loader
    for i, data in enumerate(data_loader):
        (masked_kspace, sensitivity_map, mask, _, target, fname, slice_num, _, _, _) = data
        sensitivity_map = sensitivity_map.to(device)
        y = masked_kspace.to(device)
        m = mask.to(device)

        # run the forward pass
        with torch.no_grad():
            estimate = model.forward(y, sensitivity_map, m)

            if estimate.shape[-1] == 2:
                estimate = torch.view_as_complex(estimate)

            estimate = estimate / torch.max(torch.abs(estimate))

            target, estimate = center_crop_to_smallest(target, estimate)

            output[fname[0]].append((slice_num, estimate.cpu()))

            gc.collect()
            torch.cuda.empty_cache()

            # update the progress bar
            if i % progress_bar_refresh == 0:
                sys.stdout.write("\r[{:5.2f}%]".format(100 * (i + 1) / len(data_loader)))
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)]) for fname, slice_preds in output.items()
    }

    return reconstructions


def main(args):
    """
    Main function.

    Args:
        args: Arguments from the command line.

    Returns:
        None
    """
    data_loader = DataLoader(
        dataset=SliceDataset(
            root=args.data_path,
            sense_root=args.sense_path,
            challenge=args.challenge,
            transform=PhysicsInformedDataTransform(
                mask_func=create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations),
                shift_mask=args.shift_mask,
                normalize_inputs=args.normalize_inputs,
                crop_size=args.crop_size,
                crop_before_masking=args.crop_before_masking,
                fft_type=args.fft_type,
            ),
            sample_rate=args.sample_rate,
            mask_root=args.mask_path,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    _, model, _ = load_model(args.checkpoint, args.num_cascades, args.pools, args.chans, args.fft_type, args.device)

    init_start = time.perf_counter()

    print("Reconstructing...")
    reconstructions = run_e2evn(model, data_loader, args.device, args.progress_bar_refresh)

    print("Saving...")
    save_reconstructions(reconstructions, args.out_dir)

    print("Finished! It took", time.perf_counter() - init_start, "s \n")


def create_arg_parser():
    """
    Creates an ArgumentParser to read the arguments.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="E2EVN")

    parser.add_argument("data_path", type=pathlib.Path, help="Path to the data folder")
    parser.add_argument("checkpoint", type=pathlib.Path, help="Path to the checkpoint file")
    parser.add_argument("out_dir", type=pathlib.Path, help="Path to the output folder")
    parser.add_argument("--sense_path", type=pathlib.Path, help="Path to the sense folder")
    parser.add_argument("--mask_path", type=pathlib.Path, help="Path to the mask folder")
    parser.add_argument(
        "--data-split",
        choices=["val", "test", "test_v2", "challenge"],
        help='Which data partition to run on: "val" or "test"',
    )
    parser.add_argument(
        "--challenge",
        type=str,
        choices=["singlecoil", "multicoil"],
        default="multicoil",
        help="Which challenge to run",
    )
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate for the data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the data loader")
    parser.add_argument(
        "--mask_type",
        choices=("random", "gaussian2d", "equispaced"),
        default="gaussian2d",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--accelerations", nargs="+", default=[10, 10], type=int, help="Acceleration rates to use for masks"
    )
    parser.add_argument(
        "--center_fractions", nargs="+", default=[0.7, 0.7], type=float, help="Number of center lines to use in mask"
    )
    parser.add_argument("--shift_mask", action="store_true", help="Shift the mask")
    parser.add_argument("--normalize_inputs", action="store_true", help="Normalize the inputs")
    parser.add_argument("--crop_size", default=None, help="Size of the crop to apply to the input")
    parser.add_argument("--crop_before_masking", action="store_true", help="Crop before masking")
    parser.add_argument("--num_cascades", type=int, default=1, help="Number of cascades for the model")
    parser.add_argument("--pools", type=int, default=2, help="Number of pools for the model")
    parser.add_argument("--chans", type=int, default=14, help="Number of channels for the model")
    parser.add_argument("--fft_type", type=str, default="orthogonal", help="Type of FFT to use")
    parser.add_argument("--progress_bar_refresh", type=int, default=10, help="Progress bar refresh rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the data loader")
    parser.add_argument(
        "--data_parallel", action="store_true", help="If set, use multiple GPUs using data parallelism"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Which device to run on")

    return parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args(sys.argv[1:]))