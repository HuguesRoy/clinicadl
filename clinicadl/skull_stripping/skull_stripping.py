"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""

from logging import getLogger
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from clinicadl.generate.generate_utils import load_and_check_tsv
from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import (
    CapsDataset,
    CapsDatasetImage,
)
from clinicadl.utils.clinica_utils import RemoteFileStructure, fetch_file
from clinicadl.utils.exceptions import ClinicaDLArgumentError

from .models import StripModel

logger = getLogger("clinicadl.quality-check")


def compute_padding(input_size: int, target_size: int) -> tuple:
    pad = (target_size - input_size) // 2
    if (target_size - input_size) % 2 == 1:
        return (pad, pad + 1)
    else:
        return (pad, pad)


class OrientationSynthStrip:
    def __call__(self, input_image: torch.Tensor):
        # transform RAS to LIA orientation (entry: RAS, shape (1,x,y,z))
        return input_image.permute((0, 1, 3, 2)).flip(2).flip(1)


class PaddingSynthStrip:
    def __call__(self, input_image: torch.Tensor):
        # clip to SynthStrip compatible image resolution (x64 factor)
        target_shape = tuple(
            torch.clip(
                torch.ceil(torch.tensor(input_image.shape[1:]) / 64).to(int) * 64,
                192,
                320,
            )
        )  # clip to min 192, 320 max size

        # paddind
        pad_1 = compute_padding(input_image.shape[-1], target_shape[-1])
        pad_2 = compute_padding(input_image.shape[-2], target_shape[-2])
        pad_3 = compute_padding(input_image.shape[-3], target_shape[-3])

        return F.pad(input_image, (*pad_1, *pad_2, *pad_3), mode="constant", value=0)


class NormalizeSynthStrip:
    def __call__(self, input_image: torch.Tensor):
        input_image -= input_image.min()
        quantile = torch.quantile(
            input_image, q=torch.tensor(0.99).to(input_image.dtype)
        )
        return (input_image / quantile).clip(0, 1)


class OrientationBack:
    def __call__(self, input_image: torch.Tensor):
        # transform LIA to RAS orientation (entry: LIA, shape (1,x,y,z))
        return input_image.flip(2).flip(1).permute((0, 1, 3, 2))


class Unpadd:
    def __call__(self, input_image: torch.Tensor):
        # clip to SynthStrip compatible image resolution (x64 factor)
        target_shape = tuple(
            torch.clip(
                torch.ceil(torch.tensor(input_image.shape[1:]) / 64).to(int) * 64,
                192,
                320,
            )
        )  # clip to min 192, 320 max size

        # paddind
        pad_1 = compute_padding(input_image.shape[-1], target_shape[-1])
        pad_2 = compute_padding(input_image.shape[-2], target_shape[-2])
        pad_3 = compute_padding(input_image.shape[-3], target_shape[-3])

        return F.pad(input_image, (*pad_1, *pad_2, *pad_3), mode="constant", value=0)


def skull_stripping_syntrip(
    caps_dir: Path,
    preprocessing_dict: Path,
    output_path: Path,
    tsv_path: Path = None,
    batch_size: int = 1,
    n_proc: int = 0,
    gpu: bool = True,
    amp: bool = False,
    use_tensor: bool = False,
    use_uncropped_image: bool = True,
):
    """
    Performs MR image skull stripping on caps dataset using SynthStrip model

    Parameters
    -----------
    caps_dir: str (Path)
        Path to the input caps directory
    output_path: str (Path)
        Path to the output TSV file.
    tsv_path: str (Path)
        Path to the participant.tsv if the option was added.
    threshold: float
        Threshold that indicates whether the image passes the quality check.
    batch_size: int
    n_proc: int
    gpu: int
    amp: bool
        If enabled, uses Automatic Mixed Precision (requires GPU usage).
    use_tensor: bool
        To use tensor instead of nifti images
    use_uncropped_image: bool
        To use uncropped images instead of the cropped ones.

    """

    logger = getLogger("clinicadl.skull_stripping")

    # Fetch SynStrip model
    home = Path.home()

    cache_clinicadl = Path(".")
    url_aramis = "TODO"  # https://aramislab.paris.inria.fr/files/data/models/dl/qc/"

    cache_clinicadl.mkdir(parents=True, exist_ok=True)

    FILE1 = RemoteFileStructure(
        filename="stripmodel.pt",
        url=url_aramis,
        checksum=" TODO",
    )

    model = StripModel()

    model_file = cache_clinicadl / FILE1.filename

    logger.info("Downloading skull stripping model.")

    if not (model_file.is_file()):
        try:
            model_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            print("Unable to download required model for stripping process:", err)

    # Load stripping model
    logger.debug("Loading SynthStrip model.")
    model.load_state_dict(torch.load(model_file))
    model.eval()
    if gpu:
        logger.debug("Working on GPU.")
        model = model.cuda()
    elif amp:
        raise ClinicaDLArgumentError(
            "AMP is designed to work with modern GPUs. Please add the --gpu flag."
        )

    with torch.no_grad():
        # Transform caps_dir in dict
        caps_dict = CapsDataset.create_caps_dict(caps_dir, multi_cohort=False)

        # Load DataFrame
        logger.debug("Loading data to check.")
        if tsv_path is None:
            df = load_and_check_tsv(tsv_path, caps_dict, output_path.resolve().parent)
        else:
            df = load_and_check_tsv(tsv_path, caps_dict, None)

        transforms_synthstrip = transforms.Compose(
            [
                OrientationSynthStrip(),
                PaddingSynthStrip(),
                NormalizeSynthStrip(),
            ]
        )

        dataset_synthstrip = CapsDatasetImage(
            caps_dir, df, preprocessing_dict, transformation=transforms_synthstrip
        )

        dataset_unorm = CapsDatasetImage(
            caps_dir,
            df,
            preprocessing_dict,
        )

        dataloader_synthstrip = DataLoader(
            dataset_synthstrip,
            num_workers=n_proc,
            batch_size=batch_size,
            pin_memory=True,
        )

        dataloader_unstrip = DataLoader(
            dataset_unorm,
            num_workers=n_proc,
            batch_size=batch_size,
            pin_memory=True,
        )

        logger.info(
            f"Skull stripping will be performed over {len(dataloader_synthstrip.dataset)} images."
        )

        for data_synth, data_clear in zip(dataloader_synthstrip, dataloader_unstrip):
            logger.debug(f"Processing subject {data_synth['participant_id']}.")
            inputs = data_synth["image"]

            clear_image = ""
            if gpu:
                inputs = inputs.cuda()
            with autocast(enabled=amp):
                outputs = model(inputs)
            # We cast back to 32bits. It should be a no-op as softmax is not eligible
            # to fp16 and autocast is forbidden on CPU (output would be bf16 otherwise).
            # But just in case...
            outputs = outputs.float()

            for idx, sub in enumerate(data_synth["participant_id"]):
                image_path_i = data_synth["image_path"].parent()
                name = (
                    data_synth["image_path"]
                    .parts[-1]
                    .replace(".pt", "_skull_stripped.pt")
                )
                print(outputs[idx].shape)
                torch.save(outputs[idx], image_path_i / name)

                logger.debug(f" sub {sub} - session : {data_synth["session_id"]} done")

        logger.info(f"Results are stored at {output_path}.")
