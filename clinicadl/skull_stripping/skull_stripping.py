"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""

from logging import getLogger
from pathlib import Path

import pandas as pd
import torch
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
    Performs MR image skull stripping on caps dataset

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
    logger.debug("Loading SynStrip model.")
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

        dataset = CapsDatasetImage(caps_dir, df, preprocessing_dict)
        dataloader = DataLoader(
            dataset, num_workers=n_proc, batch_size=batch_size, pin_memory=True
        )

        logger.info(f"Quality check will be performed over {len(dataloader)} images.")

        for data in dataloader:
            logger.debug(f"Processing subject {data['participant_id']}.")
            inputs = data["image"]
            if gpu:
                inputs = inputs.cuda()
            with autocast(enabled=amp):
                outputs = model(inputs)
            # We cast back to 32bits. It should be a no-op as softmax is not eligible
            # to fp16 and autocast is forbidden on CPU (output would be bf16 otherwise).
            # But just in case...
            outputs = outputs.float()

            for idx, sub in enumerate(data["participant_id"]):
                image_path_i = data["image_path"].parent()
                name = data["image_path"].parts[-1].replace(".pt", "_skull_stripped.pt")
                torch.save(outputs, image_path_i / name)

                logger.debug(f" sub {sub} - session : {data["session_id"]} done")

        logger.info(f"Results are stored at {output_path}.")
