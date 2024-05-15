from pathlib import Path
from typing import get_args

import click

from clinicadl.prepare_data.prepare_data_config import PrepareDataConfig
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)

config = PrepareDataConfig.model_fields

n_proc = click.option(
    "-np",
    "--n_proc",
    type=config["n_proc"].annotation,
    default=config["n_proc"].default,
    show_default=True,
    help="Number of cores used during the task.",
)
tsv_file = click.option(
    "--participants_tsv",
    type=get_args(config["tsv_file"].annotation)[0],
    default=config["tsv_file"].default,
    help="Path to a TSV file including a list of participants/sessions.",
    show_default=True,
)
extract_json = click.option(
    "-ej",
    "--extract_json",
    type=get_args(config["extract_json"].annotation)[0],
    default=config["extract_json"].default,
    help="Name of the JSON file created to describe the tensor extraction. "
    "Default will use format extract_{time_stamp}.json",
)
use_uncropped_image = click.option(
    "-uui",
    "--use_uncropped_image",
    is_flag=True,
    help="Use the uncropped image instead of the cropped image generated by t1-linear or pet-linear.",
    show_default=True,
)
tracer = click.option(
    "--tracer",
    type=click.Choice(Tracer),
    default=config["tracer_cls"].default.value,
    help=(
        "Acquisition label if PREPROCESSING is `pet-linear`. "
        "Name of the tracer used for the PET acquisition (trc-<tracer>). "
        "For instance it can be '18FFDG' for fluorodeoxyglucose or '18FAV45' for florbetapir."
    ),
    show_default=True,
)
suvr_reference_region = click.option(
    "-suvr",
    "--suvr_reference_region",
    type=click.Choice(SUVRReferenceRegions),
    default=config["suvr_reference_region_cls"].default.value,
    help=(
        "Regions used for normalization if PREPROCESSING is `pet-linear`. "
        "Intensity normalization using the average PET uptake in reference regions resulting in a standardized uptake "
        "value ratio (SUVR) map. It can be cerebellumPons or cerebellumPon2 (used for amyloid tracers) or pons or "
        "pons2 (used for 18F-FDG tracers)."
    ),
    show_default=True,
)
custom_suffix = click.option(
    "-cn",
    "--custom_suffix",
    type=config["custom_suffix"].annotation,
    default=config["custom_suffix"].default,
    help=(
        "Suffix of output files if PREPROCESSING is `custom`. "
        "Suffix to append to filenames, for instance "
        "`graymatter_space-Ixi549Space_modulated-off_probability.nii.gz`, or "
        "`segm-whitematter_probability.nii.gz`"
    ),
)
dti_measure = click.option(
    "--dti_measure",
    "-dm",
    type=click.Choice(DTIMeasure),
    help="Possible DTI measures.",
    default=config["dti_measure_cls"].default.value,
    show_default=True,
)
dti_space = click.option(
    "--dti_space",
    "-ds",
    type=click.Choice(DTISpace),
    help="Possible DTI space.",
    default=config["dti_space_cls"].default.value,
    show_default=True,
)
save_features = click.option(
    "--save_features",
    is_flag=True,
    help="""Extract the selected mode to save the tensor. By default, the pipeline only save images and the mode extraction
            is done when images are loaded in the train.""",
)
