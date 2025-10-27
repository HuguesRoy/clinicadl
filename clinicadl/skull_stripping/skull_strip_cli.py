from pathlib import Path

import click

from clinicadl.utils import cli_param


@click.command(name="synstrip", no_args_is_help=True)
@cli_param.argument.caps_directory
@click.argument(
    "preprocessing_dict",
    type=click.Path(path_type=Path),
)
@cli_param.option.participant_list
@cli_param.option.batch_size
@cli_param.option.n_proc
@cli_param.option.use_gpu
@cli_param.option.amp
@cli_param.option.use_uncropped_image
@click.option(
    "nifti",
    "--nifti",
    is_flag=True,
    default=False,
    help="If flagged, use nifti files instead of pytorch tensors.",
)
def synstrip_cli(
    caps_directory,
    preprocessing_dict,
    participants_tsv,
    batch_size,
    n_proc,
    gpu,
    amp,
    use_uncropped_image,
    nifti,
):
    """Performs skull stripping.

    caps_directory is the CAPS folder where t1-linear outputs are stored

    preprocessing_dict is the preprocessing dict from clinicadl extract tensor

    """
    from clinicadl.utils.cmdline_utils import check_gpu

    if gpu:
        check_gpu()

    from clinicadl.skull_stripping.skull_stripping import skull_stripping_synthstrip

    skull_stripping_synthstrip(
        caps_directory,
        preprocessing_dict,
        tsv_path=participants_tsv,
        batch_size=batch_size,
        n_proc=n_proc,
        gpu=gpu,
        amp=amp,
        use_uncropped_image=use_uncropped_image,
        nifti=nifti,
    )


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="skull-stripping", no_args_is_help=True)
def cli() -> None:
    """Skull Stripping."""
    pass


cli.add_command(synstrip_cli)
