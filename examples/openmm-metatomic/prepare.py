"""Register the metatomic backend and fetch the structures and model."""

import sys
from importlib.metadata import Distribution, DistributionFinder
from pathlib import Path

from atomistic_cookbook_utils import download_with_retry, run_command

# OpenMM-ML reads ``openmmml.potentials`` entry points at import time.
sys.path.insert(0, str(Path.cwd()))


class _MetatomicDistribution(Distribution):
    def read_text(self, filename):
        if filename == "METADATA":
            return "Metadata-Version: 2.1\nName: openmmml-metatomic\nVersion: 0\n"
        if filename == "entry_points.txt":
            return (
                "[openmmml.potentials]\n"
                "metatomic = openmm_metatomic.metatomicpotential:"
                "MetatomicPotentialImplFactory\n"
            )
        return None

    def locate_file(self, path):
        return path


class _MetatomicFinder(DistributionFinder):
    def find_distributions(self, context=DistributionFinder.Context()):
        if context.name in (None, "openmmml-metatomic"):
            yield _MetatomicDistribution()


sys.meta_path.append(_MetatomicFinder())
from openmmml import MLPotential  # noqa: E402

WORKSHOP = "https://raw.githubusercontent.com/openmm/openmm_workshops/main/section_1"
for filename in ("lysozyme.pdb", "benzene.sdf", "o-xylene.sdf"):
    download_with_retry(f"{WORKSHOP}/{filename}", filename)

model_path = Path("pet-spice-s.pt")
if not model_path.is_file():
    checkpoint = (
        "https://huggingface.co/lab-cosmo/upet/resolve/main/"
        "models/pet-spice-s-v0.2.0.ckpt"
    )
    run_command(f"mtt export {checkpoint} -o {model_path}", print_output=True)

potential = MLPotential("metatomic", model=str(model_path), device="cpu")
