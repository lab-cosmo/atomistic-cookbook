"""
Order parameters for the acidic and basic dissociation of a phosphate species.

Exports ``cv.pt``, a ``metatomic`` model that PLUMED evaluates through the
``METATOMIC`` action (see ``metad/plumed.dat``). The model returns three
quantities per structure:

1. ``r``: the signed charge-separation coordinate of Equation (9) of the
   paper, :math:`r = \\sum_i q_i |\\mathrm{mic}(\\mathbf{r}_i - \\mathbf{r}_P)|`
   over the *solvent* oxygen atoms. A hydronium (q > 0) gives r > 0, i.e. an
   acidic dissociation; a hydroxide gives r < 0, a basic one.
2. ``q``: the total apparent charge of the phosphate oxygen atoms, i.e. the
   formal charge of the solute (0, -1, -2, -3), used to detect that the solute
   is in its reference protonation state.
3. ``delta``: :math:`\\sum_i q_i^2 - (\\sum_i q_i)^2` over the solvent oxygen
   atoms, which vanishes unless the solvent autoionizes.

Apparent charges follow the recipe: each hydrogen is partitioned among the
oxygen atoms within the cutoff with Gaussian weights of width ``sigma_h``;
the phosphorus atom distributes its +5 charge over its oxygen atoms in the
same way (``sigma_p``), and each oxygen is then assigned q = (received
charge) - 2. The phosphate oxygen atoms are the first four O in the file, and
the reference point is the (single) P atom.

Run ``python cv_phosphate.py`` to export the model; the TorchScript extension
of the neighbor-list library is collected in ``extensions/``.
"""

from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)


class PhosphateChargeSeparation(torch.nn.Module):
    def __init__(self, cutoff: float = 3.5, sigma_h: float = 0.3, sigma_p: float = 1.0):
        super().__init__()
        self._sigma_h = sigma_h
        self._sigma_p = sigma_p
        self._nl_options = NeighborListOptions(cutoff=cutoff, full_list=True, strict=False)

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl_options]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "features" in outputs:
            output_name = "features"
        elif "feature" in outputs:
            output_name = "feature"
        else:
            raise ValueError("this model only computes 'features'")
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms is not implemented")

        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        values = torch.zeros((len(systems), 3), dtype=dtype, device=device)
        for i_sys, system in enumerate(systems):
            if len(system) == 0 or torch.sum(system.positions**2) == 0:
                continue  # PLUMED probes the model with an empty/zero system
            positions, cell, types = system.positions, system.cell, system.types
            nl = system.get_neighbor_list(self._nl_options)
            i = nl.samples.column("first_atom")
            j = nl.samples.column("second_atom")
            d2 = (nl.values.reshape(-1, 3) ** 2).sum(dim=1)
            n_atoms = len(system)

            # H -> O partition (normalized per hydrogen)
            oh = (types[i] == 8) & (types[j] == 1)
            w = torch.exp(-d2[oh] / (2 * self._sigma_h**2))
            norm = torch.zeros(n_atoms, dtype=dtype, device=device).index_add(0, j[oh], w)
            q = torch.zeros(n_atoms, dtype=dtype, device=device)
            q = q.index_add(0, i[oh], w / norm[j[oh]])
            # P -> O partition of the +5 charge of phosphorus (normalized per P)
            op = (types[i] == 8) & (types[j] == 15)
            w = torch.exp(-d2[op] / (2 * self._sigma_p**2))
            norm = torch.zeros(n_atoms, dtype=dtype, device=device).index_add(0, j[op], w)
            q = q.index_add(0, i[op], 5 * w / norm[j[op]])

            io = torch.where(types == 8)[0]
            ip = torch.where(types == 15)[0][0]
            q_o = q[io] - 2
            q_phosphate, q_water = q_o[:4], q_o[4:]
            # minimum-image distance of the solvent oxygens from the P atom
            dx = positions[io[4:]] - positions[ip]
            frac = dx @ torch.linalg.inv(cell)
            dist = torch.linalg.norm((frac - torch.round(frac)) @ cell, dim=1)
            values[i_sys, 0] = (q_water * dist).sum()
            values[i_sys, 1] = q_phosphate.sum()
            values[i_sys, 2] = (q_water**2).sum() - q_water.sum() ** 2

        block = TensorBlock(
            values=values,
            samples=Labels("system", torch.arange(len(systems), device=device).reshape(-1, 1)),
            components=[],
            properties=Labels("cv", torch.tensor([[0], [1], [2]], device=device)),
        )
        return {
            output_name: TensorMap(
                keys=Labels("_", torch.tensor([[0]], device=device)), blocks=[block]
            )
        }


if __name__ == "__main__":
    model = AtomisticModel(
        PhosphateChargeSeparation().eval(),
        ModelMetadata(
            name="phosphate charge separation",
            description="signed solute-ion separation, solute charge, solvent ion pairs",
        ),
        ModelCapabilities(
            outputs={"features": ModelOutput(per_atom=False)},
            interaction_range=3.5,
            supported_devices=["cpu", "cuda"],
            length_unit="angstrom",
            atomic_types=[1, 8, 11, 15],
            dtype="float64",
        ),
    )
    model.save("cv.pt", collect_extensions="extensions")
