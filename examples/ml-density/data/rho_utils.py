import numpy as np
import scipy.linalg
import torch

import metatensor.torch as mts
from metatensor.torch import TensorMap
from pyscf import df, dft, gto
from pyscf.dft import numint
from pyscf.lib import unpack_tril

from ase import Atoms

BOHR_TO_ANG = 0.529177210903


def atoms_to_pyscf(atoms: Atoms, basis: str) -> gto.Mole:
    """Build a PySCF Mole from an ASE Atoms object."""
    import pyscf

    atom_spec = list(zip(atoms.get_chemical_symbols(), atoms.get_positions()))
    parts = basis.split(":")
    if len(parts) == 3 and parts[0].lower() == "etb":
        mol_ao = gto.M(atom=atom_spec, basis=parts[1], unit="Angstrom").build()
        resolved = pyscf.df.aug_etb(mol_ao, beta=float(parts[2]))
    else:
        resolved = basis
    return gto.M(atom=atom_spec, basis=resolved, unit="Angstrom").build()


def nmae_percent(
    rho_pred: np.ndarray,
    rho_ref: np.ndarray,
    weights: np.ndarray,
) -> float:
    """NMAE% = 100 * ∫|ρ_pred - ρ_ref| dr / ∫ρ_ref dr."""
    return 100.0 * (weights @ np.abs(rho_pred - rho_ref)) / (weights @ rho_ref)


def ri_coeffs_mts_to_pyscf(tensor: TensorMap) -> list[np.ndarray]:
    """Convert a TensorMap of RI coefficients to a list of PySCF-ordered numpy arrays.

    Returns one array per system, with coefficients sorted by
    (system, atom, angular momentum, radial channel, magnetic quantum number).
    """
    sys_list, atm_list, ang_list, rad_list, m_list, val_list = [], [], [], [], [], []

    for key, block in tensor.items():
        angular = int(key[0])
        vals = block.values
        device = vals.device
        S, C, P = vals.shape

        sys_idx = block.samples.values[:, 0].to(device, dtype=torch.long)
        atm_idx = block.samples.values[:, 1].to(device, dtype=torch.long)
        rad_idx = block.properties.values[:, 0].to(device, dtype=torch.long)
        m_idx = block.components[0].values[:, 0].to(device, dtype=torch.long)

        # PySCF expects p-orbitals in (z, x, y) order
        if angular == 1:
            vals = vals[:, [2, 0, 1], :]

        sys_grid = sys_idx.view(S, 1, 1).expand(S, C, P)
        atm_grid = atm_idx.view(S, 1, 1).expand(S, C, P)
        m_grid = m_idx.view(1, C, 1).expand(S, C, P)
        rad_grid = rad_idx.view(1, 1, P).expand(S, C, P)
        ang_grid = torch.full((S, C, P), angular, dtype=torch.long, device=device)

        mask = ~torch.isnan(vals)
        sys_list.append(sys_grid[mask])
        atm_list.append(atm_grid[mask])
        ang_list.append(ang_grid[mask])
        rad_list.append(rad_grid[mask])
        m_list.append(m_grid[mask])
        val_list.append(vals[mask])

    if not val_list:
        return []

    sys_all = torch.cat(sys_list)
    atm_all = torch.cat(atm_list)
    ang_all = torch.cat(ang_list)
    rad_all = torch.cat(rad_list)
    m_all = torch.cat(m_list)
    val_all = torch.cat(val_list)

    # Stable radix sort: least-significant key first
    idx = torch.arange(len(val_all), device=device)
    for key in (m_all, rad_all, ang_all, atm_all, sys_all):
        idx = idx[torch.argsort(key[idx], stable=True)]

    sys_sorted = sys_all[idx]
    val_sorted = val_all[idx]

    changes = (sys_sorted[1:] != sys_sorted[:-1]).nonzero(as_tuple=True)[0] + 1
    result = torch.tensor_split(
        val_sorted, changes.cpu() if changes.is_cuda else changes
    )
    return [r.detach().cpu().numpy() for r in result]


def _eval_xc_mat(ao, weight, rho, vxc, xctype: str) -> np.ndarray:
    """XC matrix contribution for one quadrature block (LDA or GGA)."""
    if xctype == "LDA":
        aow = ao * (weight * np.asarray(vxc[0]).ravel())[:, None]
        return ao.T @ aow

    vrho = np.asarray(vxc[0]).ravel()
    vsigma = np.asarray(vxc[1]).ravel()
    wv0 = 0.5 * weight * vrho
    wv1 = 2.0 * weight * vsigma
    aow = (
        wv0[:, None] * ao[0]
        + (wv1 * rho[1])[:, None] * ao[1]
        + (wv1 * rho[2])[:, None] * ao[2]
        + (wv1 * rho[3])[:, None] * ao[3]
    )
    vmat = ao[0].T @ aow
    return vmat + vmat.T


def dm_from_ri_coefficients(
    atoms: Atoms,
    ri_coefficients: TensorMap,
    xc: str,
    basis: str,
    auxbasis: str,
    grid_level: int = 3,
) -> np.ndarray:
    """Density matrix from RI coefficients via single Fock diagonalisation.

    Builds F = H_core + V_J[ρ_RI] + V_xc[ρ_RI] where ρ_RI = Σ_P c_P χ_P,
    then diagonalises to get D from the occupied MOs. Works for both S-fit
    and J-fit coefficients — the density evaluation is metric-independent.
    """
    assert (
        len(
            torch.unique(
                mts.unique_metadata(ri_coefficients, "samples", "system").values
            )
        )
        == 1
    ), "ri_coefficients must be for a single system"

    mol = atoms_to_pyscf(atoms, basis)
    auxmol = atoms_to_pyscf(atoms, auxbasis)
    coeffs = ri_coeffs_mts_to_pyscf(ri_coefficients)[0]

    ni = numint.NumInt()
    xctype = ni._xc_type(xc)
    ao_deriv = 1 if xctype in ("GGA", "MGGA") else 0
    nao = mol.nao

    grid = dft.gen_grid.Grids(mol)
    grid.level = grid_level
    grid.build()

    vxc_mat = np.zeros((nao, nao))
    for ao, _, weight, coords in ni.block_loop(mol, grid, nao, ao_deriv):
        aux_ao = auxmol.eval_gto("GTOval_sph", coords)
        if xctype == "LDA":
            rho = aux_ao @ coeffs
        else:
            aux_ao_ip = auxmol.eval_gto("GTOval_ip_sph", coords)
            rho = np.vstack(
                [(aux_ao @ coeffs)[None], np.einsum("xin,n->xi", aux_ao_ip, coeffs)]
            )
        _, vxc, _, _ = ni.eval_xc(xc, rho, spin=0, deriv=1)
        vxc_mat += _eval_xc_mat(ao, weight, rho, vxc, xctype)

    eri3c = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij")
    vj = unpack_tril(eri3c @ coeffs)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    fock = h_core + vj + vxc_mat
    _, mo_coeff = scipy.linalg.eigh(fock, mol.intor("int1e_ovlp"))

    nocc = mol.nelectron // 2
    mocc = mo_coeff[:, :nocc]
    return 2.0 * mocc @ mocc.T


def run_scf(
    atoms: Atoms,
    xc: str,
    basis: str,
    dm0: np.ndarray | None = None,
) -> tuple[dft.RKS, int]:
    """Run closed-shell KS-DFT to convergence; return (mf, n_cycles)."""
    mol = atoms_to_pyscf(atoms, basis)
    mf = dft.RKS(mol)
    mf.xc = xc
    n_cycles = 0

    def _count(envs):
        nonlocal n_cycles
        n_cycles += 1

    mf.callback = _count
    mf.kernel(dm0=dm0)
    return mf, n_cycles
