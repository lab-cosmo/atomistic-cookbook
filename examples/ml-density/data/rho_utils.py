import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch

import metatensor.torch as mts
from metatensor.torch import TensorMap

import py3Dmol
from pyscf import df, dft, gto
from pyscf.dft import numint
from pyscf.lib import unpack_tril
from pyscf.tools import cubegen

from ase import Atoms


def atoms_to_pyscf(atoms: Atoms, basis: str) -> gto.Mole:
    """Build a PySCF Mole from an ASE Atoms object."""

    atom_spec = list(zip(atoms.get_chemical_symbols(), atoms.get_positions()))
    parts = basis.split(":")
    if len(parts) == 3 and parts[0].lower() == "etb":
        mol_ao = gto.M(atom=atom_spec, basis=parts[1], unit="Angstrom").build()
        resolved = df.aug_etb(mol_ao, beta=float(parts[2]))
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


def visualise_density(mol, dm, isoval, nx=20, ny=20, nz=20):
    """Return py3Dmol HTML string for the density isosurface at isoval.

    Returns an HTML string (not a displayed widget) so that callers can
    concatenate multiple viewers and display them together via
    ``IPython.display.HTML``, which sphinx-gallery captures via ``_repr_html_``.
    """
    fd, cube_path = tempfile.mkstemp(suffix=".cube")
    os.close(fd)

    cubegen.density(mol, cube_path, dm, nx=nx, ny=ny, nz=nz)

    with open(cube_path) as f:
        cube = f.read()
    os.unlink(cube_path)

    view = py3Dmol.view(width=500, height=500)
    view.addVolumetricData(
        cube, "cube", {"isoval": isoval, "color": "blue", "opacity": 0.85}
    )
    view.addModel(cube, "cube")  # molecular geometry from the cube header
    view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    view.zoomTo()
    return view._make_html()


def _rho_slice_yz(mol, dm, n_x=100, n_s=80, x_lim=(-3.5, 3.5), s_lim=(-2.5, 2.5)):
    """Electron density on the y=z molecular plane, shaped (n_s, n_x).

    Evaluates ρ(r) = Σ_μν D_μν φ_μ φ_ν on a 2D grid of points with y = z = s/√2.
    Only valid for molecules where the atoms of interest lie in this plane.
    """
    _bohr_to_ang = 0.529177210903
    x = np.linspace(*x_lim, n_x)
    s = np.linspace(*s_lim, n_s)
    xx, ss = np.meshgrid(x, s)
    pts_bohr = (
        np.column_stack([xx.ravel(), ss.ravel() / np.sqrt(2), ss.ravel() / np.sqrt(2)])
        / _bohr_to_ang
    )
    ao = mol.eval_gto("GTOval_sph", pts_bohr)
    return x, s, np.einsum("pi,ij,pj->p", ao, dm, ao).reshape(n_s, n_x)


def plot_density_slice(
    mol, atoms, dm_conv, dm_ml, dm_sad, delta_vmax=0.025, rho_vmax=0.3
):
    """Three-panel 2D electron density slice through the y=z molecular plane.

    Panel 1: converged density (absolute).
    Panel 2: ML − converged delta density.
    Panel 3: SAD − converged delta density.

    Assumes all heavy atoms lie in the y=z plane; specific to the SCFBench
    test molecule used in this recipe.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    atoms : ase.Atoms
    dm_conv, dm_ml, dm_sad : np.ndarray
        Density matrices for the converged SCF, ML guess, and SAD guess.
    delta_vmax : float
        Symmetric colour-scale limit for the delta-density panels.
    rho_vmax : float
        Upper colour-scale limit for the converged density panel.  Without
        this cap the nuclear-core peaks dominate the range and compress the
        bonding/outer-shell region into near-white.  The default 0.3 e/bohr³
        saturates the cores (clipped to dark blue) while showing the full
        molecular shape and inter-atomic bonding clearly.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_grid, s_grid, rho_conv = _rho_slice_yz(mol, dm_conv)
    _, _, rho_ml = _rho_slice_yz(mol, dm_ml)
    _, _, rho_sad = _rho_slice_yz(mol, dm_sad)

    pos = atoms.get_positions()
    x_atoms, s_atoms = pos[:, 0], pos[:, 1] * np.sqrt(2)
    numbers = atoms.get_atomic_numbers()
    _marker = {8: ("o", "red"), 6: ("o", "dimgrey"), 1: ("o", "white")}
    _ms = {8: 9, 6: 8, 1: 5}

    extent = [x_grid[0], x_grid[-1], s_grid[0], s_grid[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True, dpi=120)

    im0 = axes[0].imshow(
        rho_conv,
        extent=extent,
        origin="lower",
        cmap="Blues",
        aspect="auto",
        vmin=0,
        vmax=rho_vmax,
    )
    axes[0].set_title(r"Converged density $\rho_\mathrm{conv}$")
    plt.colorbar(im0, ax=axes[0], label=r"$\rho$ / $e\,\mathrm{bohr}^{-3}$")

    for ax, delta_rho, title in [
        (axes[1], rho_ml - rho_conv, r"$\Delta\rho$: ML $-$ converged"),
        (axes[2], rho_sad - rho_conv, r"$\Delta\rho$: SAD $-$ converged"),
    ]:
        im = ax.imshow(
            delta_rho,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            aspect="auto",
            vmin=-delta_vmax,
            vmax=delta_vmax,
        )
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=r"$\Delta\rho$ / $e\,\mathrm{bohr}^{-3}$")

    for ax in axes:
        for xi, si, Z in zip(x_atoms, s_atoms, numbers):
            m, c = _marker[Z]
            ax.plot(xi, si, m, ms=_ms[Z], color=c, mec="black", mew=0.7, zorder=5)
        ax.set_xlabel(r"$x$ / Å")
        ax.set_ylabel(r"$y$ / Å")
        ax.spines[["top", "right"]].set_visible(False)

    return fig
