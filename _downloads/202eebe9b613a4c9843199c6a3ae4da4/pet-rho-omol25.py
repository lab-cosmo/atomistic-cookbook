r"""
Electron densities and electrostatic potentials with PET-:math:`\rho`-OMol25
============================================================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo>`_

This recipe demonstrates how to predict the electron and spin densities of molecules
with a machine-learning model and to compute from them properties like the electrostatic
potential (ESP) on the molecular surface and the distribution of unpaired electrons. We
use PET-:math:`\rho`-OMol25, a point-edge transformer (`PET
<https://arxiv.org/abs/2305.19302>`_) trained on the densities of the `OMol25
<https://arxiv.org/abs/2505.08762>`_ dataset, which cover organic molecules,
biomolecules, electrolytes and metal complexes in several charge and spin states, at the
ωB97M-V/def2-TZVPD level. The model is described in `this preprint
<https://arxiv.org/abs/2611.34732>`_.

The model is evaluated via `qpet <https://github.com/ppegolo/qpet>`_, a small package
that lets you run `metatomic <https://docs.metatensor.org/metatomic/>`_ density models
on `PySCF <https://pyscf.org>`_ molecules and compute observables from the predicted
densities. If you are interested in using a predicted density to start a DFT
calculation, have a look at the `ML surrogate for the electron density
<https://atomistic-cookbook.org/examples/ml-density/ml-density.html>`__ recipe.
"""

# %%

import io
import json
import time

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
from IPython.display import HTML
from matplotlib.colors import to_hex
from qpet import CubeGrid, DensityModel, atomic_becke_grid, build_mole
from skimage.measure import marching_cubes

# %%
# Loading the model
# -----------------
#
# The model predicts two outputs: the total electron density :math:`\rho(\mathbf{r})`
# and the spin density :math:`m(\mathbf{r})` (the difference between the densities of
# the up and down electrons). Both are expanded on atom-centred functions of the
# ``def2-universal-jfit`` auxiliary basis (`Weigend, 2006
# <https://doi.org/10.1039/b515623h>`_),
#
# .. math::
#
#    \rho(\mathbf{r}) = \sum_{i} \sum_{n l m} c^{i}_{nlm} \, \chi_{nlm}(\mathbf{r} -
#    \mathbf{r}_i) \;,
#
# so that the model only predicts the coefficients :math:`c^{i}_{nlm}` of each atom
# :math:`i`, and the scalar fields can be quickly reconstructed from these coefficients
# using fixed basis functions. With ``DensityModel.from_release``, the model is
# downloaded from `Hugging Face <https://huggingface.co/lab-cosmo/pet-rho>`_. The name
# of the auxiliary basis is stored in the metadata of the model, and the basis functions
# themselves come from a basis-set library and are compatible with PySCF.

model = DensityModel.from_release("pet-rho-omol", size="l", warn_shift=0.01)

# %%
# The model takes as inputs the geometry, the total charge, and the spin multiplicity
# of a molecule. We pass this information as a PySCF ``Mole``, which can be built by a
# helper function in ``qpet`` from symbols and positions in Å. The orbital basis of the
# original DFT calculation is not used by the model, but PySCF needs one to know how
# many electrons the molecule has. That's because for heavy atoms (from rubidium,
# :math:`Z = 37`, on), the def2 basis comes with an effective core potential (ECP) that
# replaces the core electrons.


def molecule(atoms, charge=0, spin=0):
    """PySCF molecule from ASE atoms; ``spin`` is the number of unpaired electrons."""
    return build_mole(
        atoms.get_chemical_symbols(),
        atoms.positions,
        basis="def2-svp",
        ecp={"I": "def2-svp"},
        charge=charge,
        spin=spin,
    )


# %%
# The electron density as a scalar field
# --------------------------------------
#
# We start by predicting the density of iodobenzene. The prediction contains RI
# coefficients in atomic units (lengths in bohr) that, multiplied by the basis
# functions, give the total density. Integrating the total density gives the number of
# electrons of the molecule, 66 with the iodine ECP. This holds by
# construction: in qpet, the predicted coefficients are shifted by the smallest amount,
# in the Coulomb metric, that makes the density integrate to the number of electrons
# (and the spin density to the number of unpaired electrons). The applied shift is
# stored in ``prediction.shift``, and a warning is printed when it is larger than a
# fraction ``warn_shift`` of the number of electrons. We set this fraction to 1% when
# loading the model.

frames = ase.io.read("data/iodobenzenes.xyz", ":")
table = pd.read_csv("data/iodobenzenes.csv")
iodobenzene = frames[table.index[table.key == "h"][0]]

start = time.perf_counter()
prediction = model.predict(molecule(iodobenzene))
print(f"prediction took {time.perf_counter() - start:.1f} s")
print(f"number of electrons: {prediction.rho.integral():.3f}")

# %%
# From the coefficients, the density can be evaluated on any set of points with
# ``rho.density``, which takes positions in bohr. Here we evaluate it on planes parallel
# to the benzene ring, and draw each plane together with the molecule with `py3Dmol
# <https://3dmol.org>`_. We use a logarithmic scale for the density, since it changes by
# several orders of magnitude between the nuclei and the tail. The slider moves the
# plane from 3 Å below the ring to 3 Å above it.

BOHR = 0.529177210903  # Å
KCAL = 627.509474  # kcal/mol per hartree


def view_of(atoms):
    """py3Dmol view of the atoms as balls and sticks."""
    xyz = io.StringIO()
    ase.io.write(xyz, atoms, format="xyz")
    view = py3Dmol.view(width=500, height=400)
    view.addModel(xyz.getvalue(), "xyz")
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.2}})
    return view


def colorbar(cmap, left, right, label):
    """HTML color bar, with the values at its ends and a label in the middle."""
    stops = ", ".join(to_hex(c) for c in plt.colormaps[cmap](np.linspace(0, 1, 11)))
    return f"""
    <div style="width: 300px; font: 12px sans-serif">
      <div style="height: 12px; background: linear-gradient(to right, {stops})"></div>
      <div style="display: flex; justify-content: space-between">
        <span>{left}</span><span>{label}</span><span>{right}</span>
      </div>
    </div>"""


def show_slices(atoms, rho, offsets, half_width=6.0, n=70):
    """log10(rho) on planes parallel to the molecule, chosen with a slider."""
    center = atoms.positions.mean(axis=0)
    _, _, (u, v, _) = np.linalg.svd(atoms.positions - center)
    normal = np.cross(u, v)
    s = np.linspace(-half_width, half_width, n)
    a, b = (x.reshape(-1, 1) for x in np.meshgrid(s, s))
    colors = []
    for offset in offsets:
        values = rho.density((center + a * u + b * v + offset * normal) / BOHR)
        log_rho = plt.Normalize(-5, 1)(np.log10(np.clip(values, 1e-6, None)))
        colors.append([to_hex(c) for c in plt.colormaps["viridis"](log_rho)])
    view = view_of(atoms)
    view.zoomTo()
    view.setProjection("orthographic")
    data = json.dumps(
        {
            "s": s.round(3).tolist(),
            "offsets": list(offsets),
            "center": center.tolist(),
            "u": u.tolist(),
            "v": v.tolist(),
            "normal": normal.tolist(),
            "colors": colors,
        }
    )
    html = view._make_html()
    start = len(offsets) // 2
    uid = view.uniqueid
    viewer = f"viewer_{uid}"
    script = f"""
    <script>
    const data_{uid} = {data};
    function slice_{uid}(k) {{
      const d = data_{uid};
      const n = d.s.length, o = d.offsets[k];
      {viewer}.removeAllShapes();
      const vertexArr = [], normalArr = [], faceArr = [];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {{
        const p = [0, 1, 2].map(
          x => d.center[x] + d.s[j] * d.u[x] + d.s[i] * d.v[x] + o * d.normal[x]);
        vertexArr.push({{x: p[0], y: p[1], z: p[2]}});
        normalArr.push({{x: d.normal[0], y: d.normal[1], z: d.normal[2]}});
        if (i < n - 1 && j < n - 1) {{
          // counterclockwise seen from the normal, as the lighting of 3Dmol expects
          const q = i * n + j;
          faceArr.push(q, q + 1, q + n, q + 1, q + n + 1, q + n);
        }}
      }}
      {viewer}.addCustom({{vertexArr, normalArr, faceArr, color: d.colors[k],
                          opacity: 0.85}});
      {viewer}.render();
      document.getElementById("offset_{uid}").textContent = o.toFixed(2);
    }}
    viewer_ready_{uid}.then(() => slice_{uid}({start}));
    </script>
    <div style="font: 12px sans-serif">
      <input type="range" min="0" max="{len(offsets) - 1}" value="{start}"
             style="width: 300px" oninput="slice_{uid}(this.value)">
      <span id="offset_{uid}"></span> Å from the plane of the ring
    </div>"""
    bar = colorbar("viridis", "10<sup>−5</sup>", "10", "ρ (a.u.)")
    return HTML(html + script + bar)


show_slices(iodobenzene, prediction.rho, np.linspace(-3.0, 3.0, 25))

# %%
# Computing the electrostatic potential
# -------------------------------------
#
# From the density and the charges of the nuclei, the ESP can be computed at any point
# with ``rho.potential``, which also takes positions in bohr. A common way to look at
# the ESP of a molecule is to draw it on its surface, often defined as the
# :math:`\rho = 0.001` a.u. isodensity surface.
#
# To find the surface, we sample the density on a regular grid around the molecule and
# extract the :math:`\rho = 0.001` a.u. isosurface with the marching-cubes algorithm of
# `scikit-image <https://scikit-image.org>`_. The vertices it returns are interpolated
# between grid points, so we move each of them onto the exact isosurface with one
# Newton step along the gradient of the density, which also gives the normals of the
# surface.


def isosurface(density, atoms, level, spacing=0.25):
    """Vertices (bohr), triangles and normals of an isosurface of a density."""
    grid = CubeGrid.enclosing(
        atoms.positions / BOHR, padding=3.0 / BOHR, spacing=spacing / BOHR
    )
    vertices, faces, _, _ = marching_cubes(grid.sample(density.density), level=level)
    points = grid.origin + vertices @ grid.axes
    gradient = density.gradient(points)
    step = (density.density(points) - level) / (gradient**2).sum(axis=1)
    shift = step[:, None] * gradient
    # where the density is flat, a Newton step can overshoot: keep it within the grid
    length = np.linalg.norm(shift, axis=1, keepdims=True)
    points -= shift * np.minimum(1.0, 0.5 * spacing / BOHR / np.maximum(length, 1e-12))
    normals = density.gradient(points)
    return points, faces, normals / np.linalg.norm(normals, axis=1)[:, None]


# %%
# We then compute the ESP of the electrons and nuclei on the vertices of the surface of
# iodobenzene, and draw the triangles of the surface with py3Dmol, coloring each vertex
# by the ESP: red for negative, blue for positive potential.


def mesh(vertices, faces, normals, color):
    """py3Dmol shape of a surface, with one color or one color per vertex."""
    return {
        "vertexArr": [
            dict(zip("xyz", v)) for v in np.round(vertices * BOHR, 3).tolist()
        ],
        "normalArr": [dict(zip("xyz", n)) for n in np.round(normals, 3).tolist()],
        "faceArr": faces.ravel().tolist(),
        "color": color,
        "opacity": 0.8,
    }


def show_esp(atoms, rho, limit, spacing=0.25, rotation=0):
    """The rho = 0.001 a.u. surface colored by the ESP, from -limit to +limit kcal/mol,
    turned by ``rotation`` degrees about the vertical axis."""
    vertices, faces, normals = isosurface(rho, atoms, 0.001, spacing)
    esp = KCAL * rho.potential(vertices)
    colors = plt.colormaps["RdBu"](plt.Normalize(-limit, limit)(esp))
    view = view_of(atoms)
    view.addCustom(mesh(vertices, faces, normals, [to_hex(c) for c in colors]))
    view.zoomTo()
    view.rotate(rotation, "y")
    bar = colorbar("RdBu", f"&minus;{limit}", f"+{limit}", "ESP (kcal/mol)")
    return HTML(view._make_html() + bar)


show_esp(iodobenzene, prediction.rho, limit=10, rotation=-60)

# %%
# The ESP is positive around the hydrogens and negative in a belt around the iodine.
# At the tip of the iodine, however, on the extension of the C--I bond, the ESP becomes
# positive again. This blue cap is the :math:`\sigma`-hole (`Clark et al., 2007
# <https://doi.org/10.1007/s00894-006-0130-2>`_): a halogen atom bound to carbon pulls
# electron density into the C--X bond and away from the opposite side, and the red belt
# around the cap is the density pulled away from the bond axis. The :math:`\sigma`-hole
# is what lets an electron-rich group bind to the halogen, in drug--protein complexes
# and in crystals alike.
#
# The usual way to quantify it is the maximum of the ESP on the molecular surface,
# :math:`V_{S,\max}`. The maximum of the ESP sits on the iodine, but the surface has
# other local maxima, on the hydrogens. As in the paper, we take :math:`V_{S,\max}`
# as the largest value on the iodine cap, the surface points within 2.6 Å of the iodine
# and 30° of the C--I axis.


def sigma_hole(atoms, surface, esp, radius=2.6, angle=30.0):
    """Largest ESP on the cap of the iodine, opposite to its carbon."""
    iodine = atoms.get_chemical_symbols().index("I")
    distances = atoms.get_distances(iodine, range(len(atoms)))
    distances[iodine] = np.inf
    axis = atoms.positions[iodine] - atoms.positions[np.argmin(distances)]
    axis /= np.linalg.norm(axis)
    r = surface * BOHR - atoms.positions[iodine]
    d = np.linalg.norm(r, axis=1)
    cap = (d < radius) & (r @ axis > d * np.cos(np.radians(angle)))
    return esp[cap].max()


surface, faces, normals = isosurface(prediction.rho, iodobenzene, 0.001)
esp = KCAL * prediction.rho.potential(surface)
print(f"V_S,max = {sigma_hole(iodobenzene, surface, esp):.1f} kcal/mol")

# %%
# The Hammett series
# ------------------
#
# An electron-withdrawing group in *para* position deepens the :math:`\sigma`-hole,
# because it pulls even more density from the iodine, while an electron-donating group
# makes it shallower. The Hammett constant :math:`\sigma_p` is a measure of this effect
# on the acidity of benzoic acids (`Hansch, Leo and Taft, 1991
# <https://doi.org/10.1021/cr00002a004>`_), so :math:`V_{S,\max}` should grow with it.
# With the predicted density, the whole series of 20 substituents can be computed in
# less than half a minute.

start = time.perf_counter()
vs_max = []
for atoms in frames:
    rho = model.predict(molecule(atoms)).rho
    surface = isosurface(rho, atoms, 0.001)[0]
    vs_max.append(sigma_hole(atoms, surface, KCAL * rho.potential(surface)))
table["vs_max"] = vs_max
print(f"20 molecules in {time.perf_counter() - start:.1f} s")

# %%
# We plot :math:`V_{S,\max}` against :math:`\sigma_p`, together with two references
# computed on the same surface points: the ωB97M-V/def2-TZVPD density (DFT), and its fit
# on the ``def2-universal-jfit`` basis (RI), which is what the model was trained to
# reproduce. The agreement is good, and the correlation with :math:`\sigma_p` is clear.
# Some residual error is in part due to the RI basis, which describes the tail of the
# density less well than the orbital basis does. The line is a least-squares fit of the
# predicted values.

slope, intercept = np.polyfit(table.sigma_para, table.vs_max, 1)
r = np.corrcoef(table.sigma_para, table.vs_max)[0, 1]
x = np.array([-0.9, 0.9])

fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
ax.plot(x, intercept + slope * x, color="0.7", lw=1, zorder=0)
ax.scatter(table.sigma_para, table.vs_max_dft, marker="^", color="0.6", label="DFT")
ax.scatter(table.sigma_para, table.vs_max_ri, marker="s", s=20, color="C1", label="RI")
ax.scatter(table.sigma_para, table.vs_max, color="C0", label=r"PET-$\rho$-OMol25")
labels = {"NMe2": ((0, 8), "center"), "H": ((8, -3), "left"), "NO2": ((-8, 0), "right")}
for row in table[table.substituent.isin(labels)].itertuples():
    offset, ha = labels[row.substituent]
    ax.annotate(
        row.substituent,
        (row.sigma_para, row.vs_max),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va="center",
    )
ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes, ha="right")
ax.set_xlim(*x)
ax.set_xlabel(r"Hammett $\sigma_p$")
ax.set_ylabel(r"$V_{S,\max}$ (kcal/mol)")
ax.legend(frameon=False, loc="upper left")
plt.show()

# %%
# Spin densities of radicals
# --------------------------
#
# The model also predicts how the unpaired electrons of an open-shell molecule are
# distributed. To obtain this distribution, we pass the number of unpaired electrons to
# ``molecule``, and ``prediction.spin`` is then the spin density :math:`m(\mathbf{r})`,
# which, like the total density, integrates to that number by construction.
#
# A simple way to read a spin density is to assign it to atoms. To do so, we integrate
# it over the `Becke cell <https://doi.org/10.1063/1.454033>`_ of each atom, using the
# quadrature grid of PySCF. This grid can be obtained with ``atomic_becke_grid``, which
# also returns the atom that each point belongs to.


def spin_populations(atoms, charge=0, spin=1):
    mol = molecule(atoms, charge, spin)
    prediction = model.predict(mol)
    points, weights, atom = atomic_becke_grid(mol)
    populations = np.bincount(atom, weights * prediction.spin.density(points))
    return prediction, populations


allyl = ase.io.read("data/allyl.xyz")
allyl_prediction, populations = spin_populations(allyl)
print("total spin:", round(populations.sum(), 3))
for symbol, p in zip(allyl.get_chemical_symbols()[:3], populations[:3]):
    print(f"{symbol}  {p:+.3f}")

# %%
# In the allyl radical, the unpaired electron occupies a :math:`\pi` orbital with a
# node on the central carbon so, in the one-electron picture, this atom has no spin. The
# central carbon carries instead a *negative* spin population: the unpaired electron
# polarizes the other electrons, a correlation effect known as spin polarization. DFT
# gives about −0.09 on this atom. In the isosurfaces of the spin density, the negative
# spin can be seen as the blue lobes between the red ones.


def show_spin(atoms, spin, isoval=0.005, spacing=0.15):
    """Isosurfaces of the spin density at +isoval (red) and -isoval (blue)."""
    view = view_of(atoms)
    for level, color in ((isoval, "red"), (-isoval, "blue")):
        vertices, faces, normals = isosurface(spin, atoms, level, spacing)
        view.addCustom(mesh(vertices, faces, normals, color))
    view.zoomTo()
    legend = f"""
    <div style="font: 12px sans-serif">
      <span style="color: red">■</span> m = +{isoval} a.u. &nbsp;
      <span style="color: blue">■</span> m = &minus;{isoval} a.u.
    </div>"""
    return HTML(view._make_html() + legend)


show_spin(allyl, allyl_prediction.spin)

# %%
# The same alternation can be seen in a larger system, the *p*-methylphenoxyl radical:
# the spin is mostly on the oxygen and on the *ortho* and *para* carbons, while the
# *meta* carbons in between have small negative populations.

phenoxyl = ase.io.read("data/methylphenoxyl.xyz")
phenoxyl_prediction, populations = spin_populations(phenoxyl)
for i, symbol in enumerate(phenoxyl.get_chemical_symbols()):
    if symbol in ("C", "O"):
        print(f"{i:2d} {symbol}  {populations[i]:+.3f}")

show_spin(phenoxyl, phenoxyl_prediction.spin)

# %%
# Metal complexes are part of the training set too, even though models can struggle with
# them (see the `preprint <https://arxiv.org/abs/2611.34732>`_, and similar behavior for
# the ML force fields of the `OMol25 paper <https://arxiv.org/abs/2505.08762>`_). In the
# tetrachlorocuprate(II) dianion, :math:`[\mathrm{CuCl}_4]^{2-}`, the unpaired electron
# is in a Cu :math:`d` orbital that mixes with the :math:`p` orbitals of the chlorides,
# so part of the spin moves onto the ligands. Here we pass a charge of −2. With
# PET-:math:`\rho`-OMol25, about 0.7 unpaired electrons sit on copper and the rest on
# the four chlorides, in line with DFT, which gives 0.70 on Cu and 0.075 on each Cl. In
# the positive isosurface, one can see the four lobes of the Cu :math:`d_{x^2-y^2}`
# orbital pointing at the chlorides, as well as the :math:`p` lobes of the chlorides
# that it mixes with. The small blue spot at the copper nucleus is not physical. That's
# because the auxiliary basis has no functions tight enough to describe the spin density
# this close to a nucleus, so values there should not be trusted.

cucl4 = ase.io.read("data/cucl4.xyz")
cucl4_prediction, populations = spin_populations(cucl4, charge=-2)
for symbol, p in zip(cucl4.get_chemical_symbols(), populations):
    print(f"{symbol:2s}  {p:+.3f}")

show_spin(cucl4, cucl4_prediction.spin, isoval=0.003)

# %%
# A larger molecule: vitamin B\ :sub:`12`
# ---------------------------------------
#
# A convenient fact about local atomistic models is that the cost of the prediction is
# only linear in the number of atoms, because the coefficients of each atom depend only
# on the atoms within a finite distance. For this reason, the same workflow can be
# applied to systems far beyond the reach of DFT on a laptop. Cyanocobalamin, the common
# form of vitamin B\ :sub:`12`, has 181 atoms with a cobalt in the middle of a corrin
# ring.

b12 = ase.io.read("data/cyanocobalamin.xyz")
start = time.perf_counter()
b12_prediction = model.predict(molecule(b12))
print(f"{len(b12)} atoms predicted in {time.perf_counter() - start:.1f} s")

# %%
# In the ESP on its molecular surface, one can see the negative phosphate and amide
# oxygens, where the molecule accepts hydrogen bonds, next to the positive regions of
# the N--H and O--H groups.

show_esp(b12, b12_prediction.rho, limit=40, spacing=0.35)
