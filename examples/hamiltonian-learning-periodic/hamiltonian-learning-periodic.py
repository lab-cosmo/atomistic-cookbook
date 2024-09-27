"""
Periodic Hamiltonian learning Tutorial
======================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo>`__,
          Jigyasa Nigam `@curiosity54 <https://github.com/curiosity54>`__

This tutorial explains how to train a machine learning model for the
electronic Hamiltonian of a periodic system. Even though we focus on
periodic systems, the code and techniques presented here can be directly
transferred to molecules.
"""

# %%
# First, import the necessary packages
#

import os
import warnings
import zipfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from matplotlib.animation import FuncAnimation
from mlelec.data.derived_properties import compute_eigenvalues
from mlelec.data.mldataset import MLDataset
from mlelec.data.qmdataset import QMDataset
from mlelec.models.equivariant_nonlinear_lightning import (
    LitEquivariantNonlinearModel,
    MSELoss,
)
from mlelec.utils.pbc_utils import blocks_to_matrix
from mlelec.utils.plot_utils import plot_bands_frame


warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


# %%
# Get Data and Prepare Data Set
# -----------------------------
#


# %%
# The data set contains 35 distorted graphene unit cells containing 2
# atoms. The reference density functional theory (DFT) calculations are
# performed with `CP2K <https://www.cp2k.org/>`__ using a minimal
# `STO-3G <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`__ basis and
# the `PBE <https://doi.org/10.1103/PhysRevLett.77.3865>`__ functional.
# The Kohn-Sham equations are solved on a Monkhorst-Pack grid of
# :math:`15\times 15\times 1` points in the Brillouin zone of the crystal.
#


# %%
# Obtain structures and DFT data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# If you have computational resources, you can run the DFT calculations
# needed to produce the data set. `This other
# tutorial <https://tinyurl.com/cp2krun>`__ in the atomistic cookbook can
# help you set up the CP2K calculations for this data set, using the
# ``reftraj_hamiltonian.cp2k`` file that can be dowloaded in the ``data.zip``
# file. To do the same for another data set, adapt the reftraj file.
#
# We will provide here some the functions in the `batch-cp2k
# tutorial <https://tinyurl.com/cp2krun>`__ that need to be adapted to the
# current data set.
#


# %%
# Import all the modules from `batch-cp2k
# tutorial <https://tinyurl.com/cp2krun>`__ and run the cell to install
# CP2K. Run also the cells up to the one defining ``write_cp2k_in``.
#
# The following cell defines a slighly modified version of that functions,
# allowing for non-orthorombic cells, and accounting for the reftraj file
# name change.
#


# %%
# .. code:: python
#
#    def write_cp2k_in(
#        fname: str,
#        project_name: str,
#        last_snapshot: int,
#        cell_a: List[float],
#        cell_b: List[float],
#        cell_c: List[float],
#    ) -> None:
#        """Writes a cp2k input file from a template.
#
#        Importantly, it writes the location of the basis set definitions,
#        determined from the path of the system CP2K install to the input file.
#        """
#
#        cp2k_in = open("data/reftraj_hamiltonian.cp2k", "r").read()
#
#        cp2k_in = cp2k_in.replace("//PROJECT//", project_name)
#        cp2k_in = cp2k_in.replace("//LAST_SNAPSHOT//", str(last_snapshot))
#        cp2k_in = cp2k_in.replace("//CELL_A//", " ".join([f"{c:.6f}" for c in cell_a]))
#        cp2k_in = cp2k_in.replace("//CELL_B//", " ".join([f"{c:.6f}" for c in cell_b]))
#        cp2k_in = cp2k_in.replace("//CELL_C//", " ".join([f"{c:.6f}" for c in cell_c]))
#
#        with open(fname, "w") as f:
#            f.write(cp2k_in)
#


# %%
# Unlike the `batch-cp2k tutorial <https://tinyurl.com/cp2krun>`__, the
# current data set includes a single stoichiometry, :math:`\mathrm{C_2}`.
# Therefore, you can run this cell to set the calculation scripts up.
#


# %%
# .. code:: python
#
#    project_name = 'graphene'
#    frames = ase_read('data/C2.xyz', index=':')
#    os.makedirs(project_name, exist_ok=True)
#    os.makedirs(f"{project_name}/FOCK", exist_ok=True)
#    os.makedirs(f"{project_name}/OVER", exist_ok=True)
#
#    write_cp2k_in(
#            f"{project_name}/in.cp2k",
#            project_name=project_name,
#            last_snapshot=len(frames),
#            cell_a=frames[0].cell.array[0],
#            cell_b=frames[0].cell.array[1],
#            cell_c=frames[0].cell.array[2],
#        )
#
#    ase_write(f"{project_name}/init.xyz", frames[0])
#    write_reftraj(f"{project_name}/reftraj.xyz", frames)
#    write_cellfile(f"{project_name}/reftraj.cell", frames)
#


# %%
# The CP2K calculations can be simply run using:
#


# %%
# .. code:: python
#
#    subprocess.run((
#        f"cp2k.ssmp -i {project_name}/in.cp2k "
#        "> {project_name}/out.cp2k"
#        ),
#        shell=True)
#


# %%
# Once the calculations are done, we can parse the results with:
#


# %%
# .. code:: python
#
#    from scipy.sparse import csr_matrix
#
#    nao = 10
#    ifr = 1
#    fock = []
#    over = []
#    with open(f"{project_name}/out.cp2k", "r") as outfile:
#        T_lists = []  # List to hold all T_list instances
#        while True:
#            line = outfile.readline()
#            if not line:
#                break
#            if line.strip().split()[:3] != ["KS", "CSR", "write|"]:
#                continue
#            else:
#                nT = int(line.strip().split()[3])
#                outfile.readline()  # Skip the next line if necessary
#                T_list = []  # Initialize a new T_list for this block
#                for _ in range(nT):
#                    line = outfile.readline()
#                    if not line:
#                        break
#                    T_list.append([np.int32(j) for j in line.strip().split()[1:4]])
#                T_list = np.array(T_list)
#                T_lists.append(T_list)  # Append the T_list to T_lists
#                fock_ = {}
#                over_ = {}
#                for iT, T in enumerate(
#                    T_list
#                ):  # Loop through the translations and load matrices
#                    T = T.tolist()
#                    r, c, data = np.loadtxt(
#                        (
#                            f"{project_name}/FOCK/{project_name}"
#                            f"-KS_SPIN_1_R_{iT+1}-1_{ifr}.csr"
#                        ),
#                        unpack=True,
#                    )
#                    r = np.int32(r - 1)
#                    c = np.int32(c - 1)
#                    fock_[tuple(T)] = csr_matrix(
#                        (data, (r, c)), shape=(nao, nao)
#                    ).toarray()
#
#                    r, c, data = np.loadtxt(
#                        (
#                            f"{project_name}/OVER/{project_name}"
#                            f"-S_SPIN_1_R_{iT+1}-1_{ifr}.csr"
#                        ),
#                        unpack=True,
#                    )
#                    r = np.int32(r - 1)
#                    c = np.int32(c - 1)
#                    over_[tuple(T)] = csr_matrix(
#                        (data, (r, c)), shape=(nao, nao)
#                    ).toarray()
#                fock.append(fock_)
#                over.append(over_)
#                ifr += 1
#


# %%
# You can now save the matrices to ``.npy`` files, and a file with the
# k-grids used in the calculations.
#


# %%
# .. code:: python
#
#    # Save the Hamiltonians
#    np.save("graphene_fock.npy", fock)
#    # Save the overlaps
#    np.save("graphene_ovlp.npy", over)
#    # Write a file with the k-grids, one line per structure
#    np.savetxt("kmesh.dat", [[15,15,1]]*len(frames), fmt='%d')
#


# %%
# Download precomputed data
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#


# %%
# For the sake of simplicity, you can also download precomputed data and
# run just the machine learning part of the notebook using these data.
#

filename = "precomputed.zip"
if not os.path.exists(filename):
    url = (
        "https://github.com/curiosity54/mlelec/raw/"
        "tutorial_periodic/examples/periodic_tutorial/precomputed.zip"
    )
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)

with zipfile.ZipFile(filename, "r") as zip_ref:
    # TODO - this is just to strip data/, later we can update the reference file and
    # simplify this to
    zip_ref.extractall("./")


# %%
# Creating a ``QMDataset`` storing the DFT data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


# %%
# The DFT calculations for the dataset above were performed using a
# **minimal** STO-3G basis. The basis set is specified for each species
# using three quantum numbers, :math:`n`, :math:`l`, :math:`m`. :math:`n`
# is usually a natural number relating to the **radial** extent or
# resolution whereas :math:`l` and :math:`m` specify the **angular
# components** determining the shape of the orbital and its orientation in
# space. For example, :math:`1s` orbitals correspond to :math:`n=2`,
# :math:`l=0` and :math:`m=0`, while a :math:`3p_x` orbital corresponds to
# :math:`n=3`, :math:`l=1` and :math:`m=1`. For the STO-3G basis-set,
# these quantum numbers for Carbon (identified by its atomic number) are
# given as follows.
#

basis = "sto-3g"
orbitals = {
    "sto-3g": {6: [[1, 0, 0], [2, 0, 0], [2, 1, -1], [2, 1, 0], [2, 1, 1]]},
}


# %%
# We instantiate the ``QMDataset`` class that holds all the relevant data
# obtained from a quantum-mechanical (in this case, DFT) calculation. In
# particular, this instance will hold the **frames** which will form the
# train and test structures, along with the corresponding **Hamiltonian**
# (used interchangeably with Fock) and **overlap** matrices in the basis
# specified above, and the :math:`k`-point grid that was used for the
# calculation.
#
# Note that we are currently specifying these matrices in **real-space**,
# :math:`H(\mathbf{T})` , such that the element
# :math:`\langle i nlm| H(\mathbf{T})| i' n'l'm'\rangle` indicates the
# interaction between orbital :math:`nlm` on atom :math:`i` in the
# undisplaced cell (equivalently, translated by
# :math:`\mathbf{T}=\mathbf{0}`) and :math:`n'l'm'` on atom :math:`i'` in
# a periodic copy of cell translated by :math:`\mathbf{T}`.
#
# ** maybe add a figure to explain better the definition of the periodic Hamiltonian **
#
# Alternatively, we can provide the matrices in **reciprocal** (or
# Fourier, :math:`k`) space. These are related to the real-space matrices
# by a *Bloch* sum,
#
# .. math:: H(\mathbf{k})=\sum_{\mathbf{T}}e^{i\mathbf{k}\cdot\mathbf{T}}H(\mathbf{T}),
#
# where, :math:`\mathbf{T}` denotes a lattice translation vector and
# :math:`H(\mathbf{T})` is the corresponding real-space matrix.
#
# In the case the input matrices are in reciprocal space, there should be
# one matrix per :math:`k`-point in the grid.
#

qmdata = QMDataset.from_file(
    # File containing the atomistic structures
    frames_path="data/C2.xyz",
    # File containing the Hamiltonian (of Fock) matrices
    fock_realspace_path="graphene_fock.npy",
    # File containing the overlap matrices
    overlap_realspace_path="graphene_ovlp.npy",
    # File containing the k-point grids used for the DFT calculations
    kmesh_path="kmesh.dat",
    # Physical dimensionality of the system. Graphene is a 2D material
    dimension=2,
    # Device where to run the calculations
    # (can be 'cpu' or 'cuda', if GPUs are available)
    device="cpu",
    # Name of the basis set used for the calculations
    orbs_name=basis,
    # List of quantum numbers associated with the basis set orbitals
    orbs=orbitals[basis],
)


# %%
# Quantities stored in ``QMDataset`` can be accessed as attributes,
# e.g. ``qmdata.fock_realspace`` is a list (one element per structure) of
# dictionaries labeled by the indices of the unit cell real-space
# translations containing ``torch.Tensor``\ s.
#

structure_idx = 0
realspace_translation = 0, 0, 0
print(f"The real-space Hamiltonian matrix for structure {structure_idx} labeled by")
print(f"translation T={realspace_translation} is:")
print(f"{qmdata.fock_realspace[structure_idx][realspace_translation]}")


# %%
# Machine learning data set
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# Symmetries of the Hamiltonian matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


# %%
# The data stored in ``QMDataset`` can be transformed into a format that
# is optimal for machine learning modeling by leveraging the underlying
# **physical symmetries** that characterize the atomistic structure, the
# basis set, and their associated matrices.
#
# The Hamiltonian matrix is a complex learning target, indexed by two
# atoms and the orbitals centered on them. Each :math:`H(\mathbf{k})` is a
# **Hermitian** matrix, while in real space, periodicity introduces a
# **symmetry over translation pairs** such that
# :math:`H(-\mathbf{T}) = H(\mathbf{T})^\dagger`, where the dagger,
# :math:`\dagger`, denotes Hermitian conjugation.
#
# To address the symmetries associated with swapping atomic indices or
# orbital labels, we divide the matrix into **blocks labeled by pairs of
# atom types**.
#
# -  ``block_type = 0``, or **on-site** blocks, consist of elements
#    corresponding to the interaction of orbitals on the same atom,
#    :math:`i = i'`.
#
# -  ``block_type = 2``, or **cross-species** blocks, consist of elements
#    corresponding to orbitals centered on atoms of distinct species.
#    Since the two atoms can be distinguished, they can be consistently
#    arranged in a predetermined order.
#
# -  ``block_type = 1, -1``, or **same-species** blocks, consist of
#    elements corresponding to orbitals centered on distinct atoms of the
#    same species. As these atoms are indistinguishable and cannot be
#    ordered definitively, the pair must be symmetrized for permutations.
#    We construct symmetric and antisymmetric combinations
#    :math:`(
#    \langle inlm|H(\mathbf{T})|i'n'l'm'\rangle\pm\
#    \langle i'nlm|H(\mathbf{-T})|inlm\rangle
#    )`
#    that correspond to ``block_type`` :math:`+1` and :math:`-1`,
#    respectively.
#


# %%
# Equivariant structure of the Hamiltonians
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


# %%
# Even though the Hamiltonian operator under consideration is invariant,
# **its representation transforms under the action of structural rotations
# and inversions** due to the choice of the basis functions. Each of the
# blocks has elements of the form
# :math:`\langle i nlm| H | i' n'l'm'\rangle`, which are in an
# **uncoupled** representation and transform as a product of (real)
# spherical harmonics, :math:`Y_l^m \otimes Y_{l'}^{m'}`.
#
# This product can be decomposed into a direct sum of irreducible
# representations (irreps) of :math:`\mathrm{SO(3)}`,
#
# .. math:: \lambda \mu:\lambda \in [|l_1-l_2|,l_1+l_2],\mu \in [-\lambda,\lambda],
#
# which express the Hamiltonian blocks in terms of contributions that
# rotate independently and can be modeled using a feature that
# geometrically describes the pair of atoms under consideration and shares
# the same symmetry.
#
# The resulting irreps form a **coupled** representation, each of which
# transforms as a spherical harmonic :math:`Y^\mu_\lambda` under
# :math:`\mathrm{SO(3)}` rotations, but may exhibit more complex behavior
# under inversions. For example, spherical harmonics transform under
# inversion, :math:`\hat{i}`, as polar tensors:
#
# .. math:: \hat{i}Y^\mu_\lambda = (-1)^\lambda Y^\mu_\lambda.
#
# Some of the coupled basis terms instead transform as pseudotensors,
#
# .. math:: \hat{i}H_{nl,n'l',\lambda}^\mu=(-1)^{\lambda+1}H_{nl,n'l',\lambda}^\mu.
#
# For more details about the block decomposition, please refer to `Nigam
# et al., J. Chem. Phys. 156, 014115
# (2022) <https://pubs.aip.org/aip/jcp/article/156/1/014115/2839817>`__.
#
# The following is an animation of a trajectory along a `Lissajous
# curve <https://en.wikipedia.org/wiki/Lissajous_curve>`__ in 3D space,
# alongside a colormap representing the values of the real-space
# Hamiltonian matrix elements of the graphene unit cell in a minimal
# STO-3G basis. From the animation, it is evident how invariant elements,
# such as those associated with interactions between :math:`s` orbitals,
# do not change under structural rotations. On the other hand,
# interactions allowing for equivariant components, such as the
# :math:`s`-:math:`p` block, change under rotations.
#

image_files = sorted(
    [
        f"data/frames/{f}"
        for f in os.listdir("./data/frames")
        if f.startswith("rot_") and f.endswith(".png")
    ]
)
images = [mpimg.imread(img) for img in image_files]
fig, ax = plt.subplots()
img_display = ax.imshow(images[0])
ax.axis("off")


def update(frame):
    img_display.set_data(images[frame])
    return [img_display]


ani = FuncAnimation(fig, update, frames=len(images), interval=20, blit=True)


# %%
# .. code:: python
#
#    from IPython.display import HTML
#    HTML(ani.to_jshtml())
#


# %%
# Mapping geometric features to Hamiltonian targets
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


# %%
# Each Hamiltonian block obtained from the procedure `described
# above <#symmetries-of-the-hamiltonian-matrix>`__ can be modeled using
# symmetrized features,
# :math:`|\overline{\rho_{ii'}^{\otimes \nu}; \lambda\mu }\rangle`.
#
# Elements of ``block_type=0`` are indexed by a single atom and are best
# described by a symmetrized atom-centered density correlation
# (`ACDC <https://doi.org/10.1063/1.5090481>`__),
# :math:`|\overline{\rho_{i}^{\otimes \nu}; \sigma; \lambda\mu }\rangle`,
# where :math:`\nu` refers to the correlation (body)-order, and—just as
# for the blocks—:math:`\lambda \mu` indicate the :math:`\mathrm{SO(3)}`
# irrep to which the feature is symmetrized. The symbol :math:`\sigma`
# denotes the inversion parity.
#
# For other blocks, such as ``block_type=2``, which explicitly reference
# two atoms, we use `two-center <https://doi.org/10.1063/5.0072784>`__
# ACDCs, :math:`|\overline{\rho_{ii'}^{\otimes \nu}; \lambda\mu }\rangle`.
#
# For ``block_type=1, -1``, we ensure equivariance with respect to atom
# index permutation by constructing symmetric and antisymmetric pair
# features:
# :math:`(
# |\overline{\rho_{ii'}^{\otimes \nu};\lambda\mu }\rangle\pm\
# |\overline{\rho_{i'i}^{\otimes \nu};\lambda\mu }\rangle
# )`.
#


# %%
# The features are discretized on a basis of radial functions and
# spherical harmonics, and their performance may depend on the
# **resolution** of the functions included in the model. There are
# additional hyperparameters, such as the **cutoff** radius, which
# controls the extent of the atomic environment, and Gaussian widths. In
# the following, we allow for flexibility in discretizing the
# atom-centered and two-centered ACDCs by defining the hyperparameters for
# the single-center (SC) :math:`\lambda`-SOAP and two-center (TC) ACDC
# descriptors.
#
# The single and two-center descriptors have very similar hyperparameters,
# except for the cutoff radius, which is larger for the two-center
# descriptors to explicitly include distant pairs of atoms.
#
# Note that the descriptors of pairs of atoms separated by distances
# greater than the cutoff radius are identically zero. Thus, any model
# based on these descriptors would predict an identically zero value for
# these pairs.
#

SC_HYPERS = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.5,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
}

TC_HYPERS = {
    "cutoff": 6.0,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
}


# %%
# We then use the above defined hyperparameters to compute the descriptor
# and initialize a ``MLDataset`` instance, which contains, among other
# things, the Hamiltonian block decomposition and the geometric features
# described above.
#
# In addition to computing the descriptors, ``MLDataset`` takes the data
# stored in the ``QMDataset`` instance and puts it in a form required to
# train a ML model.
#


# %%
# The ``item_names`` argument is a list of names of the quantities we want
# to compute and target in the ML model training, or that we want to be
# able to access later.
#
# For example, ``fock_blocks`` is a
# `metatensor.Tensormap <https://tinyurl.com/tenmap>`__ containing the
# Hamiltonian coupled blocks. We also want to access the overlap matrices
# in :math:`k`-space (``overlap_kspace``) to be able to compute the
# Hamiltonian eigenvalues in the :math:`k`-grid.
#

mldata = MLDataset(
    # A QMDataset instance
    qmdata,
    # The names of the quantities to compute/initialize for the training
    item_names=["fock_blocks", "overlap_kspace"],
    # Hypers for the SC descriptors
    hypers_atom=SC_HYPERS,
    # Hypers for the TC descriptors
    hypers_pair=TC_HYPERS,
    # Cutoff for the angular quantum number to use in the Clebsh-Gordan iterations
    lcut=4,
    # Fraction of structures in the training set
    train_frac=0.7,
    # Fraction of structures in the validation set
    val_frac=0.2,
    # Fraction of structures in the test set
    test_frac=0.1,
    # Whether to shuffle or not the structure indices before splitting the data set
    shuffle=True,
)


# %%
# The matrix decomposition into blocks and the calculations of geometric
# features is performed by the ``MLDataset`` class.
#


# %%
# ``mldata.features`` is ``metatensor.TensorMap`` containing the
# stuctures’ descriptors
#

mldata.features


# %%
# ``mldata.items`` is a ``namedtuple`` containing the quantities defined
# in ``item_names``. e.g. the coupled Hamiltonian blocks:
#

print("The TensorMap containing the Hamiltonian coupled blocks is")
mldata.items.fock_blocks


# %%
# Or the overlap matrices:
#

structure_idx = 0
k_idx = 0
print(f"The overlap matrix for structure {structure_idx} at the {k_idx}-th k-point is")
mldata.items.overlap_kspace[structure_idx][k_idx]


# %%
# Build a machine learning model for the electronic Hamiltonian of graphene
# -------------------------------------------------------------------------
#


# %%
# For the sake of simplicity and time, the following cells define and
# train a linear model targeting the Hamiltonian coupled blocks. The
# weights are optimized via Ridge regression as implemented in
# `scikit-learn <https://scikit-learn.org/stable/>`__.
#
# In case the data set is more complex than the simple example provided
# here, or in case a more flexible model is required, one might decide to
# train a more general neural network via gradient descent, instead of
# solving a linear problem. The following cells outline how to do it using
# ``mlelec``.
#


# %%
# Model’s architecture
# ~~~~~~~~~~~~~~~~~~~~
#


# %%
# The model consists of several submodels, one for each Hamiltonian
# coupled block. Each submodel is a `multilayer
# perceptron <https://en.wikipedia.org/wiki/Multilayer_perceptron>`__
# (MLP) that maps the corresponding set of geometric features to the
# Hamiltonian coupled block. Nonlinearities are applied to the invariants
# constructed from each equivariant feature block using the
# ``EquivariantNonlinearity`` module.
#


# %%
# The architecture of ``EquivariantNonlinearity`` can be visualized with
# ``torchviz`` with the following snippet:
#
# .. code:: python
#
#    import torch
#    from mlelec.models.equivariant_nonlinear_model import EquivariantNonLinearity
#    from torchviz import make_dot
#    m = EquivariantNonLinearity(torch.nn.SiLU(), layersize = 10)
#    y = m.forward(torch.randn(3,3,10))
#    dot = make_dot(y, dict(m.named_parameters()))
#    dot.graph_attr.update(size='150,150')
#    dot.render("data/equivariantnonlinear", format="png")
#    dot
#


# %%
# .. figure:: equivariantnonlinear.png
#    :alt: Graph representing the architecture of EquivariantNonLinearity
#    :width: 300px
#
#    Graph representing the architecture of EquivariantNonLinearity
#


# %%
# The global architecture of the MLP, implemented in ``simpleMLP``, can be
# visualized with
#
# .. code:: python
#
#    import torch
#    from mlelec.models.equivariant_nonlinear_model import simpleMLP
#    from torchviz import make_dot
#    mlp = simpleMLP(
#        nin=10,
#        nout=1,
#        nhidden=1,
#        nlayers=1,
#        bias=True,
#        activation='SiLU',
#        apply_layer_norm=True
#        )
#    y = mlp.forward(torch.randn(1,1,10))
#    dot = make_dot(y, dict(mlp.named_parameters()))
#    dot.graph_attr.update(size='150,150')
#    dot.render("data/simpleMLP", format="png")
#    dot
#


# %%
# .. figure:: simpleMLP.png
#    :alt: Graph representing the architecture of simpleMLP
#    :width: 600px
#
#    Graph representing the architecture of simpleMLP
#


# %%
# Model initialization
# ~~~~~~~~~~~~~~~~~~~~
#


# %%
# The training loop is handled by `PyTorch
# Lightning <https://lightning.ai/docs/pytorch/stable/>`__, hence the name
# ``LitEquivariantNonlinearModel`` for the wrapper to
# ``EquivariantNonlinearModel``. Initializing
# ``LitEquivariantNonlinearModel`` requires a ``MLDataset`` instance, and
# the information about the model’s architecture, the optimizer, and the
# learning rate (LR) scheduler.
#
# To initialize a linear model, we ask the architecture to have no hidden
# layers with ``nlayers=0``. We pass ``init_from_ridge`` to initialize the
# weigths and biases from Ridge regression. Then, in case we want to
# further optimize the weights through gradient descent, we define the
# optimizer to be
# `LBFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`__, which is
# more robust and works well with linear models. For more general
# architectures, LBFGS might become too slow, and passing
# ``optimizer='Adam'`` might be more convenient.
#

model = LitEquivariantNonlinearModel(
    mldata=mldata,  # a MLDataset instance
    nlayers=0,  # The number of hidden layers
    nhidden=64,  # The number of neurons in each hidden layer
    init_from_ridge=True,  # If True, initialize the weights and biases of the
    # purely linear model from Ridge regression
    optimizer="LBFGS",  # Type of optimizer. Adam is likely better for
    # a more general neural network
    # activation="SiLU", # The nonlinear activation function
    learning_rate=1e-3,  # Initial learning rate (LR)
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.7,
    lr_scheduler_min_lr=1e-6,
    loss_fn=MSELoss(),  # Use the mean square error as loss function
)


# %%
# Evaluating the cell above already pre-trains the model with Ridge
# weights. In case this is not enough, or you want to try with more
# general architectures, modify the previous cell and run the following
# cells to set up a gradient descent training loop.
#


# %%
# Set up the training loop
# ^^^^^^^^^^^^^^^^^^^^^^^^
#


# %%
# Import additional modules not required in the rest of the tutorial
#


# %%
# .. code:: python
#
#    import lightning.pytorch as pl
#    from lightning.pytorch.callbacks import EarlyStopping
#    from mlelec.callbacks.logging import LoggingCallback
#    from mlelec.models.equivariant_nonlinear_lightning import MLDatasetDataModule
#


# %%
# We set up a callback for logging training information such as the
# training and validation losses.
#


# %%
# .. code:: python
#
#    logger_callback = LoggingCallback(log_every_n_epochs=5)
#


# %%
# We set up an early stopping criterion to stop the training when the
# validation loss function stops decreasing.
#


# %%
# .. code:: python
#
#    early_stopping = EarlyStopping(
#        monitor="val_loss", min_delta=1e-3, patience=10, verbose=False, mode="min"
#    )
#


# %%
# We define a ``lighting.pytorch.Trainer`` instance to handle the training
# loop. We train for 200 epochs.
#
# Note that PyTorch Lightning requires the definition of a data module to
# instantiate DataLoaders to be used during the training.
#


# %%
# .. code:: python
#
#    data_module = MLDatasetDataModule(mldata, batch_size=16, num_workers=0)
#
#    trainer = pl.Trainer(
#        max_epochs=200,
#        accelerator="cpu",
#        check_val_every_n_epoch=10,
#        callbacks=[early_stopping, logger_callback],
#        logger=False,
#        enable_checkpointing=False,
#    )
#
#    trainer.fit(model, data_module)
#


# %%
# We compute the test set loss to assess the model accuracy on an unseen
# set of structures
#


# %%
# .. code:: python
#
#    trainer.test(model, data_module)
#


# %%
# Model’s accuracy in reproducing derived properties
# --------------------------------------------------
#


# %%
# The Hamiltonian coupled blocks predicted by the ML model can be accessed
# from ``model.forward()``
#

predicted_blocks = model.forward(mldata.features)


# %%
# The real-space blocks can be transformed to Hamiltonian matrices via the
# ``mlelec.utils.pbc_utils.blocks_to_matrix`` function. The resulting
# real-space Hamiltonians can be Fourier transformed to give the
# :math:`k`-space ones.
#

HT = blocks_to_matrix(
    predicted_blocks,
    orbitals["sto-3g"],
    {A: qmdata.structures[A] for A in range(len(qmdata))},
    detach=True,
)
Hk = qmdata.bloch_sum(HT, is_tensor=True)


# %%
# Kohn-Sham eigenvalues
# ~~~~~~~~~~~~~~~~~~~~~
#


# %%
# We can then compute the eigenvalues on the :math:`k`-grid used for the
# calculation to assess the model accuracy in predicting band energies.
#

target_eigenvalues = compute_eigenvalues(qmdata.fock_kspace, qmdata.overlap_kspace)
predicted_eigenvalues = compute_eigenvalues(Hk, qmdata.overlap_kspace)

Hartree = 27.211386024367243  # eV

plt.rcParams["font.size"] = 14
fig, ax = plt.subplots()
ax.set_aspect("equal")

x_text = 0.38
y_text = 0.2
d = 0.06

for i, (idx, label) in enumerate(
    zip(
        [mldata.train_idx, mldata.val_idx, mldata.test_idx],
        ["train", "validation", "test"],
    )
):

    target = (
        torch.stack([target_eigenvalues[i] for i in idx]).flatten().detach() * Hartree
    )
    prediction = (
        torch.stack([predicted_eigenvalues[i] for i in idx]).flatten().detach()
        * Hartree
    )

    non_core_states = target > -100
    rmse = np.sqrt(
        np.mean(
            (target.numpy()[non_core_states] - prediction.numpy()[non_core_states]) ** 2
        )
    )
    ax.scatter(target, prediction, marker=".", label=label, alpha=0.5)
    ax.text(
        x=x_text,
        y=y_text - d * i,
        s=rf"$\mathrm{{RMSE_{{{label}}}={rmse:.2f}\,eV}}$",
        transform=ax.transAxes,
    )

xmin, xmax = ax.get_xlim()
ax.plot([xmin, xmax], [xmin, xmax], "--k")
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.legend()
ax.set_xlabel("Target eigenvalues (eV)")
ax.set_ylabel("Predicted eigenvalues (eV)")


# %%
# Graphene band structure
# ~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# Apart from the eigenvalues on a mesh in the Brillouin zone, we can use
# the real-space Hamiltonians predicted by the model to compute the band
# structure along high-symmetry paths.
#

fig, ax = plt.subplots(figsize=(8, 4.8))

idx = 0

handles = []

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Plot reference
    ax, h_ref = plot_bands_frame(
        qmdata.fock_realspace[idx], idx, qmdata, ax=ax, color="blue", lw=2
    )

    # Plot prediction
    ax, h_pred = plot_bands_frame(
        HT[idx], idx, qmdata, ax=ax, color="lime", ls="--", lw=2
    )

ax.set_ylim(-30, 30)
ax.legend(
    [h_ref, h_pred],
    ["Reference", "Prediction"],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
fig.tight_layout()

# %%
