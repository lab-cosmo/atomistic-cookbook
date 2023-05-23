# Tutorial for Using LAMMPS-Style Pair Potentials with GAP Models
Hello!

Welcome to the greatest (and funniest) tutorial in the pair potential + GAP model space. In this tutorial, we will create and use a LAMMPS-style pair potential alongside a Librascal-fitted Gaussian Approximation Potential (GAP) to run Path-Integral Molecular Dynamics (PIMD) simulations. The notebook will give a more complete introduction on what you will learn, but please be aware that I will not be explaining the basics of how to create a GAP potential and how to run PIMD simulations. The notebook will, however, contain all the code needed to build the models, as well as instructions on how to run LAMMPS/Quantum Espresso/i-PI from the command line. There will also be some helpful links in case your knowledge is lacking in certain areas. In case you don't want to go through the pain of compiling the aforementioned software, you can load some pre-computed files (found in this repository) in order to run the notebook independetly and in its entirety. I of course *recommend* that you run all the calculations yourself, but as the saying goes, you can take a horse to water, but you can't force it to drink.

If you have any questions regarding this tutorial, feel free to send me, Victor Principe, an email at victorprincipe1206@gmail.com.

## Installation Instructions

### Easy Installation

If you just want everything to work and have conda installed, here's all you need to do:

```
conda env create -f environment.yml
conda activate ppp
jupyter notebook
```

Otherwise, see below for the necessary and optional packages for this tutorial.

### Essential Packages
In order to run the notebook, you must have an environment with some essential packages. Most of these can be installed using

`pip install jupyter numpy pandas pickle ase matplotlib skcosmo chemiscope scipy scikit-learn tqdm`

You will also need to install Librascal, which can be done using the code below

```
git clone https://github.com/lab-cosmo/librascal
cd librascal
pip install .
```

### Optional software

If you would like to gain the most out of this tutorial, you will also need to compile the following software. More information can be found at the links provided.

- LAMMPS - https://www.lammps.org/
- Quantum Espresso - https://www.quantum-espresso.org
- i-PI - http://ipi-code.org

### Clone This Repository

Once you have installed the necessary packages (and the optional ones, if you so wish), you can clone this repository and open the Jupyter console by doing

```
git clone https://github.com/victorprincipe/pair_potentials
cd pair_potentials
jupyter notebook
```

Then you can just open the notebook and enjoy the tutorial! 

