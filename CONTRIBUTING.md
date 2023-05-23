# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in the ways listed below.

## Adding an example/tutorial

Examples are implemented as `sphinx-gallery` projects, and are compiled automatically into an HTML documentation, from which they can be downloaded as Jupyter notebooks.
You will find examples of tutorials in the `examples/` folder of the repository. 
Each new example consists of a folder, e.g. `new_example/`, which should, at the very 
least, contain a `README.rst` file that describes briefly what is contained in the example, and a python file that contains the example itself. 
This file should be formatted as expected by `sphinx-gallery` (see the [project homepage](https://sphinx-gallery.github.io/stable/index.html) for examples and documentation. 
You should also add a reference to the new example to the sphinx configuration in the `docs/src`. 

## Report Bugs

Report bugs using GitHub issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

## Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

## Write Documentation

The `software cookbok` is primarily a collection of tutorials and usage examples
for atomistic modeling techniques - see above for a brief overview of how to add
a new examples. However, the build, testing and validation process for these 
examples also requires dedicated code, and so you are also encouraged to contributing
to documenting (and/or improving) this support infrastructure.

## Submit Feedback

The best way to send feedback is to file an issue on GitHub.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started

Ready to contribute? Here's how to set up `software cookbok` for local development.

1. Fork the repo on GitHub.
2. Clone your fork locally.
3. Install your local copy into a virtualenv, e.g., using `conda`.
4. Create a branch for local development and make changes locally.
5. Commit your changes and push your branch to GitHub.
6. Submit a pull request through the GitHub website.

## Code of Conduct

Please note that the COSMO_cookbook project is released with a [Contributor Code of Conduct](CONDUCT.md). By contributing to this project you agree to abide by its terms.
