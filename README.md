# Human-Segmentation

## Tools used in this project

* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management - [article](https://mathdatasimplified.com/2023/06/12/poetry-a-better-way-to-manage-python-dependencies/)
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://mathdatasimplified.com/2023/05/25/stop-hard-coding-in-a-data-science-project-use-configuration-files-instead/)
* [sphinx](https://github.com/sphinx-doc/sphinx): The Sphinx documentation generator

## Set up the environment

1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:

```bash
make env
```

## Install dependencies

To install all dependencies for this project, run:

```bash
poetry install
```

To install a new package, run:

```bash
poetry add <package-name>
```

## Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```