# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
# SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PROJECT 						= $(shell dirname $(CURDIR))


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	-rm -rf $(BUILDDIR)/*
	-find $(SOURCEDIR) -name '*.rst' ! -name 'index.rst' -exec rm -rf {} +

build:
	sphinx-apidoc -o $(SOURCEDIR) "$(PROJECT)/src"

livehtml: build
	sphinx-autobuild -b html --port 8888 $(SOURCEDIR) $(BUILDDIR)/html & start http://127.0.0.1:8888