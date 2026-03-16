# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
SPHINXAPIDOC  ?= uv run python -m sphinx.ext.apidoc
APIOUTDIR     = $(SOURCEDIR)/api
PACKAGEDIR    = ../src/stanbkt

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help apidoc html Makefile

apidoc:
	@$(SPHINXAPIDOC) -f -e -M -o "$(APIOUTDIR)" "$(PACKAGEDIR)"
	@# Remove :members: from parent packages to avoid duplicate warnings
	@sed -i '/^   :members:/d' "$(APIOUTDIR)/stanbkt.fits.rst"
	@sed -i '/^   :members:/d' "$(APIOUTDIR)/stanbkt.models.rst"
	@sed -i '/^   :members:/d' "$(APIOUTDIR)/stanbkt.models.core.rst"
	@sed -i '/^   :undoc-members:/d' "$(APIOUTDIR)/stanbkt.fits.rst"
	@sed -i '/^   :undoc-members:/d' "$(APIOUTDIR)/stanbkt.models.rst"
	@sed -i '/^   :undoc-members:/d' "$(APIOUTDIR)/stanbkt.models.core.rst"

html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
