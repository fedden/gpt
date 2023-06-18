.PHONY: docs
SHELL=/bin/bash

# The python command; might sometimes need to be fiddled around:
PYTHON = python3

# Name (usually the same) used for the project, the subfolder to analyze, and the kernel:
KERNEL_NAME = adapting-gpt

# List all Makefile targets:
# Shamelessly borrowed ad modified from https://gist.github.com/pvdb/777954
# An alternative version for make 3.8.x:
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
.PHONY: list
list:
	@make -rpn | sed -n -e '/^$$/ { n ; /^[^ .#][^ ]*:/ { s/:.*$$// ; p ; } ; }' | sort -u


.PHONY: venv_exists
venv-exists:
	$(PYTHON) -m venv .venv; \
	source .venv/bin/activate; \
	$(PYTHON) -m pip install --upgrade pip==22.3.1; \
	deactivate


# Create or update the underlying venv:
.PHONY: venv-exists
venv: venv-exists 
	source .venv/bin/activate; \
	$(PYTHON) -m pip install click joblib torch torchvision torchaudio matplotlib pytorch_lightning numpy requests; \
	deactivate


# Expose the venv to jupyter:
.PHONY: install-kernel
install-kernel:
	source .venv/bin/activate; \
	$(PYTHON) -m pip install ipykernel; \
	$(PYTHON) -m ipykernel install --user --name=$(KERNEL_NAME); \
	deactivate


# Remove the kernel:
.PHONY: remove-kernel
remove-kernel:
	source .venv/bin/activate; \
	jupyter kernelspec uninstall $(KERNEL_NAME); \
	deactivate


# Clean the environment etc. No effect on any source controlled content including Lock.
.PHONY: clean
clean: remove-kernel
	rm -rf ./.venv
	rm -rf ./build


# Install or update the virtualenv;
# Careful that switching MODE may require a `make clean` to ensure a fresh state:
.PHONY: init
init: venv install-kernel
