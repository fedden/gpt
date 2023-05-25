.PHONY: docs
SHELL=/bin/bash

PYTHON_VERSION = 3.11.3

# The python command; might sometimes need to be fiddled around:
PYTHON = python

# Name (usually the same) used for the project, the subfolder to analyze, and the kernel:
KERNEL_NAME = gpt

# List all Makefile targets:
# Shamelessly borrowed ad modified from https://gist.github.com/pvdb/777954
# An alternative version for make 3.8.x:
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
.PHONY: list
list:
	@make -rpn | sed -n -e '/^$$/ { n ; /^[^ .#][^ ]*:/ { s/:.*$$// ; p ; } ; }' | sort -u


# Raise error if pyenv is not installed:
.PHONY: pyenv-exists
pyenv-exists:
	@command -v pyenv >/dev/null 2>&1 || { echo >&2 "pyenv is not installed. Aborting."; exit 1; }
	# pyenv is installed, check if the required python version is available:
	@pyenv versions | grep $(PYTHON_VERSION) >/dev/null 2>&1 || { echo >&2 "Python $(PYTHON_VERSION) is not installed. Aborting."; exit 1; }


.PHONY: venv_exists
venv-exists: pyenv-exists
	# First set to use the correct python version:
	pyenv local $(PYTHON_VERSION)
	# Then check if the venv exists:
	$(PYTHON) -m venv .venv; \
	source .venv/bin/activate; \
	$(PYTHON) -m pip install --upgrade pip; \
	deactivate


# Create or update the underlying venv:
.PHONY: venv-exists
venv: venv-exists 
	source .venv/bin/activate; \
	$(PYTHON) -m pip install -r requirements.txt; \
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
	rm -rf ~/nltk_data


# Install or update the virtualenv;
# Careful that switching MODE may require a `make clean` to ensure a fresh state:
.PHONY: init
init: venv install-kernel
	source .venv/bin/activate; \
	python -m nltk.downloader -d ~/nltk_data all; \
