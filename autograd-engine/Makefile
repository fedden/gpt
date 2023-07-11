# We allow the following Python versions.
ALLOWED_PYTHON_VERSION=3.11.*
# Get the version of python from the environment variable.
ACTUAL_PYTHON_VERSION=$(shell python -V | cut -d' ' -f2)


# Recipe for for checking dependencies are installed, like pyenv.
check:
	@# Check pyenv is installed.
	@which pyenv || (echo "pyenv is not installed, please install it." && exit 1)
	@# Check the correct version of python is installed.
	@pyenv versions | grep $(ALLOWED_PYTHON_VERSION) || (echo "python version $(ALLOWED_PYTHON_VERSION) is not installed, please install it." && exit 1)


use_env:
	@# Set pyenv to use the correct version of python.
	@pyenv local $(ACTUAL_PYTHON_VERSION)
	@# Activate the virtual environment.
	@. .venv/bin/activate


# Add recipe for making the environment.
init: check
	@# Set pyenv to use the correct version of python.
	@pyenv local $(ACTUAL_PYTHON_VERSION)
	@# Create the virtual environment.
	@python -m venv .venv
	@# Activate the virtual environment.
	@. .venv/bin/activate
	@# Install the requirements.
	@python -m pip install -r requirements.txt
	@# Install ag.
	@python -m pip install -e .

# Add recipe for running the tests.
test: use_env
	pytest -sx tests 


# Add recipe for cleaning the environment.
clean:
	rm -rf .venv
