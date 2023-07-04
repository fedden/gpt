PYENV_VERSION=3.11.2


# Recipe for for checking dependencies are installed, like pyenv.
check:
	# Check pyenv is installed.
	which pyenv || (echo "pyenv is not installed, please install it." && exit 1)
	# Check the correct version of python is installed.
	pyenv versions | grep $(PYENV_VERSION)  || (echo "pyenv version $(PYENV_VERSION) is not installed, please install it." && exit 1)


use_env:
	# Set pyenv to use the correct version of python.
	pyenv local $(PYENV_VERSION)
	# Activate the virtual environment.
	. .venv/bin/activate


# Add recipe for making the environment.
init: check
	# Set pyenv to use the correct version of python.
	pyenv local $(PYENV_VERSION)
	# Create the virtual environment.
	python -m venv .venv
	# Install the requirements.
	pip install -r requirements.txt
	# Install ag.
	pip install -e .

# Add recipe for running the tests.
test: use_env
	pytest -sx tests 


# Add recipe for cleaning the environment.
clean:
	rm -rf .venv
