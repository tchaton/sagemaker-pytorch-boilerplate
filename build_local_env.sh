PYTHON_VERSION=$1
#pyenv install ${PYTHON_VERSION}
pyenv local ${PYTHON_VERSION}
python -m venv .venv
cd container
poetry install
cd .. && source .venv/bin/activate