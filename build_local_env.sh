#pyenv install 3.6.9
pyenv local 3.6.9
python -m venv .venv
cd container
poetry install
cd ..
source .venv/bin/activate