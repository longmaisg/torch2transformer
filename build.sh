#!/bin/bash

# start a new git repository
git init
git add .
git commit -m "Initial torch2transformer scaffold"
git branch -M main

git remote add origin git@github.com:longmaisg/torch2transformer.git
git remote -v

git push -u origin main

# install the package in editable mode and test
uv pip install -e .
python -c "from torch2transformer import wrap_model; print(wrap_model)"

# build and upload to PyPI
uv pip install build twine
python -m build
echo $PyPY_API_TOKEN | pbcopy
twine upload dist/torch2transformer-0.1.0*


