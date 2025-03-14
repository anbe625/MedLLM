#!/bin/bash
set -x

rm -rf .venv .venv_main .venv_csv_preprocessor

python3 -m venv .venv_main
python3 -m venv .venv_csv_preprocessor

.venv_main/bin/pip install --upgrade pip
.venv_csv_preprocessor/bin/pip install --upgrade pip

.venv_main/bin/pip install -r requirements.txt
.venv_csv_preprocessor/bin/pip install -r tools/csv_preprocessor/requirements.txt
