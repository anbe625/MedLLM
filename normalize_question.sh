#!/bin/bash
set -x

.venv_csv_preprocessor/bin/python3 normalize_question.py dataset/original/kaggle_1.csv dataset/normalized/kaggle_1.csv
.venv_csv_preprocessor/bin/python3 normalize_question.py dataset/original/kaggle_2.csv dataset/normalized/kaggle_2.csv
.venv_csv_preprocessor/bin/python3 normalize_question.py dataset/original/talk_q.csv dataset/normalized/talk_q.csv

