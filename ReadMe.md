# team d main

## python環境について

基本的には `requirements.txt` を使えばいいが、 `tools/csv_preprocessor` とそれを使ったプログラム `normalize_question.py`  は `tools/csv_preprocessor/requirements.txt` を使う必要がある

これは、同じ環境にしようとするとバージョンのconflictが起きるためである

`.venv_main` と `.venv_csv_preprocessor` にそれぞれ `requirements.txt` をインストールするコードは `setup_python_environment.sh` にあるため、新しい環境ではそれを実行するとよい

## dataset

`dataset/original/` に試験データを入れること

`normalize_question.sh` が `dataset/original/` にあるファイルを正規化して `dataset/normalized/` に保存する

また、 `dataset/normalized/` の中にあるデータは `.gitignore` によって無視されるため、 `dataset/normalized/` にあるデータは `git add` しても意味がない

なので、新しい環境では毎回 `normalize_question.sh` を実行する必要がある

