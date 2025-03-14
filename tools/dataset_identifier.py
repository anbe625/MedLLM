#!/usr/bin/env python3
import unicodedata
from typing import Optional

from csv_reader import read_csv

def is_include_choic(text: str):
    """
    文字列に 'a'、'b'、'c'、'd'、'e' のいずれかが、半角・全角、大文字・小文字に関係なく含まれているか判定します。

    手順:
      1. 入力文字列を Unicode 正規化（NFKC）して、全角と半角、及び大文字と小文字を統一します。
      2. 正規化した文字列内に、'a'～'e' のいずれかが含まれていれば True を返します。

    Parameters:
        text (str): 判定対象の文字列

    Returns:
        bool: 'a'～'e' のいずれかが含まれていれば True、そうでなければ False
    """
    normalized_text = unicodedata.normalize('NFKC', text).lower()
    return any(c in normalized_text for c in 'abcde')


def is_national_exam(csv_path: str, survey_lines: Optional[int]= None):
    """
    CSVファイル内の調査データを解析し、各質問文に 'a'～'e' の文字が含まれている割合から
    国家試験のデータであるかどうかを判定します。

    手順:
      1. 指定されたCSVファイルからデータを読み込み、最初の survey_lines 行を対象にします。
      2. 各行の2番目の要素（質問文）に対して、is_include_choic() 関数を用いて判定を行います。
      3. 条件を満たさない質問については、標準出力に区切り線と共に表示します。
      4. 対象行のうち、半数以上が 'a'～'e' を含む場合は True を返し、そうでなければ False を返します。

    Parameters:
        csv_path (str): 調査データが保存されたCSVファイルのパス
        survey_lines (int, optional): 判定に使用する先頭行数（デフォルト(None)は全体）

    Returns:
        bool: 調査データが国家試験のものであると判定されれば True、そうでなければ False
    """
    data = read_csv(csv_path)

    if survey_lines is None:
        survey_lines = len(data)

    choices_included_num = 0
    for row in data[:survey_lines]:
        question = row[1]
        if is_include_choic(question):
            choices_included_num += 1

    return choices_included_num / survey_lines >= 0.5

if __name__ == '__main__':
    r = is_national_exam('../dataset/kaggle_public.csv')
    print(r)
    r = is_national_exam('../dataset/original_qa.csv')
    print(r)
