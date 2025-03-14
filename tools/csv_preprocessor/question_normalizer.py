#!/usr/bin/env python
import re
import unicodedata
import spacy
import json

from .csv_reader import read_csv

# 日本語のspaCyモデル（GiNZA）をロードする
nlp = spacy.load("ja_ginza")


def normalize_by_regular_expression(text):
    """
    正規表現を用いて、問題文と選択肢に分割する（正規化済みのテキストに対して）。

    この関数はまず、UnicodeのNFKC正規化を実施し、全角文字を半角に変換します。
    その後、指定した正規表現パターンに基づいてテキストを分割し、
    テキストの先頭部分を問題文、以降の各セクションをラベル（A～E）に対応する選択肢とします。

    パターンは、ラベル（A～Eまたはa～e）の後に「)」「．」または「:」が続き、その後に
    任意の空白があるものを想定しています。

    Parameters:
        text (str): 問題文と選択肢が含まれる元の文字列。

    Returns:
        dict: 以下のキーを持つ辞書を返す。
              - "question": 抽出された問題文部分。
              - "A", "B", "C", "D", "E": 各選択肢のテキスト。
              正規表現による分割がうまく行かない場合は、全体が問題文として返される。
    """
    normalized_text = unicodedata.normalize("NFKC", text)
    segments = {"question": "", "A": "", "B": "", "C": "", "D": "", "E": ""}
    pattern = r"(?P<label>[A-Ea-e])[).．:]\s*"
    parts = re.split(pattern, normalized_text)
    if len(parts) < 3:
        segments["question"] = normalized_text.strip()
        return segments
    segments["question"] = parts[0].strip()
    for i in range(1, len(parts) - 1, 2):
        label = parts[i]
        option_text = parts[i + 1].strip()
        if label in segments:
            segments[label] = option_text
    return segments


def normalize_by_nlp(text):
    normalized_text = unicodedata.normalize("NFKC", text)
    doc = nlp(normalized_text)

    segments = {
        "original": text,
        "question": "",
        "a": "",
        "b": "",
        "c": "",
        "d": "",
        "e": "",
    }

    searching_options = ["a", "b", "c", "d", "e"]

    # 選択肢ラベルと思われるトークンの位置を収集する
    label_indices = []
    for token in doc:
        # 1文字以外のトークンはラベルとして認めない
        if len(token.text) != 1:
            continue

        # 選択肢ラベルとして有効な文字でない場合は違う
        if not token.text in searching_options:
            continue

        # 選択肢が先頭にある場合は違う
        if token.idx == 0:
            continue

        # 選択肢が末尾にある場合は違う
        if token.idx + len(token.text) >= len(normalized_text):
            continue

        # 直前がスペースまたは句読点でない場合は違う
        prev_char = normalized_text[token.idx - 1]
        if not (prev_char.isspace() or unicodedata.category(prev_char).startswith("P")):
            continue

        # if token.i +3 < len(doc):
        #    print(f"{token.text}:{prev_char}: {unicodedata.category(prev_char)}")
        #    print("token:", normalized_text[doc[token.i - 3].idx:doc[token.i + 3].idx])

        # 直後がスペースでない場合は違う
        following_index = token.idx + len(token.text)
        following_char = normalized_text[following_index]
        if not following_char.isspace():
            continue

        label_indices.append((token.i, token.text))

        searching_options_idx = searching_options.index(token.text)
        if searching_options_idx >= 1:
            searching_options = searching_options[1:]

    # 有効なラベルが見つからなかった場合は全体をquestionとする
    if not label_indices:
        segments["question"] = normalized_text.strip()
        return segments

    # 各ラベル位置から次のラベルまたはテキスト末尾までを選択肢として抽出
    for j, (token_index, label) in enumerate(label_indices):
        start = doc[token_index].idx
        if j + 1 < len(label_indices):
            end = doc[label_indices[j + 1][0]].idx
        else:
            end = len(normalized_text)
        segments[label] = normalized_text[start:end].strip()

    last_segment_a = 0
    for token_index, label in label_indices:
        # print(label)
        if label == "a":
            last_segment_a = token_index
    # print(last_segment_a)
    segments["question"] = normalized_text[: doc[last_segment_a].idx].strip()

    # 各選択肢の先頭にあるラベル文字とその後の空白（半角・全角）を削除する
    for key in ["a", "b", "c", "d", "e"]:
        if segments[key]:
            segments[key] = re.sub(r"^[a-eA-E][\s　]+", "", segments[key])

    return segments


def normalize_question(text):
    """
    NLPを用いて問題文と選択肢を正規化するためのラッパー関数。

    Parameters:
        text (str): 元の問題文の文字列。

    Returns:
        dict: normalize_by_nlpで抽出された問題文と選択肢の辞書。
    """
    segments = normalize_by_nlp(text)
    return segments


def test_nlp_performance():
    data = read_csv("../dataset/kaggle_public.csv")
    for row in data[:100]:
        question = row[1]
        segments = normalize_by_nlp(question)

        original_len = len(segments["original"])
        segments_len = (
            len(segments["question"])
            + sum(len(segments[key]) for key in ["a", "b", "c", "d", "e"])
            + 4 * 3
        )

        if segments_len / original_len < 0.9:
            doc = nlp(question)
            for i, token in enumerate(doc):
                print(token.text, end="|")
            print("")
            print(question)
            print("")
            print("Original:", segments["original"])
            print("Question:", segments["question"])
            print("a:", segments["a"])
            print("b:", segments["b"])
            print("c:", segments["c"])
            print("d:", segments["d"])
            print("e:", segments["e"])
            print("-" * 20)
            exit()
        else:
            print(json.dumps(segments, ensure_ascii=False, indent=2))
            pass


if __name__ == "__main__":
    test_nlp_performance()
