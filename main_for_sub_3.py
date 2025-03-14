import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from collections import Counter
import re
import unicodedata
from typing import Dict, List
import asyncio
from vllm.sampling_params import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import logging

# ログ設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


### 既存の関数群

def split_question(question_text: str) -> Dict[str, str]:
    segments = {"question": "", "A": "", "B": "", "C": "", "D": "", "E": ""}
    pattern = r'(?P<label>[A-Ea-e])[).．:]\s*'
    parts = re.split(pattern, question_text)
    if len(parts) < 3:
        segments["question"] = question_text.strip()
        return segments
    segments["question"] = parts[0].strip()
    for i in range(1, len(parts) - 1, 2):
        label = parts[i].upper()
        text = parts[i+1].strip()
        if label in segments:
            segments[label] = text
    return segments

def retrieve_context(query_text: str, top_n: int, es_client, openai_client, embedding_model_name: str, team: str, index_name: str) -> str:
    try:
        response = openai_client.embeddings.create(
            input=query_text,
            model=embedding_model_name
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"Embedding API error: {e}")
        return ""
    
    response_es = es_client.options(request_timeout=60).search(
        index=index_name,
        body={
            "size": top_n,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [{"match": {"content": query_text}}],
                            "filter": [{"term": {"team": team}}]
                        }
                    },
                    "script": {
                        "source": """
                            double keyword_score = _score;
                            double vector_score = 0;
                            if (doc['embedding'].size() > 0) {
                                vector_score = cosineSimilarity(params.query_vector, 'embedding') + 1.0;
                            }
                            return keyword_score + vector_score;
                        """,
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }
    )
    hits = response_es['hits']['hits']
    docs = [hit['_source'].get('content', '') for hit in hits]
    return "\n".join(docs)

def persistent_rerank_score(prompt, request_id):
    """
    GPU3上の rinna モデルを用いて、与えられたプロンプトに対して
    0～10 の整数スコアを生成して返します。
    ※ 既存のイベントループが稼働している場合、新たなループを作成して実行します。
    """
    start_time = time.perf_counter()
    async def run_generation():
        try:
            sampling_params = SamplingParams(
                max_tokens=16,  # 短い出力で十分
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            full_text = ""
            async for output in persistent_engine.generate(prompt, sampling_params, request_id):
                if output.outputs and len(output.outputs) > 0:
                    text_chunk = output.outputs[0].text
                    full_text += text_chunk
                    # 短い応答なので1チャンクで十分と判断
                    break
            return full_text.strip()
        except Exception as e:
            logger.error(f"Generation error for prompt (first 50 chars): {prompt[:50]}... Error: {e}")
            return ""
    
    try:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                new_loop = asyncio.new_event_loop()
                score = new_loop.run_until_complete(run_generation())
                new_loop.close()
            else:
                score = asyncio.run(run_generation())
        except RuntimeError:
            score = asyncio.run(run_generation())
    except Exception as e:
        logger.error(f"Generation error for prompt (first 50 chars): {prompt[:50]}... Error: {e}")
        score = ""
    elapsed = time.perf_counter() - start_time
    return score

def rerank_documents(query_text: str, docs: str, final_top_n: int, reranker_client) -> str:
    """
    改行区切りの docs をリスト化し、各文書について採点用プロンプトを生成し、
    GPU3上の rinna モデル（reranker_client）を用いて関連度スコアを取得し、
    スコア順に上位 final_top_n 件を返します。
    ここでは、候補文書の件数のみターミナルに出力します。
    """
    doc_list = [doc.strip() for doc in docs.split("\n") if doc.strip()]
    print(f"[GPU-7] リランキング前候補文書数: {len(doc_list)}")
    scored_docs = []
    for doc in doc_list:
        prompt = (
            f"以下の入力について、関連度を0から10の整数で評価してください。\n\n"
            f"クエリ: {query_text}\n"
            f"文書: {doc}\n\n"
            f"スコア:"
        )
        try:
            score_str = reranker_client.submit(persistent_rerank_score, prompt, "score_req").result()
            score_str = score_str.strip()
            if score_str.isdigit():
                score = int(score_str)
            else:
                logger.error(f"Score conversion error for doc: {doc[:50]}... Score string: '{score_str}' is not a valid integer.")
                score = 0
        except Exception as e:
            logger.error(f"Score conversion error for doc: {doc[:50]}... Error: {e}")
            score = 0
        scored_docs.append((doc, score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored_docs[:final_top_n]]
    print(f"[GPU-7] リランキング後選出文書数: {len(top_docs)}")
    return "\n".join(top_docs)

def retrieve_context_with_reranking(query_text: str, initial_top_n: int, final_top_n: int,
                                    es_client, openai_client, embedding_model_name: str, team: str, index_name: str,
                                    reranker_client) -> str:
    initial_docs = retrieve_context(query_text, initial_top_n, es_client, openai_client, embedding_model_name, team, index_name)
    reranked = rerank_documents(query_text, initial_docs, final_top_n, reranker_client)
    return reranked

def retrieve_option_context(question_text: str, option_text: str, option_label: str,
                            initial_top_n: int, final_top_n: int,
                            es_client, openai_client, embedding_model_name: str, team: str, index_name: str,
                            reranker_client) -> str:
    combined_query = f"{question_text} {option_text}"
    docs = retrieve_context_with_reranking(combined_query, initial_top_n, final_top_n,
                                           es_client, openai_client, embedding_model_name, team, index_name,
                                           reranker_client)
    num_docs = len([d for d in docs.splitlines() if d.strip()])
    print(f"rinna: 選択肢{option_label}に対して、リランキングされた文書は {num_docs} 件でした。")
    return f"【問題文と選択肢{option_label}に関連する情報】\n{docs}"

def retrieve_question_extra_context(question_text: str, initial_top_n: int, final_top_n: int,
                                    es_client, openai_client, embedding_model_name: str, team: str, index_name: str,
                                    reranker_client) -> str:
    docs = retrieve_context_with_reranking(question_text, initial_top_n, final_top_n,
                                           es_client, openai_client, embedding_model_name, team, index_name,
                                           reranker_client)
    return f"【その他問題文に関連する追加情報】\n{docs}"

def refine_output(text: str) -> str:
    text = re.sub(r'</?think>', '', text)
    text = re.sub(r'\n+', '\n', text).strip()
    parts = text.split("[FINAL_ANSWER]:")
    if len(parts) >= 2:
        candidate = parts[-1].strip()
        candidate_lines = candidate.splitlines()
        if candidate_lines:
            return candidate_lines[0].strip()
        else:
            return candidate
    return text

def extract_final_choices(generated_text: str) -> List[str]:
    refined = refine_output(generated_text)
    parts = [p.strip() for p in refined.split(",") if p.strip()]
    choices = []
    for part in parts:
        m = re.search(r"([A-Ea-eＡ-Ｅａ-ｅ])", part)
        if m:
            letter = unicodedata.normalize("NFKC", m.group(1)).upper()
            choices.append(letter)
    if len(choices) > 2:
        choices = choices[:2]
    return choices

def compute_ensemble_vote(model_outputs: List[str]) -> str:
    all_votes = []
    for output in model_outputs:
        votes = extract_final_choices(output)
        all_votes.extend(votes)
    if all_votes:
        return Counter(all_votes).most_common(1)[0][0]
    return "None"

def compute_phi4_votes(phi4_output: str) -> List[List[str]]:
    lines = phi4_output.strip().splitlines()
    votes = []
    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            candidates = [c.strip() for c in parts[1].split(",") if c.strip()]
            votes.append(candidates[:2])
    return votes

def weighted_majority_vote(vote_lists: List[List[str]]) -> str:
    vote_counter = Counter()
    for votes in vote_lists:
        if len(votes) == 1:
            vote_counter[votes[0]] += 3
        elif len(votes) == 2:
            vote_counter[votes[0]] += 2
            vote_counter[votes[1]] += 1
    if vote_counter:
        return vote_counter.most_common(1)[0][0]
    return "None"

# グローバル変数（各ワーカーで初期化）
persistent_engine = None
persistent_prompt_template = None
persistent_max_tokens = None

def init_engine(gpu, model, prompt_template, max_tokens):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    engine_args = AsyncEngineArgs(
        model=model,
        max_model_len=16384,
        gpu_memory_utilization=0.90
    )
    global persistent_engine, persistent_prompt_template, persistent_max_tokens
    persistent_engine = AsyncLLMEngine.from_engine_args(engine_args)
    persistent_prompt_template = prompt_template
    persistent_max_tokens = max_tokens

def warmup_engine():
    """GPU-7用のウォームアップ関数。ダミープロンプトで呼び出し、モデルロード完了を待つ。"""
    dummy_prompt = "ウォームアップ"
    _ = persistent_rerank_score(dummy_prompt, "warmup")
    print("GPU-7: ウォームアップ完了。")
    return "warmup_done"

def persistent_generate_rag_answer(question, combined_context, request_id):
    start_time = time.perf_counter()
    async def run_generation():
        final_prompt = f"【関連情報】\n{combined_context}\n\n" + persistent_prompt_template.format(question=question)
        sampling_params = SamplingParams(
            max_tokens=persistent_max_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        full_text = ""
        last_output = ""
        repetition_counter = 0
        async for output in persistent_engine.generate(final_prompt, sampling_params, request_id):
            if output.outputs and len(output.outputs) > 0:
                text_chunk = output.outputs[0].text
                full_text += text_chunk
                if text_chunk == last_output:
                    repetition_counter += 1
                else:
                    repetition_counter = 0
                last_output = text_chunk
                if repetition_counter > 3:
                    break
        answer = refine_output(full_text)
        return answer
    try:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                new_loop = asyncio.new_event_loop()
                answer = new_loop.run_until_complete(run_generation())
                new_loop.close()
            else:
                answer = asyncio.run(run_generation())
        except RuntimeError:
            answer = asyncio.run(run_generation())
    except Exception as e:
        logger.error(f"Generation error for prompt (first 50 chars): {question[:50]}... Error: {e}")
        answer = ""
    elapsed = time.perf_counter() - start_time
    return {"answer": answer, "elapsed": elapsed}


### 以下、正常値データ・検査項目抽出・解析および過去問関連の関数（デバッグ出力追加）

def load_normal_ranges(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if '検査項目' not in df.columns:
        df = df.rename(columns={df.columns[0]: '検査項目'})
    return df

def parse_numeric_value(text: str) -> float:
    text = text.replace(",", "").strip()
    multiplier = 1.0
    if "万" in text:
        multiplier = 1e4
        text = text.replace("万", "")
    return float(text) * multiplier

def extract_lab_results(question_text: str, test_keywords: List[str]) -> Dict[str, float]:
    results = {}
    for test in test_keywords:
        pattern = re.compile(rf"{re.escape(test)}\s*[)）]?\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?(?:\s*万)?)")
        match = pattern.search(question_text)
        if match:
            try:
                value = parse_numeric_value(match.group(1))
                results[test] = value
            except Exception as e:
                logger.error(f"数値変換エラー: {match.group(1)} Error: {e}")
                continue
    return results

def analyze_lab_results(question_text: str, normal_df: pd.DataFrame, gender: str = None) -> str:
    test_keywords = normal_df["検査項目"].unique().tolist()
    extracted = extract_lab_results(question_text, test_keywords)
    analysis_lines = []
    for test, value in extracted.items():
        ref = normal_df[normal_df["検査項目"] == test]
        if ref.empty:
            continue
        if pd.notnull(ref.iloc[0]["性別"]) and ref.iloc[0]["性別"] != "":
            if gender is None:
                continue
            ref = ref[ref["性別"] == gender]
            if ref.empty:
                continue
        try:
            lower = float(ref.iloc[0]["下限"])
            upper = float(ref.iloc[0]["上限"])
        except Exception as e:
            logger.error(f"下限上限の変換エラー: {e}")
            continue
        if value < lower:
            result = "低め"
        elif value > upper:
            result = "高め"
        else:
            result = "正常"
        analysis_lines.append(f"{test}：{result}")
    if analysis_lines:
        print(f"【検査結果解析】: {len(analysis_lines)} 件の検査項目について解析しました。")
        return "【検査結果解析】\n" + "\n".join(analysis_lines)
    return ""

def detect_gender(question_text: str) -> str:
    male_index = question_text.find("男")
    female_index = question_text.find("女")
    if male_index >= 0 and female_index >= 0:
        return "男" if male_index < female_index else "女"
    elif male_index >= 0:
        return "男"
    elif female_index >= 0:
        return "女"
    else:
        return None

def is_lab_test_present(question_text: str, normal_df: pd.DataFrame) -> bool:
    for test in normal_df["検査項目"].unique():
        pattern = rf"{re.escape(test)}\s*[)）]?\s*[:：]?\s*[0-9]"
        if re.search(pattern, question_text):
            return True
    return False

def retrieve_past_exam_questions(query_text: str, top_n: int, es_client, index_name: str) -> List[str]:
    try:
        response = es_client.options(request_timeout=60).search(
            index=index_name,
            body={
                "size": top_n,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": query_text}}
                        ],
                        "filter": [
                            {"term": {"document_category": "medical_llm_elastic_search"}}
                        ]
                    }
                }
            }
        )
        hits = response["hits"]["hits"]
        past_questions = [hit["_source"].get("content", "") for hit in hits]
        for idx, pq in enumerate(past_questions):
            print(f"[過去問候補] {idx}: {len(pq.splitlines())} 行の文書")
        return past_questions
    except Exception as e:
        logger.error(f"Error retrieving past exam questions: {e}")
        return []

def evaluate_past_exam_questions(question_text: str, past_questions: List[str], reranker_client) -> List[str]:
    valuable_questions = []
    for idx, pq in enumerate(past_questions):
        prompt = (
            f"以下の過去問が、現在の問題を解く上で利用する価値があるかどうか、簡潔に判断してください。\n"
            f"現在の問題: {question_text}\n"
            f"過去問: {pq}\n"
            f"判断結果（「価値あり」または「価値なし」）："
        )
        try:
            result_str = reranker_client.submit(persistent_rerank_score, prompt, "past_exam_eval").result()
            if "価値あり" in result_str:
                valuable_questions.append(pq)
                print(f"[価値あり過去問] {idx}: 渡す候補として採用")
        except Exception as e:
            logger.error(f"Error evaluating past exam question: {e}")
    print(f"rinna: 類似過去問評価の結果、価値ありと判断された過去問は {len(valuable_questions)} 問でした。")
    return valuable_questions


### jmle_main() 内での処理

def jmle_main(input_path):
    japanese_prompt_template = (
        "【指示】\n"
        "あなたは熟練の医師国家試験エージェントです。\n"
        "以下の問題に対して、まず問題文と選択肢ごとの関連情報を踏まえた分析を行い、"
        "正解と思われる選択肢を教えてください。\n"
        "迷った場合は、正解と思われる選択肢をコンマで区切って2つまで出力してもよいですが、"
        "より可能性が高いと思われるものを先頭に記載してください。\n"
        "また、自己検証として推論過程を簡潔にまとめたあと、最後に以下のように"
        "最終的な推論結果を [FINAL_ANSWER]: X の形式で一度だけ出力してください。\n\n"
        "【出力例】\n"
        "[FINAL_ANSWER]: B\n"
        "または\n"
        "[FINAL_ANSWER]: A, B\n\n"
        "【問題】\n"
        "{question}\n\n"
        "【出力】\n"
    )
    english_prompt_template = (
        "【Instructions】\n"
        "You are an experienced physician licensure exam agent.\n"
        "For the following question written in Japanese, first perform an analysis based on the question text "
        "and the relevant information for each option, and then indicate the option you believe to be correct.\n"
        "If you are uncertain, you may output up to two options separated by a comma, with the option that seems more likely listed first.\n"
        "Also, as a self-check, after briefly summarizing your reasoning process, output the final reasoning result only once at the end in the following format:\n"
        "[FINAL_ANSWER]: X\n\n"
        "【Example Output】\n"
        "[FINAL_ANSWER]: B\n"
        "or\n"
        "[FINAL_ANSWER]: A, B\n\n"
        "【Question】\n"
        "{question}\n\n"
        "【Output】\n"
    )
    
    engines_config = [
        {
            "gpu": 4,
            "model": "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b",
            "prompt_template": japanese_prompt_template,
            "max_tokens": 4096
        },
        {
            "gpu": 5,
            "model": "nitky/RoguePlanet-DeepSeek-R1-Qwen-32B-RP",
            "prompt_template": japanese_prompt_template,
            "max_tokens": 2048
        },
        {
            "gpu": 6,
            "model": "Qwen/QwQ-32B",
            "prompt_template": english_prompt_template,
            "max_tokens": 2048
        }
    ]
    
    # 再ランキング用（GPU-7上の rinna モデル）
    rerank_config = {
        "gpu": 7,
        "model": "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b",
        "prompt_template": ""
    }

    print("GPU-7: rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b loading")
    rerank_executor = ProcessPoolExecutor(
        max_workers=1,
        initializer=init_engine,
        initargs=(rerank_config["gpu"], rerank_config["model"], rerank_config["prompt_template"], 4096)
    )
    # GPU-7のウォームアップ（モデルロード完了待ち）
    warmup_result = rerank_executor.submit(warmup_engine).result()
    print(f"GPU-7 warmup result: {warmup_result}")

    executors = []
    for config in engines_config:
        print(f"GPU-{config['gpu']}: {config['model']} loading")
        ex = ProcessPoolExecutor(
            max_workers=1,
            initializer=init_engine,
            initargs=(config["gpu"], config["model"], config["prompt_template"], config["max_tokens"])
        )
        executors.append(ex)
    
    main_es_client = Elasticsearch(
        "https://b5e5cf6df1c34c85bda365d8730863be.japaneast.azure.elastic-cloud.com:443",
        api_key="eWVxYU1aVUJ5TEE1X1dpX2pjdGc6OTNIUzNqQk5SeFdsdEh6bk5EMHd6QQ=="
    )
    main_openai_client = AzureOpenAI(
        api_key="fdce8b59a62d4784ba8c274b3e6a773e",
        api_version="2024-12-01-preview",
        azure_endpoint="https://apim-d-team-whgh66oxpmx72.azure-api.net/"
    )
    embedding_model_name = "text-embedding-3-large"
    team = "TeamD"
    index_name = 'medical_llm_elastic_search'
    
    normal_ranges_df = load_normal_ranges("blood_final.csv")
    df = pd.read_csv(input_path)
    
    correct_count = 0
    total = len(df)
    model_names = ["rinna", "nitky", "QwQ"]
    
    if "ensemble_vote" not in df.columns:
        df["ensemble_vote"] = ""
    if "final_answer" not in df.columns:
        df["final_answer"] = ""
    if "lab_analysis" not in df.columns:
        df["lab_analysis"] = ""
    
    for idx, row in df.iterrows():
        question_full = row["Question"]
        correct_answer = row.get("Answer", "").strip() if isinstance(row.get("Answer", ""), str) else ""
        
        segments = split_question(question_full)
        gender = detect_gender(question_full)
        
        if is_lab_test_present(question_full, normal_ranges_df):
            lab_analysis = analyze_lab_results(question_full, normal_ranges_df, gender)
        else:
            lab_analysis = ""
        
        # 問題文のコンテキストは直接取得
        context_question = retrieve_context(
            segments["question"], 5,
            main_es_client, main_openai_client,
            embedding_model_name, team, index_name
        )
        num_context = len([d for d in context_question.split("\n") if d.strip()])
        print(f"問題文関連情報として {num_context} 件の文書を渡します。")
        
        context_options = {}
        for opt in ["A", "B", "C", "D", "E"]:
            option_text = segments.get(opt, "")
            if option_text:
                context_options[opt] = retrieve_option_context(
                    segments["question"], option_text, opt,
                    initial_top_n=15, final_top_n=5,
                    es_client=main_es_client, openai_client=main_openai_client,
                    embedding_model_name=embedding_model_name, team=team, index_name=index_name,
                    reranker_client=rerank_executor
                )
                num_opt = len([d for d in context_options[opt].split("\n") if d.strip() and not d.startswith("【")])
                print(f"選択肢{opt}関連情報として {num_opt} 件の文書を渡します。")
            else:
                context_options[opt] = ""
        
        past_questions_candidates = retrieve_past_exam_questions(question_full, 10, main_es_client, index_name)
        valuable_past_questions = evaluate_past_exam_questions(question_full, past_questions_candidates, rerank_executor)
        print(f"rinna: 類似過去問として {len(valuable_past_questions)} 件の文書を渡します。")
        
        combined_context = ""
        if lab_analysis:
            combined_context += lab_analysis + "\n"
        combined_context += f"【問題文関連情報】\n{context_question}\n"
        for opt in ["A", "B", "C", "D", "E"]:
            combined_context += f"\n【選択肢{opt}関連情報】\n{context_options[opt]}\n"
        if valuable_past_questions:
            combined_context += "\n【類似過去問情報】\n"
            combined_context += "以下の過去問が見つかりました。参考にしてください。\n"
            for pq in valuable_past_questions:
                combined_context += pq + "\n"
        
        # 出力前に問題文を表示
        print(f"\n== Question {idx+1}/{total} ==")
        print("問題文:")
        print(question_full)
        
        # 推論モデルに渡す文書数（combined_context 内の改行で分割したうち空行でないもの）を出力
        num_docs = len([d for d in combined_context.split("\n") if d.strip() and not d.startswith("【")])
        print(f"推論モデルに渡す文書は合計 {num_docs} 件です。")
        
        request_ids = [f"q_{idx+1}_{i}" for i in range(len(executors))]
        futures = []
        for ex, req_id in zip(executors, request_ids):
            futures.append(ex.submit(persistent_generate_rag_answer, question_full, combined_context, req_id))
        results = [f.result() for f in futures]
        
        mechanical_votes = []
        # 各モデルの抽出答えを出力する前に問題文を再表示
        print("----- 抽出された答え（各モデルごと） -----")
        print("問題文:")
        print(question_full)
        for i, result in enumerate(results):
            extracted = extract_final_choices(result["answer"])
            print(f"Extracted Answer from Model {i+1} (elapsed {result['elapsed']:.2f} sec): {', '.join(extracted)}")
            mechanical_votes.append(extracted)
        final_vote = weighted_majority_vote(mechanical_votes)
        print(f"Final Ensemble Vote: {final_vote}")
        print(f"Correct Answer: {correct_answer}")
        print("=" * 50)
        
        df.loc[idx, "ensemble_vote"] = final_vote
        df.loc[idx, "final_answer"] = final_vote
        df.loc[idx, "lab_analysis"] = lab_analysis
        df.to_csv(f"{input_path}_tmp.csv", index=False)
        
        if final_vote != "None" and final_vote == correct_answer.upper():
            correct_count += 1
    
    accuracy = correct_count / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

    df_submit = df[['ID', 'final_answer']]
    df_submit.columns = ['ID', 'Answer']
    df_submit.to_csv(f"{input_path}_submission.csv", index=False)
    
    for ex in executors:
        ex.shutdown()
    rerank_executor.shutdown()

if __name__ == "__main__":

    input_path = "kaggle_1.csv"

    jmle_main(input_path)
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
