#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import pandas as pd
import re
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from collections import Counter

# vllm 関連（推論・サンプリング用）
from vllm.sampling_params import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# ログ設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ◆◆◆ 文書同士の区切り用ユニーク定数
DOC_DELIMITER = "\n#####DOC_DELIMITER#####\n"

# グローバル変数（各ワーカーで初期化）
persistent_engine = None
persistent_prompt_template = None
persistent_max_tokens = None

#############################################
# 1. モデルエンジン初期化関連
#############################################
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
    """各GPUでモデルが正しくロードされたか、ダミープロンプトで確認するウォームアップ処理"""
    dummy_prompt = "ウォームアップ"
    _ = persistent_rerank_score(dummy_prompt, "warmup")
    print(f"GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}: ウォームアップ完了。")
    return "warmup_done"

#############################################
# 2. 再ランキング・評価用関数
#############################################
def persistent_rerank_score(prompt, request_id):
    """
    指定プロンプトに対して、短いスコア文字列（例：0～10の整数）を生成するための関数です。
    内部では非同期処理を行い、既存のイベントループがある場合は新たなループを作成します。
    """
    start_time = time.perf_counter()
    async def run_generation():
        try:
            sampling_params = SamplingParams(
                max_tokens=16,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            full_text = ""
            async for output in persistent_engine.generate(prompt, sampling_params, request_id):
                if output.outputs and len(output.outputs) > 0:
                    text_chunk = output.outputs[0].text
                    full_text += text_chunk
                    # 1チャンクで十分と判断
                    break
            return full_text.strip()
        except Exception as e:
            logger.error(f"Generation error for prompt: {prompt[:50]}... Error: {e}")
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
        logger.error(f"Generation error for prompt: {prompt[:50]}... Error: {e}")
        score = ""
    elapsed = time.perf_counter() - start_time
    print(f"[persistent_rerank_score] プロンプト先頭: {prompt[:50]}... 返却: '{score}' (elapsed {elapsed:.2f} sec)")
    return score

#############################################
# 3. 回答生成用関数
#############################################
def persistent_generate_rag_answer(question, context, request_id):
    """
    質問と取得した関連文書（context）をもとに回答生成プロンプトを作成し、回答文を生成します。
    """
    start_time = time.perf_counter()
    async def run_generation():
        final_prompt = f"【関連情報】\n{context}\n\n" + persistent_prompt_template.format(question=question)
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
        return full_text.strip()
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
        logger.error(f"Generation error for prompt: {question[:50]}... Error: {e}")
        answer = ""
    elapsed = time.perf_counter() - start_time
    return {"answer": answer, "elapsed": elapsed}

#############################################
# 4. 文書取得＋リランキング用関数
#############################################
def retrieve_context(query_text, top_n, es_client, openai_client, embedding_model_name, team, index_name):
    try:
        response = openai_client.embeddings.create(input=query_text, model=embedding_model_name)
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
    print(f"[Elasticsearch] ヒット数: {len(hits)}")
    docs = [hit['_source'].get('content', '') for hit in hits]
    return DOC_DELIMITER.join(docs)

def rerank_documents(query_text, docs, final_top_n, reranker_client):
    doc_list = [doc.strip() for doc in docs.split(DOC_DELIMITER) if doc.strip()]
    print(f"[再ランキング前] 候補文書数: {len(doc_list)}")
    scored_docs = []
    for doc in doc_list:
        prompt = f"以下の入力について、関連度を0から10の整数で評価してください。\n\nクエリ: {query_text}\n文書: {doc}\n\nスコア:"
        try:
            score_str = reranker_client.submit(persistent_rerank_score, prompt, "score_req").result().strip()
            print(f"[再ランキング] 文書先頭: {doc[:50]}... のスコア: '{score_str}'")
            score = int(score_str) if score_str.isdigit() else 0
        except Exception as e:
            logger.error(f"Score error for doc: {doc[:50]}... Error: {e}")
            score = 0
        scored_docs.append((doc, score))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored_docs[:final_top_n]]
    print(f"[再ランキング後] 選出文書数: {len(top_docs)}")
    return DOC_DELIMITER.join(top_docs)

def retrieve_context_with_reranking(query_text, initial_top_n, final_top_n, es_client, openai_client, embedding_model_name, team, index_name, reranker_client):
    initial_docs = retrieve_context(query_text, initial_top_n, es_client, openai_client, embedding_model_name, team, index_name)
    return rerank_documents(query_text, initial_docs, final_top_n, reranker_client)

#############################################
# 5. 評価用関数
#############################################
def evaluate_answer_samples(question, answer_samples, eval_executor):
    """
    与えられた回答サンプル群から、評価プロンプトを用いて最適な回答番号を選び、その回答を返します。
    """
    prompt = "以下の質問に対する回答サンプルの中から、最も適切なものの番号を選んでください。\n"
    prompt += f"【質問】\n{question}\n\n【回答サンプル】\n"
    for i, ans in enumerate(answer_samples, 1):
        prompt += f"{i}. {ans}\n"
    prompt += "\n最も適切な回答の番号を数字のみで出力してください。"
    result_str = eval_executor.submit(persistent_rerank_score, prompt, "eval_req").result()
    print(f"[評価] 評価プロンプト結果: {result_str}")
    try:
        best_index = int(re.search(r"(\d+)", result_str).group(1)) - 1
        if 0 <= best_index < len(answer_samples):
            return answer_samples[best_index]
    except Exception as e:
        logger.error(f"評価結果の解析エラー: {e}")
    return answer_samples[0]

#############################################
# 6. メイン処理（全問題に対して、まずretrieval→次にgeneration→最後に評価）
#############################################
def new_task_main(input_path):
    # CSV読み込み（ID, Question形式）
    df = pd.read_csv(input_path)
    
    # Elasticsearch, OpenAIクライアントの設定
    es_client = Elasticsearch(
        "https://YOUR_ES_ENDPOINT:443",
        api_key="YOUR_ES_API_KEY"
    )
    openai_client = AzureOpenAI(
        api_key="YOUR_OPENAI_API_KEY",
        api_version="2024-12-01-preview",
        azure_endpoint="https://apim-d-team-whgh66oxpmx72.azure-api.net/"
    )
    embedding_model_name = "text-embedding-3-large"
    team = "TeamD"
    index_name = "medical_llm_elastic_search"
    
    #########################
    # Retrievalフェーズ
    #########################
    print("【Retrievalフェーズ開始】")
    # Retrieval用プロセスプール（GPU0～7を使用）
    retrieval_executors = []
    retrieval_warmup_futures = []
    retrieval_config = {
        "model": "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b",
        "prompt_template": "",  # retrieval時はプロンプトは関数内で組むので空でOK
        "max_tokens": 4096
    }
    for gpu in range(8):
        executor = ProcessPoolExecutor(
            max_workers=1,
            initializer=init_engine,
            initargs=(gpu, retrieval_config["model"], retrieval_config["prompt_template"], retrieval_config["max_tokens"])
        )
        retrieval_executors.append(executor)
        retrieval_warmup_futures.append(executor.submit(warmup_engine))
    for future in retrieval_warmup_futures:
        _ = future.result()
    
    # 各質問ごとに関連文書を取得（各質問はround-robinで8GPUに分散）
    retrieval_results = {}
    for idx, row in df.iterrows():
        question = row["Question"]
        qid = row["ID"]
        print(f"\n=== [Retrieval] 質問ID: {qid} ===")
        executor = retrieval_executors[idx % 8]
        context = executor.submit(
            retrieve_context_with_reranking,
            question, 10, 5,
            es_client, openai_client, embedding_model_name, team, index_name,
            executor  # 同じexecutorを利用
        ).result()
        new_context = context.replace(DOC_DELIMITER, "\n")
        print(f"[質問ID:{qid}] 取得関連文書:\n{new_context}\n")
        retrieval_results[qid] = context
    # Retrievalフェーズ終了：プール解放
    for ex in retrieval_executors:
        ex.shutdown()
    
    #########################
    # Generationフェーズ
    #########################
    print("【Generationフェーズ開始】")
    # Generation用のプロセスプールを新たに作成（同じGPU0～7を使用）
    # ここでは、回答生成用のプロンプトテンプレートを設定
    generation_executors = []
    generation_warmup_futures = []
    generation_prompt_template = (
        "【質問】\n{question}\n\n【関連情報】\n{context}\n\n【回答】\n"
    )
    generation_config = {
        "model": "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b",
        "prompt_template": generation_prompt_template,
        "max_tokens": 1024
    }
    for gpu in range(8):
        executor = ProcessPoolExecutor(
            max_workers=1,
            initializer=init_engine,
            initargs=(gpu, generation_config["model"], generation_config["prompt_template"], generation_config["max_tokens"])
        )
        generation_executors.append(executor)
        generation_warmup_futures.append(executor.submit(warmup_engine))
    for future in generation_warmup_futures:
        _ = future.result()
    
    # 各質問について、retrievalフェーズで得たcontextを用い、複数の回答サンプル（例：16サンプル）を生成
    generation_results = {}
    for idx, row in df.iterrows():
        question = row["Question"]
        qid = row["ID"]
        context = retrieval_results[qid]
        print(f"\n=== [Generation] 質問ID: {qid} ===")
        generation_futures = []
        # round-robinで各GPUに2タスクずつ割り当てて、合計16サンプル生成（8GPU×2サンプル）
        for i, executor in enumerate(generation_executors):
            req_id1 = f"q{qid}_gpu{i}_1"
            req_id2 = f"q{qid}_gpu{i}_2"
            generation_futures.append(executor.submit(persistent_generate_rag_answer, question, context, req_id1))
            generation_futures.append(executor.submit(persistent_generate_rag_answer, question, context, req_id2))
        gen_results = [f.result() for f in generation_futures]
        answer_samples = [res["answer"] for res in gen_results]
        print(f"[質問ID:{qid}] 生成された回答サンプル数: {len(answer_samples)}")
        generation_results[qid] = answer_samples
    for ex in generation_executors:
        ex.shutdown()
    
    #########################
    # Evaluationフェーズ
    #########################
    print("【Evaluationフェーズ開始】")
    final_results = []
    # 評価は、retrieval側と同じ手法（persistent_rerank_scoreを用いて評価プロンプトを作成）で実施
    # 評価はここではメインプロセスで逐次実施しています
    for idx, row in df.iterrows():
        question = row["Question"]
        qid = row["ID"]
        answer_samples = generation_results[qid]
        best_answer = evaluate_answer_samples(question, answer_samples, 
                                              # 評価用として、再びGPU0～7のいずれかのexecutorを使います（ここでは例として、GPU番号は idx % 8）
                                              ProcessPoolExecutor(max_workers=1, initializer=init_engine,
                                                                  initargs=(idx % 8, generation_config["model"],
                                                                            generation_config["prompt_template"], generation_config["max_tokens"]))
                                             )
        print(f"[質問ID:{qid}] ベストアンサー: {best_answer}")
        final_results.append({"ID": qid, "Question": question, "Answer": best_answer})
    
    # 結果をCSV出力
    df_result = pd.DataFrame(final_results)
    output_path = f"{input_path}_submission.csv"
    df_result.to_csv(output_path, index=False)
    print(f"\n全質問の回答結果を {output_path} に保存しました。")

if __name__ == "__main__":
    input_path = "talk_q.csv"  # CSVはID,Question形式
    new_task_main(input_path)
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        pass
