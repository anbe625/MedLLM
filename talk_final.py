import asyncio
import pandas as pd
import re
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

# 使用するモデルの名称
model_name = "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b"

# AsyncEngineArgs の作成
engine_args = AsyncEngineArgs(
    model=model_name,
    max_model_len=35000, 
    gpu_memory_utilization=0.95, 
)
# エンジンの初期化
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_answer(question, request_id):
    # 新しいプロンプト：評価基準は記載しつつ、見出しとして直接は出力しない工夫をする
    prompt = (
        "あなたは患者に寄り添える優秀な医師として、以下の患者の質問に回答してください。\n"
        "なお、回答する際には以下の点に留意してください。\n"
        "・診断の根拠や論理の一貫性、説得力、医学的な裏付けを丁寧に示すこと。\n"
        "・読み手が新たな知識や実践的な情報を得られるよう、わかりやすく説明すること。\n"
        "・文章は結論ファーストで、専門用語は適宜使いつつも噛み砕いた表現を用いること。\n"
        "・質問の意図や重要情報を漏れなくカバーし、必要に応じて鑑別診断や根拠を示すこと。\n"
        "・医学的誤情報や不適切な表現を避け、患者が安心できるような内容にすること。\n\n"
        f"【質問】\n{question}\n\n"
        "【回答】\n"
    )
    # SamplingParams の設定（max_tokens で最大生成トークン数を指定）
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
    )
    generated_text = ""
    # engine.generate は async generator を返すので、出力を順次まとめる
    async for output in engine.generate(prompt, sampling_params, request_id):
        # APIの仕様に合わせて、出力オブジェクトから適切にテキストを取得する
        if output.outputs and len(output.outputs) > 0:
            generated_text += output.outputs[0].text

    # generated_text内の最後の</think>タグ以降の文字列を抽出
    if "</think>" in generated_text:
        final_answer = generated_text.split("</think>")[-1].strip()
    else:
        final_answer = generated_text.strip()
    
    return final_answer

async def main():
    df = pd.read_csv("dataset/original/qa_final.csv")
    answers = []
    total = len(df)
    
    for idx, row in df.iterrows():
        question = row["Question"]
        request_id = f"q_{idx+1}"
        
        final_answer = await generate_answer(question, request_id)
        answers.append(final_answer)
        
        # リアルタイムで出力
        print(f"Question {idx+1}: {question} / {total}")
        print("Generated final answer:")
        print(final_answer)
        print("=" * 50)
    
    df["Answer"] = answers
    df.to_csv("qa_final_submission.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
