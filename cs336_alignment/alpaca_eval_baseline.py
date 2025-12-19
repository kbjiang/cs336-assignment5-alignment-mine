from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
import argparse
import json

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpaca_eval_file", type=str,
        default="/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/alpaca_eval/alpaca_eval.jsonl")
    args = parser.parse_args()

    llm = LLM(model="meta-llama/Llama-3.1-8B")

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["# Query:"]
    )

    # 1. load alpaca examples
    df = pd.read_json(args.alpaca_eval_file, lines=True)

    # 2 format the zero-shot prompts
    zero_shot_prompt_file = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/zero_shot_system_prompt.prompt"

    with open(zero_shot_prompt_file) as f:
        zero_shot_prompt_template = f.read()

    prompts = [zero_shot_prompt_template.format(instruction=instruction) for instruction in df.instruction.tolist()]

    # 3. generate outputs
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    outputs = [opt.outputs[0].text for opt in outputs]
    assert len(outputs) == len(df), "missing outputs"

    # serialization
    df["output"] = outputs
    df["generator"] = "llama-3.1-8b-base"

    output_file = Path(__file__).parent / "alpaca_eval_baseline.json"
    with open(output_file, "w") as f:
        json.dump(df.to_dict("records"), f)
    print(f"Eval output saved to: {output_file}.")