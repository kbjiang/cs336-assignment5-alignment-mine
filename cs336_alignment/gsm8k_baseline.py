from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
import argparse
from rlhf_helper_methods import parse_gsm8k_response

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gsm8k_eval_file", type=str,
        default="/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/gsm8k/test.jsonl")
    args = parser.parse_args()

    llm = LLM(model="meta-llama/Llama-3.1-8B")

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["# Query:"]
    )

    # 1. load gsm8k examples
    df = pd.read_json(args.gsm8k_eval_file, lines=True)
    df["ground_truth"] = df.answer.apply(lambda x: parse_gsm8k_response(x))
    gsm8k_examples = df.to_dict(orient="records")

    # 2.1 format the gsm8k prompts
    gsm8k_prompt_file = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/question_only.prompt"
    with open(gsm8k_prompt_file) as f:
        gsm8k_prompt_template = f.read()

    gsm8k_instructions = [
        gsm8k_prompt_template.format(
            question = example["question"],
        ) for example in gsm8k_examples
    ]

    # 2.2 format the zero-shot prompts
    zero_shot_prompt_file = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/zero_shot_system_prompt.prompt"

    with open(zero_shot_prompt_file) as f:
        zero_shot_prompt_template = f.read()

    prompts = [zero_shot_prompt_template.format(instruction=instruction) for instruction in gsm8k_instructions]

    # 3. generate outputs
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    outputs = [opt.outputs[0].text for opt in outputs]
    assert len(outputs) == len(gsm8k_examples), "missing outputs"

    # 4. calculate evaluation metrics, in this case accuracy
    answers = [parse_gsm8k_response(opt) for opt in outputs]
    assert len(answers) == len(gsm8k_examples), "missing answers; check parsing."

    ground_truths = [eg["ground_truth"] for eg in gsm8k_examples]

    accuracies = [ans == gt for ans, gt in zip(answers, ground_truths)]
    accuracy = sum(accuracies) / len(answers)
    print(f"gsm8k eval accuracy: {accuracy}.")

    # serialization
    df = pd.DataFrame({
        "instruction": gsm8k_instructions,
        "raw_answer": outputs,
        "answer": answers,
        "ground_truth": ground_truths,
        "accurate": accuracies
    })

    output_file = "./gsm8k_baseline_eval_0.jsonl"
    df.to_json(output_file, orient="records", lines=True)
    print(f"Eval output saved to: {output_file}.")