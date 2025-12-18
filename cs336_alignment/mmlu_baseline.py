from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
import argparse
from rlhf_helper_methods import parse_mmlu_response

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu_eval_dir", type=str, default="/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/mmlu/val")
    args = parser.parse_args()

    llm = LLM(model="meta-llama/Llama-3.1-8B")

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["# Query:"]
    )

    # 1. load mmlu examples
    mmlu_eval_dir = Path(args.mmlu_eval_dir)

    mmlu_examples = []
    for file in mmlu_eval_dir.glob("*.csv"):
        subject = file.name.rsplit("_", 1)[0]
        df = pd.read_csv(file, names=["question", "A", "B", "C", "D", "answer"])
        df["subject"] = subject
        df["options"] = df[["A", "B", "C", "D"]].values.tolist()
        mmlu_examples.extend(df.to_dict(orient = "records"))

    # 2.1 format the mmlu prompts
    mmlu_prompt_file = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/mmlu.prompt"

    with open(mmlu_prompt_file) as f:
        mmlu_prompt_template = f.read()

    mmlu_instructions = [
        mmlu_prompt_template.format(
            subject = example["subject"],
            question = example["question"],
            options = example["options"]
        ) for example in mmlu_examples
    ]

    # 2.2 format the zero-shot prompts
    zero_shot_prompt_file = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/zero_shot_system_prompt.prompt"

    with open(zero_shot_prompt_file) as f:
        zero_shot_prompt_template = f.read()

    prompts = [zero_shot_prompt_template.format(instruction=instruction) for instruction in mmlu_instructions]

    # 3. generate outputs
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    outputs = [opt.outputs[0].text for opt in outputs]
    assert len(outputs) == len(mmlu_examples), "missing outputs"

    # 4. calculate evaluation metrics, in this case accuracy
    answers = [parse_mmlu_response(opt) for opt in outputs]
    assert len(answers) == len(mmlu_examples), "missing answers; check parsing."

    ground_truths = [eg["answer"] for eg in mmlu_examples]
    accuracies = [ans == gt for ans, gt in zip(answers, ground_truths)]
    accuracy = sum(accuracies) / len(answers)
    print(f"MMLU eval accuracy: {accuracy}.")

    # serialization
    df = pd.DataFrame({
        "subject": [eg["subject"] for eg in mmlu_examples],
        "instruction": mmlu_instructions,
        "raw_answer": outputs,
        "answer": answers,
        "ground_truth": ground_truths,
        "accurate": accuracies
    })

    output_file = "./mmlu_baseline_eval.jsonl"
    df.to_json(output_file, orient="records", lines=True)
    print(f"Eval output saved to: {output_file}.")