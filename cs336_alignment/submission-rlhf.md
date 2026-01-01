### mmlu_baseline
* (a) Done
* (b) Done
* (c) Zero.
* (d) With VLLM, `val` folder, `00:26<00:00, 58.68it/s, est. speed input: 17305.12 toks/s, output: 1145.13 toks/s`
* (e) `MMLU eval accuracy: 0.5303723056825604.`
* (f) it made mistakes in all categories. Hard to see the sort of errors due to lack of explanation in the raw answers.

### gsm8k_baseline
* (a) Done
* (b) Done
* (c) Ran `df.answer.isna()` returned `10` rows (out of `1319`). 
    * Majority are repeating the question, a few "do not understand question", one answered "five" instead of "5".
* (d) With VLLM, `test` file, `00:35<00:00, 36.75it/s, est. speed input: 7348.95 toks/s, output: 5289.59 toks/s`
* (e) `gsm8k eval accuracy: 0.15921152388172857`. This is lower than `mmlu`, which is multi-choice.
* (f) On top of mathematical error, the model did repeat the question, instead of answering, sometimes. This is lower than `mmlu` where output format is specified.

### alpaca_eval_baseline
* (a) Done
* (b) `00:38<00:00, 21.14it/s, est. speed input: 3730.40 toks/s, output: 6533.44 toks/s]`
* (c) I found it interesting that `Llama 3.1 8B` beats `GPT-4 Turbo`. Maybe because the annotator is `Llama 3.3 70B`?
    ```
                        length_controlled_winrate  win_rate  standard_error  n_total  avg_length
    llama-3.1-8b-base                       1.54      1.61            0.44      805        1337
    ```
* (d) There are a lot of cases where `Llama 3.1` hallucinated by repeating itself but got better ranking. `Llama 3.3` apparently is not a good judge. Also, I suspect that if I set a lower limit on number of generated tokens, `Llama 3.1` might be less favorable.

### sst_baseline
* (a) Done
* (b) `00:08<00:00, 11.69it/s, est. speed input: 1794.79 toks/s, output: 1729.67 toks/s`
* (c) `34/100` are considered not safe.
* (d) There are few cases where the judge made mistakes. E.g., `sst_021`.

```
                  length_controlled_winrate  win_rate  standard_error  n_total  avg_length
llama-3.1-8b-sft                       2.71      1.86            0.48      805         885
                  length_controlled_winrate  win_rate  standard_error  n_total  avg_length
llama-3.1-8b-dpo                       2.86      1.86            0.48      805         753
```