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
* (c) Zero.
* (d) With VLLM, `test` file, `00:35<00:00, 36.75it/s, est. speed input: 7348.95 toks/s, output: 5289.59 toks/s`
* (e) `gsm8k eval accuracy: 0.15921152388172857` This is understandable, given `mmlu` provided options.
* (f) On top of mathematical error, the model did repeat the question, instead of answering, sometimes. This is reasonable, given `mmlu` specified output format.