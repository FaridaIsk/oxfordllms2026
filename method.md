# Method

We used Qwen/Qwen3-32B with temperature 0 and model thinking disabled.

For each target question, we selected 12 target-specific
survey variables from the permitted feature pool. Feature ranking used
only the labelled training data. Candidate variables were ranked by
normalized mutual information with the target response, multiplied by
the square root of answer availability to mildly penalise sparse
features.

For every respondent-target pair, we constructed a separate zero-shot
prompt. Missing feature answers were omitted. The prompt included the
respondent's country, the selected question-answer pairs, the target
question, and the exact allowed answer labels.

The model was instructed to return exactly one label. Replies were
matched back to the official label set. When an API request failed or a
reply could not be parsed, we used the target-specific majority label
calculated from the training data.

Target code-to-label mappings were validated against all observed
training codes before prediction.
