Model: Qwen/Qwen3-32B

The method is an LLM-primary target-specific survey-response pipeline.
It makes one LLM call for each of the nine target questions for every
respondent. Feature selection is fit-only and leakage-safe.

All predictive survey inputs are restricted to variables listed in the
official features dataset. The separate country and respondent_id columns
are not used for feature selection, CatBoost prediction, retrieval similarity
or LLM prompting. Country is used only to construct validation splits and
country-clustered evaluation diagnostics; respondent_id is used only as a
technical key for caching and checkpoints.

For each target, normalized mutual information and mRMR first identify
relevant non-target survey answers. CatBoost is trained on an expanded
feature pool and its feature importance is combined conservatively with
the NMI/mRMR ranking. The first eight mRMR-selected features are protected,
up to four nonlinear CatBoost-important features are added, and a compact
target-specific evidence set is formed.

Each LLM prompt contains up to 12 direct
respondent answers, 2 compact labelled
training analogues, a target-specific guide, exact official labels and a
fit-only CatBoost probability prior. Q186 and Q242 are modeled directly in
their official coarsened label spaces.

The LLM returns a probability distribution in JSON. The final distribution
uses 88% LLM probabilities and 12% CatBoost probabilities. There is no
latent-profile call and no ordinary reasoning second pass. A short repair
call is used only when the structured output is invalid. CatBoost is used
as a final fallback after technical failure, ensuring complete coverage.

API calls are parallelized with 48 workers and
all responses are cached and checkpointed.
