
# ============================================================
# AI Respondents Challenge:
# LLM prompting stage after compromise feature selection
# ============================================================
#
# This script consumes the frozen feature-selection output produced by:
#
#   ai_respondents_compromise_holdout_pipeline.py
#
# It demonstrates how to:
# 1. Read target-specific LLM features selected on development countries.
# 2. Decode survey response codes into human-readable text.
# 3. Build one prompt per respondent-target pair.
# 4. Run zero-shot LLM predictions with retries and resumable caching.
# 5. Evaluate prompt variants on development data.
# 6. Evaluate a frozen prompt once on untouched holdout countries.
# 7. Generate final predictions for the challenge test set.
# 8. Export prompts.jsonl for the submission method folder.
#
# IMPORTANT:
# - Tune prompt wording, model, country inclusion, and other settings only
#   in MODE="development_validation".
# - Freeze the prompt configuration before MODE="outer_holdout".
# - Do not repeatedly tune the prompt after inspecting outer-holdout results.
# ============================================================

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

import ai_respondents_feature_selection_pipeline as fs


# ============================================================
# 0. Configuration
# ============================================================

# Available modes:
# - "development_validation": tune the prompt on labelled development rows;
# - "outer_holdout": evaluate the frozen prompt once on untouched countries;
# - "test_prediction": generate final challenge-test predictions.
MODE = "development_validation"

# Start with one target while debugging.
# Set to None to use all target questions available in the frozen selection.
TARGETS_TO_RUN: list[str] | None = ["Q73"]

# Number of labelled development respondents sampled per target while
# comparing prompt variants.
DEVELOPMENT_SAMPLE_SIZE_PER_TARGET = 100

# Use a fixed sample so every prompt variant is compared on the same rows.
RANDOM_SEED = 42

# LLM configuration. Freeze these values before outer-holdout evaluation.
PROMPT_VERSION = "v1_strict_target_specific"
MODEL = "Qwen/Qwen3-32B"
TEMPERATURE = 0.0
MAX_TOKENS = 16
USE_COUNTRY = True

# Limit concurrency while debugging. Increase only after a successful demo.
MAX_WORKERS = 8

# Challenge API endpoint used by the starter notebook.
API_BASE_URL = "https://api.studio.nebius.com/v1/"

# Input paths produced by the compromise holdout pipeline.
COMPROMISE_DIR = Path("outputs/compromise_holdout")
SELECTION_PATH = (
    COMPROMISE_DIR
    / "development_selection"
    / "frozen_development_selection.json"
)
SPLIT_MANIFEST_PATH = COMPROMISE_DIR / "country_split_manifest.json"

# Output paths for this LLM stage.
OUTPUT_DIR = COMPROMISE_DIR / "llm_stage"
CACHE_DIR = OUTPUT_DIR / "cache"
RESULTS_DIR = OUTPUT_DIR / "results"
SUBMISSION_METHOD_DIR = OUTPUT_DIR / "submission_method"

CACHE_LOCK = Lock()

for directory in (
    OUTPUT_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    SUBMISSION_METHOD_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Fixed prompt templates
# ============================================================

SYSTEM_PROMPT = """
You are a survey-response prediction model.

Your task is to infer how a specific real respondent answered a survey
question. This is a classification task, not a request for your own
opinion.

Use only the respondent information supplied in the user message.
Prioritise directly related individual survey responses over broad
demographic or country-level assumptions.

Return exactly one label from the supplied answer list and no other text.
""".strip()


USER_PROMPT_TEMPLATE = """
{country_block}Related survey responses:

{profile}

Target question:

{target_question}

Allowed answer labels:

{labels}

Select the single most likely answer.

Output requirements:
- Return exactly one allowed answer label.
- Copy the label exactly as written.
- Do not provide an explanation.
- Do not include quotation marks.
- Do not add any other text.

Answer:
""".strip()


# ============================================================
# 2. Load frozen selections and reconstruct the data split
# ============================================================

def load_json(path: Path) -> Any:
    """Read one UTF-8 JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_frozen_selection() -> dict[str, dict]:
    """
    Load target-specific CatBoost and LLM features selected only from
    development countries.
    """
    selection = load_json(SELECTION_PATH)

    for target_id, result in selection.items():
        if "llm_features" not in result:
            raise ValueError(
                f"Frozen selection for {target_id} has no llm_features."
            )

    return selection


def recreate_development_and_holdout(
    full_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recreate the exact development and untouched-country holdout tables
    from the previously saved split manifest.
    """
    manifest = load_json(SPLIT_MANIFEST_PATH)

    development_countries = set(
        map(str, manifest["development_countries"])
    )
    holdout_countries = set(
        map(str, manifest["holdout_countries"])
    )

    country_as_text = full_train["country"].astype(str)

    development = full_train[
        country_as_text.isin(development_countries)
    ].copy()

    holdout = full_train[
        country_as_text.isin(holdout_countries)
    ].copy()

    overlap = (
        set(development["country"].astype(str).unique())
        & set(holdout["country"].astype(str).unique())
    )

    if overlap:
        raise RuntimeError(
            f"Country leakage detected: {sorted(overlap)}"
        )

    return (
        development.reset_index(drop=True),
        holdout.reset_index(drop=True),
    )


# ============================================================
# 3. Build metadata dictionaries
# ============================================================

def parse_values_json(value: Any) -> dict[str, str]:
    """Parse the feature code-to-label dictionary safely."""
    if isinstance(value, dict):
        return {
            str(key): str(label)
            for key, label in value.items()
        }

    if pd.isna(value):
        return {}

    parsed = json.loads(str(value))

    return {
        str(key): str(label)
        for key, label in parsed.items()
    }


def build_metadata(
    data: fs.ChallengeData,
) -> dict[str, Any]:
    """Create all dictionaries required for decoding and prompting."""
    value_maps = {
        str(row["variable"]): parse_values_json(row["values_json"])
        for _, row in data.features.iterrows()
    }

    target_questions = (
        data.targets
        .drop_duplicates("question_id")
        .set_index("question_id")["question"]
        .astype(str)
        .to_dict()
    )

    labels_by_target = (
        data.targets
        .groupby("question_id", sort=False)["label"]
        .apply(lambda series: [str(value) for value in series])
        .to_dict()
    )

    option_to_label: dict[str, dict[int, str]] = {}

    for target_id, group in data.targets.groupby("question_id"):
        option_to_label[str(target_id)] = {
            int(float(option)): str(label)
            for option, label in zip(
                group["option"],
                group["label"],
            )
        }

    return {
        "question_text": data.question_text,
        "value_maps": value_maps,
        "target_questions": target_questions,
        "labels_by_target": labels_by_target,
        "option_to_label": option_to_label,
    }


# ============================================================
# 4. Decode respondent answers
# ============================================================

def value_to_code(value: Any) -> str | None:
    """Convert a numeric survey value into a metadata dictionary key."""
    if pd.isna(value):
        return None

    numeric = float(value)

    if numeric.is_integer():
        return str(int(numeric))

    return str(numeric)


def decode_feature_answer(
    feature: str,
    value: Any,
    value_maps: dict[str, dict[str, str]],
) -> str | None:
    """
    Convert one coded survey answer into readable text.

    Missing answers are returned as None and will not be shown to the LLM.
    """
    code = value_to_code(value)

    if code is None:
        return None

    mapping = value_maps.get(feature, {})

    return mapping.get(code, code)


def decode_target_truth(
    target_id: str,
    value: Any,
    option_to_label: dict[str, dict[int, str]],
) -> str | None:
    """Decode one known target option into its official label."""
    if pd.isna(value):
        return None

    option = int(float(value))

    return option_to_label[target_id].get(option)


# ============================================================
# 5. Build one respondent-target prompt
# ============================================================

def build_profile(
    respondent: pd.Series,
    target_id: str,
    llm_features_by_target: dict[str, list[str]],
    question_text: dict[str, str],
    value_maps: dict[str, dict[str, str]],
) -> str:
    """
    Build a compact respondent sketch from the frozen target-specific
    LLM features.

    Unanswered survey items are skipped.
    """
    lines: list[str] = []

    for feature in llm_features_by_target[target_id]:
        decoded_answer = decode_feature_answer(
            feature=feature,
            value=respondent.get(feature),
            value_maps=value_maps,
        )

        if decoded_answer is None:
            continue

        lines.append(
            f"Question: {question_text[feature]}\n"
            f"Answer: {decoded_answer}"
        )

    if not lines:
        return (
            "No selected survey responses are available for this respondent."
        )

    return "\n\n".join(lines)


def build_user_prompt(
    respondent: pd.Series,
    target_id: str,
    llm_features_by_target: dict[str, list[str]],
    metadata: dict[str, Any],
) -> str:
    """Render the final user prompt for one respondent-target pair."""
    profile = build_profile(
        respondent=respondent,
        target_id=target_id,
        llm_features_by_target=llm_features_by_target,
        question_text=metadata["question_text"],
        value_maps=metadata["value_maps"],
    )

    labels = "\n".join(
        f"- {label}"
        for label in metadata["labels_by_target"][target_id]
    )

    if USE_COUNTRY:
        country_block = (
            "Respondent country:\n"
            f"{respondent['country']}\n\n"
        )
    else:
        country_block = ""

    return USER_PROMPT_TEMPLATE.format(
        country_block=country_block,
        profile=profile,
        target_question=metadata["target_questions"][target_id],
        labels=labels,
    )


# ============================================================
# 6. Parse LLM replies
# ============================================================

def normalise_reply(reply: str) -> str:
    """Apply conservative formatting normalisation."""
    cleaned = str(reply).strip()
    cleaned = cleaned.strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_label(
    reply: str,
    valid_labels: list[str],
) -> str | None:
    """
    Match the raw reply back to one official target label.

    Matching order:
    1. exact;
    2. case-insensitive exact;
    3. exactly one valid label contained in a longer response.
    """
    cleaned = normalise_reply(reply)

    for label in valid_labels:
        if cleaned == label:
            return label

    for label in valid_labels:
        if cleaned.lower() == label.lower():
            return label

    contained = [
        label
        for label in valid_labels
        if label.lower() in cleaned.lower()
    ]

    if len(contained) == 1:
        return contained[0]

    return None


# ============================================================
# 7. LLM client, retries, and deterministic cache keys
# ============================================================

def build_client() -> OpenAI:
    """Create an OpenAI-compatible Nebius client."""
    api_key = os.environ.get("NEBIUS_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Set the NEBIUS_API_KEY environment variable before running."
        )

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=30,
    ),
)
def call_model(
    client: OpenAI,
    user_prompt: str,
) -> str:
    """Call the LLM with deterministic classification settings."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )

    return str(response.choices[0].message.content)


def prompt_hash(
    target_id: str,
    respondent_id: Any,
    user_prompt: str,
) -> str:
    """Create a cache key that changes whenever the prompt changes."""
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "use_country": USE_COUNTRY,
            "target_id": target_id,
            "respondent_id": str(respondent_id),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    """Load existing successful jobs from a JSONL cache."""
    cache: dict[str, dict] = {}

    if not path.exists():
        return cache

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            cache[record["prompt_hash"]] = record

    return cache


def append_cache(path: Path, record: dict) -> None:
    """Append one completed prediction immediately in a thread-safe way."""
    with CACHE_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


# ============================================================
# 8. Build labelled or unlabelled prediction jobs
# ============================================================

def sample_development_rows(
    development: pd.DataFrame,
    target_id: str,
) -> pd.DataFrame:
    """
    Select a fixed labelled development sample for prompt comparison.

    The same random state should be used for every prompt variant.
    """
    available = development[
        development[target_id].notna()
    ].copy()

    sample_size = min(
        DEVELOPMENT_SAMPLE_SIZE_PER_TARGET,
        len(available),
    )

    return available.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)


def select_mode_frame(
    mode: str,
    data: fs.ChallengeData,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    target_id: str,
) -> tuple[pd.DataFrame, bool]:
    """Return the appropriate respondent table and whether truth is known."""
    if mode == "development_validation":
        return sample_development_rows(
            development=development,
            target_id=target_id,
        ), True

    if mode == "outer_holdout":
        frame = holdout[
            holdout[target_id].notna()
        ].copy()
        return frame.reset_index(drop=True), True

    if mode == "test_prediction":
        return data.test.copy().reset_index(drop=True), False

    raise ValueError(f"Unknown MODE: {mode}")


# ============================================================
# 9. Run one prompt job
# ============================================================

def predict_one(
    client: OpenAI,
    respondent: pd.Series,
    target_id: str,
    llm_features_by_target: dict[str, list[str]],
    metadata: dict[str, Any],
    cache: dict[str, dict],
    cache_path: Path,
    truth_is_known: bool,
) -> dict:
    """Predict one target for one respondent."""
    user_prompt = build_user_prompt(
        respondent=respondent,
        target_id=target_id,
        llm_features_by_target=llm_features_by_target,
        metadata=metadata,
    )

    job_hash = prompt_hash(
        target_id=target_id,
        respondent_id=respondent["respondent_id"],
        user_prompt=user_prompt,
    )

    with CACHE_LOCK:
        cached_record = cache.get(job_hash)

    if cached_record is not None:
        return cached_record

    raw_reply = call_model(
        client=client,
        user_prompt=user_prompt,
    )

    prediction = parse_label(
        reply=raw_reply,
        valid_labels=metadata["labels_by_target"][target_id],
    )

    status = "success" if prediction is not None else "invalid_reply"

    record = {
        "prompt_hash": job_hash,
        "mode": MODE,
        "respondent_id": respondent["respondent_id"],
        "country": respondent.get("country"),
        "question_id": target_id,
        "prediction": prediction,
        "raw_reply": raw_reply,
        "status": status,
        "user_prompt": user_prompt,
    }

    if truth_is_known:
        record["truth"] = decode_target_truth(
            target_id=target_id,
            value=respondent[target_id],
            option_to_label=metadata["option_to_label"],
        )

    append_cache(cache_path, record)

    with CACHE_LOCK:
        cache[job_hash] = record

    return record


# ============================================================
# 10. Run all requested respondent-target prompts
# ============================================================

def run_llm_jobs(
    data: fs.ChallengeData,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    target_ids: list[str],
    llm_features_by_target: dict[str, list[str]],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Run one prompt per respondent-target pair with resumable caching."""
    client = build_client()

    cache_path = (
        CACHE_DIR
        / f"{MODE}_{PROMPT_VERSION}_{MODEL.replace('/', '__')}.jsonl"
    )
    cache = load_cache(cache_path)

    jobs: list[tuple[pd.Series, str, bool]] = []

    for target_id in target_ids:
        frame, truth_is_known = select_mode_frame(
            mode=MODE,
            data=data,
            development=development,
            holdout=holdout,
            target_id=target_id,
        )

        for _, respondent in frame.iterrows():
            jobs.append(
                (
                    respondent,
                    target_id,
                    truth_is_known,
                )
            )

    results: list[dict] = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
    ) as executor:
        future_to_job = {
            executor.submit(
                predict_one,
                client,
                respondent,
                target_id,
                llm_features_by_target,
                metadata,
                cache,
                cache_path,
                truth_is_known,
            ): (
                respondent["respondent_id"],
                target_id,
            )
            for respondent, target_id, truth_is_known in jobs
        }

        for future in tqdm(
            as_completed(future_to_job),
            total=len(future_to_job),
            desc=f"LLM {MODE}",
        ):
            respondent_id, target_id = future_to_job[future]

            try:
                results.append(future.result())
            except Exception as error:
                results.append(
                    {
                        "mode": MODE,
                        "respondent_id": respondent_id,
                        "question_id": target_id,
                        "prediction": None,
                        "status": "error",
                        "error": repr(error),
                    }
                )

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / f"{MODE}_{PROMPT_VERSION}_raw_results.csv"
    results_df.to_csv(output_path, index=False)

    return results_df


# ============================================================
# 11. Evaluate known-label modes
# ============================================================

def calculate_skill(
    truth: pd.Series,
    prediction: pd.Series,
) -> dict[str, float]:
    """
    Calculate accuracy, majority accuracy, normalized Skill, and macro-F1.

    The local majority baseline is computed from the evaluated truth set.
    For model-selection work, a fold-specific train majority is preferable;
    this summary is primarily diagnostic for the zero-shot LLM.
    """
    valid = truth.notna() & prediction.notna()

    truth = truth.loc[valid]
    prediction = prediction.loc[valid]

    if truth.empty:
        return {
            "accuracy": np.nan,
            "majority_accuracy": np.nan,
            "skill": np.nan,
            "macro_f1": np.nan,
            "coverage": 0.0,
        }

    majority_label = truth.value_counts().idxmax()
    majority_accuracy = float(
        (truth == majority_label).mean()
    )
    accuracy = float(
        accuracy_score(truth, prediction)
    )

    if majority_accuracy >= 1.0:
        skill = 0.0
    else:
        skill = (
            accuracy - majority_accuracy
        ) / (
            1.0 - majority_accuracy
        )

    macro_f1 = float(
        f1_score(
            truth,
            prediction,
            average="macro",
            zero_division=0,
        )
    )

    coverage = float(valid.mean())

    return {
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "skill": skill,
        "macro_f1": macro_f1,
        "coverage": coverage,
    }


def summarise_known_labels(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create one validation summary row per target."""
    rows: list[dict] = []

    for target_id, group in results.groupby("question_id"):
        metrics = calculate_skill(
            truth=group["truth"],
            prediction=group["prediction"],
        )

        rows.append(
            {
                "mode": MODE,
                "question_id": target_id,
                "prompt_version": PROMPT_VERSION,
                "model": MODEL,
                "use_country": USE_COUNTRY,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        RESULTS_DIR / f"{MODE}_{PROMPT_VERSION}_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# 12. Build final predictions and submission prompt metadata
# ============================================================

def build_predictions_csv(
    results: pd.DataFrame,
    data: fs.ChallengeData,
    target_ids: list[str],
) -> pd.DataFrame:
    """Validate and save challenge-test LLM predictions."""
    predictions = results[
        [
            "respondent_id",
            "question_id",
            "prediction",
        ]
    ].copy()

    expected_rows = len(data.test) * len(target_ids)

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} test predictions, "
            f"found {len(predictions)}."
        )

    if predictions.duplicated(
        ["respondent_id", "question_id"]
    ).any():
        raise ValueError(
            "Duplicate respondent-target pairs detected."
        )

    if predictions["prediction"].isna().any():
        invalid = predictions[
            predictions["prediction"].isna()
        ].head()
        raise ValueError(
            "Invalid or missing LLM predictions remain:\n"
            f"{invalid}"
        )

    predictions.to_csv(
        RESULTS_DIR / "predictions_llm.csv",
        index=False,
    )

    return predictions


def export_prompts_jsonl(
    data: fs.ChallengeData,
    target_ids: list[str],
    llm_features_by_target: dict[str, list[str]],
    metadata: dict[str, Any],
) -> None:
    """
    Export one prompt-template record per target for method disclosure.

    The submission file contains templates and an example, not all dynamic
    respondent prompts.
    """
    example_respondent = data.test.iloc[0]

    output_path = SUBMISSION_METHOD_DIR / "prompts.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for target_id in target_ids:
            record = {
                "question_id": target_id,
                "prompt_version": PROMPT_VERSION,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "thinking_enabled": False,
                "use_country": USE_COUNTRY,
                "selected_features": (
                    llm_features_by_target[target_id]
                ),
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt_template": USER_PROMPT_TEMPLATE,
                "example_prompt": build_user_prompt(
                    respondent=example_respondent,
                    target_id=target_id,
                    llm_features_by_target=llm_features_by_target,
                    metadata=metadata,
                ),
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 13. Main
# ============================================================

def main() -> None:
    """Run the selected LLM prompting phase."""
    data = fs.load_challenge_data()
    selection = load_frozen_selection()

    development, holdout = recreate_development_and_holdout(
        full_train=data.train,
    )

    available_targets = list(selection.keys())

    target_ids = (
        TARGETS_TO_RUN
        if TARGETS_TO_RUN is not None
        else available_targets
    )

    missing_targets = set(target_ids) - set(available_targets)

    if missing_targets:
        raise ValueError(
            "The following targets are not present in the frozen "
            f"selection: {sorted(missing_targets)}"
        )

    llm_features_by_target = {
        target_id: selection[target_id]["llm_features"]
        for target_id in target_ids
    }

    metadata = build_metadata(data)

    export_prompts_jsonl(
        data=data,
        target_ids=target_ids,
        llm_features_by_target=llm_features_by_target,
        metadata=metadata,
    )

    print("\nFrozen LLM features")
    print("-------------------")
    for target_id in target_ids:
        print(
            f"{target_id}: "
            f"{llm_features_by_target[target_id]}"
        )

    print("\nPrompt configuration")
    print("--------------------")
    print(f"Mode:         {MODE}")
    print(f"Prompt ver.:  {PROMPT_VERSION}")
    print(f"Model:        {MODEL}")
    print(f"Use country:  {USE_COUNTRY}")
    print(f"Temperature:  {TEMPERATURE}")
    print(f"Max tokens:   {MAX_TOKENS}")

    results = run_llm_jobs(
        data=data,
        development=development,
        holdout=holdout,
        target_ids=target_ids,
        llm_features_by_target=llm_features_by_target,
        metadata=metadata,
    )

    if MODE in {
        "development_validation",
        "outer_holdout",
    }:
        summary = summarise_known_labels(results)

        print("\nLLM validation summary")
        print("----------------------")
        print(summary.to_string(index=False))

    elif MODE == "test_prediction":
        predictions = build_predictions_csv(
            results=results,
            data=data,
            target_ids=target_ids,
        )

        print(
            "\nFinal LLM predictions created: "
            f"{len(predictions)} rows"
        )
        print(
            f"Saved to: "
            f"{RESULTS_DIR / 'predictions_llm.csv'}"
        )


if __name__ == "__main__":
    main()
