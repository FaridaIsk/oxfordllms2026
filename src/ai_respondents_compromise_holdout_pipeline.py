
# ============================================================
# AI Respondents Challenge:
# Development-country selection + untouched country holdout
# ============================================================
#
# This script implements a practical compromise between ordinary
# cross-validation and fully nested cross-validation.
#
# Core idea:
#
# 1. Hold out a fixed set of complete countries before any feature
#    ranking or model selection is performed.
# 2. Run the full feature-selection pipeline only on the remaining
#    development countries.
# 3. Freeze the selected feature sets.
# 4. Evaluate the frozen CatBoost model once on the untouched countries.
# 5. Optionally refit the frozen model on all labelled training data
#    and generate predictions for the challenge test set.
#
# IMPORTANT:
# The outer country holdout is an evaluation set, not a tuning set.
# Do not repeatedly change the feature-selection rules after inspecting
# the outer-holdout results.
#
# This file expects the following companion module in the same folder:
#
#   ai_respondents_feature_selection_pipeline.py
#
# That module contains:
# - data loading;
# - LossFunctionChange screening;
# - repeated out-of-fold permutation importance;
# - seen-country and unseen-country inner CV;
# - correlation grouping;
# - top-k feature-set evaluation;
# - LLM feature recommendation.
# ============================================================

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import ai_respondents_feature_selection_pipeline as fs


# ============================================================
# 0. Configuration
# ============================================================

RANDOM_SEED = 42

# Hold out approximately 20% of unique countries.
OUTER_HOLDOUT_COUNTRY_FRACTION = 0.20

# The split search retries different random seeds until it finds
# a country split with adequate target coverage.
MAX_HOLDOUT_SPLIT_ATTEMPTS = 500
MIN_HOLDOUT_ROWS_PER_TARGET = 30

# Start with one target while debugging.
# Set to None to run all visible targets.
TARGETS_TO_RUN: list[str] | None = ["Q73"]

# Use a lightweight inner feature-selection run while debugging.
FAST_MODE = True

# Keep this False during method development.
# Set it to True only after the outer-holdout evaluation has been
# reviewed and the procedure has been frozen.
BUILD_FINAL_TEST_PREDICTIONS = False

OUTPUT_DIR = Path("outputs/compromise_holdout")
DEVELOPMENT_SELECTION_DIR = OUTPUT_DIR / "development_selection"
OUTER_EVALUATION_DIR = OUTPUT_DIR / "outer_holdout_evaluation"
FINAL_OUTPUT_DIR = OUTPUT_DIR / "final_test_predictions"

for directory in (
    OUTPUT_DIR,
    DEVELOPMENT_SELECTION_DIR,
    OUTER_EVALUATION_DIR,
    FINAL_OUTPUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Configure the imported feature-selection module
# ============================================================

def configure_inner_pipeline() -> None:
    """
    Configure the companion feature-selection module.

    The inner pipeline runs only on the development-country subset.
    """
    fs.OUTPUT_DIR = DEVELOPMENT_SELECTION_DIR
    fs.RANDOM_SEED = RANDOM_SEED

    if FAST_MODE:
        fs.N_SCREEN_FOLDS = 3
        fs.N_IMPORTANCE_FOLDS = 3
        fs.N_TOPK_FOLDS = 3
        fs.N_PERMUTATION_REPEATS = 3
        fs.CATBOOST_ITERATIONS = 300
    else:
        fs.N_SCREEN_FOLDS = 5
        fs.N_IMPORTANCE_FOLDS = 5
        fs.N_TOPK_FOLDS = 5
        fs.N_PERMUTATION_REPEATS = 10
        fs.CATBOOST_ITERATIONS = 700

    fs.CATBOOST_PARAMS["iterations"] = fs.CATBOOST_ITERATIONS
    fs.CATBOOST_PARAMS["random_seed"] = RANDOM_SEED

    # Equal weighting treats seen-country and unseen-country inner CV
    # as equally important during feature-set selection.
    fs.SEEN_WEIGHT = 0.50
    fs.UNSEEN_WEIGHT = 0.50

    # Example optional semantic grouping.
    # Extend this dictionary for other target questions if needed.
    fs.MANUAL_SEMANTIC_GROUPS_BY_TARGET = {
        "Q73": {
            "institutional_trust": [
                "Q69",
                "Q70",
                "Q71",
                "Q72",
                "Q74",
                "Q76",
            ],
        },
    }

    # Example optional manual additions for the LLM prompt.
    # Add at most one to three very close questions per target.
    fs.MANUAL_LLM_ADDITIONS = {
        "Q73": ["Q71", "Q72", "Q76"],
    }


# ============================================================
# 2. Choose one fixed outer holdout of complete countries
# ============================================================

def target_class_set(
    frame: pd.DataFrame,
    target_id: str,
) -> set[int]:
    """Return the observed integer class codes for one target."""
    values = frame[target_id].dropna()
    if values.empty:
        return set()
    return set(values.astype(float).astype(int).tolist())


def split_is_acceptable(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    target_ids: list[str],
) -> bool:
    """
    Check whether a candidate country holdout is usable.

    Requirements:
    - every target has enough labelled holdout rows;
    - every class observed in the holdout also appears in development;
    - every target has at least two classes in development;
    - the holdout contains at least two classes when possible.
    """
    for target_id in target_ids:
        development_values = development[target_id].dropna()
        holdout_values = holdout[target_id].dropna()

        if len(holdout_values) < MIN_HOLDOUT_ROWS_PER_TARGET:
            return False

        development_classes = target_class_set(development, target_id)
        holdout_classes = target_class_set(holdout, target_id)

        if len(development_classes) < 2:
            return False

        if not holdout_classes.issubset(development_classes):
            return False

        if len(holdout_classes) < 2:
            return False

    return True


def choose_fixed_country_holdout(
    train: pd.DataFrame,
    target_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], int]:
    """
    Search for a reproducible country split that satisfies class coverage.

    GroupShuffleSplit samples complete countries rather than individual rows.
    The returned split is selected before any feature importance is calculated.
    """
    groups = train["country"].astype(str)

    for attempt in range(MAX_HOLDOUT_SPLIT_ATTEMPTS):
        split_seed = RANDOM_SEED + attempt

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=OUTER_HOLDOUT_COUNTRY_FRACTION,
            random_state=split_seed,
        )

        development_idx, holdout_idx = next(
            splitter.split(train, groups=groups)
        )

        development = train.iloc[development_idx].copy()
        holdout = train.iloc[holdout_idx].copy()

        if not split_is_acceptable(
            development=development,
            holdout=holdout,
            target_ids=target_ids,
        ):
            continue

        development_countries = sorted(
            development["country"].astype(str).unique().tolist()
        )
        holdout_countries = sorted(
            holdout["country"].astype(str).unique().tolist()
        )

        overlap = set(development_countries) & set(holdout_countries)
        if overlap:
            raise RuntimeError(
                f"Country leakage detected: {sorted(overlap)}"
            )

        return (
            development.reset_index(drop=True),
            holdout.reset_index(drop=True),
            development_countries,
            holdout_countries,
            split_seed,
        )

    raise RuntimeError(
        "Could not find an acceptable country holdout. "
        "Reduce MIN_HOLDOUT_ROWS_PER_TARGET or the holdout fraction."
    )


def save_split_manifest(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    development_countries: list[str],
    holdout_countries: list[str],
    split_seed: int,
    target_ids: list[str],
) -> None:
    """Save the frozen country split and target coverage summary."""
    manifest = {
        "split_seed": split_seed,
        "outer_holdout_country_fraction": (
            OUTER_HOLDOUT_COUNTRY_FRACTION
        ),
        "development_countries": development_countries,
        "holdout_countries": holdout_countries,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
    }

    with (OUTPUT_DIR / "country_split_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    coverage_rows = []

    for target_id in target_ids:
        development_values = development[target_id].dropna()
        holdout_values = holdout[target_id].dropna()

        coverage_rows.append(
            {
                "target_id": target_id,
                "development_labelled_rows": len(development_values),
                "holdout_labelled_rows": len(holdout_values),
                "development_classes": sorted(
                    target_class_set(development, target_id)
                ),
                "holdout_classes": sorted(
                    target_class_set(holdout, target_id)
                ),
            }
        )

    pd.DataFrame(coverage_rows).to_csv(
        OUTPUT_DIR / "country_split_target_coverage.csv",
        index=False,
    )


# ============================================================
# 3. Build a development-only ChallengeData object
# ============================================================

def make_development_data(
    full_data: fs.ChallengeData,
    development_train: pd.DataFrame,
) -> fs.ChallengeData:
    """
    Create a ChallengeData object whose train table contains only
    development countries.

    The untouched holdout rows are not available to the inner pipeline.
    """
    return fs.ChallengeData(
        train=development_train.copy(),
        test=full_data.test,
        features=full_data.features,
        targets=full_data.targets,
        allowed_features=full_data.allowed_features,
        target_ids=full_data.target_ids,
        question_text=full_data.question_text,
    )


# ============================================================
# 4. Run feature selection on development countries only
# ============================================================

def run_development_feature_selection(
    development_data: fs.ChallengeData,
    target_ids: list[str],
) -> dict[str, dict]:
    """
    Run the complete feature-selection pipeline without accessing
    the outer country holdout.
    """
    results: dict[str, dict] = {}

    for target_id in target_ids:
        results[target_id] = fs.run_target_pipeline(
            data=development_data,
            target_id=target_id,
        )

    with (
        DEVELOPMENT_SELECTION_DIR
        / "frozen_development_selection.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    return results


# ============================================================
# 5. Evaluate the frozen model once on untouched countries
# ============================================================

def build_option_to_label(
    targets: pd.DataFrame,
) -> dict[str, dict[int, str]]:
    """Create target-specific mappings from numeric option to label."""
    mappings: dict[str, dict[int, str]] = {}

    for target_id, group in targets.groupby("question_id"):
        mappings[str(target_id)] = {
            int(float(option)): str(label)
            for option, label in zip(
                group["option"],
                group["label"],
            )
        }

    return mappings


def evaluate_frozen_features_on_holdout(
    full_data: fs.ChallengeData,
    development_train: pd.DataFrame,
    holdout_train: pd.DataFrame,
    selection_results: dict[str, dict],
    target_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one fresh CatBoost model on all development countries and evaluate
    it once on the untouched holdout countries.

    No feature ranking, top-k selection, threshold selection, or model
    comparison is performed on the outer holdout.
    """
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    option_to_label = build_option_to_label(full_data.targets)

    for target_id in target_ids:
        selected_features = selection_results[target_id][
            "chosen_catboost_features"
        ]

        development_mask = development_train[target_id].notna()
        holdout_mask = holdout_train[target_id].notna()

        X_development = development_train.loc[
            development_mask,
            full_data.allowed_features,
        ].reset_index(drop=True)

        y_development = (
            development_train.loc[development_mask, target_id]
            .astype(float)
            .astype(int)
            .reset_index(drop=True)
        )

        X_holdout = holdout_train.loc[
            holdout_mask,
            full_data.allowed_features,
        ].reset_index(drop=True)

        y_holdout = (
            holdout_train.loc[holdout_mask, target_id]
            .astype(float)
            .astype(int)
            .reset_index(drop=True)
        )

        holdout_metadata = holdout_train.loc[
            holdout_mask,
            ["respondent_id", "country"],
        ].reset_index(drop=True)

        model, active_features = fs.fit_catboost(
            X_train=X_development,
            y_train=y_development,
            feature_names=selected_features,
            random_seed=RANDOM_SEED,
        )

        y_pred = fs.predict_classes(
            model=model,
            X=X_holdout,
            active_features=active_features,
        )

        metrics = fs.calculate_fold_metrics(
            y_train=y_development,
            y_valid=y_holdout,
            y_pred=y_pred,
        )

        metric_rows.append(
            {
                "target_id": target_id,
                "evaluation": "untouched_country_holdout",
                "selected_feature_set": selection_results[target_id][
                    "chosen_catboost_feature_set"
                ],
                "requested_n_features": len(selected_features),
                "active_n_features": len(active_features),
                "development_countries": (
                    development_train["country"].nunique()
                ),
                "holdout_countries": holdout_train["country"].nunique(),
                **metrics,
            }
        )

        for row_number in range(len(y_holdout)):
            truth_code = int(y_holdout.iloc[row_number])
            prediction_code = int(y_pred[row_number])

            prediction_rows.append(
                {
                    "respondent_id": holdout_metadata.loc[
                        row_number,
                        "respondent_id",
                    ],
                    "country": holdout_metadata.loc[
                        row_number,
                        "country",
                    ],
                    "question_id": target_id,
                    "truth_code": truth_code,
                    "prediction_code": prediction_code,
                    "truth_label": option_to_label[target_id].get(
                        truth_code,
                        str(truth_code),
                    ),
                    "prediction_label": option_to_label[target_id].get(
                        prediction_code,
                        str(prediction_code),
                    ),
                    "correct": prediction_code == truth_code,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    metrics_df.to_csv(
        OUTER_EVALUATION_DIR / "outer_holdout_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTER_EVALUATION_DIR / "outer_holdout_predictions.csv",
        index=False,
    )

    return metrics_df, predictions_df


# ============================================================
# 6. Optional final refit on all labelled train rows
# ============================================================

def build_final_catboost_predictions(
    full_data: fs.ChallengeData,
    selection_results: dict[str, dict],
    target_ids: list[str],
) -> pd.DataFrame:
    """
    Refit the frozen feature sets on all labelled training rows and predict
    the challenge test set.

    The feature lists are not reselected after viewing the outer holdout.
    """
    option_to_label = build_option_to_label(full_data.targets)

    prediction_rows: list[dict] = []
    feature_rows: list[dict] = []

    for target_id in target_ids:
        selected_features = selection_results[target_id][
            "chosen_catboost_features"
        ]

        labelled_mask = full_data.train[target_id].notna()

        X_train = full_data.train.loc[
            labelled_mask,
            full_data.allowed_features,
        ].reset_index(drop=True)

        y_train = (
            full_data.train.loc[labelled_mask, target_id]
            .astype(float)
            .astype(int)
            .reset_index(drop=True)
        )

        X_test = full_data.test[
            full_data.allowed_features
        ].reset_index(drop=True)

        model, active_features = fs.fit_catboost(
            X_train=X_train,
            y_train=y_train,
            feature_names=selected_features,
            random_seed=RANDOM_SEED,
        )

        prediction_codes = fs.predict_classes(
            model=model,
            X=X_test,
            active_features=active_features,
        )

        for row_number, prediction_code in enumerate(
            prediction_codes
        ):
            prediction_code = int(prediction_code)

            prediction_rows.append(
                {
                    "respondent_id": full_data.test.loc[
                        row_number,
                        "respondent_id",
                    ],
                    "question_id": target_id,
                    "prediction": option_to_label[target_id][
                        prediction_code
                    ],
                }
            )

        for feature in selected_features:
            feature_rows.append(
                {
                    "question_id": target_id,
                    "feature_variable_code": feature,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    used_features = pd.DataFrame(feature_rows).drop_duplicates()

    expected_rows = len(full_data.test) * len(target_ids)

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} predictions, "
            f"found {len(predictions)}."
        )

    if predictions.duplicated(
        ["respondent_id", "question_id"]
    ).any():
        raise ValueError(
            "Duplicate respondent-target prediction pairs detected."
        )

    if predictions["prediction"].isna().any():
        raise ValueError("Missing predictions detected.")

    predictions.to_csv(
        FINAL_OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    used_features.to_csv(
        FINAL_OUTPUT_DIR / "features.csv",
        index=False,
    )

    with (
        FINAL_OUTPUT_DIR
        / "llm_features_by_target.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                target_id: selection_results[target_id]["llm_features"]
                for target_id in target_ids
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return predictions


# ============================================================
# 7. Main execution
# ============================================================

def main() -> None:
    """Run the compromise holdout pipeline."""
    configure_inner_pipeline()

    full_data = fs.load_challenge_data()

    target_ids = (
        TARGETS_TO_RUN
        if TARGETS_TO_RUN is not None
        else full_data.target_ids
    )

    invalid_targets = set(target_ids) - set(full_data.target_ids)
    if invalid_targets:
        raise ValueError(
            f"Unknown targets requested: {sorted(invalid_targets)}"
        )

    (
        development_train,
        holdout_train,
        development_countries,
        holdout_countries,
        split_seed,
    ) = choose_fixed_country_holdout(
        train=full_data.train,
        target_ids=target_ids,
    )

    save_split_manifest(
        development=development_train,
        holdout=holdout_train,
        development_countries=development_countries,
        holdout_countries=holdout_countries,
        split_seed=split_seed,
        target_ids=target_ids,
    )

    print("\nFrozen outer split")
    print("------------------")
    print(f"Development countries: {len(development_countries)}")
    print(f"Holdout countries:     {len(holdout_countries)}")
    print(f"Development rows:      {len(development_train)}")
    print(f"Holdout rows:          {len(holdout_train)}")
    print(f"Split seed:            {split_seed}")
    print(f"Holdout list:          {holdout_countries}")

    development_data = make_development_data(
        full_data=full_data,
        development_train=development_train,
    )

    # All ranking and top-k selection happens only here.
    selection_results = run_development_feature_selection(
        development_data=development_data,
        target_ids=target_ids,
    )

    # The outer holdout is opened only after the selection is frozen.
    metrics_df, _ = evaluate_frozen_features_on_holdout(
        full_data=full_data,
        development_train=development_train,
        holdout_train=holdout_train,
        selection_results=selection_results,
        target_ids=target_ids,
    )

    print("\nUntouched country-holdout results")
    print("---------------------------------")
    print(
        metrics_df[
            [
                "target_id",
                "skill",
                "accuracy",
                "majority_accuracy",
                "macro_f1",
            ]
        ].to_string(index=False)
    )

    print(
        "\nMean outer-holdout Skill: "
        f"{metrics_df['skill'].mean():.4f}"
    )

    if BUILD_FINAL_TEST_PREDICTIONS:
        predictions = build_final_catboost_predictions(
            full_data=full_data,
            selection_results=selection_results,
            target_ids=target_ids,
        )

        print(
            "\nFinal test predictions created: "
            f"{len(predictions)} rows"
        )
        print(
            f"Saved to: {FINAL_OUTPUT_DIR / 'predictions.csv'}"
        )
    else:
        print(
            "\nBUILD_FINAL_TEST_PREDICTIONS is False. "
            "Review the untouched holdout once, freeze the method, "
            "then set it to True for the final refit."
        )


if __name__ == "__main__":
    main()
