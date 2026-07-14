
# ============================================================
# AI Respondents Challenge: feature-selection pipeline
# ============================================================
# This script implements:
# 1. CatBoost on all allowed features.
# 2. Fast screening with LossFunctionChange.
# 3. Repeated out-of-fold permutation importance on top candidates.
# 4. Aggregation across seen-country and held-out-country CV.
# 5. Correlation/manual semantic grouping.
# 6. Top-k feature-set evaluation by CV Skill.
# 7. CatBoost feature-set selection.
# 8. LLM feature-set recommendation.
# 9. Optional manual additions for semantically close questions.
# ============================================================

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from tqdm.auto import tqdm


# ============================================================
# 0. Configuration
# ============================================================

DATASET_ID = "oxford-llms/ai-respondents-challenge"
OUTPUT_DIR = Path("outputs/feature_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Start with one target while debugging, for example ["Q73"].
# Set to None to run all visible targets.
TARGETS_TO_RUN: list[str] | None = ["Q73"]

# FAST_MODE is suitable for debugging and an initial submission.
# Set it to False for a more thorough final run.
FAST_MODE = True

if FAST_MODE:
    N_SCREEN_FOLDS = 3
    N_IMPORTANCE_FOLDS = 3
    N_TOPK_FOLDS = 3
    N_PERMUTATION_REPEATS = 3
    CATBOOST_ITERATIONS = 300
else:
    N_SCREEN_FOLDS = 5
    N_IMPORTANCE_FOLDS = 5
    N_TOPK_FOLDS = 5
    N_PERMUTATION_REPEATS = 10
    CATBOOST_ITERATIONS = 700

TOP_CANDIDATES = 50
TOP_K_VALUES = [5, 10, 15, 20, 30]
CORRELATION_THRESHOLD = 0.75
MIN_CORRELATION_OBSERVATIONS = 100

# Weight used to combine seen-country and unseen-country results.
SEEN_WEIGHT = 0.50
UNSEEN_WEIGHT = 0.50

# LLM profile constraints.
LLM_MIN_FEATURES = 8
LLM_MAX_FEATURES = 15
LLM_MAX_FEATURES_PER_GROUP = 2
MIN_FEATURE_COVERAGE_FOR_LLM = 0.20

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": CATBOOST_ITERATIONS,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}

# Add manually curated semantic groups here.
# Features in the same manual group are treated as one redundancy block.
MANUAL_SEMANTIC_GROUPS_BY_TARGET: dict[str, dict[str, list[str]]] = {
    # Example:
    # "Q73": {
    #     "institutional_trust": ["Q69", "Q70", "Q71", "Q72", "Q74", "Q76"],
    # }
}

# Add one to three semantically close features after automatic ranking.
MANUAL_LLM_ADDITIONS: dict[str, list[str]] = {
    # Example:
    # "Q73": ["Q71", "Q72", "Q76"],
}


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class ChallengeData:
    train: pd.DataFrame
    test: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    allowed_features: list[str]
    target_ids: list[str]
    question_text: dict[str, str]


# ============================================================
# 2. Load and validate the challenge data
# ============================================================

def load_challenge_data() -> ChallengeData:
    """Load all four challenge configurations from Hugging Face."""
    train = load_dataset(DATASET_ID, "train", split="train").to_pandas()
    test = load_dataset(DATASET_ID, "test", split="train").to_pandas()
    features = load_dataset(DATASET_ID, "features", split="train").to_pandas()
    targets = load_dataset(DATASET_ID, "targets", split="train").to_pandas()

    required_feature_columns = {"variable", "question", "values_json"}
    required_target_columns = {"question_id", "question", "option", "label"}

    missing_feature_columns = required_feature_columns - set(features.columns)
    missing_target_columns = required_target_columns - set(targets.columns)

    if missing_feature_columns:
        raise ValueError(
            f"Missing feature metadata columns: {sorted(missing_feature_columns)}"
        )
    if missing_target_columns:
        raise ValueError(
            f"Missing target metadata columns: {sorted(missing_target_columns)}"
        )

    allowed_features = features["variable"].astype(str).tolist()
    target_ids = targets["question_id"].drop_duplicates().astype(str).tolist()
    question_text = dict(
        zip(features["variable"].astype(str), features["question"].astype(str))
    )

    missing_train_features = set(allowed_features) - set(train.columns)
    missing_test_features = set(allowed_features) - set(test.columns)

    if missing_train_features:
        raise ValueError(
            f"Allowed features missing from train: {sorted(missing_train_features)}"
        )
    if missing_test_features:
        raise ValueError(
            f"Allowed features missing from test: {sorted(missing_test_features)}"
        )

    if "country" not in train.columns:
        raise ValueError("The train table must contain the 'country' column.")

    return ChallengeData(
        train=train,
        test=test,
        features=features,
        targets=targets,
        allowed_features=allowed_features,
        target_ids=target_ids,
        question_text=question_text,
    )


# ============================================================
# 3. Target-specific modeling table
# ============================================================

def prepare_target_data(
    train: pd.DataFrame,
    target_id: str,
    allowed_features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create X, y, and country groups for one target.

    Target rows with missing labels are removed.
    Target values are converted to integer class codes.
    """
    if target_id not in train.columns:
        raise ValueError(f"Target {target_id} is not present in train.")

    mask = train[target_id].notna()

    X = train.loc[mask, allowed_features].copy()
    y = train.loc[mask, target_id].astype(float).astype(int).copy()
    groups = train.loc[mask, "country"].astype(str).copy()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError(f"Target {target_id} has fewer than two observed classes.")

    return X, y, groups


# ============================================================
# 4. Cross-validation splitters
# ============================================================

def _safe_stratified_splits(y: pd.Series, requested_splits: int) -> int:
    """Reduce the number of folds when a class is too rare."""
    minimum_class_count = int(y.value_counts().min())
    return max(2, min(requested_splits, minimum_class_count))


def make_cv_splits(
    y: pd.Series,
    groups: pd.Series,
    scheme: Literal["seen", "unseen"],
    n_splits: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create CV splits.

    'seen' uses StratifiedKFold for held-out respondents.
    'unseen' uses StratifiedGroupKFold to hold out entire countries.
    GroupKFold is used as a fallback if stratified grouping is infeasible.
    """
    if scheme == "seen":
        actual_splits = _safe_stratified_splits(y, n_splits)
        splitter = StratifiedKFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=random_seed,
        )
        return list(splitter.split(np.zeros(len(y)), y))

    actual_splits = min(n_splits, groups.nunique())
    if actual_splits < 2:
        raise ValueError("At least two countries are required for grouped CV.")

    try:
        splitter = StratifiedGroupKFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=random_seed,
        )
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
    except (TypeError, ValueError):
        splitter = GroupKFold(n_splits=actual_splits)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))

    return splits


# ============================================================
# 5. Metrics
# ============================================================

def calculate_fold_metrics(
    y_train: pd.Series,
    y_valid: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate accuracy, majority-baseline accuracy, normalized skill,
    and macro-F1 for one validation fold.
    """
    majority_label = y_train.value_counts().idxmax()
    majority_accuracy = float((y_valid == majority_label).mean())
    accuracy = float(accuracy_score(y_valid, y_pred))

    if majority_accuracy >= 1.0:
        skill = 0.0
    else:
        skill = (accuracy - majority_accuracy) / (1.0 - majority_accuracy)

    macro_f1 = float(
        f1_score(
            y_valid,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "skill": skill,
        "macro_f1": macro_f1,
    }


# ============================================================
# 6. CatBoost helpers
# ============================================================

def get_active_features(
    X_train: pd.DataFrame,
    feature_names: Iterable[str],
) -> list[str]:
    """
    Keep features that contain at least two observed values in the training fold.

    This prevents fold-specific all-missing or constant columns from causing
    unstable models and meaningless importance values.
    """
    active = []

    for feature in feature_names:
        observed = X_train[feature].dropna()
        if len(observed) > 0 and observed.nunique() > 1:
            active.append(feature)

    return active


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    random_seed: int,
) -> tuple[CatBoostClassifier, list[str]]:
    """Train one CatBoost multiclass model on active numeric features."""
    active_features = get_active_features(X_train, feature_names)

    if not active_features:
        raise ValueError("No active features are available in this training fold.")

    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = random_seed

    model = CatBoostClassifier(**params)
    train_pool = Pool(
        data=X_train[active_features],
        label=y_train,
        feature_names=active_features,
    )
    model.fit(train_pool)

    return model, active_features


def predict_classes(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    active_features: list[str],
) -> np.ndarray:
    """Return a one-dimensional array of predicted class codes."""
    predictions = model.predict(X[active_features])
    return np.asarray(predictions).reshape(-1).astype(int)


# ============================================================
# 7. Step 1-2: all-feature CatBoost and LossFunctionChange screening
# ============================================================

def compute_loss_function_change_screening(
    target_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    all_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train CatBoost on all allowed features across seen-country folds and
    calculate LossFunctionChange on each held-out fold.

    The resulting ranking is used only as a fast screening stage.
    """
    splits = make_cv_splits(
        y=y,
        groups=groups,
        scheme="seen",
        n_splits=N_SCREEN_FOLDS,
        random_seed=RANDOM_SEED,
    )

    importance_records: list[dict] = []
    metric_records: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        if y_train.nunique() < 2:
            continue

        model, active_features = fit_catboost(
            X_train=X_train,
            y_train=y_train,
            feature_names=all_features,
            random_seed=RANDOM_SEED + fold,
        )

        valid_pool = Pool(
            data=X_valid[active_features],
            label=y_valid,
            feature_names=active_features,
        )

        importance_values = model.get_feature_importance(
            data=valid_pool,
            type="LossFunctionChange",
        )

        fold_importance = dict(zip(active_features, importance_values))

        for feature in all_features:
            importance_records.append(
                {
                    "target_id": target_id,
                    "fold": fold,
                    "feature": feature,
                    "loss_change": float(fold_importance.get(feature, 0.0)),
                }
            )

        y_pred = predict_classes(model, X_valid, active_features)
        metrics = calculate_fold_metrics(y_train, y_valid, y_pred)
        metric_records.append(
            {
                "target_id": target_id,
                "stage": "loss_change_screening",
                "scheme": "seen",
                "fold": fold,
                "n_features": len(active_features),
                **metrics,
            }
        )

    importance_df = pd.DataFrame(importance_records)
    metrics_df = pd.DataFrame(metric_records)

    ranking = (
        importance_df.groupby(["target_id", "feature"], as_index=False)
        .agg(
            loss_change_mean=("loss_change", "mean"),
            loss_change_std=("loss_change", "std"),
            loss_change_positive_share=(
                "loss_change",
                lambda values: float((values > 0).mean()),
            ),
        )
        .sort_values(
            ["target_id", "loss_change_mean"],
            ascending=[True, False],
        )
    )

    return ranking, metrics_df


# ============================================================
# 8. Step 3: repeated OOF permutation importance
# ============================================================

def permutation_importance_for_candidates(
    model: CatBoostClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    active_features: list[str],
    candidate_features: list[str],
    n_repeats: int,
    random_seed: int,
) -> list[dict]:
    """
    Calculate repeated permutation importance for a subset of candidates.

    The model still receives its complete active feature matrix.
    Only one candidate column is shuffled at a time.
    """
    baseline_prediction = predict_classes(model, X_valid, active_features)
    baseline_accuracy = float(accuracy_score(y_valid, baseline_prediction))

    records: list[dict] = []
    rng = np.random.default_rng(random_seed)

    for feature in candidate_features:
        if feature not in active_features:
            for repeat in range(n_repeats):
                records.append(
                    {
                        "feature": feature,
                        "repeat": repeat,
                        "baseline_accuracy": baseline_accuracy,
                        "permuted_accuracy": baseline_accuracy,
                        "importance": 0.0,
                    }
                )
            continue

        original_values = X_valid[feature].to_numpy(copy=True)

        for repeat in range(n_repeats):
            X_permuted = X_valid[active_features].copy()
            X_permuted[feature] = rng.permutation(original_values)

            permuted_prediction = np.asarray(
                model.predict(X_permuted)
            ).reshape(-1).astype(int)

            permuted_accuracy = float(
                accuracy_score(y_valid, permuted_prediction)
            )

            records.append(
                {
                    "feature": feature,
                    "repeat": repeat,
                    "baseline_accuracy": baseline_accuracy,
                    "permuted_accuracy": permuted_accuracy,
                    "importance": baseline_accuracy - permuted_accuracy,
                }
            )

    return records


def compute_oof_permutation_importance(
    target_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    all_features: list[str],
    candidate_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate repeated permutation importance under both validation schemes.

    Models are trained on all allowed active features.
    Permutation importance is calculated only for the shortlisted candidates.
    """
    importance_records: list[dict] = []
    metric_records: list[dict] = []

    for scheme in ("seen", "unseen"):
        splits = make_cv_splits(
            y=y,
            groups=groups,
            scheme=scheme,
            n_splits=N_IMPORTANCE_FOLDS,
            random_seed=RANDOM_SEED,
        )

        for fold, (train_idx, valid_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            if y_train.nunique() < 2:
                continue

            model, active_features = fit_catboost(
                X_train=X_train,
                y_train=y_train,
                feature_names=all_features,
                random_seed=RANDOM_SEED + 1000 * (scheme == "unseen") + fold,
            )

            y_pred = predict_classes(model, X_valid, active_features)
            metrics = calculate_fold_metrics(y_train, y_valid, y_pred)

            metric_records.append(
                {
                    "target_id": target_id,
                    "stage": "permutation_importance",
                    "scheme": scheme,
                    "fold": fold,
                    "n_features": len(active_features),
                    **metrics,
                }
            )

            fold_records = permutation_importance_for_candidates(
                model=model,
                X_valid=X_valid,
                y_valid=y_valid,
                active_features=active_features,
                candidate_features=candidate_features,
                n_repeats=N_PERMUTATION_REPEATS,
                random_seed=RANDOM_SEED + fold,
            )

            for record in fold_records:
                importance_records.append(
                    {
                        "target_id": target_id,
                        "scheme": scheme,
                        "fold": fold,
                        **record,
                    }
                )

    return pd.DataFrame(importance_records), pd.DataFrame(metric_records)


# ============================================================
# 9. Step 4: aggregate seen and unseen importance
# ============================================================

def aggregate_permutation_importance(
    permutation_df: pd.DataFrame,
    loss_change_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate repeated importance values and build a combined ranking."""
    by_scheme = (
        permutation_df.groupby(
            ["target_id", "feature", "scheme"],
            as_index=False,
        )
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            positive_share=(
                "importance",
                lambda values: float((values > 0).mean()),
            ),
            n_measurements=("importance", "size"),
        )
    )

    mean_wide = by_scheme.pivot(
        index=["target_id", "feature"],
        columns="scheme",
        values="importance_mean",
    ).fillna(0.0)

    positive_wide = by_scheme.pivot(
        index=["target_id", "feature"],
        columns="scheme",
        values="positive_share",
    ).fillna(0.0)

    std_wide = by_scheme.pivot(
        index=["target_id", "feature"],
        columns="scheme",
        values="importance_std",
    ).fillna(0.0)

    ranking = mean_wide.reset_index()
    ranking = ranking.rename(
        columns={
            "seen": "seen_importance",
            "unseen": "unseen_importance",
        }
    )

    positive_reset = positive_wide.reset_index().rename(
        columns={
            "seen": "seen_positive_share",
            "unseen": "unseen_positive_share",
        }
    )

    std_reset = std_wide.reset_index().rename(
        columns={
            "seen": "seen_importance_std",
            "unseen": "unseen_importance_std",
        }
    )

    ranking = ranking.merge(
        positive_reset,
        on=["target_id", "feature"],
        how="left",
    ).merge(
        std_reset,
        on=["target_id", "feature"],
        how="left",
    )

    for column in [
        "seen_importance",
        "unseen_importance",
        "seen_positive_share",
        "unseen_positive_share",
        "seen_importance_std",
        "unseen_importance_std",
    ]:
        if column not in ranking:
            ranking[column] = 0.0

    ranking["combined_importance"] = (
        SEEN_WEIGHT * ranking["seen_importance"]
        + UNSEEN_WEIGHT * ranking["unseen_importance"]
    )

    ranking["combined_positive_share"] = (
        SEEN_WEIGHT * ranking["seen_positive_share"]
        + UNSEEN_WEIGHT * ranking["unseen_positive_share"]
    )

    ranking = ranking.merge(
        loss_change_ranking[
            [
                "target_id",
                "feature",
                "loss_change_mean",
                "loss_change_std",
                "loss_change_positive_share",
            ]
        ],
        on=["target_id", "feature"],
        how="left",
    )

    ranking = ranking.sort_values(
        ["target_id", "combined_importance", "loss_change_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    ranking["combined_rank"] = (
        ranking.groupby("target_id")["combined_importance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return ranking


# ============================================================
# 10. Step 5: correlation and manual semantic groups
# ============================================================

class UnionFind:
    """Small union-find implementation for correlation components."""

    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def build_feature_groups(
    target_id: str,
    X: pd.DataFrame,
    candidate_features: list[str],
    manual_groups: dict[str, list[str]] | None = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Build redundancy groups using absolute Spearman correlation.

    Manual semantic groups can additionally connect features that should be
    reviewed together even when pairwise correlation is below the threshold.
    """
    candidate_features = [
        feature for feature in candidate_features if feature in X.columns
    ]

    union_find = UnionFind(candidate_features)

    correlation_matrix = X[candidate_features].corr(
        method="spearman",
        min_periods=MIN_CORRELATION_OBSERVATIONS,
    ).abs()

    correlation_records: list[dict] = []

    for i, first in enumerate(candidate_features):
        for second in candidate_features[i + 1 :]:
            value = correlation_matrix.loc[first, second]

            if pd.notna(value):
                correlation_records.append(
                    {
                        "target_id": target_id,
                        "feature_1": first,
                        "feature_2": second,
                        "abs_spearman": float(value),
                    }
                )

                if value >= CORRELATION_THRESHOLD:
                    union_find.union(first, second)

    manual_groups = manual_groups or {}

    for _, members in manual_groups.items():
        valid_members = [
            feature for feature in members if feature in union_find.parent
        ]
        for first, second in zip(valid_members, valid_members[1:]):
            union_find.union(first, second)

    components: dict[str, list[str]] = defaultdict(list)
    for feature in candidate_features:
        components[union_find.find(feature)].append(feature)

    feature_to_group: dict[str, str] = {}

    for group_number, members in enumerate(
        sorted(components.values(), key=lambda values: values[0]),
        start=1,
    ):
        group_id = f"{target_id}_group_{group_number:02d}"
        for feature in members:
            feature_to_group[feature] = group_id

    return feature_to_group, pd.DataFrame(correlation_records)


def make_group_limited_ranking(
    ranking: pd.DataFrame,
    feature_to_group: dict[str, str],
    max_per_group: int,
) -> list[str]:
    """Create a ranked list while limiting redundancy within each group."""
    selected: list[str] = []
    counts: dict[str, int] = defaultdict(int)

    for feature in ranking["feature"]:
        group_id = feature_to_group.get(feature, f"singleton_{feature}")

        if counts[group_id] >= max_per_group:
            continue

        selected.append(feature)
        counts[group_id] += 1

    return selected


# ============================================================
# 11. Step 6: top-k CV evaluation
# ============================================================

def evaluate_feature_set_cv(
    target_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    selected_features: list[str],
    feature_set_name: str,
    scheme: Literal["seen", "unseen"],
) -> pd.DataFrame:
    """Evaluate one feature set using one CV scheme."""
    splits = make_cv_splits(
        y=y,
        groups=groups,
        scheme=scheme,
        n_splits=N_TOPK_FOLDS,
        random_seed=RANDOM_SEED + 100,
    )

    records: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        active_features = get_active_features(X_train, selected_features)

        if y_train.nunique() < 2 or not active_features:
            majority_label = y_train.value_counts().idxmax()
            y_pred = np.repeat(majority_label, len(y_valid))
        else:
            model, active_features = fit_catboost(
                X_train=X_train,
                y_train=y_train,
                feature_names=selected_features,
                random_seed=RANDOM_SEED + fold,
            )
            y_pred = predict_classes(model, X_valid, active_features)

        metrics = calculate_fold_metrics(y_train, y_valid, y_pred)

        records.append(
            {
                "target_id": target_id,
                "feature_set": feature_set_name,
                "scheme": scheme,
                "fold": fold,
                "requested_n_features": len(selected_features),
                "active_n_features": len(active_features),
                **metrics,
            }
        )

    return pd.DataFrame(records)


def summarize_topk_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level scores and calculate a combined seen/unseen score."""
    summary = (
        fold_results.groupby(
            [
                "target_id",
                "feature_set",
                "scheme",
                "requested_n_features",
            ],
            as_index=False,
        )
        .agg(
            skill_mean=("skill", "mean"),
            skill_std=("skill", "std"),
            skill_count=("skill", "size"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
        )
    )

    summary["skill_se"] = (
        summary["skill_std"].fillna(0.0)
        / np.sqrt(summary["skill_count"].clip(lower=1))
    )

    index_columns = [
        "target_id",
        "feature_set",
        "requested_n_features",
    ]

    skill_mean = summary.pivot(
        index=index_columns,
        columns="scheme",
        values="skill_mean",
    ).reset_index()

    skill_se = summary.pivot(
        index=index_columns,
        columns="scheme",
        values="skill_se",
    ).reset_index()

    skill_mean = skill_mean.rename(
        columns={
            "seen": "seen_skill",
            "unseen": "unseen_skill",
        }
    )
    skill_se = skill_se.rename(
        columns={
            "seen": "seen_skill_se",
            "unseen": "unseen_skill_se",
        }
    )

    final = skill_mean.merge(skill_se, on=index_columns, how="left")

    for column in [
        "seen_skill",
        "unseen_skill",
        "seen_skill_se",
        "unseen_skill_se",
    ]:
        if column not in final:
            final[column] = 0.0

    final["combined_skill"] = (
        SEEN_WEIGHT * final["seen_skill"]
        + UNSEEN_WEIGHT * final["unseen_skill"]
    )

    final["combined_skill_se"] = np.sqrt(
        (SEEN_WEIGHT * final["seen_skill_se"]) ** 2
        + (UNSEEN_WEIGHT * final["unseen_skill_se"]) ** 2
    )

    return final.sort_values(
        ["target_id", "combined_skill"],
        ascending=[True, False],
    ).reset_index(drop=True)


def select_feature_set_one_se(summary: pd.DataFrame) -> pd.Series:
    """
    Apply a one-standard-error rule.

    Choose the smallest feature set whose combined score is within one
    standard error of the best observed score.
    """
    best_row = summary.loc[summary["combined_skill"].idxmax()]
    threshold = (
        best_row["combined_skill"]
        - best_row["combined_skill_se"]
    )

    eligible = summary[
        summary["combined_skill"] >= threshold
    ].copy()

    return eligible.sort_values(
        ["requested_n_features", "combined_skill"],
        ascending=[True, False],
    ).iloc[0]


# ============================================================
# 12. Step 8-9: recommend LLM features and add manual questions
# ============================================================

def recommend_llm_features(
    target_id: str,
    ranking: pd.DataFrame,
    feature_to_group: dict[str, str],
    X: pd.DataFrame,
    allowed_features: set[str],
    manual_additions: list[str] | None = None,
) -> list[str]:
    """
    Recommend a compact, stable, de-redundant feature list for an LLM prompt.

    Manual additions are retained when they are valid allowed features.
    """
    manual_additions = manual_additions or []

    valid_manual_additions = [
        feature
        for feature in manual_additions
        if feature in allowed_features and feature != target_id
    ][:3]

    coverage = X.notna().mean()

    automated_candidates = ranking[
        ranking["feature"].map(coverage).fillna(0.0)
        >= MIN_FEATURE_COVERAGE_FOR_LLM
    ].copy()

    automated_candidates = automated_candidates.sort_values(
        ["combined_importance", "combined_positive_share"],
        ascending=[False, False],
    )

    automated_limit = max(
        0,
        LLM_MAX_FEATURES - len(valid_manual_additions),
    )

    selected: list[str] = []
    group_counts: dict[str, int] = defaultdict(int)

    for feature in automated_candidates["feature"]:
        if feature in valid_manual_additions:
            continue

        group_id = feature_to_group.get(feature, f"singleton_{feature}")

        if group_counts[group_id] >= LLM_MAX_FEATURES_PER_GROUP:
            continue

        selected.append(feature)
        group_counts[group_id] += 1

        if len(selected) >= automated_limit:
            break

    selected.extend(
        feature
        for feature in valid_manual_additions
        if feature not in selected
    )

    # Fill the list when strict stability/group filters leave too few features.
    if len(selected) < LLM_MIN_FEATURES:
        for feature in ranking["feature"]:
            if feature not in selected and feature != target_id:
                selected.append(feature)
            if len(selected) >= LLM_MIN_FEATURES:
                break

    return selected[:LLM_MAX_FEATURES]


# ============================================================
# 13. Run the complete pipeline for one target
# ============================================================

def run_target_pipeline(
    data: ChallengeData,
    target_id: str,
) -> dict:
    """Run all feature-selection stages for one target question."""
    print(f"\n{'=' * 72}\nTarget: {target_id}\n{'=' * 72}")

    target_output_dir = OUTPUT_DIR / target_id
    target_output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = prepare_target_data(
        train=data.train,
        target_id=target_id,
        allowed_features=data.allowed_features,
    )

    # --------------------------------------------------------
    # Steps 1-2: all-feature model and LossFunctionChange
    # --------------------------------------------------------
    loss_ranking, screening_metrics = (
        compute_loss_function_change_screening(
            target_id=target_id,
            X=X,
            y=y,
            groups=groups,
            all_features=data.allowed_features,
        )
    )

    loss_ranking.to_csv(
        target_output_dir / "01_loss_function_change_ranking.csv",
        index=False,
    )
    screening_metrics.to_csv(
        target_output_dir / "01_screening_metrics.csv",
        index=False,
    )

    candidate_features = (
        loss_ranking.head(TOP_CANDIDATES)["feature"].tolist()
    )

    # --------------------------------------------------------
    # Steps 3-4: repeated OOF permutation importance
    # --------------------------------------------------------
    permutation_raw, permutation_metrics = (
        compute_oof_permutation_importance(
            target_id=target_id,
            X=X,
            y=y,
            groups=groups,
            all_features=data.allowed_features,
            candidate_features=candidate_features,
        )
    )

    permutation_raw.to_csv(
        target_output_dir / "02_permutation_importance_raw.csv",
        index=False,
    )
    permutation_metrics.to_csv(
        target_output_dir / "02_permutation_model_metrics.csv",
        index=False,
    )

    combined_ranking = aggregate_permutation_importance(
        permutation_df=permutation_raw,
        loss_change_ranking=loss_ranking,
    )

    # Add metadata for manual review.
    feature_metadata = data.features[
        ["variable", "question"]
    ].rename(columns={"variable": "feature"})

    combined_ranking = combined_ranking.merge(
        feature_metadata,
        on="feature",
        how="left",
    )

    # --------------------------------------------------------
    # Step 5: correlation/manual semantic groups
    # --------------------------------------------------------
    manual_groups = MANUAL_SEMANTIC_GROUPS_BY_TARGET.get(
        target_id,
        {},
    )

    feature_to_group, correlation_pairs = build_feature_groups(
        target_id=target_id,
        X=X,
        candidate_features=combined_ranking["feature"].tolist(),
        manual_groups=manual_groups,
    )

    combined_ranking["group_id"] = combined_ranking["feature"].map(
        feature_to_group
    )

    combined_ranking.to_csv(
        target_output_dir / "03_combined_feature_ranking.csv",
        index=False,
    )
    correlation_pairs.to_csv(
        target_output_dir / "03_correlation_pairs.csv",
        index=False,
    )

    raw_order = combined_ranking["feature"].tolist()
    grouped_order = make_group_limited_ranking(
        ranking=combined_ranking,
        feature_to_group=feature_to_group,
        max_per_group=1,
    )

    # --------------------------------------------------------
    # Step 6: evaluate top-k feature sets
    # --------------------------------------------------------
    feature_sets: dict[str, list[str]] = {
        "all_278": data.allowed_features,
    }

    for k in TOP_K_VALUES:
        feature_sets[f"raw_top_{k}"] = raw_order[:k]
        feature_sets[f"grouped_top_{k}"] = grouped_order[:k]

    topk_fold_results = []

    for feature_set_name, selected_features in feature_sets.items():
        if not selected_features:
            continue

        for scheme in ("seen", "unseen"):
            fold_df = evaluate_feature_set_cv(
                target_id=target_id,
                X=X,
                y=y,
                groups=groups,
                selected_features=selected_features,
                feature_set_name=feature_set_name,
                scheme=scheme,
            )
            topk_fold_results.append(fold_df)

    topk_fold_results_df = pd.concat(
        topk_fold_results,
        ignore_index=True,
    )

    topk_summary = summarize_topk_results(topk_fold_results_df)
    chosen_row = select_feature_set_one_se(topk_summary)

    chosen_feature_set_name = str(chosen_row["feature_set"])
    chosen_catboost_features = feature_sets[chosen_feature_set_name]

    topk_fold_results_df.to_csv(
        target_output_dir / "04_topk_fold_results.csv",
        index=False,
    )
    topk_summary.to_csv(
        target_output_dir / "04_topk_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Steps 8-9: compact LLM feature recommendation
    # --------------------------------------------------------
    llm_features = recommend_llm_features(
        target_id=target_id,
        ranking=combined_ranking,
        feature_to_group=feature_to_group,
        X=X,
        allowed_features=set(data.allowed_features),
        manual_additions=MANUAL_LLM_ADDITIONS.get(target_id, []),
    )

    llm_review = combined_ranking[
        combined_ranking["feature"].isin(llm_features)
    ].copy()

    llm_review["llm_order"] = llm_review["feature"].map(
        {feature: index + 1 for index, feature in enumerate(llm_features)}
    )

    llm_review = llm_review.sort_values("llm_order")
    llm_review.to_csv(
        target_output_dir / "05_llm_feature_review.csv",
        index=False,
    )

    result = {
        "target_id": target_id,
        "chosen_catboost_feature_set": chosen_feature_set_name,
        "chosen_catboost_features": chosen_catboost_features,
        "llm_features": llm_features,
        "chosen_combined_skill": float(chosen_row["combined_skill"]),
        "chosen_seen_skill": float(chosen_row["seen_skill"]),
        "chosen_unseen_skill": float(chosen_row["unseen_skill"]),
    }

    with (target_output_dir / "06_selected_features.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(
        f"Selected CatBoost set: {chosen_feature_set_name} "
        f"({len(chosen_catboost_features)} features)"
    )
    print(
        f"CV Skill: seen={result['chosen_seen_skill']:.3f}, "
        f"unseen={result['chosen_unseen_skill']:.3f}, "
        f"combined={result['chosen_combined_skill']:.3f}"
    )
    print(f"Recommended LLM features: {llm_features}")

    return result


# ============================================================
# 14. Main entry point
# ============================================================

def main() -> None:
    """Run the pipeline and save global JSON configurations."""
    data = load_challenge_data()

    targets_to_run = (
        TARGETS_TO_RUN
        if TARGETS_TO_RUN is not None
        else data.target_ids
    )

    invalid_targets = set(targets_to_run) - set(data.target_ids)
    if invalid_targets:
        raise ValueError(
            f"Unknown target IDs: {sorted(invalid_targets)}"
        )

    results = []

    for target_id in tqdm(targets_to_run, desc="Targets"):
        results.append(run_target_pipeline(data, target_id))

    catboost_config = {
        result["target_id"]: result["chosen_catboost_features"]
        for result in results
    }

    llm_config = {
        result["target_id"]: result["llm_features"]
        for result in results
    }

    with (OUTPUT_DIR / "catboost_features_by_target.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(catboost_config, file, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "llm_features_by_target.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(llm_config, file, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {
                "target_id": result["target_id"],
                "catboost_feature_set": result[
                    "chosen_catboost_feature_set"
                ],
                "catboost_n_features": len(
                    result["chosen_catboost_features"]
                ),
                "llm_n_features": len(result["llm_features"]),
                "seen_skill": result["chosen_seen_skill"],
                "unseen_skill": result["chosen_unseen_skill"],
                "combined_skill": result["chosen_combined_skill"],
            }
            for result in results
        ]
    ).to_csv(
        OUTPUT_DIR / "selection_summary.csv",
        index=False,
    )

    print("\nPipeline completed.")
    print(f"Outputs were written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
