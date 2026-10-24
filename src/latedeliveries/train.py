"""Train a model fairly using only training data for decisions."""

from sklearn.model_selection import GridSearchCV

from latedeliveries.evaluation import candidates
from latedeliveries.pipeline import build_pipeline, training_folds


def make_grid_search(model_name):
    """Try at least 6 recipes with 2 settings for the chosen learner."""
    if model_name in ("Logistic regression", "Balanced logistic"):
        parameters = {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        }
    elif model_name == "Decision tree":
        parameters = {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [10, 25],
        }
    elif model_name == "Random forest":
        parameters = {
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [2, 8],
        }
    else:
        raise ValueError("Do not tune the dummy baseline")
    return GridSearchCV(
        build_pipeline(candidates()[model_name]),
        param_grid=parameters,
        scoring="roc_auc",
        cv=training_folds(),
        n_jobs=1,
        refit=True,
        error_score="raise",
    )


def choose_training_threshold(best_pipeline, train_x, train_y):
    """Ask each held-out TRAINING fold for a chance, never touch test."""
    from sklearn.model_selection import cross_val_predict

    from latedeliveries.evaluation import choose_threshold, cost_by_threshold

    oof_probabilities = cross_val_predict(
        best_pipeline,
        train_x,
        train_y,
        cv=training_folds(),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    cutoff = choose_threshold(train_y, oof_probabilities)
    return cutoff, cost_by_threshold(train_y, oof_probabilities)


def run_training():
    """Rebuild model, graphs' data, and reports from only the teacher's data."""
    import json
    from pathlib import Path

    import joblib
    import numpy as np
    import sklearn
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from latedeliveries.data import make_orders
    from latedeliveries.evaluation import (
        choose_model,
        compare_models,
        decision_cost,
    )
    from latedeliveries.pipeline import (
        CATEGORICAL_FEATURES,
        EXCLUDED_FEATURES,
        FEATURE_COLUMNS,
        NUMERIC_FEATURES,
        SEED,
        split_orders,
    )

    # This is the root of the REPOSITORY, not the current command's location.
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models"
    report_dir = root / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    orders = make_orders(seed=SEED)
    train_x, test_x, train_y, test_y = split_orders(orders)

    # Stages 2-4: all comparison/tuning/threshold choices use training ONLY.
    cv_table = compare_models(train_x, train_y)
    chosen_name = choose_model(cv_table)
    search = make_grid_search(chosen_name)
    search.fit(train_x, train_y)

    # These use the same five folds and are for interpretation, NOT test tuning.
    original_auc = cv_table.loc[cv_table["model"] == chosen_name, "roc_auc_mean"].iloc[
        0
    ]
    original_spread = cv_table.loc[
        cv_table["model"] == chosen_name, "roc_auc_std"
    ].iloc[0]
    tuned_auc = float(search.best_score_)
    gain = tuned_auc - float(original_auc)
    gain_is_bigger_than_fold_spread = gain > float(original_spread)

    # Stage 4: get held-out TRAINING-fold chances for the voucher threshold.
    threshold, threshold_table = choose_training_threshold(
        search.best_estimator_, train_x, train_y
    )

    # Stage 5: ONLY HERE do we score our final 750 held-out rows.
    final_pipeline = clone(search.best_estimator_)
    final_pipeline.fit(train_x, train_y)
    test_probabilities = final_pipeline.predict_proba(test_x)[:, 1]
    test_flags = (test_probabilities >= threshold).astype(int)
    cm = confusion_matrix(test_y, test_flags, labels=[0, 1])

    test_count = len(test_y)
    cost_none = decision_cost(test_y, np.zeros(test_count, dtype=bool))
    cost_all = decision_cost(test_y, np.ones(test_count, dtype=bool))
    cost_model = decision_cost(test_y, test_flags)
    metrics = {
        "train_orders": len(train_y),
        "test_orders": test_count,
        "train_late_rate": float(train_y.mean()),
        "test_late_rate": float(test_y.mean()),
        "selected_model": chosen_name,
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
        "accuracy": float(accuracy_score(test_y, test_flags)),
        "precision": float(precision_score(test_y, test_flags, zero_division=0)),
        "recall": float(recall_score(test_y, test_flags, zero_division=0)),
        "f1": float(f1_score(test_y, test_flags, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y, test_probabilities)),
        "cost_flag_nobody": cost_none,
        "cost_flag_everybody": cost_all,
        "cost_model": cost_model,
        "money_saved_vs_nobody": cost_none - cost_model,
        "money_saved_vs_everybody": cost_all - cost_model,
        "chosen_params": search.best_params_,
        "grid_scoring": "roc_auc",
        "grid_roc_auc_mean": tuned_auc,
        "untuned_cv_roc_auc_mean": float(original_auc),
        "untuned_cv_roc_auc_std": float(original_spread),
        "apparent_tuning_roc_auc_gain": float(gain),
        "tuning_gain_beats_untuned_fold_spread": bool(gain_is_bigger_than_fold_spread),
        "threshold_training_cost": int(threshold_table["training_cost"].min()),
    }

    # Stage 6: save the WHOLE fitted pipeline and the evidence.
    cv_table.to_csv(report_dir / "cv_results.csv", index=False, float_format="%.6f")
    threshold_table.to_csv(report_dir / "threshold_costs.csv", index=False)
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(final_pipeline, model_dir / "late_orders.joblib")
    model_info = {
        "package": "latedeliveries",
        "data_generator": "teacher's unedited make_orders(n=3000, seed=7)",
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_FEATURES,
        "target": "late",
        "model_name": chosen_name,
        "best_params": search.best_params_,
        "test_results": metrics,
    }
    (model_dir / "model_info.json").write_text(
        json.dumps(model_info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Chosen model: {chosen_name}")
    print(f"Best settings: {search.best_params_}")
    print(f"Best OOF training threshold: {threshold:.2f}")
    print(f"Test F1: {metrics['f1']:.3f}; test ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Test cost: none ${cost_none}, all ${cost_all}, model ${cost_model}")
    print("Saved models/ and reports/.")
    return metrics


def main():
    """Run all steps with python -m latedeliveries.train."""
    run_training()


if __name__ == "__main__":
    main()
