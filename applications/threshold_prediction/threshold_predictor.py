import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def build_feature_matrix(
    trajectory_features: dict[str, np.ndarray] | None = None,
    distribution_features: dict[str, np.ndarray] | None = None,
    inter_neuron_features: dict[str, np.ndarray] | None = None,
    basic_features: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate all feature dicts into a single matrix.

    Also computes interaction features between groups.

    Args:
        trajectory_features: From NeuronTracker.compute_trajectory_features().
        distribution_features: From compute_distribution_features().
        inter_neuron_features: From compute_inter_neuron_features().
        basic_features: Previous experiment metrics (threshold_drift, etc.).

    Returns:
        (X, feature_names) where X is (n_neurons, n_features).
    """
    columns = []
    names = []

    for group_name, group in [
        ("trajectory", trajectory_features),
        ("distribution", distribution_features),
        ("inter_neuron", inter_neuron_features),
        ("basic", basic_features),
    ]:
        if group is None:
            continue
        for feat_name, values in group.items():
            columns.append(values.reshape(-1, 1))
            names.append(f"{group_name}.{feat_name}")

    if not columns:
        return np.empty((0, 0)), []

    X = np.hstack(columns)

    # Interaction features
    interactions = _compute_interactions(
        trajectory_features, distribution_features, inter_neuron_features
    )
    for feat_name, values in interactions.items():
        X = np.hstack([X, values.reshape(-1, 1)])
        names.append(f"interaction.{feat_name}")

    return X, names


def _compute_interactions(
    trajectory: dict[str, np.ndarray] | None,
    distribution: dict[str, np.ndarray] | None,
    inter_neuron: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Compute cross-group interaction features."""
    interactions = {}

    if inter_neuron and distribution:
        if "weight_overlap_mean" in inter_neuron and "v_kurtosis" in distribution:
            interactions["overlap_x_kurtosis"] = (
                inter_neuron["weight_overlap_mean"] * distribution["v_kurtosis"]
            )

    if trajectory and distribution:
        if "win_entropy" in trajectory and "v_quantile_99" in distribution:
            interactions["entropy_x_quantile99"] = (
                trajectory["win_entropy"] * distribution["v_quantile_99"]
            )
        if "threshold_integral" in trajectory and "v_mean" in distribution:
            interactions["integral_x_vmean"] = (
                trajectory["threshold_integral"] * distribution["v_mean"]
            )
        if "convergence_epoch" in trajectory and "weight_velocity_late" in trajectory:
            interactions["convergence_x_velocity"] = (
                trajectory["convergence_epoch"] * trajectory["weight_velocity_late"]
            )

    return interactions


def _create_models() -> dict:
    """Create model instances for evaluation."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "GBR": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "RF": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        ),
    }


def fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Fit multiple regression models with leave-one-seed-out cross-validation.

    Args:
        X: (n_samples, n_features) feature matrix.
        y: (n_samples,) target values (optimal absolute thresholds).
        groups: (n_samples,) group labels (seed IDs) for GroupKFold.
        feature_names: List of feature names.

    Returns:
        Dict with per_model results, best_model_name, and feature_importance.
    """
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    scaler = StandardScaler()
    models = _create_models()
    per_model = {}

    for model_name, model in tqdm(models.items(), desc="Fitting models", leave=False):
        fold_r2 = []
        fold_mae = []
        all_preds = np.zeros_like(y)

        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler_fold = StandardScaler()
            X_train_s = scaler_fold.fit_transform(X_train)
            X_val_s = scaler_fold.transform(X_val)

            model_clone = _clone_model(model)
            model_clone.fit(X_train_s, y_train)
            preds = model_clone.predict(X_val_s)

            fold_r2.append(r2_score(y_val, preds))
            fold_mae.append(mean_absolute_error(y_val, preds))
            all_preds[val_idx] = preds

        per_model[model_name] = {
            "r2_mean": float(np.mean(fold_r2)),
            "r2_std": float(np.std(fold_r2)),
            "mae_mean": float(np.mean(fold_mae)),
            "mae_std": float(np.std(fold_mae)),
            "predictions": all_preds.tolist(),
        }

    # Find best model by R²
    best_name = max(per_model, key=lambda k: per_model[k]["r2_mean"])

    # Permutation importance from best model (full data)
    X_scaled = scaler.fit_transform(X)
    best_model = _clone_model(models[best_name])
    best_model.fit(X_scaled, y)
    perm_imp = permutation_importance(
        best_model, X_scaled, y, n_repeats=10, random_state=42
    )

    feature_importance = {
        name: float(imp) for name, imp in zip(feature_names, perm_imp.importances_mean)
    }

    return {
        "per_model": per_model,
        "best_model_name": best_name,
        "feature_importance": feature_importance,
    }


def feature_group_ablation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """Train GBR using only subsets of features to assess group contribution.

    Groups are determined by the prefix before the first dot in feature names.

    Returns:
        List of dicts with keys: group, r2_mean, r2_std, mae_mean, mae_std.
    """
    # Identify feature groups
    group_map: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        group_name = name.split(".")[0]
        group_map.setdefault(group_name, []).append(i)

    # Add "all" group
    group_map["all"] = list(range(len(feature_names)))

    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    results = []
    pbar = tqdm(group_map.items(), desc="Feature ablation", leave=False)
    for group_name, feat_indices in pbar:
        pbar.set_postfix_str(f"{group_name} ({len(feat_indices)} features)")
        X_sub = X[:, feat_indices]

        fold_r2 = []
        fold_mae = []

        for train_idx, val_idx in gkf.split(X_sub, y, groups):
            X_train, X_val = X_sub[train_idx], X_sub[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
            model.fit(X_train_s, y_train)
            preds = model.predict(X_val_s)

            fold_r2.append(r2_score(y_val, preds))
            fold_mae.append(mean_absolute_error(y_val, preds))

        results.append(
            {
                "group": group_name,
                "n_features": len(feat_indices),
                "r2_mean": float(np.mean(fold_r2)),
                "r2_std": float(np.std(fold_r2)),
                "mae_mean": float(np.mean(fold_mae)),
                "mae_std": float(np.std(fold_mae)),
            }
        )

    return results


def _clone_model(model):
    """Create a fresh copy of a model with the same hyperparameters."""
    from sklearn.base import clone

    return clone(model)
