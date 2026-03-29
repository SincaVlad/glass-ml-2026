import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    param_distributions: dict,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict]:
    """
    Trains a RandomForestRegressor using RandomizedSearchCV.

    Returns:
        best_model: the fitted estimator with best found params
        results: dict with train/test RMSE, R², and best params
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    results = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    results["best_params"] = search.best_params_

    return best_model, results


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluates a fitted model on train and test sets.

    Returns:
        dict with train_rmse, test_rmse, train_r2, test_r2
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return {
        "train_rmse": root_mean_squared_error(y_train, train_preds),
        "test_rmse": root_mean_squared_error(y_test, test_preds),
        "train_r2": r2_score(y_train, train_preds),
        "test_r2": r2_score(y_test, test_preds),
    }


def save_model(model, path: str | Path, metadata: dict | None = None) -> None:
    """
    Saves a model (and optional metadata) to disk using joblib.

    Args:
        model: fitted sklearn estimator
        path: where to save (e.g. "outputs/models/tg_rf.joblib")
        metadata: optional dict (e.g. feature names, RMSE) saved alongside
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "metadata": metadata or {}}
    joblib.dump(payload, path)
    print(f"Saved to {path}")


def load_model(path: str | Path) -> tuple:
    """
    Loads a model saved with save_model().

    Returns:
        (model, metadata) tuple
    """
    payload = joblib.load(path)
    return payload["model"], payload["metadata"]
