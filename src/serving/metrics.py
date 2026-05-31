from prometheus_client import Gauge

# === Drift na cechach (covariate) ===
DRIFT_SCORE = Gauge(
    "evidently_drift_score",
    "Drift score per column (KS statistic dla numerycznych)",
    ["column"],
)
DRIFT_DETECTED = Gauge(
    "evidently_drift_detected",
    "1 jesli wykryto drift dla kolumny, 0 w przeciwnym razie",
    ["column"],
)
DRIFT_DATASET_SHARE = Gauge(
    "evidently_drifted_columns_share",
    "Udzial kolumn z driftem (0-1)",
)
DRIFT_WASSERSTEIN = Gauge(
    "evidently_wasserstein_distance",
    "Dystans Wassersteina dla rozkladu cechy wejsciowej",
    ["column"],
)

# === Model performance (concept drift signal) ===
MODEL_MAE = Gauge("evidently_regression_mae", "MAE na biezacym oknie")
MODEL_RMSE = Gauge("evidently_regression_rmse", "RMSE na biezacym oknie")
MODEL_R2 = Gauge("evidently_regression_r2", "R2 na biezacym oknie")

# === Metadane ===
WINDOW_SAMPLE_COUNT = Gauge(
    "evidently_window_samples",
    "Liczba probek w biezacym oknie",
)
LAST_RUN_TIMESTAMP = Gauge(
    "evidently_last_run_timestamp",
    "Unix timestamp ostatniego pomyslnego runu",
)
LAST_RUN_DURATION = Gauge(
    "evidently_last_run_duration_seconds",
    "Czas trwania ostatniego runu (do debugowania)",
)