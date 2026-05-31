import asyncio
import logging
import time
import pandas as pd
import json

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import RegressionQualityMetric, ColumnDriftMetric

from src.serving.db import engine
from src.serving.metrics import (
    DRIFT_SCORE,
    DRIFT_DETECTED,
    DRIFT_DATASET_SHARE,
    MODEL_MAE,
    MODEL_RMSE,
    MODEL_R2,
    WINDOW_SAMPLE_COUNT,
    LAST_RUN_TIMESTAMP,
    LAST_RUN_DURATION,
    DRIFT_WASSERSTEIN
)

logger = logging.getLogger(__name__)

COLUMN_MAPPING = ColumnMapping(
    target="y_ground_truth",
    prediction="y_pred",
    numerical_features=["x"],
)

REFERENCE_DATA: pd.DataFrame | None = None


def load_reference() -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT x as x, y_prediction as y_pred, y_train as y_ground_truth FROM training_dataset",
        engine,
    )
    logger.info("Loaded reference: %d rows", len(df))
    return df


def fetch_current_window(minutes: int = 1) -> pd.DataFrame:
    if not isinstance(minutes, int) or minutes < 0:
        raise ValueError(f"minutes musi być nieujemnym int, dostałam: {minutes!r}")

    return pd.read_sql(
        f"""
        SELECT x as x, y_pred as y_pred, y_true as y_ground_truth
        FROM predictions
        WHERE timestamp >= NOW() - INTERVAL '{minutes}' MINUTE
        ORDER BY timestamp DESC
        LIMIT 5000
        """,
        engine,
    )

def compute_evidently(current_df: pd.DataFrame, reference_df: pd.DataFrame) -> dict:
    report = Report(metrics=[
        DataDriftPreset(stattest="ks", stattest_threshold=0.05),
        ColumnDriftMetric(column_name='x', stattest='wasserstein'),
        RegressionQualityMetric(),
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=COLUMN_MAPPING,
    )

    return report.as_dict()


def update_prometheus_metrics(result: dict) -> None:
    """
    Wyciagnij konkretne wartosci z dict Evidently i zaktualizuj Gauge'y.

    DataDriftPreset zwraca dwa metryki:
      - DatasetDriftMetric    -> share_of_drifted_columns, number_of_drifted_columns
      - DataDriftTable        -> drift_by_columns per kolumna

    RegressionPreset zwraca m.in.:
      - RegressionQualityMetric -> current.mean_abs_error, rmse, r2_score
      - RegressionErrorPlot, RegressionErrorDistribution (wykresy - ignorujemy)
      - RegressionAbsPercentageErrorPlot itp.
    """
    metrics_list = result.get("metrics", [])

    for m in metrics_list:
        metric_name = m.get("metric")
        r = m.get("result", {})

        if metric_name == "DatasetDriftMetric":
            share = r.get("share_of_drifted_columns", 0)
            DRIFT_DATASET_SHARE.set(share)
            logger.info("Dataset drift share: %.3f", share)

        elif metric_name == "DataDriftTable":
            drift_by_columns = r.get("drift_by_columns", {})
            for col_name, col_data in drift_by_columns.items():
                drift_score = col_data.get("drift_score", 0)
                drift_detected = col_data.get("drift_detected", False)

                DRIFT_SCORE.labels(column=col_name).set(drift_score)
                DRIFT_DETECTED.labels(column=col_name).set(
                    1 if drift_detected else 0
                )

        elif metric_name == "RegressionQualityMetric":
            current_metrics = r.get("current", {})
            if "mean_abs_error" in current_metrics:
                MODEL_MAE.set(current_metrics["mean_abs_error"])
            if "rmse" in current_metrics:
                MODEL_RMSE.set(current_metrics["rmse"])
            if "r2_score" in current_metrics:
                MODEL_R2.set(current_metrics["r2_score"])

        elif metric_name == "ColumnDriftMetric":
            column = r.get("column_name")
            drift_score = r.get("drift_score", 0)
            stattest = r.get("stattest_name", "").lower()
            if "wasserstein" in stattest:
                DRIFT_WASSERSTEIN.labels(column=column).set(drift_score)


async def drift_monitor_loop(
        interval_seconds: int = 30,
        window_minutes: int = 2,
        min_samples: int = 500,
):
    global REFERENCE_DATA

    if REFERENCE_DATA is None:
        REFERENCE_DATA = load_reference() #załaduj dane referencyjne z bazy

    logger.info(
        "Starting drift monitor: interval=%ds, window=%dmin, min_samples=%d",
        interval_seconds, window_minutes, min_samples,
    )

    while True:
        start = time.monotonic()
        try:
            current = fetch_current_window(minutes=window_minutes)
            WINDOW_SAMPLE_COUNT.set(len(current))

            if len(current) < min_samples:
                logger.debug(
                    "Skip: %d samples < min %d",
                    len(current), min_samples,
                )
            else:
                print(current, REFERENCE_DATA)
                result = compute_evidently(current, REFERENCE_DATA)
                print(json.dumps(result, indent=2, default=str))
                update_prometheus_metrics(result)
                LAST_RUN_TIMESTAMP.set(time.time())
                logger.info(
                    "Drift check done: %d samples, took %.2fs",
                    len(current), time.monotonic() - start,
                                  )

        except Exception:
            logger.exception("Drift monitor loop failed (will retry)")

        finally:
            LAST_RUN_DURATION.set(time.monotonic() - start)

        await asyncio.sleep(interval_seconds)