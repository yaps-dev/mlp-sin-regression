import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generator import SineConfig, generate

# === Konfiguracja ===
MODEL_VERSION = "v1"
ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = Path("training_plots")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# === Dane treningowe ===
config = SineConfig(
    A=2.0, B=2.0,
    x_min=0.0, x_max=4 * np.pi,
    noise_std=0.1, n_samples=500, random_seed=42,
)

X, y = generate(config, str(PLOTS_DIR) + "/")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

# === Trening ===
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        learning_rate_init=0.01,
        solver="adam",
        max_iter=2000,
        tol=1e-6,
        random_state=42,
        verbose=True,
    )),
])

model.fit(X_train, y_train)

# === Metryki — pipeline sam skaluje, nie podajemy "scaled" wersji ===
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

metrics = {
    "train_mse": float(mean_squared_error(y_train, y_pred_train)),
    "test_mse": float(mean_squared_error(y_test, y_pred_test)),
    "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
    "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
    "test_r2": float(r2_score(y_test, y_pred_test)),
}

# Dostęp do atrybutów regresora przez named_steps
regressor = model.named_steps['regressor']
training_info = {
    "n_iter": int(regressor.n_iter_),
    "final_loss": float(regressor.loss_),
}

print("=== Trening ===")
for k, v in {**training_info, **metrics}.items():
    print(f"{k:>15}: {v:.6f}" if isinstance(v, float) else f"{k:>15}: {v}")

# === Wykres ===
x_smooth = np.linspace(
    config.x_min - np.pi, config.x_max + np.pi, 1000
).reshape(-1, 1)
y_pred_smooth = model.predict(x_smooth)
y_true_smooth = config.A * np.sin(config.B * x_smooth.ravel())

plt.figure(figsize=(12, 4))
plt.scatter(X.ravel(), y, s=10, alpha=0.4, label="dane treningowe")
plt.plot(x_smooth, y_true_smooth, "r-", linewidth=2, label="prawda")
plt.plot(x_smooth, y_pred_smooth, "g--", linewidth=2, label="predykcja MLP")
plt.xlabel("x")
plt.ylabel("y")
plt.title("MLPRegressor ze skalowaniem")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "mlp_regressor.png", dpi=100)
plt.close()

# === Zapis modelu i metadanych ===
model_path = ARTIFACTS_DIR / f"sine_model_{MODEL_VERSION}.joblib"
metadata_path = ARTIFACTS_DIR / f"sine_model_{MODEL_VERSION}_metadata.json"

joblib.dump(model, model_path)

metadata = {
    "model_version": MODEL_VERSION,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "sklearn_version": sklearn.__version__,
    "training_config": {
        "A": config.A,
        "B": config.B,
        "x_min": config.x_min,
        "x_max": config.x_max,
        "noise_std": config.noise_std,
        "n_samples": config.n_samples,
        "random_seed": config.random_seed,
    },
    "model_architecture": {
        "type": "MLPRegressor",
        "hidden_layer_sizes": [32, 16],
        "activation": "tanh",
        "solver": "adam",
        "learning_rate_init": 0.01,
        "max_iter": 2000,
    },
    "training_info": training_info,
    "metrics": metrics,
    "input_schema": {
        "type": "float",
        "shape": [1],
        "description": "Argument funkcji A*sin(B*x), pojedyncza liczba rzeczywista",
    },
    "output_schema": {
        "type": "float",
        "shape": [1],
        "description": "Predykcja wrtosci funkcji",
    },
}

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)