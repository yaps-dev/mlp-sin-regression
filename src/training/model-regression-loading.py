"""Załaduj model, użyj generatora do ewaluacji, policz MSE, zrób wykres."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generator import SineConfig, generate

MODEL_PATH = Path("../artifacts/sine_model_v1.joblib")
PLOT_PATH = Path("../artifacts/predictions.png")

# Załaduj model
model = joblib.load(MODEL_PATH)

# Wygeneruj dane ewaluacyjne (twój generator)
eval_config = SineConfig(
    A=2.0,
    B=2.0,
    x_min=0.0,
    x_max=4 * np.pi,
    noise_std=0.1,
    n_samples=200,
    random_seed=123,  # inny seed niż przy treningu
)
X_eval, y_true = generate(eval_config, "")  # "" = bez wizualizacji z generatora

# Predykcja + metryki
y_pred = model.predict(X_eval)

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE:  {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.6f}")
print(f"R²:   {r2:.6f}")

# Plot — X teraz jest losowe, więc sortujemy do linii predykcji
sort_idx = np.argsort(X_eval.ravel())
X_sorted = X_eval[sort_idx].ravel()
y_pred_sorted = y_pred[sort_idx]

# Czysta prawda na gęstej siatce (do gładkiej linii)
x_smooth = np.linspace(eval_config.x_min, eval_config.x_max, 500)
y_clean = eval_config.A * np.sin(eval_config.B * x_smooth)

plt.figure(figsize=(12, 4))
plt.scatter(X_eval.ravel(), y_true, s=12, alpha=0.4, label="dane testowe (z szumem)")
plt.plot(x_smooth, y_clean, "r-", linewidth=2, label="prawda (czysty sinus)")
plt.plot(X_sorted, y_pred_sorted, "g--", linewidth=2, label="predykcja modelu")
plt.axvline(2 * np.pi, color="gray", linestyle="--", alpha=0.4, label="2π (granica gęstości)")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Predykcje na danych z generatora — MSE={mse:.4f}, R²={r2:.4f}")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=100)
plt.close()

print(f"\nWykres zapisany: {PLOT_PATH}")