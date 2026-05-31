from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import matplotlib.pyplot as plt

"""
@dataclass z frozen=True jako odpowiednik record z Java.
obiekt immutable idealny do przenoszenia konfiguracji 
"""
@dataclass(frozen=True)
class Uniform:
    pass  # korzysta z x_min, x_max z SineConfig

@dataclass(frozen=True)
class Normal:
    mean: float
    std: float

@dataclass(frozen=True)
class Linspace:
    pass  # deterministyczny, równomierny rozkład

Distribution = Uniform | Normal | Linspace

@dataclass(frozen=True)
class SineConfig:
    A: float
    B: float
    x_min: float
    x_max: float
    distribution: Distribution = field(default_factory=Uniform)
    noise_std: float = 0.0
    n_samples: int = 1000
    random_seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def generate(config: SineConfig, visualisation_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    generuje syntentyczne dane odpowiadające funkcji: y = A·sin(B·x) + szum o rozkładzie Normalnym.
    próbki są równomiernie rozłożone od [x_min, x_max], dodawany jest szum o rozkładzie normalnym N(0, noise_std²).

    Args:
      config: SineConfig A, B, x range, noise level,
                sample count and optional random seed.

    Returns:
        X: Input features, shape (n_samples, 1). Reshaped to 2D for sklearn.
        y: Target values, shape (n_samples,).
    """
    rng = np.random.default_rng(config.random_seed)

    match config.distribution:
        case Uniform():
            x = rng.uniform(config.x_min, config.x_max, config.n_samples)
        case Normal(mean=m, std=s):
            x = rng.normal(m, s, config.n_samples)
            x = np.clip(x, config.x_min, config.x_max)
        case Linspace():
            x = np.linspace(config.x_min, config.x_max, config.n_samples)

    if config.noise_std > 0:
        noise = rng.normal(0.0, config.noise_std, config.n_samples)
    else:
        noise = 0.0

    y = config.A * np.sin(config.B * x) + noise

    if visualisation_path:
        x_smooth = np.linspace(config.x_min, config.x_max, 1000)
        y_smooth = config.A * np.sin(config.B * x_smooth)

        # Plot 1: dane treningowe
        plt.figure(figsize=(12, 4))
        plt.scatter(x, y, s=10, alpha=0.6, label="próbki (z szumem)")
        plt.plot(x_smooth, y_smooth, "r-", linewidth=2, label="A·sin(B·x)")
        plt.axvline(2 * np.pi, color="gray", linestyle="--", alpha=0.5, label="2π")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"y = {config.A}·sin({config.B}·x), n={config.n_samples}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(visualisation_path + "plot_dane.png", dpi=100)

        # Plot 2: rozkład x
        plt.figure(figsize=(8, 4))
        plt.hist(x, bins=30, edgecolor="black", alpha=0.7)
        plt.xlabel("x")
        plt.ylabel("liczba próbek")
        plt.title("Rozkład x (równomierny)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(visualisation_path + "plot_rozklad_x.png", dpi=100)

        # Plot 3: rozkład y
        plt.figure(figsize=(8, 4))
        plt.hist(y, bins=30, edgecolor="black", alpha=0.7, color="coral")
        plt.xlabel("y")
        plt.ylabel("liczba próbek")
        plt.title("Rozkład y")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(visualisation_path + "plot_rozklad_y.png", dpi=100)

        # Plot 4: rozkład szumu
        plt.figure(figsize=(8, 4))
        plt.hist(noise, bins=30, edgecolor="black", alpha=0.7, color="green")
        plt.xlabel("wartość szumu")
        plt.ylabel("liczba próbek")
        plt.title("Rozkład szumu")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(visualisation_path + "plot_rozklad_szumu.png", dpi=100)

    return x.reshape(-1, 1), y
