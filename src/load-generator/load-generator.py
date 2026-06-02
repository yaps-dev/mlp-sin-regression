"""
Load generator: wysyła requesty do FastAPI, zbiera predykcje + ground truth, zapisuje do DB.
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import List
import httpx
import os
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from generator.generator import SineConfig, Uniform, Normal, generate

# ===== DATABASE SETUP =====

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    x = Column(Float, nullable=False)
    y_true = Column(Float, nullable=False)  # ground truth
    y_pred = Column(Float, nullable=False)  # model prediction

    def __repr__(self):
        return f"<Prediction(x={self.x:.3f}, y_true={self.y_true:.3f}, y_pred={self.y_pred:.3f})>"

db_user = os.getenv("DB_USER", "username")
db_user_password = os.getenv("DB_PASSWORD", "password")
db_addr = os.getenv("DB_ADDR", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_schema = os.getenv("DB_SCHEMA", "ml_data")

engine = create_engine(
    f'postgresql://{db_user}:{db_user_password}@{db_addr}:{db_port}/{db_schema}',
    echo=False
)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


class LoadGenerator:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def send_prediction_request(self, x: float) -> float:
        """Wysyła POST /predict, zwraca y_pred."""

        response = await self.client.post(
            f"{self.api_url}/predict",
            json={"x": x}
        )
        response.raise_for_status()
        data = response.json()
        return data['prediction']['y']

    async def run_batch(self, config: SineConfig, requests_per_second: float = 10.0):
        """
        Generuje dane z config, wysyła requesty, zapisuje do DB.

        Args:
            config: SineConfig z rozkładem (Uniform/Normal/Linspace)
            requests_per_second: tempo wysyłania requestów
        """
        # 1. Wygeneruj ground truth
        x_values, y_true_values = generate(config, "")  # bez wizualizacji

        distribution_type = config.distribution.__class__.__name__

        print(f"\n=== Starting load generation ===")
        print(f"Distribution: {distribution_type}")
        print(f"Samples: {len(x_values)}")
        print(f"Rate: {requests_per_second} req/s")
        print(f"Duration: ~{len(x_values) / requests_per_second:.1f}s\n")

        session = Session()

        delay = 1.0 / requests_per_second  # czas między requestami

        for i, (x, y_true) in enumerate(zip(x_values, y_true_values)):
            start = time.time()

            # 2. Wyślij request do modelu
            try:
                x = x.item()
                y_pred = await self.send_prediction_request(x)
                prediction = Prediction(
                    timestamp=datetime.now(timezone.utc),
                    x=float(x),
                    y_true=float(y_true),
                    y_pred=float(y_pred)
                )
                session.add(prediction)

                if (i + 1) % 100 == 0:  # commit co 10 rekordów
                    session.commit()
                    print(f"Sent {i + 1}/{len(x_values)} requests | "
                          f"x={x:.3f}, y_true={y_true:.3f}, y_pred={y_pred:.3f}, "
                          f"error={abs(y_true - y_pred):.4f}")

            except Exception as e:
                print(f"Error on request {i}: {e}")

            # 4. Rate limiting (czekaj do następnego requesta)
            elapsed = time.time() - start
            sleep_time = max(0, delay - elapsed)
            await asyncio.sleep(sleep_time)

        session.commit()
        session.close()

        print(f"\n=== Completed {len(x_values)} requests ===\n")

    async def close(self):
        await self.client.aclose()


async def main():
    generator = LoadGenerator(api_url="http://localhost:8000")

    # Phase 1: Baseline (dane o rozkładzie i wartościach identyczne z danymi treningowymi)
    # generator tworzy stałe obciążenie ~10 req/s
    baseline_config = SineConfig(
        A=2.0,
        B=2.0,
        x_min=0.0,
        x_max=12.566370614359172,
        distribution=Uniform(),
        #distribution=Normal(mean=5.0, std=1.5),
        noise_std=0.1,
        n_samples=100000,
        random_seed=1
    )

    await generator.run_batch(baseline_config, requests_per_second=10.85)

if __name__ == "__main__":
    asyncio.run(main())