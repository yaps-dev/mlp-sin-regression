FROM python:3.12-slim

WORKDIR /app

# Skopiuj pyproject.toml
COPY pyproject.toml .

# Skopiuj src/ PRZED pip install (setuptools tego potrzebuje)
COPY src ./src
RUN echo "=== After copying src ===" && ls -la && ls -la src/

# Teraz dopiero zainstaluj
RUN pip install --no-cache-dir -e ".[drift]"
RUN echo "=== After pip install ===" && pip list

# Skopiuj model
COPY artifacts/sine_model_v1.joblib /app/artifacts/sine_model_v1.joblib
COPY artifacts/sine_model_v1_metadata.json /app/artifacts/sine_model_v1_metadata.json
RUN echo "=== After model ===" && ls -la /app/artifacts/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]