FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# 先裝 CPU-only torch，避免 ultralytics 拉 CUDA wheels（~4GB+）
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# ── Non-root user（Aegis HIGH finding fix）─────────────────────────────────────
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
