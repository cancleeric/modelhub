FROM python:3.11-slim

# 安裝系統依賴（Pillow / ultralytics 需要 libGL）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 -s /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8951", "--reload"]
