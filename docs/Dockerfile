FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch with CUDA 13.0 (cu130 wheel)
RUN pip install --no-cache-dir torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
