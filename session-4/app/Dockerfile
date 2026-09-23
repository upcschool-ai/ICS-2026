FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Install the CPU-only build of PyTorch: the default one includes ~2 GB of CUDA
# libraries that we don't need, and it makes the deploy much slower
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 0 main:app
