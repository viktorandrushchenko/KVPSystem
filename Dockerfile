FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
RUN grep -vE '^torch([<>= ].*)?$' requirements.txt > /tmp/requirements-docker.txt \
    && pip install --no-cache-dir torch==2.6.0 --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY app ./app
COPY scripts ./scripts
COPY checkpoint-1 ./checkpoint-1
COPY README.md .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
