FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
  ffmpeg \
  curl \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY model.pt .
COPY app.py .
COPY inference.py .
COPY requirements.txt .
COPY model/ /app/model

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
