# FROM registry.gitlab.com/telecom-argentina/coo/smarthome/backend/smarthome-soundclass:latest
FROM python:3.11-slim-bookworm
 
RUN useradd -m dev
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y \
  ffmpeg \
  wget \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER dev

COPY . /app/
 
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
  
RUN wget -O model.pt http://tinyurl.com/5a3b7ec5
 
CMD ["python", "app.py"]

