FROM python:3.11-slim

# accept proxy args from docker-compose build args
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# make them available during build steps (pip) and runtime if needed
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]